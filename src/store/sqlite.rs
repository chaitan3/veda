//! SQLite backend: documents + chunks tables, an FTS5 index for keyword search,
//! two `sqlite-vec` `vec0` tables (text and image spaces), and per-document
//! stage fingerprints.
//!
//! The `vec0` extension is statically linked (the `sqlite-vec` crate) and
//! registered via `sqlite3_auto_extension`, so every connection created after
//! [`register_sqlite_vec`] can `CREATE VIRTUAL TABLE … USING vec0`.
//!
//! `rusqlite` is synchronous; each method takes the connection lock and runs to
//! completion without awaiting, which is fine for a single-user local store.
//! WAL mode lets the out-of-process ingest CLI write while the service reads.

use super::VectorStore;
use crate::math;
use crate::model::{Chunk, ChunkKind, Document, Scored, now_rfc3339};
use crate::{Error, Result};
use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

/// SQLite-backed [`VectorStore`].
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    /// Text-space dimensionality.
    dim: usize,
    /// Image-space dimensionality, when the image channel is enabled.
    image_dim: Option<usize>,
}

const SCHEMA: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    source_uri  TEXT NOT NULL,
    title       TEXT NOT NULL,
    hash        TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT 'null',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash   ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_uri);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'text',
    metadata    TEXT NOT NULL DEFAULT 'null'
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS document_stages (
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (doc_id, stage)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

/// Register the statically linked `sqlite-vec` extension for all future
/// connections. Idempotent and thread-safe.
#[allow(clippy::missing_transmute_annotations)] // registration idiom from sqlite-vec
fn register_sqlite_vec() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is the extension's entry point, whose
        // signature matches `sqlite3_auto_extension`'s expected callback (the
        // same registration the sqlite-vec crate documents and tests).
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

impl SqliteStore {
    /// Open (creating if needed) the database at `path`, with a text vector
    /// index sized for `dim` and, optionally, an image vector index.
    pub async fn connect(path: &str, dim: usize, image_dim: Option<usize>) -> Result<Self> {
        register_sqlite_vec();
        if let Some(parent) = std::path::Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let conn = Connection::open(path)
            .map_err(|e| Error::store(format!("cannot open database {path}: {e}")))?;
        Ok(SqliteStore {
            conn: Arc::new(Mutex::new(conn)),
            dim: dim.max(1),
            image_dim: image_dim.map(|d| d.max(1)),
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|_| Error::store("database mutex poisoned"))
    }
}

fn read_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| {
            r.get(0)
        })
        .optional()?)
}

fn write_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

/// Columns selected for [`Document`] rows, in [`row_to_document`] order.
const DOCUMENT_COLS: &str = "id, source_uri, title, hash, metadata, created_at";

fn row_to_document(row: &rusqlite::Row<'_>) -> rusqlite::Result<Document> {
    let metadata: String = row.get(4)?;
    Ok(Document {
        id: row.get(0)?,
        source_uri: row.get(1)?,
        title: row.get(2)?,
        hash: row.get(3)?,
        metadata: serde_json::from_str(&metadata).unwrap_or(serde_json::Value::Null),
        created_at: row.get(5)?,
    })
}

fn row_to_chunk_base(row: &rusqlite::Row<'_>) -> rusqlite::Result<Chunk> {
    let kind: String = row.get(5)?;
    let metadata: String = row.get(6)?;
    Ok(Chunk {
        id: row.get(0)?,
        doc_id: row.get(1)?,
        ordinal: row.get(2)?,
        text: row.get(3)?,
        token_count: row.get(4)?,
        kind: kind.parse().unwrap_or(ChunkKind::Text),
        metadata: serde_json::from_str(&metadata).unwrap_or(serde_json::Value::Null),
        embedding: None,
    })
}

const CHUNK_COLS_QUALIFIED: &str = "chunks.id, chunks.doc_id, chunks.ordinal, chunks.text, chunks.token_count, chunks.kind, chunks.metadata";

#[async_trait]
impl VectorStore for SqliteStore {
    async fn migrate(&self) -> Result<()> {
        let conn = self.lock()?;
        conn.execute_batch(SCHEMA)
            .map_err(|e| Error::store(format!("migration failed: {e}")))?;

        // Text space. If the configured dimension differs from what the vec0
        // table was created for, recreate it and forget the text-embed stage.
        let stored = read_meta(&conn, "dim.text")?.and_then(|s| s.parse::<usize>().ok());
        if stored.is_some() && stored != Some(self.dim) {
            conn.execute_batch("DROP TABLE IF EXISTS chunk_vectors;")?;
            conn.execute("DELETE FROM document_stages WHERE stage = 'text_embed'", [])?;
        }
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                 chunk_id TEXT PRIMARY KEY,
                 embedding float[{}] distance_metric=cosine
             );",
            self.dim
        ))
        .map_err(|e| Error::store(format!("creating text vector index failed: {e}")))?;
        write_meta(&conn, "dim.text", &self.dim.to_string())?;

        // Image space, only when the channel is enabled.
        if let Some(image_dim) = self.image_dim {
            let stored = read_meta(&conn, "dim.image")?.and_then(|s| s.parse::<usize>().ok());
            if stored.is_some() && stored != Some(image_dim) {
                conn.execute_batch("DROP TABLE IF EXISTS image_vectors;")?;
                conn.execute(
                    "DELETE FROM document_stages WHERE stage = 'image_embed'",
                    [],
                )?;
            }
            conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS image_vectors USING vec0(
                     chunk_id TEXT PRIMARY KEY,
                     embedding float[{}] distance_metric=cosine
                 );",
                image_dim
            ))
            .map_err(|e| Error::store(format!("creating image vector index failed: {e}")))?;
            write_meta(&conn, "dim.image", &image_dim.to_string())?;
        }
        Ok(())
    }

    async fn upsert_document(&self, doc: &Document) -> Result<()> {
        let metadata = serde_json::to_string(&doc.metadata)?;
        self.lock()?.execute(
            "INSERT INTO documents (id, source_uri, title, hash, metadata, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO UPDATE SET
                source_uri = excluded.source_uri,
                title      = excluded.title,
                hash       = excluded.hash,
                metadata   = excluded.metadata",
            params![
                doc.id,
                doc.source_uri,
                doc.title,
                doc.hash,
                metadata,
                doc.created_at
            ],
        )?;
        Ok(())
    }

    async fn find_document_by_hash(&self, hash: &str) -> Result<Option<String>> {
        let conn = self.lock()?;
        let id = conn
            .query_row(
                "SELECT id FROM documents WHERE hash = ?1 LIMIT 1",
                params![hash],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
        Ok(id)
    }

    async fn replace_text_chunks(&self, doc_id: &str, chunks: &[Chunk]) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        {
            tx.execute(
                "DELETE FROM chunk_vectors WHERE chunk_id IN
                    (SELECT id FROM chunks WHERE doc_id = ?1 AND kind = 'text')",
                params![doc_id],
            )?;
            tx.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN
                    (SELECT id FROM chunks WHERE doc_id = ?1 AND kind = 'text')",
                params![doc_id],
            )?;
            tx.execute(
                "DELETE FROM chunks WHERE doc_id = ?1 AND kind = 'text'",
                params![doc_id],
            )?;
            let mut ins = tx.prepare(
                "INSERT OR REPLACE INTO chunks
                   (id, doc_id, ordinal, text, token_count, kind, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            let mut fts = tx.prepare("INSERT INTO chunks_fts (chunk_id, text) VALUES (?1, ?2)")?;
            for c in chunks {
                let metadata = serde_json::to_string(&c.metadata)?;
                ins.execute(params![
                    c.id,
                    c.doc_id,
                    c.ordinal,
                    c.text,
                    c.token_count,
                    c.kind.as_str(),
                    metadata
                ])?;
                fts.execute(params![c.id, c.text])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    async fn set_text_vectors(&self, pairs: &[(String, Vec<f32>)]) -> Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        {
            let mut del = tx.prepare("DELETE FROM chunk_vectors WHERE chunk_id = ?1")?;
            let mut ins =
                tx.prepare("INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?1, ?2)")?;
            for (id, vector) in pairs {
                if vector.len() != self.dim {
                    return Err(Error::store(format!(
                        "text embedding has {} dimensions but the index expects {} (LS_EMBED_DIM); \
                         re-ingest after fixing the embedder",
                        vector.len(),
                        self.dim
                    )));
                }
                del.execute(params![id])?;
                ins.execute(params![id, math::to_bytes(vector)])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    async fn text_chunks(&self, doc_id: &str) -> Result<Vec<Chunk>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(&format!(
            "SELECT {CHUNK_COLS_QUALIFIED} FROM chunks
             WHERE doc_id = ?1 AND kind = 'text' ORDER BY ordinal"
        ))?;
        let rows = stmt.query_map(params![doc_id], row_to_chunk_base)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    async fn count_text_chunks_for(&self, doc_id: &str) -> Result<usize> {
        let n: i64 = self.lock()?.query_row(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?1 AND kind = 'text'",
            params![doc_id],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }

    async fn upsert_image_chunk(&self, chunk: &Chunk, embedding: &[f32]) -> Result<()> {
        let image_dim = self.image_dim.ok_or_else(|| {
            Error::store("image chunks require an image vector space (LS_IMAGE_EMBED_PROVIDER)")
        })?;
        if embedding.len() != image_dim {
            return Err(Error::store(format!(
                "image embedding has {} dimensions but the index expects {} (LS_IMAGE_EMBED_DIM)",
                embedding.len(),
                image_dim
            )));
        }
        let metadata = serde_json::to_string(&chunk.metadata)?;
        let blob = math::to_bytes(embedding);
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        {
            tx.execute(
                "INSERT OR REPLACE INTO chunks
                   (id, doc_id, ordinal, text, token_count, kind, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    chunk.id,
                    chunk.doc_id,
                    chunk.ordinal,
                    chunk.text,
                    chunk.token_count,
                    chunk.kind.as_str(),
                    metadata
                ],
            )?;
            tx.execute(
                "DELETE FROM image_vectors WHERE chunk_id = ?1",
                params![chunk.id],
            )?;
            tx.execute(
                "INSERT INTO image_vectors(chunk_id, embedding) VALUES (?1, ?2)",
                params![chunk.id, blob],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    async fn vector_search(&self, query: &[f32], k: usize) -> Result<Vec<Scored>> {
        if query.len() != self.dim {
            return Err(Error::store(format!(
                "query vector has {} dimensions but the index expects {} (LS_EMBED_DIM)",
                query.len(),
                self.dim
            )));
        }
        let blob = math::to_bytes(query);
        let conn = self.lock()?;
        let mut stmt = conn.prepare(&format!(
            "SELECT {CHUNK_COLS_QUALIFIED}, v.distance
             FROM chunk_vectors v JOIN chunks ON chunks.id = v.chunk_id
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance"
        ))?;
        let rows = stmt.query_map(params![blob, k as i64], |row| {
            let distance: f64 = row.get(7)?;
            let chunk = row_to_chunk_base(row)?;
            // vec0's cosine distance is 1 - cosine similarity; report similarity.
            Ok(Scored::new(chunk, (1.0 - distance) as f32))
        })?;
        collect(rows)
    }

    async fn image_search(&self, query: &[f32], k: usize) -> Result<Vec<Scored>> {
        let image_dim = self
            .image_dim
            .ok_or_else(|| Error::store("image search requires an image vector space"))?;
        if query.len() != image_dim {
            return Err(Error::store(format!(
                "image query vector has {} dimensions but the index expects {} \
                 (LS_IMAGE_EMBED_DIM)",
                query.len(),
                image_dim
            )));
        }
        let blob = math::to_bytes(query);
        let conn = self.lock()?;
        let mut stmt = conn.prepare(&format!(
            "SELECT {CHUNK_COLS_QUALIFIED}, v.distance
             FROM image_vectors v JOIN chunks ON chunks.id = v.chunk_id
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance"
        ))?;
        let rows = stmt.query_map(params![blob, k as i64], |row| {
            let distance: f64 = row.get(7)?;
            let chunk = row_to_chunk_base(row)?;
            Ok(Scored::new(chunk, (1.0 - distance) as f32))
        })?;
        collect(rows)
    }

    async fn keyword_search(&self, query: &str, k: usize) -> Result<Vec<Scored>> {
        let Some(match_expr) = fts_match_expr(query) else {
            return Ok(Vec::new());
        };
        let conn = self.lock()?;
        let mut stmt = conn.prepare(&format!(
            "SELECT {CHUNK_COLS_QUALIFIED}, bm25(chunks_fts)
             FROM chunks_fts JOIN chunks ON chunks.id = chunks_fts.chunk_id
             WHERE chunks_fts MATCH ?1
             ORDER BY bm25(chunks_fts)
             LIMIT ?2"
        ))?;
        let rows = stmt.query_map(params![match_expr, k as i64], |row| {
            let rank: f64 = row.get(7)?;
            let chunk = row_to_chunk_base(row)?;
            Ok(Scored::new(chunk, (-rank) as f32))
        })?;
        collect(rows)
    }

    async fn count_chunks(&self) -> Result<usize> {
        let n: i64 = self
            .lock()?
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    async fn count_chunks_for(&self, doc_id: &str) -> Result<usize> {
        let n: i64 = self.lock()?.query_row(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?1",
            params![doc_id],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }

    async fn count_documents(&self) -> Result<usize> {
        let n: i64 = self
            .lock()?
            .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    async fn list_documents(&self) -> Result<Vec<Document>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(&format!(
            "SELECT {DOCUMENT_COLS} FROM documents ORDER BY created_at DESC"
        ))?;
        let rows = stmt.query_map([], row_to_document)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    async fn get_document(&self, doc_id: &str) -> Result<Option<Document>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(&format!(
            "SELECT {DOCUMENT_COLS} FROM documents WHERE id = ?1"
        ))?;
        let doc = stmt
            .query_row(params![doc_id], row_to_document)
            .optional()?;
        Ok(doc)
    }

    async fn get_documents_by_ids(&self, ids: &[String]) -> Result<HashMap<String, Document>> {
        let mut out = HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        let conn = self.lock()?;
        let placeholders = vec!["?"; ids.len()].join(",");
        let mut stmt = conn.prepare(&format!(
            "SELECT {DOCUMENT_COLS} FROM documents WHERE id IN ({placeholders})"
        ))?;
        let rows = stmt.query_map(rusqlite::params_from_iter(ids.iter()), row_to_document)?;
        for r in rows {
            let doc = r?;
            out.insert(doc.id.clone(), doc);
        }
        Ok(out)
    }

    async fn get_chunks_range(
        &self,
        doc_id: &str,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<Chunk>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(&format!(
            "SELECT {CHUNK_COLS_QUALIFIED} FROM chunks
             WHERE doc_id = ?1 ORDER BY ordinal LIMIT ?2 OFFSET ?3"
        ))?;
        let rows = stmt.query_map(
            params![doc_id, limit as i64, offset as i64],
            row_to_chunk_base,
        )?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<Chunk>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(&format!(
            "SELECT {CHUNK_COLS_QUALIFIED} FROM chunks WHERE id = ?1"
        ))?;
        let chunk = stmt
            .query_row(params![chunk_id], row_to_chunk_base)
            .optional()?;
        Ok(chunk)
    }

    async fn all_source_uris(&self) -> Result<Vec<String>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare("SELECT DISTINCT source_uri FROM documents")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    async fn delete_document(&self, doc_id: &str) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?1)",
            params![doc_id],
        )?;
        tx.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?1)",
            params![doc_id],
        )?;
        if self.image_dim.is_some() {
            tx.execute(
                "DELETE FROM image_vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?1)",
                params![doc_id],
            )?;
        }
        tx.execute("DELETE FROM chunks WHERE doc_id = ?1", params![doc_id])?;
        tx.execute(
            "DELETE FROM document_stages WHERE doc_id = ?1",
            params![doc_id],
        )?;
        tx.execute("DELETE FROM documents WHERE id = ?1", params![doc_id])?;
        tx.commit()?;
        Ok(())
    }

    async fn delete_documents_by_source(&self, source_uri: &str) -> Result<()> {
        let ids: Vec<String> = {
            let conn = self.lock()?;
            let mut stmt = conn.prepare("SELECT id FROM documents WHERE source_uri = ?1")?;
            let rows = stmt.query_map(params![source_uri], |r| r.get::<_, String>(0))?;
            let mut v = Vec::new();
            for r in rows {
                v.push(r?);
            }
            v
        };
        for id in ids {
            self.delete_document(&id).await?;
        }
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let conn = self.lock()?;
        conn.execute_batch(
            "DELETE FROM chunk_vectors; DELETE FROM chunks_fts; DELETE FROM chunks;
             DELETE FROM document_stages; DELETE FROM documents;",
        )?;
        if self.image_dim.is_some() {
            conn.execute("DELETE FROM image_vectors", [])?;
        }
        Ok(())
    }

    async fn corpus_version(&self) -> Result<i64> {
        let conn = self.lock()?;
        let v = read_meta(&conn, "corpus_version")?;
        Ok(v.and_then(|s| s.parse().ok()).unwrap_or(0))
    }

    async fn bump_corpus_version(&self) -> Result<i64> {
        let conn = self.lock()?;
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('corpus_version', '1')
             ON CONFLICT(key) DO UPDATE
               SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)",
            [],
        )?;
        let v = read_meta(&conn, "corpus_version")?.unwrap_or_else(|| "0".to_string());
        Ok(v.parse().unwrap_or(0))
    }

    async fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let conn = self.lock()?;
        read_meta(&conn, key)
    }

    async fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.lock()?;
        write_meta(&conn, key, value)
    }

    async fn document_stage(&self, doc_id: &str, stage: &str) -> Result<Option<String>> {
        let conn = self.lock()?;
        let fp = conn
            .query_row(
                "SELECT fingerprint FROM document_stages WHERE doc_id = ?1 AND stage = ?2",
                params![doc_id, stage],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
        Ok(fp)
    }

    async fn set_document_stage(&self, doc_id: &str, stage: &str, fingerprint: &str) -> Result<()> {
        self.lock()?.execute(
            "INSERT INTO document_stages (doc_id, stage, fingerprint, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(doc_id, stage) DO UPDATE SET
                fingerprint = excluded.fingerprint,
                updated_at  = excluded.updated_at",
            params![doc_id, stage, fingerprint, now_rfc3339()],
        )?;
        Ok(())
    }

    async fn clear_document_stage(&self, doc_id: &str, stage: &str) -> Result<()> {
        self.lock()?.execute(
            "DELETE FROM document_stages WHERE doc_id = ?1 AND stage = ?2",
            params![doc_id, stage],
        )?;
        Ok(())
    }

    async fn clear_stage_all(&self, stage: &str) -> Result<()> {
        self.lock()?.execute(
            "DELETE FROM document_stages WHERE stage = ?1",
            params![stage],
        )?;
        Ok(())
    }
}

fn collect(rows: impl Iterator<Item = rusqlite::Result<Scored>>) -> Result<Vec<Scored>> {
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Build an FTS5 MATCH expression from a free-text query.
///
/// Terms are quoted and OR-ed so partial matches are not lost to strict AND.
/// Returns `None` for an empty query.
fn fts_match_expr(query: &str) -> Option<String> {
    let terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_expr_quotes_terms() {
        assert_eq!(
            fts_match_expr("hello world").unwrap(),
            "\"hello\" OR \"world\""
        );
        assert!(fts_match_expr("   ").is_none());
        assert_eq!(fts_match_expr("a\"b").unwrap(), "\"a\" OR \"b\"");
    }
}
