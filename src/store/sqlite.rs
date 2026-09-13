//! SQLite backend: documents + chunks tables, an FTS5 index for keyword search,
//! and a `sqlite-vec` `vec0` virtual table for KNN vector search.
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
use crate::model::{Chunk, ChunkKind, Document, Locator, MediaRef, Scored};
use crate::{Error, Result};
use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};
use std::sync::{Arc, Mutex, Once};

/// SQLite-backed [`VectorStore`].
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    /// Embedding dimensionality the `vec0` table was created for.
    dim: usize,
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
    media       TEXT,
    locator     TEXT,
    metadata    TEXT NOT NULL DEFAULT 'null',
    embedding   BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
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
    /// Open (creating if needed) the database at `path`, with a vector index
    /// sized for `dim`-dimensional embeddings.
    pub async fn connect(path: &str, dim: usize) -> Result<Self> {
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
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|_| Error::store("database mutex poisoned"))
    }

    /// Rebuild the `vec0` index from the embeddings stored in `chunks` when the
    /// two are out of sync (e.g. after enabling sqlite-vec on an existing DB).
    /// Best-effort: a dimensionality mismatch leaves the index empty and warns.
    fn backfill_vectors(conn: &Connection) -> Result<()> {
        let stored: i64 = conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        let indexed: i64 =
            conn.query_row("SELECT COUNT(*) FROM chunk_vectors", [], |r| r.get(0))?;
        if stored == indexed {
            return Ok(());
        }
        conn.execute("DELETE FROM chunk_vectors", [])?;
        let rows: Vec<(String, Vec<u8>)> = {
            let mut stmt =
                conn.prepare("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL")?;
            let mapped = stmt.query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
            })?;
            let mut out = Vec::new();
            for row in mapped {
                out.push(row?);
            }
            out
        };
        let mut insert =
            conn.prepare("INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?1, ?2)")?;
        for (id, embedding) in rows {
            insert.execute(params![id, embedding])?;
        }
        Ok(())
    }
}

fn parse_locator(raw: Option<String>) -> Option<Locator> {
    raw.and_then(|s| serde_json::from_str(&s).ok())
}

fn parse_media(raw: Option<String>) -> Option<MediaRef> {
    raw.and_then(|s| serde_json::from_str(&s).ok())
}

fn row_to_chunk_base(row: &rusqlite::Row<'_>) -> rusqlite::Result<Chunk> {
    let kind: String = row.get(5)?;
    let media: Option<String> = row.get(6)?;
    let locator: Option<String> = row.get(7)?;
    let metadata: String = row.get(8)?;
    Ok(Chunk {
        id: row.get(0)?,
        doc_id: row.get(1)?,
        ordinal: row.get(2)?,
        text: row.get(3)?,
        token_count: row.get(4)?,
        kind: kind.parse().unwrap_or(ChunkKind::Text),
        media: parse_media(media),
        locator: parse_locator(locator),
        metadata: serde_json::from_str(&metadata).unwrap_or(serde_json::Value::Null),
        embedding: None,
    })
}

const CHUNK_COLS_QUALIFIED: &str = "chunks.id, chunks.doc_id, chunks.ordinal, chunks.text, chunks.token_count, chunks.kind, chunks.media, chunks.locator, chunks.metadata";

#[async_trait]
impl VectorStore for SqliteStore {
    async fn migrate(&self) -> Result<()> {
        let conn = self.lock()?;
        conn.execute_batch(SCHEMA)
            .map_err(|e| Error::store(format!("migration failed: {e}")))?;
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                 chunk_id TEXT PRIMARY KEY,
                 embedding float[{}] distance_metric=cosine
             );",
            self.dim
        ))
        .map_err(|e| Error::store(format!("creating vector index failed: {e}")))?;
        if let Err(e) = Self::backfill_vectors(&conn) {
            tracing::warn!(
                error = %e,
                "vector index is out of sync and could not be rebuilt; re-ingest to populate it"
            );
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

    async fn insert_chunks(&self, chunks: &[Chunk]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        {
            let mut ins = tx.prepare(
                "INSERT OR REPLACE INTO chunks
                   (id, doc_id, ordinal, text, token_count, kind, media, locator, metadata, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;
            let mut fts = tx.prepare("INSERT INTO chunks_fts (chunk_id, text) VALUES (?1, ?2)")?;
            let mut vec_del = tx.prepare("DELETE FROM chunk_vectors WHERE chunk_id = ?1")?;
            let mut vec_ins =
                tx.prepare("INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?1, ?2)")?;
            for c in chunks {
                let media = match &c.media {
                    Some(m) => Some(serde_json::to_string(m)?),
                    None => None,
                };
                let locator = match &c.locator {
                    Some(l) => Some(serde_json::to_string(l)?),
                    None => None,
                };
                let metadata = serde_json::to_string(&c.metadata)?;
                let embedding = c.embedding.as_deref().map(math::to_bytes);
                if let Some(blob) = &embedding
                    && blob.len() != self.dim * 4
                {
                    return Err(Error::store(format!(
                        "embedding has {} dimensions but the index expects {} (LS_EMBED_DIM); \
                         re-ingest after fixing the embedder",
                        blob.len() / 4,
                        self.dim
                    )));
                }
                ins.execute(params![
                    c.id,
                    c.doc_id,
                    c.ordinal,
                    c.text,
                    c.token_count,
                    c.kind.as_str(),
                    media,
                    locator,
                    metadata,
                    embedding
                ])?;
                // Refresh the FTS and vector rows for this chunk id.
                fts.execute(params![c.id, c.text])?;
                if let Some(blob) = embedding {
                    vec_del.execute(params![c.id])?;
                    vec_ins.execute(params![c.id, blob])?;
                }
            }
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
            let distance: f64 = row.get(9)?;
            let chunk = row_to_chunk_base(row)?;
            // vec0's cosine distance is 1 - cosine similarity; report similarity.
            Ok(Scored::new(chunk, (1.0 - distance) as f32))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
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
            let rank: f64 = row.get(9)?;
            let chunk = row_to_chunk_base(row)?;
            Ok(Scored::new(chunk, (-rank) as f32))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
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
        let mut stmt = conn.prepare(
            "SELECT id, source_uri, title, hash, metadata, created_at
             FROM documents ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            let metadata: String = row.get(4)?;
            Ok(Document {
                id: row.get(0)?,
                source_uri: row.get(1)?,
                title: row.get(2)?,
                hash: row.get(3)?,
                metadata: serde_json::from_str(&metadata).unwrap_or(serde_json::Value::Null),
                created_at: row.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
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
        tx.execute("DELETE FROM chunks WHERE doc_id = ?1", params![doc_id])?;
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
            "DELETE FROM chunk_vectors; DELETE FROM chunks_fts; DELETE FROM chunks; DELETE FROM documents;",
        )?;
        Ok(())
    }

    async fn corpus_version(&self) -> Result<i64> {
        let conn = self.lock()?;
        let v: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'corpus_version'",
                [],
                |r| r.get(0),
            )
            .optional()?;
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
        let v: String = conn.query_row(
            "SELECT value FROM meta WHERE key = 'corpus_version'",
            [],
            |r| r.get(0),
        )?;
        Ok(v.parse().unwrap_or(0))
    }

    async fn pipeline_version(&self) -> Result<u32> {
        let conn = self.lock()?;
        let v: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'pipeline_version'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        Ok(v.and_then(|s| s.parse().ok()).unwrap_or(0))
    }

    async fn set_pipeline_version(&self, version: u32) -> Result<()> {
        self.lock()?.execute(
            "INSERT INTO meta (key, value) VALUES ('pipeline_version', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![version.to_string()],
        )?;
        Ok(())
    }
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
