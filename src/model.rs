//! Core data types shared across the pipeline: documents, chunks, scored hits.
//!
//! The model mirrors `docling-rag`'s `Document`/`Chunk` shape, extended with a
//! [`ChunkKind`], an optional [`MediaRef`] and a [`Locator`] so image fragments
//! and provenance (page / bounding box) are first-class from day one — even
//! though Phase 0 only stores text chunks.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// A source asset, stored once with its metadata. Its text lives in [`Chunk`]s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Stable id (UUID v4, hyphenated) — primary key.
    pub id: String,
    /// Where the asset came from (file path, upload URI, …).
    pub source_uri: String,
    /// Human-readable title (first heading, or file stem).
    pub title: String,
    /// Content hash (sha256 hex) used to skip re-ingesting unchanged assets.
    pub hash: String,
    /// Arbitrary metadata (format, byte length, custom fields).
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// RFC-3339 creation timestamp.
    pub created_at: String,
}

impl Document {
    /// Build a document with a fresh UUID and the given content hash.
    pub fn new(
        source_uri: impl Into<String>,
        title: impl Into<String>,
        hash: impl Into<String>,
    ) -> Self {
        Document {
            id: new_id(),
            source_uri: source_uri.into(),
            title: title.into(),
            hash: hash.into(),
            metadata: serde_json::Value::Null,
            created_at: now_rfc3339(),
        }
    }

    pub fn with_metadata(mut self, meta: serde_json::Value) -> Self {
        self.metadata = meta;
        self
    }
}

/// What a fragment represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChunkKind {
    /// A passage of text (document text, OCR, or a caption in Phase 1).
    #[default]
    Text,
    /// An image / page raster (Phase 1).
    Image,
}

impl ChunkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ChunkKind::Text => "text",
            ChunkKind::Image => "image",
        }
    }
}

impl FromStr for ChunkKind {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "text" => Ok(ChunkKind::Text),
            "image" => Ok(ChunkKind::Image),
            other => Err(crate::Error::Store(format!("unknown chunk kind '{other}'"))),
        }
    }
}

/// A pointer to blobs on disk (relative to the configured blob root). Phase 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaRef {
    pub path: String,
    pub mime: String,
}

/// Where a fragment sits in its source (Phase 0: page for PDFs).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Locator {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f32; 4]>,
}

/// One retrievable fragment of a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Stable id (UUID v4, hyphenated) — primary key.
    pub id: String,
    /// Owning [`Document::id`].
    pub doc_id: String,
    /// 0-based position of this chunk within its document.
    pub ordinal: i64,
    /// The text (already includes any prepended heading context).
    pub text: String,
    /// Number of units (words/tokens) counted at chunk time.
    pub token_count: i64,
    /// Text or image fragment.
    #[serde(default)]
    pub kind: ChunkKind,
    /// Blob pointer for image fragments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media: Option<MediaRef>,
    /// Provenance within the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<Locator>,
    /// Arbitrary per-chunk metadata (heading path, …).
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// The embedding vector; `None` until embedded / when loaded without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

impl Chunk {
    /// Override the generated id (tests, deterministic re-ingest).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Construct a text chunk (without embedding).
    pub fn new(
        doc_id: impl Into<String>,
        ordinal: i64,
        text: impl Into<String>,
        token_count: i64,
    ) -> Self {
        Chunk {
            id: new_id(),
            doc_id: doc_id.into(),
            ordinal,
            text: text.into(),
            token_count,
            kind: ChunkKind::Text,
            media: None,
            locator: None,
            metadata: serde_json::Value::Null,
            embedding: None,
        }
    }
}

/// A retrieval hit: a chunk plus the score that ranked it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scored {
    pub chunk: Chunk,
    /// Higher is better (cosine similarity, negated BM25, or fused RRF score).
    pub score: f32,
}

impl Scored {
    pub fn new(chunk: Chunk, score: f32) -> Self {
        Scored { chunk, score }
    }
}

/// The retrieval strategy to run for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RetrievalMode {
    /// Dense vector (semantic) search only.
    Vector,
    /// Sparse keyword search only (SQLite FTS5 / BM25).
    Bm25,
    /// Vector + BM25, fused with Reciprocal Rank Fusion.
    Hybrid,
}

impl RetrievalMode {
    pub const ALL: [RetrievalMode; 3] = [
        RetrievalMode::Vector,
        RetrievalMode::Bm25,
        RetrievalMode::Hybrid,
    ];
}

impl std::fmt::Display for RetrievalMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            RetrievalMode::Vector => "vector",
            RetrievalMode::Bm25 => "bm25",
            RetrievalMode::Hybrid => "hybrid",
        };
        f.write_str(s)
    }
}

impl FromStr for RetrievalMode {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "vector" | "dense" | "semantic" => Ok(RetrievalMode::Vector),
            "bm25" | "keyword" | "sparse" | "fts" => Ok(RetrievalMode::Bm25),
            "hybrid" => Ok(RetrievalMode::Hybrid),
            other => Err(crate::Error::config(format!(
                "unknown retrieval mode '{other}'"
            ))),
        }
    }
}

/// Generate a fresh hyphenated UUID v4.
pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// sha256 hex digest of the given bytes — used for asset dedup.
pub fn content_hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

/// Current time as an RFC-3339 string (UTC), without a `chrono` dependency.
pub fn now_rfc3339() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => format_epoch_utc(d.as_secs()),
        Err(_) => "1970-01-01T00:00:00Z".to_string(),
    }
}

/// Minimal civil-date formatter (UTC), correct for all dates after the epoch.
fn format_epoch_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Howard Hinnant's civil_from_days algorithm.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// File stem of a name (`report.md` → `report`).
pub fn stem(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    base.rsplit_once('.')
        .map(|(s, _)| s)
        .unwrap_or(base)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_hex() {
        let h = content_hash(b"hello world");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(h, content_hash(b"hello world"));
        assert_ne!(h, content_hash(b"hello worlds"));
    }

    #[test]
    fn epoch_formats_known_dates() {
        assert_eq!(format_epoch_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_epoch_utc(1_609_459_200), "2021-01-01T00:00:00Z");
    }

    #[test]
    fn stem_and_modes() {
        assert_eq!(stem("/a/b/report.md"), "report");
        assert_eq!(
            "hybrid".parse::<RetrievalMode>().unwrap(),
            RetrievalMode::Hybrid
        );
        assert!("nope".parse::<RetrievalMode>().is_err());
        assert_eq!("image".parse::<ChunkKind>().unwrap(), ChunkKind::Image);
    }

    #[test]
    fn math_roundtrip() {
        let v = vec![0.25f32, -1.5, 3.0];
        let b = crate::math::to_bytes(&v);
        assert_eq!(crate::math::from_bytes(&b), v);
        assert!((crate::math::cosine(&v, &v) - 1.0).abs() < 1e-5);
    }
}
