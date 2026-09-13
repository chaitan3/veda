//! Per-document processing metrics, stored in the document's JSON metadata.
//!
//! Mirrors docling-rag's approach: metrics live under a `"metrics"` key in the
//! document metadata (a plain JSON object), so new fields need no migration.

use serde::{Deserialize, Serialize};

/// Busy-time timings for the parse / chunk / embed phases, in seconds.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Timings {
    pub parse_secs: f64,
    pub chunk_secs: f64,
    pub embed_secs: f64,
}

/// Processing metrics recorded for one ingested document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    pub file_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<usize>,
    pub words: usize,
    pub chunks: usize,
    pub embedded_words: usize,
    pub timings: Timings,
}

impl ProcessingMetrics {
    pub fn compute(
        file_bytes: u64,
        pages: Option<usize>,
        words: usize,
        chunks: usize,
        embedded_words: usize,
        timings: Timings,
    ) -> Self {
        ProcessingMetrics {
            file_bytes,
            pages,
            words,
            chunks,
            embedded_words,
            timings,
        }
    }

    /// Words processed per second during parsing.
    pub fn parse_words_per_sec(&self) -> f64 {
        rate(self.words as f64, self.timings.parse_secs)
    }

    /// Words embedded per second.
    pub fn embed_words_per_sec(&self) -> f64 {
        rate(self.embedded_words as f64, self.timings.embed_secs)
    }
}

fn rate(units: f64, secs: f64) -> f64 {
    if secs > 0.0 { units / secs } else { 0.0 }
}
