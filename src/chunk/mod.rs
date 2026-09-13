//! Markdown-aware chunking.
//!
//! * `window` — a streaming sliding window over the converted Markdown.
//! * `hierarchical` / `hybrid` — docling's structure-aware chunkers, which walk
//!   the converted document tree (heading paths, whole lists, triplet tables).
//!   The hybrid variant needs a HuggingFace `tokenizer.json`
//!   (`DOCLING_CHUNK_TOKENIZER`, default `.models/chunk/tokenizer.json`).

use crate::Result;
use crate::config::{ChunkerKind, Config};
use crate::model::Chunk;

/// Splits converted Markdown (or a docling document) into retrievable chunks.
#[derive(Clone)]
pub struct Chunker {
    kind: ChunkerKind,
    size: usize,
    overlap: f32,
}

impl Chunker {
    pub fn from_config(cfg: &Config) -> Result<Self> {
        Ok(Chunker {
            kind: cfg.chunker,
            size: cfg.chunk_size.max(1),
            overlap: cfg.chunk_overlap,
        })
    }

    pub fn kind(&self) -> ChunkerKind {
        self.kind
    }

    /// Window-chunk a Markdown string.
    pub fn chunk_markdown(&self, doc_id: &str, markdown: &str) -> Result<Vec<Chunk>> {
        Ok(self.window(doc_id, markdown))
    }

    /// Structure-aware chunking over the docling document tree.
    #[cfg(feature = "docling-extract")]
    pub fn chunk_document(
        &self,
        doc_id: &str,
        doc: &docling::DoclingDocument,
    ) -> Result<Vec<Chunk>> {
        use docling::chunker::{HierarchicalChunker, HuggingFaceTokenizer, HybridChunker};

        let doc_chunks = match self.kind {
            ChunkerKind::Window => return self.chunk_markdown(doc_id, ""),
            ChunkerKind::Hierarchical => HierarchicalChunker.chunk(doc),
            ChunkerKind::Hybrid => {
                let path = std::env::var("DOCLING_CHUNK_TOKENIZER")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| format!("{}/chunk/tokenizer.json", ".models"));
                let tokenizer = HuggingFaceTokenizer::from_file(&path, self.size).map_err(|e| {
                    crate::Error::conversion(format!("chunk tokenizer '{path}': {e}"))
                })?;
                HybridChunker::new(tokenizer).chunk(doc)
            }
        };

        Ok(doc_chunks
            .into_iter()
            .enumerate()
            .map(|(i, dc)| {
                let mut chunk = Chunk::new(doc_id, i as i64, dc.text.clone(), 0);
                chunk.token_count = dc.text.split_whitespace().count().max(1) as i64;
                if let Some(headings) = dc.headings {
                    chunk.metadata = serde_json::json!({ "headings": headings });
                }
                chunk
            })
            .collect())
    }

    /// Fixed-size sliding window over words, never producing empty chunks.
    fn window(&self, doc_id: &str, markdown: &str) -> Vec<Chunk> {
        let words: Vec<&str> = markdown.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }
        let step = {
            let overlap_words = (self.size as f32 * self.overlap).round() as usize;
            self.size.saturating_sub(overlap_words).max(1)
        };

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut ordinal = 0i64;
        while start < words.len() {
            let end = (start + self.size).min(words.len());
            let text = words[start..end].join(" ");
            let token_count = (end - start) as i64;
            chunks.push(Chunk::new(doc_id, ordinal, text, token_count));
            ordinal += 1;
            if end == words.len() {
                break;
            }
            start += step;
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunker(size: usize, overlap: f32) -> Chunker {
        Chunker {
            kind: ChunkerKind::Window,
            size,
            overlap,
        }
    }

    #[test]
    fn window_covers_and_overlaps() {
        let md = "one two three four five six seven eight nine ten";
        let chunks = chunker(4, 0.5).window("d", md);
        // step = 4 - 2 = 2 -> windows [0..4],[2..6],[4..8],[6..10]
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].token_count, 4);
        assert_eq!(chunks[3].token_count, 4);
        assert_eq!(chunks[3].text, "seven eight nine ten");
    }

    #[test]
    fn empty_markdown_yields_no_chunks() {
        assert!(chunker(4, 0.0).window("d", "   \n  ").is_empty());
    }
}
