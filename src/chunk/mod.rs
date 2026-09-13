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
                // docling-rag parity: keep the heading path *and* the source item
                // refs (`#/texts/12`, `#/tables/0`, …). The refs are the pointers
                // that resolve to the document's per-item provenance (page/bbox).
                let doc_items: Vec<String> =
                    dc.doc_items.iter().map(|d| d.self_ref.clone()).collect();
                chunk.metadata = serde_json::json!({
                    "headings": dc.headings,
                    "doc_items": doc_items,
                });
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

    /// docling-rag parity: docling chunks carry both the heading path and the
    /// source item refs (`self_ref`s) in metadata.
    #[cfg(feature = "docling-extract")]
    #[test]
    fn docling_chunks_store_headings_and_doc_items() {
        let md = "# Guide\n\n## Setup\n\nInstall the tools.\n\n- clone\n- build\n";
        let src = docling::SourceDocument::from_bytes(
            "t.md",
            docling::InputFormat::Md,
            md.as_bytes().to_vec(),
        );
        let doc = docling::DocumentConverter::new()
            .convert(src)
            .expect("convert")
            .document;

        let chunker = Chunker {
            kind: ChunkerKind::Hierarchical,
            size: 300,
            overlap: 0.0,
        };
        let chunks = chunker.chunk_document("d", &doc).expect("chunk");
        assert!(!chunks.is_empty());

        let setup = chunks
            .iter()
            .find(|c| c.text.contains("Install"))
            .expect("setup chunk");
        assert_eq!(setup.metadata["headings"][1], "Setup");
        assert!(
            setup.metadata["doc_items"][0]
                .as_str()
                .unwrap()
                .starts_with("#/"),
            "doc_items should hold the source self_refs: {}",
            setup.metadata
        );
    }
}
