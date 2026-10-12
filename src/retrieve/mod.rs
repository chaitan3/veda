//! Retrieval: dense vector, sparse keyword (FTS5/BM25), and Hybrid (RRF).

use crate::Result;
use crate::embed::Embedder;
use crate::embed::image::ImageEmbedder;
use crate::model::{RetrievalMode, Scored};
use crate::store::VectorStore;
use std::collections::HashMap;
use std::sync::Arc;

/// Runs retrieval modes over a store + embedder.
pub struct Retriever {
    store: Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    /// Image→text embedder; adds the image channel to hybrid search.
    image: Option<Arc<dyn ImageEmbedder>>,
    rrf_k: f32,
}

impl Retriever {
    pub fn new(store: Arc<dyn VectorStore>, embedder: Arc<dyn Embedder>, rrf_k: f32) -> Self {
        Retriever {
            store,
            embedder,
            image: None,
            rrf_k,
        }
    }

    /// Attach the image embedder so hybrid search also queries the image space.
    pub fn with_image(mut self, image: Option<Arc<dyn ImageEmbedder>>) -> Self {
        self.image = image;
        self
    }

    /// Retrieve the top `k` chunks for `query` under `mode`.
    pub async fn retrieve(
        &self,
        mode: RetrievalMode,
        query: &str,
        k: usize,
    ) -> Result<Vec<Scored>> {
        match mode {
            RetrievalMode::Vector => self.vector(query, k).await,
            RetrievalMode::Bm25 => self.store.keyword_search(query, k).await,
            RetrievalMode::Hybrid => self.hybrid(query, k).await,
            RetrievalMode::Image => self.image_text(query, k).await,
        }
    }

    /// Text query, image results only. In a unified space (EG2 both sides) the
    /// query is embedded by the *text* embedder and filtered to image chunks —
    /// one model, one space. With two spaces the image embedder's text tower
    /// embeds it instead.
    async fn image_text(&self, query: &str, k: usize) -> Result<Vec<Scored>> {
        if self.unified() {
            let q = self.embedder.embed_query(query).await?;
            return self.store.image_search(&q, k).await;
        }
        let image = self.image.as_ref().ok_or_else(|| {
            crate::Error::embedding(
                "image search needs an image embedder (LS_IMAGE_EMBED_PROVIDER is off)",
            )
        })?;
        let iq = image.embed_query_text(query).await?;
        self.store.image_search(&iq, k).await
    }

    /// Whether text and image chunks share one vector space.
    fn unified(&self) -> bool {
        self.store.image_space() == crate::store::ImageSpace::Unified
    }

    async fn vector(&self, query: &str, k: usize) -> Result<Vec<Scored>> {
        let q = self.embedder.embed_query(query).await?;
        self.store.vector_search(&q, k).await
    }

    async fn hybrid(&self, query: &str, k: usize) -> Result<Vec<Scored>> {
        // Widen each channel a little before fusing, so RRF has material to work with.
        let wide = (k * 4).max(k);
        let q = self.embedder.embed_query(query).await?;
        let dense = self.store.vector_search(&q, wide).await?;
        let sparse = self.store.keyword_search(query, wide).await?;
        let mut lists = vec![dense, sparse];
        // Two-space layouts need an explicit image channel: the text query,
        // embedded by the image model's text tower, against image chunks.
        // In a unified space the dense list already contains image chunks, so
        // no extra channel is needed (and the query is embedded once).
        if !self.unified()
            && let Some(image) = &self.image
        {
            let iq = image.embed_query_text(query).await?;
            lists.push(self.store.image_search(&iq, wide).await?);
        }
        Ok(rrf_fuse(&lists, self.rrf_k, k))
    }

    /// Image→image: embed the query image and search the image space.
    pub async fn retrieve_image(&self, image: &[u8], k: usize) -> Result<Vec<Scored>> {
        let embedder = self.image.as_ref().ok_or_else(|| {
            crate::Error::config(
                "image search needs an image embedder (LS_IMAGE_EMBED_PROVIDER is off)",
            )
        })?;
        let mut vectors = embedder.embed_images(&[image.to_vec()]).await?;
        let query = vectors
            .pop()
            .ok_or_else(|| crate::Error::embedding("image embedder returned no vector"))?;
        self.store.image_search(&query, k).await
    }
}

/// Reciprocal Rank Fusion: score(id) = Σ 1 / (k + rank).
///
/// Fusion is rank-based, so it is agnostic to the wildly different score scales
/// of cosine similarity and BM25 — which is what lets the two channels share a
/// single result list.
pub fn rrf_fuse(lists: &[Vec<Scored>], rrf_k: f32, k: usize) -> Vec<Scored> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    let mut chunks: HashMap<String, crate::Chunk> = HashMap::new();

    for list in lists {
        for (rank, scored) in list.iter().enumerate() {
            let contribution = 1.0 / (rrf_k + (rank as f32) + 1.0);
            *scores.entry(scored.chunk.id.clone()).or_insert(0.0) += contribution;
            chunks
                .entry(scored.chunk.id.clone())
                .or_insert_with(|| scored.chunk.clone());
        }
    }

    let mut fused: Vec<Scored> = scores
        .into_iter()
        .filter_map(|(id, score)| chunks.remove(&id).map(|chunk| Scored::new(chunk, score)))
        .collect();
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused.truncate(k);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Chunk;

    fn s(id: &str) -> Scored {
        Scored::new(Chunk::new("doc", 0, "x", 1).with_id(id), 1.0)
    }

    #[test]
    fn rrf_prefers_agreement() {
        let a = vec![s("a"), s("b")];
        let b = vec![s("b"), s("c")];
        let fused = rrf_fuse(&[a, b], 60.0, 3);
        assert_eq!(fused[0].chunk.id, "b", "item present in both lists wins");
        assert_eq!(fused.len(), 3);
    }
}
