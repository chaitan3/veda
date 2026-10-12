//! Image embedding for cross-modal search.
//!
//! `siglip` runs SigLIP2 (a **shared image+text space**) in-process via ONNX:
//! the vision tower embeds image bytes and the text tower embeds a text query,
//! so text→image and image→image both reduce to a KNN in one space. `hash` is a
//! deterministic offline stub for tests. `off` disables the channel entirely
//! (no image vector table is created).

pub mod hash;

#[cfg(feature = "image-embed")]
pub mod eg2;
pub mod siglip;
use crate::config::{Config, ImageEmbedProvider};
use crate::{Error, Result};
use async_trait::async_trait;
use std::sync::Arc;

/// A source of embedding vectors in a shared image+text space.
#[async_trait]
pub trait ImageEmbedder: Send + Sync {
    /// Embed a batch of **images**, given as encoded bytes (PNG/JPEG/…), one
    /// vector per input, in order.
    async fn embed_images(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>>;

    /// Embed a batch of **text queries** into the same space.
    async fn embed_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Embed one text query.
    async fn embed_query_text(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self
            .embed_text(std::slice::from_ref(&text.to_string()))
            .await?;
        v.pop()
            .ok_or_else(|| Error::embedding("image embedder returned no vector"))
    }

    /// The dimensionality of every returned vector (0 if not yet known).
    fn dim(&self) -> usize;

    /// A short identifier (`"siglip:…"`, `"hash-image"`) for logs.
    fn id(&self) -> &str;
}

/// Build the image embedder selected by `cfg.image.provider`, if enabled.
pub fn from_config(cfg: &Config) -> Result<Option<Arc<dyn ImageEmbedder>>> {
    match cfg.image.provider {
        ImageEmbedProvider::Off => Ok(None),
        ImageEmbedProvider::Hash => Ok(Some(Arc::new(hash::HashImageEmbedder::new(cfg.image.dim)))),
        ImageEmbedProvider::Siglip => {
            #[cfg(feature = "image-embed")]
            {
                Ok(Some(Arc::new(siglip::SigLipEmbedder::from_config(cfg)?)))
            }
            #[cfg(not(feature = "image-embed"))]
            {
                Err(Error::FeatureDisabled(
                    "image embedder (LS_IMAGE_EMBED_PROVIDER=siglip)".into(),
                    "image-embed".into(),
                ))
            }
        }
        ImageEmbedProvider::Eg2 => {
            #[cfg(feature = "image-embed")]
            {
                Ok(Some(Arc::new(eg2::Eg2ImageEmbedder::from_config(cfg)?)))
            }
            #[cfg(not(feature = "image-embed"))]
            {
                Err(Error::FeatureDisabled(
                    "EG2 image embedder (LS_IMAGE_EMBED_PROVIDER=eg2)".into(),
                    "image-embed".into(),
                ))
            }
        }
    }
}
