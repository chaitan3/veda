//! Deterministic hash image embedder — offline smoke tests only.
//!
//! Mirrors the text [`super::super::hash::HashEmbedder`]: no semantics, just a
//! stable expansion of a sha256 digest into `dim` normalized floats, so the
//! image store/retrieval paths can be tested without any model files.

use super::ImageEmbedder;
use crate::Result;
use async_trait::async_trait;
use sha2::{Digest, Sha256};

/// Image embedder that hashes its input bytes.
pub struct HashImageEmbedder {
    dim: usize,
    id: String,
}

impl HashImageEmbedder {
    pub fn new(dim: usize) -> Self {
        let dim = dim.max(1);
        HashImageEmbedder {
            dim,
            id: format!("hash-image[{dim}]"),
        }
    }

    fn vector(&self, bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dim);
        let mut counter = 0u64;
        while out.len() < self.dim {
            let mut hasher = Sha256::new();
            hasher.update(bytes);
            hasher.update(counter.to_le_bytes());
            for chunk in hasher.finalize().as_chunks::<4>().0 {
                if out.len() == self.dim {
                    break;
                }
                let mut b = [0u8; 4];
                b.copy_from_slice(chunk);
                out.push(u32::from_le_bytes(b) as f32 / u32::MAX as f32 - 0.5);
            }
            counter += 1;
        }
        crate::math::normalize(&mut out);
        out
    }
}

#[async_trait]
impl ImageEmbedder for HashImageEmbedder {
    async fn embed_images(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
        Ok(images.iter().map(|b| self.vector(b)).collect())
    }

    async fn embed_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.vector(t.as_bytes())).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deterministic_and_normalized() {
        let e = HashImageEmbedder::new(64);
        let a = e.embed_images(&[b"img".to_vec()]).await.unwrap();
        let b = e.embed_images(&[b"img".to_vec()]).await.unwrap();
        assert_eq!(a, b);
        assert_eq!(a[0].len(), 64);
        let norm: f32 = a[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}
