//! Deterministic hashing embedder — offline smoke tests only.
//!
//! It is *not* semantic: identical text always yields the same vector, and
//! unrelated text is unrelated. Useful to exercise the pipeline without any
//! model or service.

use super::Embedder;
use crate::Result;
use crate::math::normalize;
use async_trait::async_trait;

/// Deterministic, dependency-free embedder.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        HashEmbedder { dim: dim.max(1) }
    }
}

fn seed(text: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    let out = h.finalize();
    let mut b = [0u8; 8];
    b.copy_from_slice(&out[..8]);
    let s = u64::from_le_bytes(b);
    if s == 0 { 0x9e37_79b9_7f4a_7c15 } else { s }
}

fn next(state: &mut u64) -> u64 {
    // xorshift64*
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

#[async_trait]
impl Embedder for HashEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut state = seed(t);
                let mut v = Vec::with_capacity(self.dim);
                for _ in 0..self.dim {
                    let x = next(&mut state);
                    // map to [-1, 1)
                    let f = ((x >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
                    v.push(f);
                }
                normalize(&mut v);
                v
            })
            .collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        "hash"
    }
}
