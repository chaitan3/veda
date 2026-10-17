//! Idle-evicting embedder wrappers.
//!
//! `serve` keeps the embedding models resident while queries are arriving and
//! releases them after a shared idle window, so a quiet service holds no model
//! weights. An in-flight call holds an `Arc` clone of the loaded model, so
//! dropping the slot never pulls a session out from under a query. Text and
//! image share one clock, so either kind of embedding call refreshes the window.
//!
//! The ingest CLI and the ingest child process never spawn a reaper, so their
//! models stay loaded for the whole run.

use crate::config::Config;
use crate::embed::image::{self, ImageEmbedder};
use crate::embed::{self, Embedder};
use crate::{Error, Result};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The shared idle window: one clock, touched by any embedding call.
pub struct IdleWindow {
    start: Instant,
    /// Millis since `start` at the last embedding call.
    last_used_ms: AtomicU64,
    /// `None` keeps the models for the process lifetime.
    timeout: Option<Duration>,
}

impl IdleWindow {
    pub fn new(timeout: Option<Duration>) -> Arc<Self> {
        Arc::new(IdleWindow {
            start: Instant::now(),
            last_used_ms: AtomicU64::new(0),
            timeout,
        })
    }

    /// Record an embedding call now.
    pub fn touch(&self) {
        let ms = self.start.elapsed().as_millis() as u64;
        self.last_used_ms.store(ms, Ordering::Relaxed);
    }

    /// Whether the window has been idle longer than the timeout.
    fn expired(&self) -> bool {
        match self.timeout {
            Some(timeout) => {
                let last = Duration::from_millis(self.last_used_ms.load(Ordering::Relaxed));
                self.start.elapsed().saturating_sub(last) >= timeout
            }
            None => false,
        }
    }
}

/// Lazily (re)loads and idle-drops a text embedder.
pub struct IdleEmbedder {
    cfg: Config,
    window: Arc<IdleWindow>,
    inner: tokio::sync::Mutex<Option<Arc<dyn Embedder>>>,
    id: String,
    dim: usize,
}

impl IdleEmbedder {
    /// Wrap an already-built embedder. `id`/`dim` are cached so they stay
    /// available (for logs and `/api/stats`) even when the model is unloaded.
    pub fn new(cfg: &Config, embedder: Arc<dyn Embedder>, window: Arc<IdleWindow>) -> Self {
        IdleEmbedder {
            cfg: cfg.clone(),
            window,
            id: embedder.id().to_string(),
            dim: embedder.dim(),
            inner: tokio::sync::Mutex::new(Some(embedder)),
        }
    }

    async fn get(&self) -> Result<Arc<dyn Embedder>> {
        let mut guard = self.inner.lock().await;
        if let Some(e) = guard.as_ref() {
            return Ok(e.clone());
        }
        let e = embed::from_config(&self.cfg)?;
        *guard = Some(e.clone());
        tracing::info!(embedder = %self.id, "embedding model loaded");
        Ok(e)
    }

    /// Whether the model is currently resident (used by tests; the idle
    /// reaper clears it and the next call reloads).
    pub async fn is_loaded(&self) -> bool {
        self.inner.lock().await.is_some()
    }

    /// Drop the model if the window expired and nothing is in flight.
    pub async fn reap(&self) -> bool {
        if !self.window.expired() {
            return false;
        }
        let mut guard = self.inner.lock().await;
        // Keep it if a query still holds a reference.
        if guard.as_ref().is_some_and(|e| Arc::strong_count(e) > 1) {
            return false;
        }
        if guard.take().is_some() {
            tracing::info!(embedder = %self.id, "embedding model released (idle)");
            return true;
        }
        false
    }
}

#[async_trait]
impl Embedder for IdleEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let e = self.get().await?;
        self.window.touch();
        e.embed(texts).await
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// Lazily (re)loads and idle-drops an image embedder.
pub struct IdleImageEmbedder {
    cfg: Config,
    window: Arc<IdleWindow>,
    inner: tokio::sync::Mutex<Option<Arc<dyn ImageEmbedder>>>,
    id: String,
    dim: usize,
}

impl IdleImageEmbedder {
    pub fn new(cfg: &Config, embedder: Arc<dyn ImageEmbedder>, window: Arc<IdleWindow>) -> Self {
        IdleImageEmbedder {
            cfg: cfg.clone(),
            window,
            id: embedder.id().to_string(),
            dim: embedder.dim(),
            inner: tokio::sync::Mutex::new(Some(embedder)),
        }
    }

    async fn get(&self) -> Result<Arc<dyn ImageEmbedder>> {
        let mut guard = self.inner.lock().await;
        if let Some(e) = guard.as_ref() {
            return Ok(e.clone());
        }
        let e =
            image::from_config(&self.cfg)?.ok_or_else(|| Error::config("image embedder is off"))?;
        *guard = Some(e.clone());
        tracing::info!(embedder = %self.id, "image model loaded");
        Ok(e)
    }

    /// Whether the model is currently resident (used by tests).
    pub async fn is_loaded(&self) -> bool {
        self.inner.lock().await.is_some()
    }

    pub async fn reap(&self) -> bool {
        if !self.window.expired() {
            return false;
        }
        let mut guard = self.inner.lock().await;
        if guard.as_ref().is_some_and(|e| Arc::strong_count(e) > 1) {
            return false;
        }
        if guard.take().is_some() {
            tracing::info!(embedder = %self.id, "image model released (idle)");
            return true;
        }
        false
    }
}

#[async_trait]
impl ImageEmbedder for IdleImageEmbedder {
    async fn embed_images(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
        let e = self.get().await?;
        self.window.touch();
        e.embed_images(images).await
    }

    async fn embed_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let e = self.get().await?;
        self.window.touch();
        e.embed_text(texts).await
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        &self.id
    }
}
