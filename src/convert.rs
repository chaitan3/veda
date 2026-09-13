//! Convert a source file to Markdown.
//!
//! With the `docling-extract` feature this calls docling.rs:
//!
//! * `LS_EXTRACT_PIPELINE=onnx` (default) — the local ONNX pipeline
//!   (layout + PP-OCR + TableFormer). PDF/image inputs use a **warm
//!   `docling_pdf::Pipeline`** held for the process, so models load once per
//!   ingest run rather than once per file. Native assets (`pdfium` + `.models/`)
//!   are resolved by docling, CWD-relative; `LS_MODELS_DIR` fans out to
//!   docling's per-model `DOCLING_*` variables.
//! * `LS_EXTRACT_PIPELINE=vlm` — docling's remote OpenAI-compatible VLM
//!   pipeline (`docling::vlm::convert_vlm`), configured through
//!   `DOCLING_RS_VLM_ENDPOINT` / `_MODEL` / `_PROMPT` / `_API_KEY`.
//!
//! Plain text/Markdown inputs bypass docling entirely (no model load).

use crate::config::{Config, ExtractPipeline};
use crate::{Error, Result};

/// Result of converting one source document.
pub struct Converted {
    pub markdown: String,
    pub pages: Option<usize>,
    /// The structured document, retained so docling's chunkers can walk it.
    #[cfg(feature = "docling-extract")]
    pub document: Option<docling::DoclingDocument>,
}

/// Convert `bytes` (named `name`) to Markdown.
pub fn convert(cfg: &Config, name: &str, bytes: &[u8]) -> Result<Converted> {
    let _ = &cfg; // used by the docling backend; silence when that feature is off
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();

    if matches!(ext.as_str(), "md" | "markdown" | "txt" | "text") {
        return Ok(Converted {
            markdown: String::from_utf8_lossy(bytes).to_string(),
            pages: None,
            #[cfg(feature = "docling-extract")]
            document: None,
        });
    }

    #[cfg(feature = "docling-extract")]
    {
        docling_backend::convert(cfg, name, &ext, bytes)
    }

    #[cfg(not(feature = "docling-extract"))]
    {
        let _ = ExtractPipeline::Onnx;
        Err(Error::FeatureDisabled(
            format!("converting '.{ext}' files"),
            "docling-extract".into(),
        ))
    }
}

#[cfg(feature = "docling-extract")]
mod docling_backend {
    use super::*;
    use docling::vlm::{self, VlmOptions};
    use docling::{DocumentConverter, InputFormat, SourceDocument};
    use std::sync::{Mutex, OnceLock};

    /// The process-wide warm PDF/image pipeline.
    ///
    /// `DocumentConverter` builds a fresh `docling_pdf::Pipeline` on every call,
    /// reloading all ONNX models each time; holding one amortizes that across
    /// the run. pdfium is not thread-safe, so access is serialized by the mutex
    /// (our ingestion is sequential anyway).
    fn ml_pipeline(cfg: &Config) -> Result<&'static Mutex<docling_pdf::Pipeline>> {
        static PIPELINE: OnceLock<Mutex<docling_pdf::Pipeline>> = OnceLock::new();
        if PIPELINE.get().is_none() {
            let pipeline = docling_pdf::Pipeline::new().map_err(|e| {
                Error::conversion(format!("initialising the docling PDF pipeline: {e}"))
            })?;
            // Opt-in enrichment passes (docling's do_code_enrichment /
            // do_formula_enrichment). A missing model warns once and is skipped,
            // so enabling is safe before the models are downloaded.
            let pipeline = if cfg.code_formula {
                pipeline.enrichments(docling_pdf::EnrichmentOptions {
                    picture_classification: false,
                    code: true,
                    formula: true,
                })
            } else {
                pipeline
            };
            tracing::info!(
                code_formula = cfg.code_formula,
                "docling PDF/image pipeline created; ONNX models stay warm for this process"
            );
            let _ = PIPELINE.set(Mutex::new(pipeline));
        }
        PIPELINE
            .get()
            .ok_or_else(|| Error::conversion("docling pipeline missing after init"))
    }

    fn lock_pipeline(
        cfg: &Config,
    ) -> Result<std::sync::MutexGuard<'static, docling_pdf::Pipeline>> {
        ml_pipeline(cfg)?
            .lock()
            .map_err(|_| Error::conversion("docling pipeline mutex poisoned"))
    }

    pub(super) fn convert(cfg: &Config, name: &str, ext: &str, bytes: &[u8]) -> Result<Converted> {
        let fmt = InputFormat::from_extension(ext)
            .ok_or_else(|| Error::Unsupported(format!("no docling backend for '.{ext}'")))?;
        let source = SourceDocument::from_bytes(name.to_string(), fmt, bytes.to_vec());

        let document = match cfg.extract {
            ExtractPipeline::Onnx => convert_local(cfg, &fmt, source)?,
            ExtractPipeline::Vlm => {
                let opts = VlmOptions::resolve(None, None)
                    .map_err(|e| Error::conversion(e.to_string()))?;
                vlm::convert_vlm(&source, &opts).map_err(|e| Error::conversion(e.to_string()))?
            }
        };

        let pages = page_count(&fmt, bytes);
        let markdown = document.export_to_markdown();
        Ok(Converted {
            markdown,
            pages,
            document: Some(document),
        })
    }

    /// Local ONNX conversion: the warm pipeline for PDF/image, the lightweight
    /// per-format backends (no ML models) for everything else.
    fn convert_local(
        cfg: &Config,
        fmt: &InputFormat,
        source: SourceDocument,
    ) -> Result<docling::DoclingDocument> {
        if matches!(fmt, InputFormat::Pdf) {
            lock_pipeline(cfg)?
                .convert(&source.bytes, None, &source.name)
                .map_err(|e| Error::conversion(format!("pdf: {e}")))
        } else if matches!(fmt, InputFormat::Image) {
            lock_pipeline(cfg)?
                .convert_image(&source.bytes, &source.name)
                .map_err(|e| Error::conversion(format!("image: {e}")))
        } else {
            DocumentConverter::new()
                .convert(source)
                .map(|r| r.document)
                .map_err(|e| Error::conversion(e.to_string()))
        }
    }

    fn page_count(fmt: &InputFormat, bytes: &[u8]) -> Option<usize> {
        if matches!(fmt, InputFormat::Pdf) {
            docling::pdf_page_count(bytes, None).ok()
        } else {
            None
        }
    }
}
