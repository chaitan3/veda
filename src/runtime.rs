//! ONNX execution-provider selection, reporting, and docling runtime defaults.
//!
//! docling's default when a GPU provider is compiled in is `auto` — but ORT's
//! `auto` only falls back to CPU when the provider fails to *register*. If the
//! CUDA provider registers and then fails to load its runtime libraries lazily
//! (the common `libcudnn.so` case), the kernel errors out and there is no
//! fallback. So before docling caches its choice we probe for the CUDA runtime
//! ourselves and, when it is missing, pin the provider to CPU and say so.
//!
//! We also apply two docling defaults the operator did not set:
//! - `DOCLING_RS_OCR_ORIENTATION=off` (orientation detection re-runs the
//!   recognizer on rotated hypotheses and is a large share of OCR time),
//! - on CUDA, `DOCLING_RS_PDF_WORKERS=1` (parallel page workers each build
//!   their own CUDA sessions and exhaust GPU memory).

/// Resolve and report the execution provider, disabling CUDA up front when its
/// runtime libraries are not loadable.
pub fn log_execution_provider() {
    #[cfg(not(any(feature = "docling-extract", feature = "onnx-embed")))]
    {
        tracing::info!("ONNX backends not compiled in (no extraction/embedding)");
    }

    #[cfg(any(feature = "docling-extract", feature = "onnx-embed"))]
    {
        let forced_to_cpu = cuda_forced_cpu();

        let choice = docling_onnx::choice();
        apply_runtime_defaults(choice);
        if forced_to_cpu {
            tracing::info!(resolved = ?choice, "ONNX execution provider (CPU fallback)");
        } else {
            match std::env::var("DOCLING_RS_EP")
                .ok()
                .filter(|v| !v.trim().is_empty())
            {
                Some(requested) => {
                    tracing::info!(requested = %requested, resolved = ?choice, "ONNX execution provider")
                }
                None => tracing::info!(
                    resolved = ?choice,
                    "ONNX execution provider (DOCLING_RS_EP unset; `auto` prefers CUDA, CPU otherwise)"
                ),
            }
        }

        #[cfg(feature = "cuda")]
        if !matches!(choice, docling_onnx::Ep::Cpu) {
            probe_cuda();
        }

        #[cfg(not(feature = "cuda"))]
        tracing::info!("built without the 'cuda' feature; ONNX runs on CPU");
    }
}

/// Apply docling defaults the operator did not set. Runs at startup, before
/// docling reads (and caches) them.
#[cfg(any(feature = "docling-extract", feature = "onnx-embed"))]
fn apply_runtime_defaults(choice: docling_onnx::Ep) {
    // Orientation detection re-runs the OCR recognizer over rotated hypotheses —
    // a large share of OCR time — for pages that are rarely rotated. Off by
    // default; set `DOCLING_RS_OCR_ORIENTATION=auto` to re-enable.
    set_default("DOCLING_RS_OCR_ORIENTATION", "off");

    // The default parallel page workers each build their own CUDA sessions and
    // exhaust GPU memory (cuBLAS allocation failure / OOM). A single serial
    // worker is the reliable default on CUDA; override with
    // `DOCLING_RS_PDF_WORKERS` if the GPU has headroom.
    if !matches!(choice, docling_onnx::Ep::Cpu) {
        set_default("DOCLING_RS_PDF_WORKERS", "1");
    }
}

/// Set a docling variable only when the operator has not already set it.
#[cfg(any(feature = "docling-extract", feature = "onnx-embed"))]
fn set_default(name: &str, value: &str) {
    if std::env::var_os(name).is_none() {
        // SAFETY: startup-time, single-threaded; set before docling reads it.
        unsafe { std::env::set_var(name, value) };
        tracing::info!(var = name, value, "applying docling default");
    }
}

/// Whether the resolved execution provider is a GPU. Used to pick GPU-safe
/// defaults (code/formula enrichment off, since CodeFormulaV2's fp32 models do
/// not fit alongside the rest of the pipeline on a modest card).
pub fn gpu_selected() -> bool {
    #[cfg(any(feature = "docling-extract", feature = "onnx-embed"))]
    {
        !matches!(docling_onnx::choice(), docling_onnx::Ep::Cpu)
    }
    #[cfg(not(any(feature = "docling-extract", feature = "onnx-embed")))]
    {
        false
    }
}

/// True when CUDA was compiled in but its runtime is unusable, so the provider
/// was pinned to CPU.
#[cfg(feature = "cuda")]
fn cuda_forced_cpu() -> bool {
    maybe_disable_cuda()
}

#[cfg(not(feature = "cuda"))]
fn cuda_forced_cpu() -> bool {
    false
}

/// If CUDA is compiled in but its runtime is unusable, pin `DOCLING_RS_EP=cpu`
/// *before* docling reads it. Returns `true` when it did so. An explicit
/// `DOCLING_RS_EP` is always respected.
#[cfg(feature = "cuda")]
fn maybe_disable_cuda() -> bool {
    if std::env::var("DOCLING_RS_EP")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .is_some()
    {
        return false; // an explicit request is the operator's call
    }
    if cuda_runtime_available() {
        return false;
    }
    tracing::warn!(
        "CUDA is compiled in but the CUDA runtime (libcudnn) is not loadable — \
         cuDNN 9 (and the CUDA runtime matching the ONNX Runtime build) is required. \
         Falling back to CPU for ONNX inference."
    );
    // SAFETY: startup-time, single-threaded; nothing has read the provider
    // choice yet (this runs before docling's `choice()` caches it).
    unsafe { std::env::set_var("DOCLING_RS_EP", "cpu") };
    true
}

/// True when the CUDA runtime ONNX Runtime needs is loadable.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_runtime_available() -> bool {
    // ORT's CUDA EP dlopens cuDNN lazily at kernel execution; probe it now.
    // cuDNN 9 is what the ORT 1.28 line requires.
    ["libcudnn.so.9", "libcudnn.so"]
        .iter()
        .any(|name| dlopen_ok(name))
}

#[cfg(all(feature = "cuda", not(target_os = "linux")))]
fn cuda_runtime_available() -> bool {
    // Non-Linux: rely on ORT's own registration behaviour.
    true
}

#[cfg(target_os = "linux")]
fn dlopen_ok(name: &str) -> bool {
    use std::ffi::{CString, c_void};

    unsafe extern "C" {
        fn dlopen(filename: *const i8, flag: i32) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> i32;
    }
    const RTLD_LAZY: i32 = 1;

    let Ok(cname) = CString::new(name) else {
        return false;
    };
    let handle = unsafe { dlopen(cname.as_ptr(), RTLD_LAZY) };
    if handle.is_null() {
        false
    } else {
        unsafe { dlclose(handle) };
        true
    }
}

#[cfg(all(
    feature = "cuda",
    any(feature = "docling-extract", feature = "onnx-embed")
))]
fn probe_cuda() {
    use ort::ep::{ExecutionProvider as _, cuda::CUDA};
    match ort::session::Session::builder() {
        Ok(mut builder) => match CUDA::default().register(&mut builder) {
            Ok(()) => tracing::info!("CUDA execution provider registered; GPU inference enabled"),
            Err(e) => tracing::warn!(
                error = %e,
                "CUDA execution provider unavailable; ONNX sessions will fall back to CPU"
            ),
        },
        Err(e) => {
            tracing::warn!(error = %e, "could not create an ONNX session builder to probe CUDA")
        }
    }
}
