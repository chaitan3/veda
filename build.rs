//! Emit a build id so the extractor identity can tell a rebuilt binary apart
//! from a mere restart (failed documents are retried only when the file or the
//! extractor changed). `.git` is excluded from the container build context, so
//! the sha falls back to "unknown" there and the model/pdfium identities carry
//! the signal instead.

fn main() {
    let sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=BUILD_GIT_SHA={sha}");
    // Re-run when the checked-out commit changes.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=build.rs");
}
