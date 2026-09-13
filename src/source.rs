//! Document sources. Phase 0 ships the local folder source; FTP/SFTP are out of
//! scope. A source can span **several roots** (`LS_SOURCE_PATH` is a
//! path-separated list, and `ingest` accepts multiple paths).

use crate::config::Config;
use crate::{Error, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Extensions we hand to the converter.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "md", "markdown", "txt", "text", "pdf", "docx", "doc", "html", "htm", "pptx", "xlsx", "csv",
    "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp",
];

/// A document to ingest.
#[derive(Debug, Clone)]
pub struct SourceRef {
    /// Stable identity stored as `Document::source_uri`.
    pub uri: String,
    /// File name (used to pick a converter).
    pub name: String,
    /// Path relative to the root it was found under.
    pub rel_path: String,
    /// Absolute path on disk.
    pub path: PathBuf,
}

/// Walks one or more local folders.
pub struct FolderSource {
    roots: Vec<PathBuf>,
}

impl FolderSource {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        FolderSource {
            roots: vec![root.into()],
        }
    }

    pub fn from_roots(roots: Vec<PathBuf>) -> Self {
        FolderSource { roots }
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Canonicalize every root, erroring if one does not exist (a missing root
    /// must not silently shrink the corpus, which would make `--sync` delete it).
    pub fn canonical_roots(&self) -> Result<Vec<PathBuf>> {
        self.roots
            .iter()
            .map(|r| {
                if !r.exists() {
                    return Err(Error::source(format!(
                        "source path does not exist: {}",
                        r.display()
                    )));
                }
                r.canonicalize().map_err(|e| Error::io(r, e))
            })
            .collect()
    }

    /// List every supported file under every root (recursively), sorted by uri
    /// and de-duplicated when roots overlap.
    pub fn list(&self) -> Result<Vec<SourceRef>> {
        let mut refs = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for root in self.canonical_roots()? {
            for entry in WalkDir::new(&root).follow_links(false) {
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::warn!(error = %e, "skipping unreadable directory entry");
                        continue;
                    }
                };
                if !entry.file_type().is_file() {
                    continue;
                }
                let path = entry.path();
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase())
                    .unwrap_or_default();
                if !SUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
                    continue;
                }
                let uri = path.to_string_lossy().to_string();
                if !seen.insert(uri.clone()) {
                    continue;
                }
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("document")
                    .to_string();
                let rel_path = path
                    .strip_prefix(&root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string();
                refs.push(SourceRef {
                    uri,
                    name,
                    rel_path,
                    path: path.to_path_buf(),
                });
            }
        }

        refs.sort_by(|a, b| a.uri.cmp(&b.uri));
        Ok(refs)
    }
}

/// True when `uri` lies under one of `roots` (component-wise, so `/srv/docsx`
/// is not under `/srv/docs`). Used to scope `--sync` to the ingested roots.
pub fn uri_under_roots(uri: &str, roots: &[PathBuf]) -> bool {
    let path = Path::new(uri);
    roots.iter().any(|root| path.starts_with(root))
}

/// Build the folder source from config.
pub fn from_config(cfg: &Config) -> FolderSource {
    FolderSource::from_roots(cfg.source_paths.iter().map(PathBuf::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_base(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ls-src-{tag}-{}-{nanos}", std::process::id()))
    }

    #[test]
    fn lists_multiple_roots_with_relative_paths() {
        let base = temp_base("multi");
        let a = base.join("a");
        let b = base.join("b");
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(b.join("sub")).unwrap();
        fs::write(a.join("one.md"), "one").unwrap();
        fs::write(b.join("sub").join("two.txt"), "two").unwrap();
        fs::write(b.join("ignore.bin"), "x").unwrap();

        let src = FolderSource::from_roots(vec![a.clone(), b.clone()]);
        let refs = src.list().unwrap();
        let names: Vec<&str> = refs.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["one.md", "two.txt"]);
        let rels: Vec<&str> = refs.iter().map(|r| r.rel_path.as_str()).collect();
        assert!(rels.contains(&"one.md"));
        assert!(rels.contains(&"sub/two.txt"));
        assert_eq!(src.canonical_roots().unwrap().len(), 2);

        fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn missing_root_is_an_error() {
        let base = temp_base("missing");
        fs::create_dir_all(&base).unwrap();
        let src = FolderSource::from_roots(vec![base.join("nope")]);
        assert!(src.list().is_err());
        assert!(src.canonical_roots().is_err());
        fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn containment_is_component_wise() {
        let roots = vec![PathBuf::from("/srv/docs"), PathBuf::from("/srv/archive")];
        assert!(uri_under_roots("/srv/docs/a/b.md", &roots));
        assert!(uri_under_roots("/srv/archive/z.md", &roots));
        assert!(!uri_under_roots("/srv/other/z.md", &roots));
        assert!(!uri_under_roots("/srv/docsx/z.md", &roots));
    }
}
