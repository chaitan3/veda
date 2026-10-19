//! Periodic ingest while the service runs.
//!
//! Enabled with `LS_INGEST_INTERVAL` (e.g. `24h`): `serve` runs an ingest once
//! immediately (catch-up) and then every interval, in a single background task
//! — one run at a time by construction, and errors never stop the server.
//! `LS_INGEST_SYNC=true` also drops documents whose source files disappeared
//! (the same scope as `ingest --sync`).
//!
//! The ingest runs in a **separate process** (the same binary, `ingest`
//! subcommand) rather than sharing the server's warm pipeline. That keeps
//! docling's models and GPU memory out of `serve` entirely: the child loads
//! them, ingests, and releases everything on exit. The child inherits the
//! environment and CWD (so `LS_*` config and `.models/`/`.pdfium/` resolve
//! identically), and is killed if the service shuts down mid-run.

use crate::config::Config;
use crate::{Error, Result};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::sync::watch;

/// Spawn the scheduled-ingest supervisor. Runs immediately, then every
/// `interval`, until `shutdown` flips to `true`.
pub fn spawn(
    cfg: Config,
    interval: Duration,
    sync: bool,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            if *shutdown.borrow() {
                break;
            }
            let started = Instant::now();
            match run_once(&cfg, sync, &mut shutdown).await {
                Ok(Some(status)) => tracing::info!(
                    code = status.code(),
                    secs = started.elapsed().as_secs_f64(),
                    "scheduled ingest finished"
                ),
                Ok(None) => {
                    tracing::info!("scheduled ingest stopped (shutdown)");
                    break;
                }
                Err(e) => tracing::warn!(
                    error = %e,
                    secs = started.elapsed().as_secs_f64(),
                    "scheduled ingest failed; will retry next interval"
                ),
            }
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                _ = shutdown.changed() => break,
            }
        }
    })
}

/// Run one ingest in a child process, killing it if shutdown arrives mid-run.
async fn run_once(
    cfg: &Config,
    sync: bool,
    shutdown: &mut watch::Receiver<bool>,
) -> Result<Option<std::process::ExitStatus>> {
    let mut cmd = tokio::process::Command::new(ingest_binary()?);
    cmd.arg("ingest");
    cmd.args(&cfg.source_paths);
    if sync {
        cmd.arg("--sync");
    }
    cmd.stdin(Stdio::null());
    let mut child = cmd
        .spawn()
        .map_err(|e| Error::Other(format!("spawning the ingest process: {e}")))?;
    tokio::select! {
        status = child.wait() => Ok(Some(
            status.map_err(|e| Error::Other(format!("waiting for the ingest process: {e}")))?
        )),
        _ = shutdown.changed() => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            Ok(None)
        }
    }
}

/// The binary to run for the child ingest: `LS_INGEST_BIN` when set (tests,
/// unusual deployments), otherwise this process's own executable.
fn ingest_binary() -> Result<std::path::PathBuf> {
    if let Ok(path) = std::env::var("LS_INGEST_BIN")
        && !path.trim().is_empty()
    {
        return Ok(path.into());
    }
    std::env::current_exe().map_err(|e| Error::Other(format!("cannot locate the veda binary: {e}")))
}

/// Parse `LS_INGEST_INTERVAL`: `<n>s|m|h|d`, a bare number of seconds, or
/// `off`/empty for "no scheduled ingest".
pub fn parse_interval(raw: &str) -> std::result::Result<Option<Duration>, String> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() || s == "off" || s == "none" || s == "0" {
        return Ok(None);
    }
    let (num, unit) = match s.strip_suffix("ms") {
        Some(n) => (n, "ms"),
        None => match s.chars().last() {
            Some(c) if c.is_ascii_alphabetic() => (
                &s[..s.len() - 1],
                match c {
                    's' => "s",
                    'm' => "m",
                    'h' => "h",
                    'd' => "d",
                    other => return Err(format!("unknown interval unit '{other}'")),
                },
            ),
            _ => (s.as_str(), "s"),
        },
    };
    let value: u64 = num
        .trim()
        .parse()
        .map_err(|_| format!("'{raw}' is not a valid interval (use e.g. 24h, 30m, 90s)"))?;
    if value == 0 {
        return Ok(None);
    }
    let secs = match unit {
        "ms" => return Ok(Some(Duration::from_millis(value))),
        "s" => value,
        "m" => value * 60,
        "h" => value * 3600,
        "d" => value * 86_400,
        _ => unreachable!(),
    };
    Ok(Some(Duration::from_secs(secs)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intervals_parse() {
        assert_eq!(
            parse_interval("24h").unwrap(),
            Some(Duration::from_secs(86_400))
        );
        assert_eq!(
            parse_interval("30m").unwrap(),
            Some(Duration::from_secs(1800))
        );
        assert_eq!(
            parse_interval("90s").unwrap(),
            Some(Duration::from_secs(90))
        );
        assert_eq!(
            parse_interval("2d").unwrap(),
            Some(Duration::from_secs(172_800))
        );
        assert_eq!(parse_interval("45").unwrap(), Some(Duration::from_secs(45)));
        assert_eq!(parse_interval("off").unwrap(), None);
        assert_eq!(parse_interval("").unwrap(), None);
        assert_eq!(parse_interval("0").unwrap(), None);
        assert!(parse_interval("24x").is_err());
        assert!(parse_interval("soon").is_err());
    }
}
