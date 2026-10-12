//! Periodic ingest while the service runs.
//!
//! Enabled with `LS_INGEST_INTERVAL` (e.g. `24h`): `serve` ingests once
//! immediately (catch-up) and then every interval, in a single background task
//! — one ingest at a time by construction, and errors never stop the server.
//! `LS_INGEST_SYNC=true` also drops documents whose source files disappeared
//! (the same scope as `ingest --sync`: the configured `LS_SOURCE_PATH` roots).
//!
//! Queries stay available throughout: SQLite is in WAL mode, each document is
//! committed as it finishes, and the ingest task shares the warm pipeline.

use crate::pipeline::{Pipeline, StageSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Spawn the scheduled-ingest task. Runs immediately, then every `interval`.
pub fn spawn(
    pipeline: Arc<Pipeline>,
    interval: Duration,
    sync: bool,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let started = Instant::now();
            match pipeline.ingest_all(sync, false, StageSet::all()).await {
                Ok(report) => tracing::info!(
                    ingested = report.documents_ingested,
                    skipped = report.documents_skipped,
                    failed = report.documents_failed,
                    deleted = report.documents_deleted,
                    chunks = report.chunks_added,
                    secs = started.elapsed().as_secs_f64(),
                    "scheduled ingest finished"
                ),
                Err(e) => tracing::warn!(
                    error = %e,
                    secs = started.elapsed().as_secs_f64(),
                    "scheduled ingest failed; will retry next interval"
                ),
            }
            tokio::time::sleep(interval).await;
        }
    })
}

/// Parse `LS_INGEST_INTERVAL`: `<n>s|m|h|d`, a bare number of seconds, or
/// `off`/empty for "no scheduled ingest".
pub fn parse_interval(raw: &str) -> Result<Option<Duration>, String> {
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
