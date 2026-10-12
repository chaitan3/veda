//! Retrieval evaluation harness (Phase 0).
//!
//! Two pieces:
//!
//! * [`generate`] mines a **gold set** from the corpus: for sampled text
//!   chunks it writes queries whose relevant target is known (a distinctive
//!   phrase from the chunk, the document title, or the file name). These are
//!   *synthetic* — phrase queries are lexically identical to the corpus, which
//!   flatters bm25 and is deliberately included as a control for the
//!   file-name case, which no vector channel covers. Hand-written queries can
//!   be appended to the same JSONL at any time.
//! * [`run`] scores a gold set through the real pipeline (every mode) and
//!   reports recall@k, MRR and nDCG@k.
//!
//! JSONL schema (one object per line):
//! ```json
//! {"query": "…", "kind": "phrase|title|filename",
//!  "relevant_chunk_ids": ["…"], "relevant_doc_ids": ["…"]}
//! ```
//! A hit is any result whose `chunk.id` is in `relevant_chunk_ids` **or**
//! whose `chunk.doc_id` is in `relevant_doc_ids`.

use crate::model::{Chunk, ChunkKind, Scored};
use crate::pipeline::Pipeline;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};

/// A relevant target expressed with **stable** keys, so one gold set can be
/// scored against several stores (chunk/document ids are per-ingest UUIDs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRef {
    /// The document's `source_uri` (exact match).
    pub source_uri: String,
    /// When set, only this chunk ordinal of that document is relevant;
    /// otherwise any chunk of the document counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<i64>,
}

/// One labelled query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldQuery {
    pub query: String,
    /// Free-form label for grouping the report (`phrase`, `title`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Portable targets (preferred): resolved against the store under test.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relevant_sources: Vec<SourceRef>,
    /// Direct ids, for hand-written queries against one specific store.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relevant_chunk_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relevant_doc_ids: Vec<String>,
}

/// A gold query with its portable targets resolved to this store's ids.
#[derive(Debug, Clone)]
pub struct ResolvedGold {
    pub query: String,
    pub kind: Option<String>,
    chunk_ids: std::collections::HashSet<String>,
    doc_ids: std::collections::HashSet<String>,
    /// (doc_id, ordinal) targets for chunk-level relevance.
    ordinals: std::collections::HashSet<(String, i64)>,
    relevant_count: usize,
}

impl ResolvedGold {
    fn is_hit(&self, hit: &Scored) -> bool {
        self.chunk_ids.contains(&hit.chunk.id)
            || self.doc_ids.contains(&hit.chunk.doc_id)
            || self
                .ordinals
                .contains(&(hit.chunk.doc_id.clone(), hit.chunk.ordinal))
    }

    fn relevant_count(&self) -> usize {
        self.relevant_count.max(1)
    }
}

/// Resolve every query's `relevant_sources` against the store's documents.
pub async fn resolve(pipeline: &Pipeline, gold: &[GoldQuery]) -> Result<Vec<ResolvedGold>> {
    let documents = pipeline.store().list_documents().await?;
    let by_uri: std::collections::HashMap<&str, &str> = documents
        .iter()
        .map(|d| (d.source_uri.as_str(), d.id.as_str()))
        .collect();
    let mut out = Vec::with_capacity(gold.len());
    for g in gold {
        let mut chunk_ids: std::collections::HashSet<String> =
            g.relevant_chunk_ids.iter().cloned().collect();
        let mut doc_ids: std::collections::HashSet<String> =
            g.relevant_doc_ids.iter().cloned().collect();
        let mut ordinals = std::collections::HashSet::new();
        let mut missing = 0usize;
        for r in &g.relevant_sources {
            match by_uri.get(r.source_uri.as_str()) {
                Some(id) => {
                    let id = id.to_string();
                    match r.ordinal {
                        Some(ordinal) => {
                            ordinals.insert((id, ordinal));
                        }
                        None => {
                            doc_ids.insert(id);
                        }
                    }
                }
                None => missing += 1,
            }
        }
        if missing > 0 {
            tracing::warn!(
                query = %g.query,
                missing,
                "gold-set source_uri(s) not present in this store; scoring them as unreachable"
            );
        }
        let relevant_count = chunk_ids.len() + doc_ids.len() + ordinals.len();
        // An unresolvable query must still count as a miss, not vanish.
        let relevant_count = relevant_count.max(1);
        let _ = &mut chunk_ids;
        out.push(ResolvedGold {
            query: g.query.clone(),
            kind: g.kind.clone(),
            chunk_ids,
            doc_ids,
            ordinals,
            relevant_count,
        });
    }
    Ok(out)
}

/// Per-query scores at the requested cutoffs.
#[derive(Debug, Clone, Default, Serialize)]
pub struct QueryScore {
    pub recall: BTreeMap<usize, f32>,
    pub ndcg: BTreeMap<usize, f32>,
    pub mrr: f32,
}

/// Aggregate report for one retrieval mode.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ModeReport {
    pub mode: String,
    pub queries: usize,
    pub recall: BTreeMap<usize, f32>,
    pub ndcg: BTreeMap<usize, f32>,
    pub mrr: f32,
    /// Same aggregates, split by the gold query's `kind` label.
    pub by_kind: BTreeMap<String, KindReport>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct KindReport {
    pub queries: usize,
    pub recall: BTreeMap<usize, f32>,
    pub ndcg: BTreeMap<usize, f32>,
    pub mrr: f32,
}

/// Score one ranked result list against one query.
pub fn score_query(ranked: &[Scored], gold: &ResolvedGold, cutoffs: &[usize]) -> QueryScore {
    let mut out = QueryScore::default();
    let mut first_hit: Option<usize> = None;
    let mut hits_at: Vec<bool> = Vec::with_capacity(ranked.len());
    for (i, hit) in ranked.iter().enumerate() {
        let hit = gold.is_hit(hit);
        hits_at.push(hit);
        if hit && first_hit.is_none() {
            first_hit = Some(i + 1);
        }
    }
    out.mrr = first_hit.map(|r| 1.0 / r as f32).unwrap_or(0.0);
    for &k in cutoffs {
        let top = &hits_at[..hits_at.len().min(k)];
        let found = top.iter().filter(|h| **h).count();
        out.recall
            .insert(k, found as f32 / gold.relevant_count().min(k) as f32);
        // Binary-relevance nDCG: ideal puts min(relevant, k) hits at the top.
        let dcg: f32 = top
            .iter()
            .enumerate()
            .filter(|(_, h)| **h)
            .map(|(i, _)| 1.0 / ((i + 2) as f32).log2())
            .sum();
        let ideal: f32 = (0..gold.relevant_count().min(k))
            .map(|i| 1.0 / ((i + 2) as f32).log2())
            .sum();
        out.ndcg
            .insert(k, if ideal > 0.0 { dcg / ideal } else { 0.0 });
    }
    out
}

/// Load a gold set from JSONL.
pub fn load(path: &str) -> Result<Vec<GoldQuery>> {
    let f = std::fs::File::open(path)
        .map_err(|e| Error::config(format!("opening gold set '{path}': {e}")))?;
    let mut out = Vec::new();
    for (n, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line.map_err(|e| Error::config(format!("reading '{path}': {e}")))?;
        if line.trim().is_empty() {
            continue;
        }
        let q: GoldQuery = serde_json::from_str(&line)
            .map_err(|e| Error::config(format!("{path}:{}: invalid gold query: {e}", n + 1)))?;
        out.push(q);
    }
    Ok(out)
}

/// Run every mode over the gold set and aggregate.
pub async fn run(
    pipeline: &Pipeline,
    gold: &[ResolvedGold],
    modes: &[crate::model::RetrievalMode],
    cutoffs: &[usize],
) -> Result<Vec<ModeReport>> {
    let k_max = cutoffs.iter().copied().max().unwrap_or(10);
    let mut reports = Vec::new();
    for &mode in modes {
        let mut report = ModeReport {
            mode: mode.to_string(),
            ..Default::default()
        };
        let mut sums: BTreeMap<usize, f32> = cutoffs.iter().map(|k| (*k, 0.0)).collect();
        let mut ndcg_sums = sums.clone();
        let mut mrr = 0.0f32;
        // kind -> (query count, accumulating report)
        let mut kinds: BTreeMap<String, (usize, KindReport)> = BTreeMap::new();
        for g in gold {
            let ranked = pipeline.query(mode, &g.query, k_max).await?;
            let s = score_query(&ranked, g, cutoffs);
            report.queries += 1;
            for k in cutoffs {
                *sums.get_mut(k).unwrap() += s.recall[k];
                *ndcg_sums.get_mut(k).unwrap() += s.ndcg[k];
            }
            mrr += s.mrr;
            let kind = g.kind.clone().unwrap_or_else(|| "unlabelled".to_string());
            let entry = kinds.entry(kind).or_insert_with(|| {
                (
                    0,
                    KindReport {
                        recall: cutoffs.iter().map(|k| (*k, 0.0)).collect(),
                        ndcg: cutoffs.iter().map(|k| (*k, 0.0)).collect(),
                        ..Default::default()
                    },
                )
            });
            entry.0 += 1;
            for k in cutoffs {
                *entry.1.recall.get_mut(k).unwrap() += s.recall[k];
                *entry.1.ndcg.get_mut(k).unwrap() += s.ndcg[k];
            }
            entry.1.mrr += s.mrr;
        }
        let n = report.queries.max(1) as f32;
        for k in cutoffs {
            report.recall.insert(*k, sums[k] / n);
            report.ndcg.insert(*k, ndcg_sums[k] / n);
        }
        report.mrr = mrr / n;
        for (kind, (count, mut kr)) in kinds {
            let c = count.max(1) as f32;
            kr.queries = count;
            kr.recall = kr.recall.into_iter().map(|(k, v)| (k, v / c)).collect();
            kr.ndcg = kr.ndcg.into_iter().map(|(k, v)| (k, v / c)).collect();
            kr.mrr /= c;
            report.by_kind.insert(kind, kr);
        }
        reports.push(report);
    }
    Ok(reports)
}

/// Print a report as an aligned table.
pub fn print_reports(reports: &[ModeReport], cutoffs: &[usize]) {
    println!(
        "{:<10} {:>7} {:>9} {:>9} {:>9}",
        "mode", "queries", "MRR", "recall@k", "ndcg@k"
    );
    for r in reports {
        let recall: Vec<String> = cutoffs
            .iter()
            .map(|k| format!("{k}:{:.3}", r.recall.get(k).copied().unwrap_or(0.0)))
            .collect();
        let ndcg: Vec<String> = cutoffs
            .iter()
            .map(|k| format!("{k}:{:.3}", r.ndcg.get(k).copied().unwrap_or(0.0)))
            .collect();
        println!(
            "{:<10} {:>7} {:>9.3} {:>9} {:>9}",
            r.mode,
            r.queries,
            r.mrr,
            recall.join(" "),
            ndcg.join(" ")
        );
        for (kind, kr) in &r.by_kind {
            let recall: Vec<String> = cutoffs
                .iter()
                .map(|k| format!("{:.3}", kr.recall.get(k).copied().unwrap_or(0.0)))
                .collect();
            println!(
                "  {:<8} {:>7} {:>9.3}  recall {}",
                kind,
                kr.queries,
                kr.mrr,
                recall.join(" ")
            );
        }
    }
}

/// Generate a synthetic gold set from the corpus.
///
/// For up to `per_doc` text chunks per document, emit a `phrase` query (a
/// distinctive run of words from the chunk) and, once per document, a `title`
/// and a `filename` query. Returns the number of queries written.
pub async fn generate(pipeline: &Pipeline, per_doc: usize, out: &str) -> Result<usize> {
    let documents = pipeline.store().list_documents().await?;
    let mut file =
        std::fs::File::create(out).map_err(|e| Error::config(format!("creating '{out}': {e}")))?;
    let mut written = 0usize;
    for doc in &documents {
        let chunks = pipeline.store().text_chunks(&doc.id).await?;
        // Documents with no text chunks (image-only folders) still get
        // title/file-name queries: those are exactly the "which file is
        // called X" cases that no search channel covers, so they belong in
        // the gold set as a control.
        for chunk in chunks.iter().take(per_doc) {
            if let Some(query) = distinctive_phrase(&chunk.text) {
                write_query(
                    &mut file,
                    &GoldQuery {
                        query,
                        kind: Some("phrase".into()),
                        relevant_sources: vec![SourceRef {
                            source_uri: doc.source_uri.clone(),
                            ordinal: Some(chunk.ordinal),
                        }],
                        relevant_chunk_ids: Vec::new(),
                        relevant_doc_ids: Vec::new(),
                    },
                )?;
                written += 1;
            }
        }
        // Title query: any chunk of the document counts.
        if doc.title.split_whitespace().count() >= 2 {
            write_query(
                &mut file,
                &GoldQuery {
                    query: doc.title.clone(),
                    kind: Some("title".into()),
                    relevant_sources: vec![SourceRef {
                        source_uri: doc.source_uri.clone(),
                        ordinal: None,
                    }],
                    relevant_chunk_ids: Vec::new(),
                    relevant_doc_ids: Vec::new(),
                },
            )?;
            written += 1;
        }
        // File-name query: a control — no retrieval channel indexes names
        // (see the `api/documents?query=` filter for that), so a low score
        // here is expected and informative.
        let name = doc
            .source_uri
            .rsplit('/')
            .next()
            .unwrap_or(&doc.source_uri)
            .rsplit_once('.')
            .map(|(stem, _)| stem)
            .unwrap_or(&doc.source_uri)
            .to_string();
        if name.split_whitespace().count() >= 1 && name.len() >= 6 {
            write_query(
                &mut file,
                &GoldQuery {
                    query: name,
                    kind: Some("filename".into()),
                    relevant_sources: vec![SourceRef {
                        source_uri: doc.source_uri.clone(),
                        ordinal: None,
                    }],
                    relevant_chunk_ids: Vec::new(),
                    relevant_doc_ids: Vec::new(),
                },
            )?;
            written += 1;
        }
    }
    file.flush()
        .map_err(|e| Error::config(format!("writing '{out}': {e}")))?;
    Ok(written)
}

fn write_query(f: &mut std::fs::File, q: &GoldQuery) -> Result<()> {
    let line = serde_json::to_string(q)?;
    writeln!(f, "{line}").map_err(|e| Error::config(format!("writing gold set: {e}")))
}

/// A distinctive, self-contained run of words from `text`: prefer a sentence
/// with unusual terms (digits, long words) over the opening words, which tend
/// to be generic.
fn distinctive_phrase(text: &str) -> Option<String> {
    const STOP: &[&str] = &[
        "the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "which",
        "there", "their", "have", "has", "not", "but", "you", "can", "will", "its", "into", "than",
        "then", "when", "also", "such", "been", "other", "more", "most", "some", "only", "over",
    ];
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 8 {
        return None;
    }
    // Sliding window of 8 words; score by rare/long/digit tokens.
    let mut best: (usize, usize, f32) = (0, 0, f32::MIN);
    let window = 8usize;
    for start in 0..words.len().saturating_sub(window) {
        let slice = &words[start..start + window];
        let score: f32 = slice
            .iter()
            .map(|w| {
                let clean: String = w
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();
                let mut s = (clean.len().min(12) as f32) / 12.0;
                if clean.chars().any(|c| c.is_ascii_digit()) {
                    s += 0.5;
                }
                if STOP.contains(&clean.as_str()) {
                    s -= 0.4;
                }
                s
            })
            .sum();
        if score > best.2 {
            best = (start, start + window, score);
        }
    }
    let phrase = words[best.0..best.1].join(" ");
    let phrase = phrase
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_string();
    (phrase.split_whitespace().count() >= 4).then_some(phrase)
}

/// Convenience for the CLI: parse `--modes` and `--k`.
pub fn parse_modes(spec: &str) -> Result<Vec<crate::model::RetrievalMode>> {
    spec.split(',')
        .map(|m| {
            m.trim()
                .parse::<crate::model::RetrievalMode>()
                .map_err(|e| Error::config(e.to_string()))
        })
        .collect()
}

/// Parse `--k` cutoffs, e.g. "1,5,10".
pub fn parse_cutoffs(spec: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let k: usize = part
            .trim()
            .parse()
            .map_err(|_| Error::config(format!("invalid --k value '{part}'")))?;
        if k == 0 {
            return Err(Error::config("--k cutoffs must be >= 1"));
        }
        out.push(k);
    }
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        return Err(Error::config("--k needs at least one cutoff"));
    }
    Ok(out)
}

/// Count image chunks in the store (used to warn when an image-mode eval has
/// nothing to search).
pub async fn image_chunk_count(pipeline: &Pipeline) -> Result<usize> {
    let mut n = 0usize;
    for doc in pipeline.store().list_documents().await? {
        n += pipeline
            .store()
            .get_chunks_range(&doc.id, 0, usize::MAX)
            .await?
            .iter()
            .filter(|c: &&Chunk| c.kind == ChunkKind::Image)
            .count();
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Chunk;

    fn hit(chunk_id: &str, doc_id: &str) -> Scored {
        let mut chunk = Chunk::new(doc_id, 0, "x", 1);
        chunk.id = chunk_id.to_string();
        Scored { chunk, score: 1.0 }
    }

    fn gold(chunk_ids: &[&str], doc_ids: &[&str]) -> ResolvedGold {
        ResolvedGold {
            query: "q".into(),
            kind: None,
            chunk_ids: chunk_ids.iter().map(|s| s.to_string()).collect(),
            doc_ids: doc_ids.iter().map(|s| s.to_string()).collect(),
            ordinals: Default::default(),
            relevant_count: (chunk_ids.len() + doc_ids.len()).max(1),
        }
    }

    #[test]
    fn metrics_are_computed_over_the_ranked_list() {
        let g = gold(&["b"], &[]);
        // Relevant target at rank 2.
        let ranked = vec![hit("a", "d1"), hit("b", "d2"), hit("c", "d3")];
        let s = score_query(&ranked, &g, &[1, 3]);
        assert_eq!(s.recall[&1], 0.0);
        assert_eq!(s.recall[&3], 1.0);
        assert!((s.mrr - 0.5).abs() < 1e-6);
        // nDCG@3: one hit at rank 2 -> (1/log2(3)) / (1/log2(2)).
        let expected = (1.0f32 / 3.0f32.log2()) / 1.0;
        assert!((s.ndcg[&3] - expected).abs() < 1e-6);
    }

    #[test]
    fn doc_level_relevance_counts_any_chunk() {
        let g = gold(&[], &["d9"]);
        let ranked = vec![hit("x", "d1"), hit("y", "d9")];
        let s = score_query(&ranked, &g, &[2]);
        assert_eq!(s.recall[&2], 1.0);
    }

    #[test]
    fn distinctive_phrase_skips_short_text() {
        assert!(distinctive_phrase("too short here").is_none());
        let text = "The process control system uses a depropanizer column 42 with a \
                    composition analyzer to regulate distillate composition despite feed \
                    disturbances and other operational upsets in the plant";
        let p = distinctive_phrase(text).unwrap();
        assert!(p.split_whitespace().count() >= 4);
    }

    #[test]
    fn cutoffs_parse_and_dedupe() {
        assert_eq!(parse_cutoffs("10,1,5,5").unwrap(), vec![1, 5, 10]);
        assert!(parse_cutoffs("0").is_err());
    }

    #[test]
    fn modes_parse() {
        let m = parse_modes("hybrid, bm25,image").unwrap();
        assert_eq!(m.len(), 3);
    }
}
