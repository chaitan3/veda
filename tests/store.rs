//! End-to-end store test: proves the bundled SQLite has FTS5 and that
//! keyword + vector search work over inserted chunks.

use local_search::embed::{Embedder, hash::HashEmbedder};
use local_search::model::{Chunk, Document};
use local_search::store::{VectorStore, sqlite::SqliteStore};

async fn temp_store() -> SqliteStore {
    let mut path = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("ls-test-{}-{nanos}.db", std::process::id()));
    let store = SqliteStore::connect(path.to_str().unwrap(), 64)
        .await
        .unwrap();
    store.migrate().await.unwrap();
    store
}

#[tokio::test]
async fn fts5_and_vector_search() {
    let store = temp_store().await;
    let embedder = HashEmbedder::new(64);
    let texts = [
        "the quick brown fox jumps",
        "a lazy dog sleeps all day",
        "a mountain lake at sunset",
    ];

    let doc = Document::new("file:///sample.txt", "sample", "deadbeef");
    store.upsert_document(&doc).await.unwrap();

    let mut chunks = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let mut chunk = Chunk::new(
            &doc.id,
            i as i64,
            *text,
            text.split_whitespace().count() as i64,
        );
        let vector = embedder
            .embed(&[text.to_string()])
            .await
            .unwrap()
            .pop()
            .unwrap();
        chunk.embedding = Some(vector);
        chunks.push(chunk);
    }
    store.insert_chunks(&chunks).await.unwrap();
    assert_eq!(store.count_chunks().await.unwrap(), 3);

    // FTS5 keyword search
    let hits = store.keyword_search("mountain sunset", 5).await.unwrap();
    assert_eq!(hits.len(), 1, "one chunk mentions both mountain and sunset");
    assert!(hits[0].chunk.text.contains("mountain"));

    // Vector search returns all candidates, ranked
    let query = embedder.embed_query("mountain lake").await.unwrap();
    let ranked = store.vector_search(&query, 3).await.unwrap();
    assert_eq!(ranked.len(), 3);

    // Hash embedder is deterministic, so the same text ranks first.
    let exact = embedder.embed_query(texts[2]).await.unwrap();
    let top = store.vector_search(&exact, 1).await.unwrap();
    assert_eq!(top[0].chunk.text, texts[2]);

    // Corpus version increments.
    let before = store.corpus_version().await.unwrap();
    assert_eq!(store.bump_corpus_version().await.unwrap(), before + 1);

    // Pipeline version round-trips (used to trigger re-ingest on upgrades).
    assert_eq!(store.pipeline_version().await.unwrap(), 0);
    store.set_pipeline_version(7).await.unwrap();
    assert_eq!(store.pipeline_version().await.unwrap(), 7);

    // Deleting the source removes its chunks and FTS rows.
    store
        .delete_documents_by_source("file:///sample.txt")
        .await
        .unwrap();
    assert_eq!(store.count_chunks().await.unwrap(), 0);
    assert!(
        store
            .keyword_search("mountain", 5)
            .await
            .unwrap()
            .is_empty()
    );
}
