// Diagnostic: convert a PDF with the real onnx pipeline, chunk it with the
// real hybrid chunker, and count each chunk's BPE tokens with the *embedder's*
// tokenizer — i.e. exactly what the embedder will see.
use local_search::chunk::Chunker;
use local_search::config::{Config, EmbedProvider};
use tokenizers::Tokenizer;

#[tokio::main]
async fn main() -> local_search::Result<()> {
    let cfg = Config::from_env()?;
    println!(
        "chunker={:?} size={} overlap={}",
        cfg.chunker, cfg.chunk_size, cfg.chunk_overlap
    );
    let path = std::env::args().nth(1).expect("usage: chunk_audit <pdf>");
    let bytes = std::fs::read(&path)?;
    let converted = local_search::convert::convert(&cfg, &path, &bytes)?;
    println!(
        "tree={} markdown_words={}",
        converted.document.is_some(),
        converted.markdown.split_whitespace().count()
    );
    let chunker = Chunker::from_config(&cfg)?;
    let chunks = match &converted.document {
        Some(doc) => chunker.chunk_document("audit", doc)?,
        None => chunker.chunk_markdown("audit", &converted.markdown)?,
    };
    // Count with the *configured* embedder's tokenizer: bge uses a 30k
    // WordPiece vocab, EG2 a 262k SentencePiece one, and the counts differ.
    // A second argument overrides the path (for comparisons).
    let tok_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| match cfg.embed.provider {
            EmbedProvider::Eg2 => cfg
                .embed
                .eg2_tokenizer
                .clone()
                .unwrap_or_else(|| ".models/eg2/tokenizer.json".to_string()),
            _ => cfg
                .embed
                .onnx_tokenizer
                .clone()
                .unwrap_or_else(|| ".models/embed/tokenizer.json".to_string()),
        });
    println!("tokenizer={tok_path}");
    let tok = Tokenizer::from_file(&tok_path)
        .map_err(|e| local_search::Error::embedding(e.to_string()))?;
    let mut over = 0;
    let mut max_n = 0;
    for (i, c) in chunks.iter().enumerate() {
        let n = tok
            .encode(&*c.text, true)
            .map_err(|e| local_search::Error::embedding(e.to_string()))?
            .get_ids()
            .len();
        max_n = max_n.max(n);
        if n > 512 {
            over += 1;
            println!(
                "chunk {i}: {n} BPE tokens  (words={})",
                c.text.split_whitespace().count()
            );
            println!("  text: {:180}", c.text.replace('\n', " "));
        }
    }
    println!(
        "chunks={} max_bpe_tokens={} over_512={over}",
        chunks.len(),
        max_n
    );
    Ok(())
}
