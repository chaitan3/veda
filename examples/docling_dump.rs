//! Test helper: dump docling's Markdown for one file under several options, to
//! diagnose extraction behaviour. Not part of the product surface.
//!
//! Usage: `cargo run --example docling_dump -- <file>`

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: docling_dump <file>");
    let name = std::path::Path::new(&path)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let ext = name.rsplit('.').next().unwrap().to_string();

    let variants: Vec<(&str, docling::DocumentConverter)> = vec![
        ("default", docling::DocumentConverter::new()),
        (
            "force_full_page_ocr",
            docling::DocumentConverter::new().force_full_page_ocr(true),
        ),
        (
            "ocr_lang=ch",
            docling::DocumentConverter::new().ocr_lang("ch"),
        ),
        (
            "no_table_former",
            docling::DocumentConverter::new().no_table_former(true),
        ),
        (
            "no_text_panels",
            docling::DocumentConverter::new().no_text_panels(true),
        ),
    ];

    for (label, converter) in variants {
        let bytes = std::fs::read(&path)?;
        let fmt = docling::InputFormat::from_extension(&ext).unwrap();
        let source = docling::SourceDocument::from_bytes(name.clone(), fmt, bytes);
        match converter.convert(source) {
            Ok(result) => {
                let md = result.document.export_to_markdown();
                println!("=== {label} ({} chars) ===\n{md}\n", md.len());
            }
            Err(e) => println!("=== {label} ERROR: {e} ==="),
        }
    }
    Ok(())
}
