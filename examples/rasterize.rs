//! Test helper: rasterize a PDF's pages to PNG so OCR can be exercised on an
//! image with no text layer. Not part of the product surface.
//!
//! Usage: `cargo run --example rasterize -- <input.pdf> <out-dir>`

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let pdf = args.next().expect("usage: rasterize <input.pdf> <out-dir>");
    let out = args.next().expect("usage: rasterize <input.pdf> <out-dir>");

    let bytes = std::fs::read(&pdf)?;
    // scale = pixels per PDF point; 2.0 == 144 dpi (docling's render scale).
    let pages = docling::render_pdf_pages(&bytes, None, None, 2.0)?;
    std::fs::create_dir_all(&out)?;
    for page in pages {
        let path = format!("{out}/page_{:04}.png", page.page_no);
        std::fs::write(&path, &page.png)?;
        println!("wrote {path} ({}x{})", page.width, page.height);
    }
    Ok(())
}
