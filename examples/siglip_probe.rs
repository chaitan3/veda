//! Probe the SigLIP image+text embedder: load the towers, embed a synthetic
//! image and a few prompts, and print dimensions plus cosine similarities.
//! Development tool for verifying the ONNX graphs and preprocessing.
//!
//!     LS_SIGLIP_DIR=.models/image cargo run --release --example siglip_probe

use local_search::embed::image::{ImageEmbedder, siglip::SigLipEmbedder};
use local_search::math::cosine;

fn solid_png(width: u32, height: u32, rgb: [u8; 3]) -> anyhow::Result<Vec<u8>> {
    let mut img = image::RgbImage::new(width, height);
    for px in img.pixels_mut() {
        *px = image::Rgb(rgb);
    }
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img).write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    let dir = std::env::var("LS_SIGLIP_DIR").unwrap_or_else(|_| ".models/image".to_string());
    let embedder = SigLipEmbedder::from_paths(
        &format!("{dir}/vision_model.onnx"),
        &format!("{dir}/text_model.onnx"),
        &format!("{dir}/tokenizer.json"),
        &format!("{dir}/preprocessor_config.json"),
        768,
    )?;

    let red = solid_png(224, 224, [220, 20, 20])?;
    let image_vec = embedder.embed_images(&[red]).await?.remove(0);
    let texts = vec!["a red square".to_string(), "a blue circle".to_string()];
    let text_vecs = embedder.embed_text(&texts).await?;

    println!("image dim: {}", image_vec.len());
    println!("text dim:  {}", text_vecs[0].len());
    println!(
        "cos(image, 'a red square') = {:.4}",
        cosine(&image_vec, &text_vecs[0])
    );
    println!(
        "cos(image, 'a blue circle') = {:.4}",
        cosine(&image_vec, &text_vecs[1])
    );
    Ok(())
}
