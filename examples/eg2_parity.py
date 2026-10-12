#!/usr/bin/env python3
"""EmbeddingGemma 2 ONNX parity check (layer 1 of the EG2 evaluation).

Verifies that the ONNX artifacts, tokenizer and image preprocessing reproduce
the official model, on both modalities:

  text  — embeddings compared against sentence-transformers' implementation of
          google/embeddinggemma-2 (the model card's reference); expected
          cosine ~1.0 and elementwise agreement ~1e-7.
  image — image→text similarities compared against the numbers published in
          the onnx-community/embeddinggemma-2-ONNX model card (the cats /
          speech / turtle example).

Also reports fp32-vs-q8 deltas, so quantisation quality is measured, not
assumed.

Setup (once), with the artifacts in ./eg2-work:
    python3 -m venv .venv
    .venv/bin/pip install numpy pillow onnxruntime transformers \
        sentence-transformers torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu   # torch only
    # download from onnx-community/embeddinggemma-2-ONNX:
    #   config.json tokenizer.json tokenizer_config.json
    #   preprocessor_config.json processor_config.json
    #   onnx/model.onnx(+_data) onnx/model_quantized.onnx(+_data)
    #   onnx/vision_encoder.onnx(+_data) onnx/vision_encoder_quantized.onnx(+_data)
    # plus cats.jpg from datasets/Xenova/transformers.js-docs

Run from the artifact directory:  ../.venv/bin/python eg2_parity.py
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

QP = "task: search result | query: "
DP = "title: none | text: "

TEXT_DOCS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
IMAGE_QUERIES = [
    "cats sleeping on a couch",
    "a president's speech about serving your country",
    "a turtle swimming in the ocean",
]
# Published in the onnx-community/embeddinggemma-2-ONNX model card.
REF_IMAGE_SIMS = [0.740, 0.462, 0.508]


def session(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s, {i.name for i in s.get_inputs()}, [o.name for o in s.get_outputs()]


def text_embedding(path, enc):
    s, names, outs = session(path)
    feeds = {k: np.asarray(v) for k, v in enc.items() if k in names}
    for k in ("image_features", "video_features", "audio_features"):
        if k in names:
            feeds[k] = np.zeros((0, 512), dtype=np.float32)
    return s.run(None, feeds)[outs.index("sentence_embedding")]


def main():
    proc = AutoProcessor.from_pretrained(".")
    print(f"processor: {type(proc).__name__}  max_soft_tokens={proc.image_processor.max_soft_tokens}")

    # ---- text parity against the official implementation -------------------
    texts = [QP + "Which planet is known as the Red Planet?"] + [DP + d for d in TEXT_DOCS]
    enc = proc(text=texts, padding=True, return_tensors="np")
    try:
        from sentence_transformers import SentenceTransformer

        import torch

        # Pin fp32: the card requires bf16/fp32, and the ST default may pick a
        # lower precision (which shows up as ~1e-3 drift instead of ~1e-7).
        st = SentenceTransformer(
            "google/embeddinggemma-2", model_kwargs={"torch_dtype": torch.float32}
        )
        ref = st.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        onnx = text_embedding("onnx/model.onnx", enc)
        cos = (ref * onnx).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(onnx, axis=1))
        print(f"TEXT parity: cosine per text = {np.round(cos, 6).tolist()}")
        print(f"             max |elementwise diff| = {float(np.abs(ref - onnx).max()):.2e}")
        print(f"             sims = {np.round(onnx[1:] @ onnx[0], 4).tolist()}")
    except ImportError:
        print("TEXT parity: skipped (sentence-transformers not installed)")
    q8 = text_embedding("onnx/model_quantized.onnx", enc)
    fp32 = text_embedding("onnx/model.onnx", enc)
    print(f"TEXT fp32-vs-q8 sims: fp32={np.round(fp32[1:] @ fp32[0], 4).tolist()} "
          f"q8={np.round(q8[1:] @ q8[0], 4).tolist()}")

    # ---- image parity against the published example ------------------------
    enc_img = proc(images=[Image.open("cats.jpg").convert("RGB")], return_tensors="np")
    enc_q = proc(text=[QP + t for t in IMAGE_QUERIES], padding=True, return_tensors="np")
    vs, _, vouts = session("onnx/vision_encoder.onnx")
    img_feat = vs.run(None, {
        "pixel_values": np.asarray(enc_img["pixel_values"]),
        "pixel_position_ids": np.asarray(enc_img["image_position_ids"]),
    })[vouts.index("image_features")]
    print(f"IMAGE features: {img_feat.shape} (patch input {np.asarray(enc_img['pixel_values']).shape})")
    for tag, path in (("fp32", "onnx/model.onnx"), ("q8", "onnx/model_quantized.onnx")):
        s, names, outs = session(path)
        feeds = {k: np.asarray(enc_img[k]) for k in ("input_ids", "attention_mask") if k in names}
        feeds["image_features"] = img_feat.astype(np.float32)
        for k in ("video_features", "audio_features"):
            if k in names:
                feeds[k] = np.zeros((0, 512), dtype=np.float32)
        img_emb = s.run(None, feeds)[outs.index("sentence_embedding")][0]
        feeds = {k: np.asarray(enc_q[k]) for k in ("input_ids", "attention_mask") if k in names}
        for k in ("image_features", "video_features", "audio_features"):
            if k in names:
                feeds[k] = np.zeros((0, 512), dtype=np.float32)
        q_emb = s.run(None, feeds)[outs.index("sentence_embedding")]
        sims = q_emb @ img_emb
        print(f"IMAGE {tag}: {np.round(sims, 3).tolist()}  published {REF_IMAGE_SIMS}  "
              f"max|diff| = {float(np.abs(sims - np.array(REF_IMAGE_SIMS)).max()):.3f}")


if __name__ == "__main__":
    main()
