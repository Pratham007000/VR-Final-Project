"""
index_builder.py — Offline indexing pipeline.

Builds a FAISS HNSW index from gallery embeddings:
1. Load gallery images
2. Crop with YOLO
3. Generate captions with BLIP-2
4. Encode with CLIP (image + text → fused vector)
5. Build FAISS index + save metadata
"""

import os
import json
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm

import config
from utils import load_image, save_json, load_json, normalize_embedding, fuse_embeddings, ensure_dir
from data_preparation import prepare_dataset
from clip_embedder import CLIPEmbedder

def build_embeddings(gallery_df, clip_embedder, captions, alpha, crop_dir=None):
    embeddings = []
    metadata = []

    for _, row in tqdm(gallery_df.iterrows(), total=len(gallery_df), desc=f"Encoding (α={alpha})"):
        img_path = row["image_path"]
        item_id = row["item_id"]

        try:
            if crop_dir:
                flat = img_path.replace("/", "_").replace("\\", "_")
                if not flat.lower().endswith((".jpg", ".png")):
                    flat += ".jpg"
                cp = os.path.join(crop_dir, flat)
                img = load_image(cp) if os.path.exists(cp) else load_image(img_path)
            else:
                img = load_image(img_path)

            if alpha == 1.0:
                emb = clip_embedder.encode_image(img)
            else:
                caption = captions.get(img_path, "clothing item")
                emb = clip_embedder.get_fused_embedding(img, caption, alpha)

            embeddings.append(emb)
            metadata.append({"image_path": img_path, "item_id": item_id})

        except Exception as e:
            pass

    return np.array(embeddings, dtype=np.float32), metadata

def build_faiss_index(embeddings, dim=None):
    dim = dim or embeddings.shape[1]
    print(f"Building FAISS HNSW index: {embeddings.shape[0]} vectors, dim={dim}")

    index = faiss.IndexHNSWFlat(dim, config.HNSW_M)
    index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = config.HNSW_EF_SEARCH
    index.add(embeddings)

    print(f"Index built. Total vectors: {index.ntotal}")
    return index

def save_index(index, metadata, alpha, suffix=""):
    ensure_dir(config.INDEX_DIR)

    tag = f"alpha{alpha}{suffix}"
    index_path = os.path.join(config.INDEX_DIR, f"index_{tag}.faiss")
    meta_path = os.path.join(config.INDEX_DIR, f"metadata_{tag}.json")

    faiss.write_index(index, index_path)
    save_json(metadata, meta_path)

    print(f"Saved index: {index_path}")
    print(f"Saved metadata: {meta_path}")
    return index_path, meta_path

def load_index(alpha, suffix=""):
    tag = f"alpha{alpha}{suffix}"
    index_path = os.path.join(config.INDEX_DIR, f"index_{tag}.faiss")
    meta_path = os.path.join(config.INDEX_DIR, f"metadata_{tag}.json")

    index = faiss.read_index(index_path)
    metadata = load_json(meta_path)

    print(f"Loaded index: {index.ntotal} vectors")
    return index, metadata

def build_index_pipeline(alpha=0.7, finetuned_path=None, suffix=""):
    print("=" * 60)
    print(f"Offline Indexing Pipeline (α={alpha}, finetuned={finetuned_path is not None})")
    print("=" * 60)

    data = prepare_dataset()
    gallery_df = data["gallery"]

    captions = {}
    if alpha < 1.0:
        cap_path = os.path.join(config.CAPTIONS_DIR, "captions_gallery.json")
        if os.path.exists(cap_path):
            captions = load_json(cap_path)
            print(f"Loaded {len(captions)} captions")
        else:
            print("WARNING: No captions found. Run blip2_captioner.py first or use α=1.0")

    clip_embedder = CLIPEmbedder(finetuned_path=finetuned_path)

    crop_dir = os.path.join(config.CROPS_DIR, "gallery")
    embeddings, metadata = build_embeddings(
        gallery_df, clip_embedder, captions, alpha,
        crop_dir=crop_dir if os.path.exists(crop_dir) else None
    )

    index = build_faiss_index(embeddings)
    save_index(index, metadata, alpha, suffix=suffix)

    return index, metadata

if __name__ == "__main__":
    print("\n>>> Config A: Vision-only CLIP (α=1.0)")
    build_index_pipeline(alpha=1.0, suffix="_configA")

    for alpha in config.ALPHA_VALUES:
        print(f"\n>>> Config B: Frozen CLIP + BLIP-2 (α={alpha})")
        build_index_pipeline(alpha=alpha, suffix=f"_configB")

    for seed in config.RANDOM_SEEDS:
        ft_path = os.path.join(config.CLIP_FT_DIR, f"clip_finetuned_seed{seed}.pt")
        if os.path.exists(ft_path):
            for alpha in config.ALPHA_VALUES:
                print(f"\n>>> Config C: FT CLIP (seed={seed}) + BLIP-2 (α={alpha})")
                build_index_pipeline(alpha=alpha, finetuned_path=ft_path, suffix=f"_configC_seed{seed}")
