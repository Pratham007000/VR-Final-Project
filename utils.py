"""
utils.py — Shared utilities for the Visual Product Search Engine.
"""

import os
import re
import json
import numpy as np
from PIL import Image

def extract_item_id(image_path: str) -> str:
    parts = image_path.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("id_"):
            return part
    match = re.search(r"(id_\d+)", image_path)
    if match:
        return match.group(1)
    return "unknown_id"

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")

def save_json(data: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)

def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    if emb.ndim == 1:
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)
    else:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-8)

def fuse_embeddings(img_emb: np.ndarray, txt_emb: np.ndarray, alpha: float) -> np.ndarray:
    """Fuses visual and text embeddings via: v = alpha * img_emb + (1 - alpha) * txt_emb"""
    fused = alpha * img_emb + (1.0 - alpha) * txt_emb
    return normalize_embedding(fused)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
