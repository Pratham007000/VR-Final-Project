"""
clip_embedder.py — CLIP embedding module.

Provides functions to encode images and text using CLIP,
fuse embeddings with configurable alpha weighting, and stream full catalog datasets.
"""

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

import config
from utils import normalize_embedding, fuse_embeddings as fuse_np, load_image

class CLIPEmbedder:
    """Wraps OpenCLIP for image/text encoding with optional fine-tuned weights."""

    def __init__(self, model_name=None, pretrained=None, device=None, finetuned_path=None):
        self.model_name = model_name or config.CLIP_MODEL_NAME
        self.pretrained = pretrained or config.CLIP_PRETRAINED
        self.device = device or config.DEVICE

        print(f"Loading CLIP: {self.model_name} ({self.pretrained}) on {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        if finetuned_path and os.path.exists(finetuned_path):
            print(f"Loading fine-tuned weights from {finetuned_path}")
            state_dict = torch.load(finetuned_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(img_tensor).cpu().numpy().flatten()
        return normalize_embedding(emb)

    @torch.no_grad()
    def encode_images_batch(self, images: list, batch_size=32) -> np.ndarray:
        all_embs = []
        for i in range(0, len(images), batch_size):
            tensors = torch.stack([self.preprocess(img) for img in images[i:i + batch_size]]).to(self.device)
            all_embs.append(self.model.encode_image(tensors).cpu().numpy())
        result = np.vstack(all_embs)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / (norms + 1e-8)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(tokens).cpu().numpy().flatten()
        return normalize_embedding(emb)

    @torch.no_grad()
    def encode_texts_batch(self, texts: list, batch_size=64) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[i:i + batch_size]).to(self.device)
            all_embs.append(self.model.encode_text(tokens).cpu().numpy())
        result = np.vstack(all_embs)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / (norms + 1e-8)

    def get_fused_embedding(self, image: Image.Image, caption: str, alpha: float) -> np.ndarray:
        """Computes cross-modal fusion: v = alpha * phi_V(image) + (1-alpha) * phi_T(caption)."""
        return fuse_np(self.encode_image(image), self.encode_text(caption), alpha)

    def extract_fused_embeddings(self, image_paths: list, captions: dict, crop_dir: str = None, alpha: float = 1.0) -> dict:
        """End-to-end pipeline extraction streaming directly to a persistent dictionary."""
        embeddings = {}
        for path in tqdm(image_paths, desc=f"Extracting Fused Embeddings (alpha={alpha})"):
            try:
                img = None
                if crop_dir:
                    flat = path.replace("/", "_").replace("\\", "_") + (".jpg" if not path.lower().endswith((".jpg", ".png")) else "")
                    cp = os.path.join(crop_dir, flat)
                    if os.path.exists(cp): img = load_image(cp)
                if img is None: img = load_image(path)
                
                img_emb = self.encode_image(img)
                embeddings[path] = fuse_np(img_emb, self.encode_text(captions.get(path, "")), alpha) if alpha < 1.0 else img_emb
            except Exception:
                embeddings[path] = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
        return embeddings

    def get_model(self): return self.model
    def get_preprocess(self): return self.preprocess
    def get_tokenizer(self): return self.tokenizer

if __name__ == "__main__":
    embedder = CLIPEmbedder()
