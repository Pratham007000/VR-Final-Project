"""
retrieval.py — Online query pipeline.

Given a query image:
1. YOLO crop
2. CLIP visual encode
3. FAISS search (top-K candidates)
4. BLIP-2 ITM re-ranking
"""

import os
import numpy as np
import faiss
from PIL import Image

import config
from utils import load_image, load_json, normalize_embedding
from yolo_detector import YOLODetector
from clip_embedder import CLIPEmbedder
from blip2_captioner import BLIP2Captioner

class ProductRetriever:
    """End-to-end product retrieval from a query image."""

    def __init__(self, index_path=None, metadata_path=None,
                 finetuned_clip_path=None, use_reranking=True):
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Loaded FAISS index: {self.index.ntotal} vectors")
        else:
            self.index = None
            print("WARNING: No FAISS index loaded")

        if metadata_path and os.path.exists(metadata_path):
            self.metadata = load_json(metadata_path)
        else:
            self.metadata = []

        cap_path = os.path.join(config.CAPTIONS_DIR, "captions_gallery.json")
        self.captions = load_json(cap_path) if os.path.exists(cap_path) else {}

        self.detector = YOLODetector()
        self.clip = CLIPEmbedder(finetuned_path=finetuned_clip_path)
        self.use_reranking = use_reranking
        self.blip2 = BLIP2Captioner() if use_reranking else None

    def search(self, query_image_path: str, top_k: int = None,
               rerank_top_n: int = None) -> list:
        top_k = top_k or config.DEFAULT_TOP_K
        rerank_top_n = rerank_top_n or config.ITM_RERANK_CANDIDATES

        if self.index is None:
            raise ValueError("No FAISS index loaded. Build the index first.")

        query_crop = self.detector.crop_main_item(query_image_path)

        query_emb = self.clip.encode_image(query_crop)
        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        search_k = max(top_k, rerank_top_n) if self.use_reranking else top_k
        distances, indices = self.index.search(query_emb, search_k)

        candidates = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            candidates.append({
                "image_path": meta["image_path"],
                "item_id": meta["item_id"],
                "score": float(dist),
                "faiss_rank": rank + 1
            })

        if self.use_reranking and self.blip2 and len(candidates) > 0:
            candidates = self.blip2.rerank_candidates(
                query_crop, candidates[:rerank_top_n], self.captions
            )

        results = candidates[:top_k]
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    def search_image(self, query_image: Image.Image, top_k=None) -> list:
        top_k = top_k or config.DEFAULT_TOP_K

        if self.index is None:
            raise ValueError("No FAISS index loaded.")

        query_emb = self.clip.encode_image(query_image)
        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "image_path": meta["image_path"],
                "item_id": meta["item_id"],
                "score": float(dist),
                "rank": rank + 1
            })

        return results
