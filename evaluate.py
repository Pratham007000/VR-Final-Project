"""
evaluate.py — Batch evaluation script.

Runs the retrieval pipeline on query images and computes:
- Recall@K
- NDCG@K
- mAP@K
for K in {5, 10, 15}.

Supports all three ablation configurations (A, B, C).
"""

import os
import json
import random
import numpy as np
import faiss
from tqdm import tqdm

import config
from utils import load_json, normalize_embedding, load_image
from data_preparation import prepare_dataset
from clip_embedder import CLIPEmbedder
from yolo_detector import YOLODetector

# ──────────────────────────────────────────────
# Metric functions
# ──────────────────────────────────────────────

def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of queries where at least one relevant item is in top-K."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top_k) else 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """NDCG@K: position-aware ranking metric."""
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)  

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def average_precision_at_k(retrieved_ids, relevant_ids, k):
    """Average precision at K for a single query."""
    hits = 0
    sum_precision = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / min(len(relevant_ids), k) if relevant_ids else 0.0

def compute_metrics(all_retrieved, all_relevant, k_values):
    results = {}
    for k in k_values:
        recalls = [recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]
        ndcgs = [ndcg_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]
        aps = [average_precision_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]

        results[f"Recall@{k}"] = np.mean(recalls)
        results[f"NDCG@{k}"] = np.mean(ndcgs)
        results[f"mAP@{k}"] = np.mean(aps)
    return results

# ──────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────

def evaluate_config(config_name, alpha, finetuned_path=None, seed=None, max_queries=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Evaluating Config {config_name} (α={alpha}, seed={seed})")
    print(f"{'='*60}")

    data = prepare_dataset()
    query_df = data["query"]
    gallery_df = data["gallery"]

    if max_queries:
        query_df = query_df.head(max_queries)

    clip_embedder = CLIPEmbedder(finetuned_path=finetuned_path)

    suffix = f"_config{config_name}"
    if config_name == "C" and seed is not None:
        suffix += f"_seed{seed}"
    tag = f"alpha{alpha}{suffix}"
    index_path = os.path.join(config.INDEX_DIR, f"index_{tag}.faiss")
    meta_path = os.path.join(config.INDEX_DIR, f"metadata_{tag}.json")

    if not os.path.exists(index_path):
        print(f"Index not found: {index_path}. Building on the fly...")
        from index_builder import build_index_pipeline
        build_index_pipeline(alpha=alpha, finetuned_path=finetuned_path, suffix=suffix)

    index = faiss.read_index(index_path)
    metadata = load_json(meta_path)
    print(f"Index loaded: {index.ntotal} vectors")

    detector = YOLODetector()
    crop_dir = os.path.join(config.CROPS_DIR, "query")
    max_k = max(config.TOP_K_VALUES)
    
    all_retrieved = []
    all_relevant = []

    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Evaluating queries"):
        query_path = row["image_path"]
        query_item_id = row["item_id"]
        relevant = {query_item_id}

        try:
            img = None
            if crop_dir and os.path.exists(crop_dir):
                flat = query_path.replace("/", "_").replace("\\", "_") + (".jpg" if not query_path.lower().endswith((".jpg", ".png")) else "")
                cp = os.path.join(crop_dir, flat)
                if os.path.exists(cp): img = load_image(cp)
            if img is None:
                img = detector.crop_main_item(query_path)

            query_emb = clip_embedder.encode_image(img)
            query_emb = query_emb.reshape(1, -1).astype(np.float32)

            distances, indices = index.search(query_emb, max_k)

            retrieved_ids = []
            for idx in indices[0]:
                if 0 <= idx < len(metadata):
                    retrieved_ids.append(metadata[idx]["item_id"])

            all_retrieved.append(retrieved_ids)
            all_relevant.append(relevant)

        except Exception as e:
            pass

    metrics = compute_metrics(all_retrieved, all_relevant, config.TOP_K_VALUES)
    print(f"\nResults for Config {config_name} (α={alpha}, seed={seed}):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics

def run_ablation_study():
    all_results = {}

    # Config A: Vision-only CLIP (α=1.0)
    print("\n" + "#" * 60)
    print("# CONFIG A: Vision-only CLIP (α=1.0)")
    print("#" * 60)
    config_a_results = []
    for seed in config.RANDOM_SEEDS:
        config_a_results.append(evaluate_config("A", alpha=1.0, seed=seed))
    all_results["A"] = config_a_results

    # Config B: Frozen CLIP + frozen BLIP-2
    print("\n" + "#" * 60)
    print("# CONFIG B: Frozen CLIP + BLIP-2 captions")
    print("#" * 60)
    config_b_results = {}
    for alpha in config.ALPHA_VALUES:
        alpha_results = []
        for seed in config.RANDOM_SEEDS:
            alpha_results.append(evaluate_config("B", alpha=alpha, seed=seed))
        config_b_results[str(alpha)] = alpha_results
    all_results["B"] = config_b_results

    # Config C: Fine-tuned CLIP + frozen BLIP-2
    print("\n" + "#" * 60)
    print("# CONFIG C: Fine-tuned CLIP + BLIP-2 captions")
    print("#" * 60)
    config_c_results = {}
    for alpha in config.ALPHA_VALUES:
        alpha_results = []
        for seed in config.RANDOM_SEEDS:
            ft_path = os.path.join(config.CLIP_FT_DIR, f"clip_finetuned_seed{seed}.pt")
            if not os.path.exists(ft_path):
                continue
            alpha_results.append(evaluate_config("C", alpha=alpha, finetuned_path=ft_path, seed=seed))
        config_c_results[str(alpha)] = alpha_results
    all_results["C"] = config_c_results

    print_summary(all_results)
    save_path = os.path.join(config.OUTPUT_DIR, "ablation_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")
    return all_results

def print_summary(all_results):
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    def mean_std(results_list, metric):
        vals = [r[metric] for r in results_list if metric in r]
        return f"{np.mean(vals):.4f}±{np.std(vals):.4f}" if vals else "N/A"

    header = f"{'Config':<20} | {'α':<5}"
    for k in config.TOP_K_VALUES:
        header += f" | {'Recall@'+str(k):<15} | {'NDCG@'+str(k):<15} | {'mAP@'+str(k):<15}"
    print(header)
    print("-" * len(header))

    if "A" in all_results:
        row = f"{'A (Vision-only)':<20} | {'1.0':<5}"
        for k in config.TOP_K_VALUES:
            for m in [f"Recall@{k}", f"NDCG@{k}", f"mAP@{k}"]: row += f" | {mean_std(all_results['A'], m):<15}"
        print(row)

    if "B" in all_results:
        for alpha_str, res in all_results["B"].items():
            row = f"{'B (Frozen+BLIP)':<20} | {alpha_str:<5}"
            for k in config.TOP_K_VALUES:
                for m in [f"Recall@{k}", f"NDCG@{k}", f"mAP@{k}"]: row += f" | {mean_std(res, m):<15}"
            print(row)

    if "C" in all_results:
        for alpha_str, res in all_results["C"].items():
            row = f"{'C (FT+BLIP)':<20} | {alpha_str:<5}"
            for k in config.TOP_K_VALUES:
                for m in [f"Recall@{k}", f"NDCG@{k}", f"mAP@{k}"]: row += f" | {mean_std(res, m):<15}"
            print(row)

if __name__ == "__main__":
    run_ablation_study()
