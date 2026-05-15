"""
=========================================================
DEMO BATCH EVALUATION SCRIPT
=========================================================

Best Performing Configuration:
Config C (Fine-tuned CLIP + BLIP)
Alpha = 0.7

This script performs:
1. YOLO-based product localization
2. CLIP embedding generation
3. FAISS ANN retrieval
4. Retrieval metric computation

Metrics:
- Recall@K
- NDCG@K
- mAP@K

K ∈ {5, 10, 15}

=========================================================
HOW TO RUN
=========================================================

RETRIEVAL ONLY:
python demo_eval.py --query_folder demo_queries

EVALUATION MODE:
python demo_eval.py \
    --query_folder demo_queries \
    --ground_truth gt.csv

=========================================================
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt

# =========================================================
# MODEL 1: YOLOv8
# Purpose:
# Detect and crop main clothing product
# =========================================================

from yolo_detector import YOLODetector

# =========================================================
# MODEL 2: Fine-tuned CLIP
# Purpose:
# Generate image embeddings for retrieval
# Best Model:
# Config C (FT+BLIP), alpha = 0.7
# =========================================================

from clip_embedder import CLIPEmbedder

# =========================================================
# MODEL 3: FAISS HNSW Index
# Purpose:
# Fast approximate nearest neighbor retrieval
# =========================================================

from retrieval import RetrievalEngine

from utils import normalize_embedding

# =========================================================
# CONFIG
# =========================================================

TOP_K_VALUES = [5, 10, 15]

OUTPUT_DIR = "demo_outputs"
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")
RESULT_DIR = os.path.join(OUTPUT_DIR, "retrieval_results")

os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================================================
# LOAD MODELS
# =========================================================

print("\n==================================================")
print("LOADING MODELS")
print("==================================================")

detector = YOLODetector()

embedder = CLIPEmbedder(
    model_path="outputs/clip_finetuned"
)

retrieval_engine = RetrievalEngine(
    index_path="outputs/faiss_index"
)

print("All models loaded successfully.\n")

# =========================================================
# METRICS
# =========================================================

def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top_k) else 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):

    dcg = 0.0
    seen = set()

    for i, rid in enumerate(retrieved_ids[:k]):

        if rid in relevant_ids and rid not in seen:
            seen.add(rid)
            dcg += 1.0 / np.log2(i + 2)

    ideal_hits = min(len(relevant_ids), k)

    idcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(ideal_hits)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg

def average_precision_at_k(retrieved_ids, relevant_ids, k):

    hits = 0
    sum_precision = 0.0
    seen = set()

    for i, rid in enumerate(retrieved_ids[:k]):

        if rid in relevant_ids and rid not in seen:
            seen.add(rid)

            hits += 1
            sum_precision += hits / (i + 1)

    if len(relevant_ids) == 0:
        return 0.0

    return sum_precision / min(len(relevant_ids), k)

# =========================================================
# VISUALIZATION
# =========================================================

def save_visualization(
    query_image,
    retrieved_paths,
    scores,
    save_path,
    top_k=5
):

    fig, axes = plt.subplots(1, top_k + 1, figsize=(18, 5))

    axes[0].imshow(query_image)
    axes[0].set_title("QUERY")
    axes[0].axis("off")

    for i in range(top_k):

        img = Image.open(retrieved_paths[i]).convert("RGB")

        axes[i + 1].imshow(img)
        axes[i + 1].set_title(
            f"Rank {i+1}\n{scores[i]:.3f}"
        )
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =========================================================
# MAIN PIPELINE
# =========================================================

def process_query(query_path):

    query_name = os.path.basename(query_path)

    print("==================================================")
    print(f"QUERY: {query_name}")
    print("==================================================")

    # ---------------------------------------------------------
    # STEP 1: LOAD IMAGE
    # ---------------------------------------------------------

    image = Image.open(query_path).convert("RGB")

    # ---------------------------------------------------------
    # STEP 2: YOLO Product Localization
    # ---------------------------------------------------------

    crop = detector.detect_and_crop(image)

    crop_path = os.path.join(
        CROP_DIR,
        f"{os.path.splitext(query_name)[0]}_crop.jpg"
    )

    crop.save(crop_path)

    print(f"[1] YOLO crop saved -> {crop_path}")

    # ---------------------------------------------------------
    # STEP 3: CLIP Embedding Generation
    # ---------------------------------------------------------

    embedding = embedder.embed_image(crop)

    embedding = normalize_embedding(embedding)

    print(f"[2] Embedding generated -> shape {embedding.shape}")

    # ---------------------------------------------------------
    # STEP 4: ANN Retrieval using FAISS
    # ---------------------------------------------------------

    retrieved_paths, scores, retrieved_ids = retrieval_engine.search(
        embedding,
        top_k=max(TOP_K_VALUES)
    )

    print(f"[3] Retrieval complete")

    for rank in range(5):

        print(
            f"Rank {rank+1} | "
            f"Score: {scores[rank]:.4f} | "
            f"Item ID: {retrieved_ids[rank]}"
        )

    # ---------------------------------------------------------
    # STEP 5: SAVE VISUALIZATION
    # ---------------------------------------------------------

    result_path = os.path.join(
        RESULT_DIR,
        f"{os.path.splitext(query_name)[0]}_results.jpg"
    )

    save_visualization(
        image,
        retrieved_paths,
        scores,
        result_path,
        top_k=5
    )

    print(f"[4] Visualization saved -> {result_path}\n")

    return retrieved_ids

# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None
    )

    args = parser.parse_args()

    valid_exts = [".jpg", ".jpeg", ".png"]

    query_images = sorted([
        os.path.join(args.query_folder, f)
        for f in os.listdir(args.query_folder)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    print(f"\nFound {len(query_images)} query images.\n")

    all_predictions = {}

    # =========================================================
    # RUN PIPELINE
    # =========================================================

    for query_path in query_images:

        retrieved_ids = process_query(query_path)

        query_name = os.path.basename(query_path)

        all_predictions[query_name] = retrieved_ids

    # =========================================================
    # EVALUATION MODE
    # =========================================================

    if args.ground_truth is not None:

        print("\n==================================================")
        print("COMPUTING METRICS")
        print("==================================================")

        gt_df = pd.read_csv(args.ground_truth)

        metrics = {
            5: {"recall": [], "ndcg": [], "map": []},
            10: {"recall": [], "ndcg": [], "map": []},
            15: {"recall": [], "ndcg": [], "map": []}
        }

        for _, row in gt_df.iterrows():

            query_image = row["query_image"]
            gt_item_id = row["item_id"]

            retrieved_ids = all_predictions[query_image]

            relevant_ids = {gt_item_id}

            for k in TOP_K_VALUES:

                metrics[k]["recall"].append(
                    recall_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k
                    )
                )

                metrics[k]["ndcg"].append(
                    ndcg_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k
                    )
                )

                metrics[k]["map"].append(
                    average_precision_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k
                    )
                )

        print("\n==================================================")
        print("FINAL METRICS")
        print("==================================================\n")

        for k in TOP_K_VALUES:

            recall = np.mean(metrics[k]["recall"])
            ndcg = np.mean(metrics[k]["ndcg"])
            mapk = np.mean(metrics[k]["map"])

            print(f"Recall@{k:<2}: {recall:.4f}")
            print(f"NDCG@{k:<4}: {ndcg:.4f}")
            print(f"mAP@{k:<5}: {mapk:.4f}\n")

    print("==================================================")
    print("DEMO EVALUATION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()
