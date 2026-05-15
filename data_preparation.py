"""
data_preparation.py — Load and prepare the DeepFashion In-Shop dataset.
Automatically parses raw files or pre-computed CSVs, dynamically guaranteeing absolute image paths.
"""

import os
import pandas as pd
import config
from utils import extract_item_id

def load_partition_file(filepath: str) -> pd.DataFrame:
    rows = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            rows.append({"image_name": parts[0], "item_id": parts[1], "split": parts[2]})
    return pd.DataFrame(rows)

def load_bbox_file(filepath: str) -> pd.DataFrame:
    rows = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 7:
            rows.append({
                "image_name": parts[0], "clothes_type": int(parts[1]), "pose_type": int(parts[2]),
                "x1": int(parts[3]), "y1": int(parts[4]), "x2": int(parts[5]), "y2": int(parts[6])
            })
    return pd.DataFrame(rows)

def prepare_dataset() -> dict:
    print("=" * 60)
    print("Preparing DeepFashion In-Shop Dataset")
    print("=" * 60)
    print(f"Target Dataset Root: {config.DATASET_ROOT}")
    
    result = {}
    csv_mapping = [("train", config.TRAIN_CSV), ("query", config.QUERY_CSV), ("gallery", config.GALLERY_CSV)]
    
    # ── Strategy 1: Check if CSV splits exist (either pre-computed in input or cached in working) ──
    if all(os.path.exists(path) for _, path in csv_mapping):
        print("Loading pre-processed CSV splits...")
        for name, path in csv_mapping:
            df = pd.read_csv(path)
            
            # Guarantees the absolute image_path column exists
            if "image_path" not in df.columns:
                img_col = "image_name" if "image_name" in df.columns else df.columns[0]
                df["image_path"] = df[img_col].apply(
                    lambda x: os.path.join(config.IMG_DIR, str(x)[4:] if str(x).startswith("img/") else str(x))
                )
            
            # Guarantees item_id exists
            if "item_id" not in df.columns:
                df["item_id"] = df["image_path"].apply(extract_item_id)
                
            result[name] = df
            print(f"  Loaded {name}: {len(df)} rows, {df['item_id'].nunique()} unique items")
        return result

    # ── Strategy 2: Fallback to parsing list_eval_partition.txt ──
    print("\nCached CSVs not found. Parsing raw partition files...")
    if not os.path.exists(config.PARTITION_FILE):
        raise FileNotFoundError(f"Could not locate partition file at: {config.PARTITION_FILE}")
        
    partition_df = load_partition_file(config.PARTITION_FILE)
    
    # Resolve absolute paths properly based on the dynamic IMG_DIR
    partition_df["image_path"] = partition_df["image_name"].apply(
        lambda x: os.path.join(config.IMG_DIR, str(x)[4:] if str(x).startswith("img/") else str(x))
    )
    
    # Merge bounding boxes if available
    if os.path.exists(config.BBOX_FILE):
        bbox_df = load_bbox_file(config.BBOX_FILE)
        partition_df = partition_df.merge(bbox_df, on="image_name", how="left")
        print(f"Successfully merged bounding boxes for {len(bbox_df)} entries.")

    # Split, verify, and cache to disk
    for name, path in csv_mapping:
        df = partition_df[partition_df["split"] == name].copy().reset_index(drop=True)
        result[name] = df
        try:
            df.to_csv(path, index=False)
            print(f"  Processed and cached {name} split -> {path} ({len(df)} images)")
        except Exception:
            print(f"  Processed {name} split ({len(df)} images)")
        
    return result

def get_item_to_images(df: pd.DataFrame) -> dict:
    """Builds a mapping from item_id to list of absolute image paths."""
    item_to_imgs = {}
    for _, row in df.iterrows():
        item_id = row["item_id"]
        img_path = row["image_path"]
        if item_id not in item_to_imgs:
            item_to_imgs[item_id] = []
        item_to_imgs[item_id].append(img_path)
    return item_to_imgs

if __name__ == "__main__":
    data = prepare_dataset()
