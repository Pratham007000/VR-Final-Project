"""
config.py — Central configuration for the Visual Product Search Engine.
Uses recursive auto-discovery to locate dataset files reliably regardless of mount paths.
"""

import os
import torch

# ──────────────────────────────────────────────
# Deep Auto-Discovery of Dataset Root
# ──────────────────────────────────────────────
DATASET_ROOT = None
PARTITION_FILE = None

# Recursively walk through /kaggle/input to find exactly where the partition file lives
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "list_eval_partition.txt" in files:
            DATASET_ROOT = root
            PARTITION_FILE = os.path.join(root, "list_eval_partition.txt")
            break

# Fallback default if running outside Kaggle
if DATASET_ROOT is None:
    DATASET_ROOT = "/kaggle/input/datasets/sahilakolte/deepfashion"
    PARTITION_FILE = os.path.join(DATASET_ROOT, "list_eval_partition.txt")

IMG_DIR      = os.path.join(DATASET_ROOT, "img")      
OUTPUT_DIR   = "/kaggle/working/outputs"

CROPS_DIR    = os.path.join(OUTPUT_DIR, "crops")
CAPTIONS_DIR = os.path.join(OUTPUT_DIR, "captions")
INDEX_DIR    = os.path.join(OUTPUT_DIR, "faiss_index")
CLIP_FT_DIR  = os.path.join(OUTPUT_DIR, "clip_finetuned")

BBOX_FILE    = os.path.join(DATASET_ROOT, "list_bbox_inshop.txt")

# Dynamically check if pre-computed CSVs exist right inside the discovered root
if os.path.exists(os.path.join(DATASET_ROOT, "train.csv")):
    MASTER_CSV  = os.path.join(DATASET_ROOT, "master_df.csv")
    TRAIN_CSV   = os.path.join(DATASET_ROOT, "train.csv")
    QUERY_CSV   = os.path.join(DATASET_ROOT, "query.csv")
    GALLERY_CSV = os.path.join(DATASET_ROOT, "gallery.csv")
else:
    # Route generated CSVs to persistent writable storage
    MASTER_CSV  = os.path.join(OUTPUT_DIR, "master_df.csv")
    TRAIN_CSV   = os.path.join(OUTPUT_DIR, "train.csv")
    QUERY_CSV   = os.path.join(OUTPUT_DIR, "query.csv")
    GALLERY_CSV = os.path.join(OUTPUT_DIR, "gallery.csv")

# Create persistent output dirs
for d in [CROPS_DIR, CAPTIONS_DIR, INDEX_DIR, CLIP_FT_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Device & Models
# ──────────────────────────────────────────────
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_MODEL       = "yolov8n.pt"                           
BLIP2_MODEL      = "Salesforce/blip2-opt-2.7b"            
CLIP_MODEL_NAME  = "ViT-B-32"                             
CLIP_PRETRAINED  = "openai"                               

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
CLIP_FT_LR           = 1e-5
CLIP_FT_EPOCHS       = 3
CLIP_FT_BATCH_SIZE   = 32
CLIP_FT_MARGIN       = 0.2       
CLIP_FT_UNFREEZE     = 4         
CLIP_FT_NUM_WORKERS  = 2         

EMBEDDING_DIM        = 512              
ALPHA_VALUES         = [0.7, 0.5]      

HNSW_M               = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH       = 100

TOP_K_VALUES         = [5, 10, 15]
DEFAULT_TOP_K        = 10
RANDOM_SEEDS         = [2023534, 2023066, 2023612]

CAPTION_PROMPT        = "Describe this clothing item in detail including color, pattern, material, fit, and style."
CAPTION_MAX_LENGTH    = 50
ITM_RERANK_CANDIDATES = 50
