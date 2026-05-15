# Visual Product Search Engine

A multimodal fashion retrieval system using YOLOv8, BLIP-2, CLIP, and FAISS HNSW for query-by-image fashion search.

## Features

- YOLOv8 clothing localization
- BLIP-2 semantic caption generation
- Fine-tuned CLIP embeddings
- FAISS HNSW retrieval
- Streamlit interactive demo
- Batch evaluation pipeline

## Files

- `app.py` → Streamlit demo
- `evaluate.py` → metric computation
- `demo_eval.py` → batch evaluation
- `retrieval.py` → retrieval pipeline
- `clip_finetune.py` → CLIP fine-tuning
- `index_builder.py` → FAISS indexing

## Evaluation Metrics

Implemented metrics:
- Recall@K
- NDCG@K
- mAP@K

Metric implementations are available in:
- `evaluate.py`
- `demo_eval.py`

## Run Streamlit App

```bash
streamlit run app.py