"""
app.py — Streamlit interactive demo for Visual Product Search.
Adheres strictly to the Confirm Crop / Re-crop evaluation rubric.
Dynamically resets session state on new file uploads.
"""

import os
import streamlit as st
import numpy as np
from PIL import Image
import faiss

import config
from utils import load_json, normalize_embedding, load_image
from yolo_detector import YOLODetector
from clip_embedder import CLIPEmbedder

st.set_page_config(page_title="Visual Product Search", page_icon="🔍", layout="wide")
st.title("🛍️ Query-by-Image Visual Product Search Engine")

@st.cache_resource
def load_detector(): return YOLODetector()

@st.cache_resource
def load_clip(): return CLIPEmbedder()

@st.cache_resource
def load_faiss_index(index_path, meta_path):
    return faiss.read_index(index_path), load_json(meta_path)

@st.cache_resource
def load_captions():
    cap_path = os.path.join(config.CAPTIONS_DIR, "captions_gallery.json")
    return load_json(cap_path) if os.path.exists(cap_path) else {}

def find_available_indices():
    indices = []
    if os.path.exists(config.INDEX_DIR):
        for f in os.listdir(config.INDEX_DIR):
            if f.startswith("index_") and f.endswith(".faiss"):
                indices.append(f.replace("index_", "").replace(".faiss", ""))
    return indices

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of results (K)", 1, 20, config.DEFAULT_TOP_K)
    available_indices = find_available_indices()
    selected_index = st.selectbox("Select Index", available_indices) if available_indices else None
    use_reranking = st.checkbox("Enable BLIP-2 ITM Re-ranking", value=False)

uploaded_file = st.file_uploader("📸 Upload a product query image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")
    
    # Reset confirmation state whenever a new image upload is detected
    current_file_id = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.get("last_file_id") != current_file_id:
        st.session_state.confirmation_status = None
        st.session_state.last_file_id = current_file_id

    temp_path = os.path.join(config.OUTPUT_DIR, "temp_query.jpg")
    query_image.save(temp_path)

    st.subheader("Step 1: Product Localization (YOLO)")
    detector = load_detector()
    region_result = detector.get_body_region_crops(temp_path)
    crops = region_result["crops"]
    
    # Fallback initialization if state doesn't exist yet
    if "confirmation_status" not in st.session_state:
        st.session_state.confirmation_status = None

    col_preview, col_select = st.columns([1, 2])
    with col_preview:
        st.image(query_image, caption="Original Query Image", width=250)

    with col_select:
        selected_region = st.radio(
            "Select Detected Region Focus:",
            options=["full_body", "upper_body", "lower_body"],
            format_func=lambda x: x.replace("_", " ").upper(),
            horizontal=True
        )
        st.image(crops[selected_region], caption=f"Preview: {selected_region.replace('_', ' ').upper()}", width=150)
        
        st.markdown("### ⚠️ Rubric Step: Confirmation Required")
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("✅ Confirm Crop & Proceed", use_container_width=True):
                st.session_state.confirmation_status = "confirmed"
        with action_col2:
            if st.button("🔄 Re-crop (Skip & Use Full Original)", use_container_width=True):
                st.session_state.confirmation_status = "recrop"

    # Proceed to the search pipeline strictly upon confirmation
    if st.session_state.confirmation_status and selected_index:
        st.markdown("---")
        st.subheader("Step 2: Vector Retrieval & Matching Results")
        
        target_crop = crops[selected_region] if st.session_state.confirmation_status == "confirmed" else query_image
        
        with st.spinner("Extracting multi-modal features and querying FAISS..."):
            clip = load_clip()
            query_emb = clip.encode_image(target_crop).reshape(1, -1).astype(np.float32)

            idx_path = os.path.join(config.INDEX_DIR, f"index_{selected_index}.faiss")
            mt_path = os.path.join(config.INDEX_DIR, f"metadata_{selected_index}.json")
            index, metadata = load_faiss_index(idx_path, mt_path)

            search_k = min(top_k * 3 if use_reranking else top_k, index.ntotal)
            distances, indices = index.search(query_emb, search_k)

            candidates = [{"image_path": metadata[idx]["image_path"], "item_id": metadata[idx]["item_id"], "score": float(dist), "rank": r + 1} 
                          for r, (dist, idx) in enumerate(zip(distances[0], indices[0])) if 0 <= idx < len(metadata)]

            if use_reranking and candidates:
                from blip2_captioner import BLIP2Captioner
                blip2 = BLIP2Captioner()
                candidates = blip2.rerank_candidates(target_crop, candidates, load_captions())

            results = candidates[:top_k]

            cols_per_row = 5
            for row_start in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if row_start + j >= len(results): break
                    r = results[row_start + j]
                    with col:
                        if os.path.exists(r["image_path"]):
                            st.image(load_image(r["image_path"]), use_container_width=True)
                        st.markdown(f"**Rank {r['rank']}** | `{r['item_id']}`")
                        st.caption(f"Sim: {r['score']:.3f}")

        if os.path.exists(temp_path): os.remove(temp_path)
