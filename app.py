"""
app.py — Streamlit interactive demo for Visual Product Search.

Online Pipeline (4 explicit steps, per project spec):
  Step 1 — YOLO:     Detect product region and crop query image.
  Step 2 — CLIP:     Encode cropped query into a fused embedding.
  Step 3 — FAISS:    Retrieve top-K candidates via cosine similarity (ANN).
  Step 4 — BLIP-2:   Re-rank candidates by Image-Text Matching (ITM) score.
"""

import os
import time
import streamlit as st
import numpy as np
from PIL import Image
import faiss

import config
from utils import load_json, normalize_embedding, fuse_embeddings, load_image
from yolo_detector import YOLODetector
from clip_embedder import CLIPEmbedder

st.set_page_config(page_title="Visual Product Search", page_icon="🔍", layout="wide")
st.title("🛍️ Query-by-Image Visual Product Search Engine")
st.markdown(
    "**End-to-end pipeline:** Upload Image → "
    "**①** YOLO Crop → **Confirm** → "
    "**②** CLIP Encode → "
    "**③** FAISS Retrieve → "
    "**④** BLIP-2 Re-rank → **Results**"
)

# ── Cached resource loaders ──────────────────────────────────────────

@st.cache_resource
def load_detector():
    return YOLODetector()

@st.cache_resource
def load_clip(finetuned_path: str = ""):
    """Cache key includes finetuned_path so swapping weights reloads CLIP."""
    path = finetuned_path if finetuned_path and os.path.exists(finetuned_path) else None
    return CLIPEmbedder(finetuned_path=path)

@st.cache_resource
def load_faiss_index(index_path: str, meta_path: str):
    return faiss.read_index(index_path), load_json(meta_path)

@st.cache_resource
def load_captions():
    cap_path = os.path.join(config.CAPTIONS_DIR, "captions_gallery.json")
    return load_json(cap_path) if os.path.exists(cap_path) else {}

@st.cache_resource
def load_blip2():
    from blip2_captioner import BLIP2Captioner
    return BLIP2Captioner()

def find_available_indices():
    indices = []
    if os.path.exists(config.INDEX_DIR):
        for f in sorted(os.listdir(config.INDEX_DIR)):
            if f.startswith("index_") and f.endswith(".faiss"):
                indices.append(f.replace("index_", "").replace(".faiss", ""))
    return indices

def find_finetuned_checkpoints():
    ckpts = {"None (frozen / pre-trained)": ""}
    if os.path.exists(config.CLIP_FT_DIR):
        for f in sorted(os.listdir(config.CLIP_FT_DIR)):
            if f.endswith(".pt"):
                ckpts[f] = os.path.join(config.CLIP_FT_DIR, f)
    return ckpts

# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("Top-K results", 1, 20, config.DEFAULT_TOP_K,
                      help="Number of products to retrieve and display.")

    available_indices = find_available_indices()
    if available_indices:
        selected_index = st.selectbox(
            "Ablation config / Index",
            available_indices,
            help=(
                "Choose the FAISS index built under each ablation setting:\n"
                "  A – Vision-only CLIP (α=1)\n"
                "  B – Frozen CLIP + BLIP-2\n"
                "  C – Fine-tuned CLIP + BLIP-2"
            ),
        )
    else:
        selected_index = None
        st.error("No FAISS index found in outputs/faiss_index/")

    alpha = st.slider(
        "Alpha α (image weight in fused embedding)",
        0.0, 1.0, 0.7, 0.05,
        help="α=1 → pure visual embedding  |  α=0 → pure caption embedding\n"
             "Matches Equation 1 in the project spec: v = α·φ_V(x̂) + (1−α)·φ_T(c)"
    )

    ckpts = find_finetuned_checkpoints()
    selected_ckpt_label = st.selectbox(
        "CLIP checkpoint (fine-tuned weights)",
        list(ckpts.keys()),
        help="Select a fine-tuned CLIP checkpoint from outputs/clip_finetuned/. "
             "Choosing 'None' uses frozen pre-trained weights."
    )
    selected_ckpt_path = ckpts[selected_ckpt_label]

    use_reranking = st.checkbox(
        "Enable BLIP-2 ITM Re-ranking (Step 4)",
        value=True,
        help="Re-rank FAISS candidates using BLIP-2 Image-Text Matching. "
             "Recommended: ON for ablation configs B and C."
    )
    show_query_caption = st.checkbox(
        "Generate BLIP-2 caption for query",
        value=True,
        help="Show the BLIP-2-generated caption for your query image (for inspection)."
    )

    st.markdown("---")
    st.caption(f"Device: `{config.DEVICE}`")
    st.markdown("""
**Ablation legend**
| Config | Description |
|--------|-------------|
| A | Vision-only CLIP, α=1 |
| B | Frozen CLIP + BLIP-2 |
| C | Fine-tuned CLIP + BLIP-2 |
""")

# ── Image Upload ──────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "📸 Upload a product query image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")

    # Reset pipeline state on new upload
    current_file_id = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.get("_file_id") != current_file_id:
        st.session_state._file_id = current_file_id
        st.session_state.confirmed = None       # None | "confirmed" | "recrop"
        st.session_state.query_caption = None

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    temp_path = os.path.join(config.OUTPUT_DIR, "temp_query.jpg")
    query_image.save(temp_path)

    # ════════════════════════════════════════════════════════════════
    # STEP 1 — YOLO: Product Localization
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("① Product Localization  —  YOLO")
    st.caption(
        "YOLO detects the primary clothing item and crops the image to reduce "
        "background noise before encoding."
    )

    with st.spinner("Running YOLO detection..."):
        detector = load_detector()
        region_result = detector.get_body_region_crops(temp_path)
        crops = region_result["crops"]
        detections = region_result["detections"]

    col_orig, col_crop, col_ctrl = st.columns([1, 1, 2])

    with col_orig:
        st.image(query_image, caption="Original upload", use_container_width=True)
        st.caption(f"Objects detected by YOLO: **{len(detections)}**")

    with col_crop:
        selected_region = st.radio(
            "Crop region:",
            ["full_body", "upper_body", "lower_body"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        st.image(
            crops[selected_region],
            caption=f"YOLO crop — {selected_region.replace('_', ' ').title()}",
            use_container_width=True,
        )

    with col_ctrl:
        st.markdown("#### Confirmation Required")
        st.info(
            "Inspect the YOLO crop. "
            "**Confirm** to use it as the query, or **Re-crop** to fall back to the "
            "full original image."
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Confirm Crop & Proceed", use_container_width=True, type="primary"):
                st.session_state.confirmed = "confirmed"
                st.session_state.query_caption = None   # reset caption for fresh crop
        with b2:
            if st.button("Re-crop (Use Full Image)", use_container_width=True):
                st.session_state.confirmed = "recrop"
                st.session_state.query_caption = None

    if st.session_state.confirmed == "confirmed":
        st.success(f"Using YOLO crop — region: **{selected_region.replace('_', ' ').title()}**")
    elif st.session_state.confirmed == "recrop":
        st.warning("Using full original image (YOLO crop bypassed)")

    # ════════════════════════════════════════════════════════════════
    # STEPS 2–4: Run only after user confirms
    # ════════════════════════════════════════════════════════════════
    if st.session_state.confirmed and selected_index:

        target = (
            crops[selected_region]
            if st.session_state.confirmed == "confirmed"
            else query_image
        )

        # ────────────────────────────────────────────────────────────
        # STEP 2 — CLIP: Query Encoding
        # ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("② Query Encoding  —  CLIP")
        st.caption(
            f"The cropped image is embedded by CLIP (ViT-B/32). "
            f"When α < 1, a BLIP-2 caption is used to produce the fused embedding "
            f"v = α·φ_V(x̂) + (1−α)·φ_T(c).  "
            f"Current **α = {alpha:.2f}** (checkpoint: *{selected_ckpt_label}*)."
        )

        with st.spinner("Loading CLIP and encoding query image..."):
            t0 = time.time()
            clip = load_clip(selected_ckpt_path)
            img_emb = clip.encode_image(target)

            # Optionally fuse with a caption embedding when α < 1
            if alpha < 1.0 and show_query_caption:
                try:
                    blip2 = load_blip2()
                    if st.session_state.query_caption is None:
                        st.session_state.query_caption = blip2.generate_caption(target)
                    query_emb = fuse_embeddings(img_emb, clip.encode_text(st.session_state.query_caption), alpha)
                except Exception:
                    query_emb = img_emb   # fall back to image-only
            else:
                query_emb = img_emb

            query_emb_2d = query_emb.reshape(1, -1).astype(np.float32)
            elapsed_clip = time.time() - t0

        # Show embedding preview
        preview = query_emb[:16]
        st.code(
            f"Embedding (first 16 / {config.EMBEDDING_DIM} dims):\n"
            f"[{', '.join(f'{v:.4f}' for v in preview)}, ...]",
            language=None,
        )
        st.success(
            f"Query embedded  |  shape: {query_emb_2d.shape}  |  "
            f"L2 norm: {float(np.linalg.norm(query_emb_2d)):.4f}  |  "
            f"time: {elapsed_clip*1000:.0f} ms"
        )

        # Optionally generate + display BLIP-2 query caption
        if show_query_caption:
            if st.session_state.query_caption is None:
                with st.spinner("Generating BLIP-2 caption for query image..."):
                    try:
                        blip2 = load_blip2()
                        st.session_state.query_caption = blip2.generate_caption(target)
                    except Exception as e:
                        st.session_state.query_caption = f"(unavailable — {e})"
            st.markdown(
                f"**BLIP-2 query caption:** *{st.session_state.query_caption}*"
            )

        # ────────────────────────────────────────────────────────────
        # STEP 3 — FAISS: Candidate Retrieval
        # ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("③ Candidate Retrieval  —  FAISS / HNSW")
        st.caption(
            "The query embedding is compared against all gallery embeddings "
            "using cosine similarity (inner product on L2-normalised vectors) "
            "via an Approximate Nearest Neighbour index."
        )

        with st.spinner("Searching FAISS index..."):
            t0 = time.time()
            idx_path = os.path.join(config.INDEX_DIR, f"index_{selected_index}.faiss")
            mt_path  = os.path.join(config.INDEX_DIR, f"metadata_{selected_index}.json")
            index, metadata = load_faiss_index(idx_path, mt_path)

            # Fetch extra candidates when re-ranking will follow
            search_k = min(top_k * 3 if use_reranking else top_k, index.ntotal)
            distances, faiss_indices = index.search(query_emb_2d, search_k)
            elapsed_faiss = time.time() - t0

        candidates = [
            {
                "image_path": metadata[idx]["image_path"],
                "item_id":    metadata[idx]["item_id"],
                "score":      float(dist),
                "rank":       r + 1,
                "itm_score":  None,
            }
            for r, (dist, idx) in enumerate(zip(distances[0], faiss_indices[0]))
            if 0 <= idx < len(metadata)
        ]

        c1, c2, c3 = st.columns(3)
        c1.metric("Gallery size", f"{index.ntotal:,}")
        c2.metric("Candidates fetched", len(candidates))
        c3.metric("Search time", f"{elapsed_faiss*1000:.0f} ms")
        st.success(f"Top-{len(candidates)} candidates retrieved (will show top-{top_k} after re-ranking)")

        # ────────────────────────────────────────────────────────────
        # STEP 4 — BLIP-2: ITM Re-ranking
        # ────────────────────────────────────────────────────────────
        st.markdown("---")
        if use_reranking:
            st.subheader("④ Semantic Re-ranking  —  BLIP-2 ITM")
            st.caption(
                "A BLIP-2 Image-Text Matching score is computed between the query image "
                "and each candidate's pre-computed caption. Candidates are re-ranked by "
                "ITM score to boost semantic precision."
            )
            with st.spinner(f"Re-ranking {len(candidates)} candidates via BLIP-2 ITM..."):
                t0 = time.time()
                try:
                    blip2 = load_blip2()
                    captions = load_captions()
                    candidates = blip2.rerank_candidates(target, candidates, captions)
                    elapsed_itm = time.time() - t0
                    st.success(
                        f"Re-ranked {len(candidates)} candidates by ITM score  |  "
                        f"time: {elapsed_itm*1000:.0f} ms"
                    )
                except Exception as e:
                    st.warning(f"Re-ranking failed ({e}) — displaying FAISS-ranked results.")
        else:
            st.subheader("④ Semantic Re-ranking  —  BLIP-2 ITM  *(disabled)*")
            st.caption("Enable **BLIP-2 ITM Re-ranking** in the sidebar to activate Step 4.")

        # ════════════════════════════════════════════════════════════
        # Results grid
        # ════════════════════════════════════════════════════════════
        st.markdown("---")
        results = candidates[:top_k]
        st.subheader(f"Top-{len(results)} Retrieved Products")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Results shown", len(results))
        m2.metric("Best similarity", f"{results[0]['score']:.4f}" if results else "—")
        m3.metric("Re-ranking", "ON" if use_reranking else "OFF")
        m4.metric("α (image weight)", f"{alpha:.2f}")

        cols_per_row = 5
        for row_start in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx >= len(results):
                    break
                r = results[idx]
                with col:
                    if os.path.exists(r["image_path"]):
                        st.image(load_image(r["image_path"]), use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='background:#f0f0f0;height:160px;display:flex;"
                            "align-items:center;justify-content:center;border-radius:6px;'>"
                            "🖼️<br><small>not available</small></div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(f"**Rank {r['rank']}**")
                    st.caption(f"`{r['item_id']}`")
                    st.caption(f"Sim: `{r['score']:.4f}`")
                    if r.get("itm_score") is not None:
                        st.caption(f"ITM: `{r['itm_score']:.4f}`")

    elif st.session_state.confirmed and not selected_index:
        st.error(
            "No FAISS index found. "
            "Place your `index_*.faiss` and `metadata_*.json` files inside "
            "`outputs/faiss_index/` and re-run Cell 4."
        )

    # Cleanup temp file
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

else:
    st.markdown("---")
    st.info("Upload a clothing image above to start the pipeline.")
    st.markdown("""
**How the pipeline works once you upload:**

| Step | Module | What happens |
|------|--------|--------------|
| ① | **YOLO** | Detects clothing region; crops background noise |
| ✅ | **You** | Confirm crop or choose full image |
| ② | **CLIP** | Encodes query into a 512-d fused embedding (Eq. 1) |
| ③ | **FAISS** | ANN search returns top-K candidates via cosine similarity |
| ④ | **BLIP-2 ITM** | Re-ranks candidates using image–text matching score |
""")
