# app.py (FAST RAG - FINAL STABLE VERSION - FIXED DESCRIPTION)
import os
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import faiss
from collections import Counter
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration
)

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="RAG Image Retrieval (Fast)",
    page_icon="🐶🐱",
    layout="wide"
)
st.title("🐶🐱 Image Retrieval & Description (Fast RAG)")
st.caption("CLIP + FAISS + BLIP + FLAN-T5 (Optimized, Non-hallucinative)")

DATASET_ROOT = os.path.join("archive", "images")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    labels = np.load("models/labels_clean.npy", allow_pickle=True)
    image_paths = np.load("models/image_paths.npy", allow_pickle=True)
    index = faiss.read_index("models/faiss.index")

    # CLIP
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=dtype
    ).to(device)
    clip_model.eval()

    # BLIP BASE (FAST & STABLE)
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=dtype
    ).to(device)
    blip_model.eval()

    # FLAN-T5 BASE (FAST & CONTROLLED)
    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    t5_model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-base",
        torch_dtype=dtype
    ).to(device)
    t5_model.eval()

    return (
        labels, image_paths, index,
        clip_processor, clip_model,
        blip_processor, blip_model,
        t5_tokenizer, t5_model
    )

(
    labels, image_paths, index,
    clip_processor, clip_model,
    blip_processor, blip_model,
    t5_tokenizer, t5_model
) = load_artifacts()

# =========================
# HELPERS
# =========================
def resolve_image_path(p):
    return os.path.join(DATASET_ROOT, os.path.basename(p))

def center_crop(img, size):
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m)//2, (h - m)//2, (w + m)//2, (h + m)//2))
    return img.resize((size, size), Image.LANCZOS)

def search_similar(img, k):
    inputs = clip_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        q = clip_model.get_image_features(**inputs)
        q = F.normalize(q, dim=-1)
        q = q.cpu().numpy().astype("float32")
    faiss.normalize_L2(q)
    d, i = index.search(q, k)
    return d[0], i[0]

def dominant_breed(indices):
    breeds = [labels[int(i)] for i in indices]
    b, c = Counter(breeds).most_common(1)[0]
    return b, c / len(breeds)

# =========================
# GENERATIVE (FINAL FIX)
# =========================
def generate_caption_fast(img):
    inputs = blip_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = blip_model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=1,
            do_sample=False
        )
    return blip_processor.decode(out[0], skip_special_tokens=True)

def generate_description_fast(breed, confidence, visual_caption, top_indices=None):
    """
    Perbaikan:
    - Tambahkan Top-K retrieval breeds
    - Pisahkan action dan location dari BLIP caption
    - Gunakan beam search tanpa sampling
    """

    if confidence >= 0.85:
        certainty_sentence = (
            "The identification is supported by strong visual similarity to reference images."
        )
    else:
        certainty_sentence = (
            "The identification is based on visual similarity to reference images, and some uncertainty remains."
        )

    # Extract action & location
    if " in " in visual_caption:
        parts = visual_caption.split(" in ", 1)
        action_part = parts[0]
        location_part = "in " + parts[1]
    else:
        action_part = visual_caption
        location_part = ""

    # Top-K retrieval breeds
    if top_indices is not None:
        retrieval_context = ", ".join([labels[int(i)] for i in top_indices[:5]])
        retrieval_text = f" Other similar breeds: {retrieval_context}."
    else:
        retrieval_text = ""

    base_text = (
        f"The image shows a {breed.lower()} {action_part} {location_part}. "
        f"{certainty_sentence}{retrieval_text}"
    )

    prompt = f"""
Rewrite the following text into a natural, concise descriptive paragraph.
Keep it factual and coherent.
Do NOT add new details or assumptions.

Text:
{base_text}
"""

    inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = t5_model.generate(
            **inputs,
            max_new_tokens=180,
            num_beams=4,
            do_sample=False
        )

    return t5_tokenizer.decode(out[0], skip_special_tokens=True)

# =========================
# UI
# =========================
with st.sidebar:
    top_k = st.slider("Top-K retrieval", 3, 8, 5)
    img_size = st.selectbox("Top-K image size", [160, 200, 240], index=1)

uploaded = st.file_uploader("Upload cat/dog image", ["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(center_crop(img, 260), width=260, caption="Query Image")

    with st.spinner("Processing..."):
        distances, indices = search_similar(img, top_k)
        breed, conf = dominant_breed(indices)
        caption = generate_caption_fast(img)
        description = generate_description_fast(breed, conf, caption, top_indices=indices)

    with col2:
        st.subheader("Result")
        st.write(f"**Predicted Breed:** {breed}")
        st.write(f"**Confidence:** {conf*100:.0f}%")
        st.write(f"**Visual Caption (BLIP):** {caption}")
        st.markdown(description)

    st.markdown("---")
    st.subheader("Top Retrieved Images")

    cols = st.columns(top_k)
    for c, idx, score in zip(cols, indices, distances):
        with c:
            path = resolve_image_path(image_paths[int(idx)])
            if os.path.exists(path):
                ref = center_crop(Image.open(path).convert("RGB"), img_size)
                st.image(ref, width=img_size)
                st.caption(f"{labels[int(idx)]} ({score:.2f})")

else:
    st.info("Upload an image to start.")
