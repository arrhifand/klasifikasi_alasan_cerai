import streamlit as st
import joblib
import numpy as np
import io
from pathlib import Path
import pdfplumber
import pandas as pd
import re
import matplotlib.pyplot as plt

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Klasifikasi Putusan Perceraian",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# SESSION STATE FOR THEME
# ==========================
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# ==========================
# CUSTOM CSS - LIGHT MODE
# ==========================
light_mode_css = """
<style>
.stApp { background-color: #f8fafc; color: #0f172a; }
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
}
.header-container {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    padding: 1rem;
    border-radius: 10px;
}
.result-card {
    background-color: #ffffff;
    border-left: 5px solid #4f46e5;
    padding: 1rem;
    border-radius: 10px;
}
.section-title {
    color: #4f46e5;
    border-bottom: 2px solid #4f46e5;
    margin-bottom: 1rem;
}
.stButton > button {
    background-color: #4f46e5;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}
</style>
"""

# ==========================
# CUSTOM CSS - DARK MODE
# ==========================
dark_mode_css = """
<style>
.stApp { background-color: #0f172a; color: #e5e7eb; }
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
}
.header-container {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    color: white;
    padding: 1rem;
    border-radius: 10px;
}
.result-card {
    background-color: #020617;
    border-left: 5px solid #7c3aed;
    padding: 1rem;
    border-radius: 10px;
}
.section-title {
    color: #a855f7;
    border-bottom: 2px solid #7c3aed;
    margin-bottom: 1rem;
}
.stButton > button {
    background-color: #7c3aed;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}
</style>
"""

# Apply theme
st.markdown(
    light_mode_css if st.session_state.theme == "light" else dark_mode_css,
    unsafe_allow_html=True
)

# ==========================
# LOAD MODEL & VECTORIZER
# ==========================
@st.cache_resource
def load_models():
    base_dir = Path(__file__).parent
    model_dir = base_dir / "save_models"

    svm_path = model_dir / "svm_linear_tfidf_model.joblib"
    logreg_path = model_dir / "log_reg_tfidf_model.joblib"
    vec_path = model_dir / "tfidf_vectorizer.joblib"

    missing = [p.name for p in [svm_path, logreg_path, vec_path] if not p.exists()]

    if missing:
        st.error(
            "❌ File model tidak ditemukan:\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nPastikan file `.joblib` ada di folder `save_models/`."
        )
        st.stop()

    svm_model = joblib.load(svm_path)
    logreg_model = joblib.load(logreg_path)
    vectorizer = joblib.load(vec_path)

    return svm_model, logreg_model, vectorizer


svm_model, logreg_model, vectorizer = load_models()

# ==========================
# SIMPLE STEMMER
# ==========================
def simple_stem(word):
    word = word.lower()
    prefixes = ['di', 'ke', 'me', 'be', 'ter', 'per']
    suffixes = ['kan', 'an', 'i', 'lah', 'nya', 'kah']

    for p in prefixes:
        if word.startswith(p) and len(word) > len(p) + 2:
            word = word[len(p):]
            break

    for s in suffixes:
        if word.endswith(s) and len(word) > len(s) + 2:
            word = word[:-len(s)]
            break

    return word


def preprocess_text(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return " ".join(simple_stem(t) for t in tokens)

# ==========================
# KATA KUNCI KATEGORI
# ==========================
KEYWORDS_BY_CATEGORY = {
    "KDRT": ["aniaya", "pukul", "kekerasan", "tampar", "cekik"],
    "Perselingkuhan": ["selingkuh", "zina", "wil"],
    "Ekonomi": ["nafkah", "uang", "hutang", "penghasilan"],
    "Pertengkaran": ["cekcok", "bertengkar", "ribut"],
    "Pisah Rumah": ["pisah", "meninggalkan", "pulang"],
    "Moral": ["judi", "mabuk", "narkoba"]
}
KEYWORDS_BY_CATEGORY = {
    k: [simple_stem(w) for w in v]
    for k, v in KEYWORDS_BY_CATEGORY.items()
}

# ==========================
# KLASIFIKASI
# ==========================
def classify_text(text, model, vectorizer):
    processed = preprocess_text(text)
    X = vectorizer.transform([processed])
    pred = model.predict(X)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(np.max(model.predict_proba(X)))
    elif hasattr(model, "decision_function"):
        d = model.decision_function(X)
        d = d[0] if isinstance(d, np.ndarray) else d
        prob = float(1 / (1 + np.exp(-d)))

    return pred, prob

# ==========================
# EKSTRAK TEKS FILE
# ==========================
def extract_text(uploaded):
    if uploaded.name.endswith(".txt"):
        return uploaded.read().decode("utf-8", errors="ignore")

    if uploaded.name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)

    return ""

# ==========================
# UI
# ==========================
st.markdown("""
<div class="header-container">
    <h1>⚖️ Sistem Klasifikasi Putusan Perceraian</h1>
    <p>Analisis otomatis alasan perceraian berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    if st.button("☀️ Terang"):
        st.session_state.theme = "light"
        st.rerun()
    if st.button("🌙 Gelap"):
        st.session_state.theme = "dark"
        st.rerun()

    model_choice = st.selectbox(
        "Pilih Model",
        ["SVM", "Logistic Regression"]
    )

uploaded = st.file_uploader(
    "Upload dokumen putusan (TXT / PDF)",
    type=["txt", "pdf"]
)

if st.button("🔍 Mulai Klasifikasi"):
    if uploaded is None:
        st.error("❌ Silakan upload file terlebih dahulu")
    else:
        text = extract_text(uploaded)
        model = svm_model if model_choice == "SVM" else logreg_model
        pred, prob = classify_text(text, model, vectorizer)

        st.markdown('<div class="section-title">📌 Hasil Klasifikasi</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-card">
            <h2>{pred}</h2>
            <p>Confidence: <strong>{prob*100:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<p style="text-align:center;margin-top:2rem">
Dikembangkan oleh <strong>Arya</strong> © 2025
</p>
""", unsafe_allow_html=True)

