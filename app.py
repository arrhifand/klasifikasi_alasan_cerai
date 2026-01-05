import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ===== BERT =====
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Klasifikasi Alasan Perceraian",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Klasifikasi Alasan Perceraian")
st.caption("SVM • Logistic Regression • IndoBERT")

# ======================================================
# LOAD MODEL SVM & LOGREG
# ======================================================
@st.cache_resource
def load_models():
    base_dir = Path(__file__).parent
    model_dir = base_dir / "save_models"

    svm_path = model_dir / "svm_linear_tfidf_model.joblib"
    logreg_path = model_dir / "logreg_tfidf_model.joblib"
    tfidf_path = model_dir / "tfidf_vectorizer.joblib"

    svm_model = joblib.load(svm_path)
    logreg_model = joblib.load(logreg_path)
    vectorizer = joblib.load(tfidf_path)

    return svm_model, logreg_model, vectorizer


# ======================================================
# LOAD MODEL BERT
# ======================================================
@st.cache_resource
def load_models():
    base_dir = Path(__file__).parent
    model_dir = base_dir / "save_models"

    svm_path = model_dir / "svm_linear_tfidf_model.joblib"
    logreg_path = model_dir / "log_reg_tfidf_model.joblib"
    vec_path = model_dir / "tfidf_vectorizer.joblib"

    missing = [p.name for p in [svm_path, logreg_path, vec_path] if not p.exists()]

    if missing:
        st.error("❌ Model tidak ditemukan:")
        for m in missing:
            st.code(f"save_models/{m}")
        st.stop()

    return (
        joblib.load(svm_path),
        joblib.load(logreg_path),
        joblib.load(vec_path)
    )


# ======================================================
# FUNGSI KLASIFIKASI
# ======================================================
def classify_text_ml(text, model, vectorizer):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X).max()
    return pred, proba


def classify_text_bert(text, tokenizer, model):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return pred.item(), conf.item()


# ======================================================
# LOAD SEMUA MODEL
# ======================================================
svm_model, logreg_model, vectorizer = load_models()

# ======================================================
# INPUT USER
# ======================================================
st.subheader("📝 Input Teks Putusan")
text_input = st.text_area(
    "Masukkan teks alasan perceraian:",
    height=200,
    placeholder="Contoh: Penggugat mengajukan cerai karena terjadi perselisihan..."
)

model_choice = st.selectbox(
    "Pilih Model Klasifikasi",
    ["SVM", "Logistic Regression", "BERT (IndoBERT)"]
)

# ======================================================
# PROSES KLASIFIKASI
# ======================================================
if st.button("🔍 Klasifikasikan") and text_input.strip() != "":
    with st.spinner("⏳ Memproses klasifikasi..."):

        if model_choice == "SVM":
            pred, prob = classify_text_ml(text_input, svm_model, vectorizer)
            st.success("✅ Hasil Klasifikasi (SVM)")
            st.write(f"**Label:** {pred}")
            st.write(f"**Confidence:** {prob*100:.2f}%")

        elif model_choice == "Logistic Regression":
            pred, prob = classify_text_ml(text_input, logreg_model, vectorizer)
            st.success("✅ Hasil Klasifikasi (Logistic Regression)")
            st.write(f"**Label:** {pred}")
            st.write(f"**Confidence:** {prob*100:.2f}%")

        else:
            if bert_model is None:
                st.error("❌ Model BERT tidak tersedia.")
            else:
                pred, prob = classify_text_bert(text_input, bert_tokenizer, bert_model)
                st.success("✅ Hasil Klasifikasi (IndoBERT)")
                st.write(f"**Label:** {pred}")
                st.write(f"**Confidence:** {prob*100:.2f}%")

# ======================================================
# EVALUASI MODEL (AKURASI & CONFUSION MATRIX)
# ======================================================
st.divider()
st.subheader("📊 Evaluasi Model (Contoh Dataset Uji)")

uploaded_file = st.file_uploader(
    "Upload CSV (kolom: text, label)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X_text = df["text"]
    y_true = df["label"]

    if model_choice in ["SVM", "Logistic Regression"]:
        model = svm_model if model_choice == "SVM" else logreg_model
        X_vec = vectorizer.transform(X_text)
        y_pred = model.predict(X_vec)

    else:
        if bert_model is None:
            st.error("❌ Model BERT tidak tersedia.")
            st.stop()

        y_pred = []
        for text in X_text:
            pred, _ = classify_text_bert(text, bert_tokenizer, bert_model)
            y_pred.append(pred)

    # ===== METRIK =====
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    st.write(f"### ✅ Akurasi: **{acc*100:.2f}%**")

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)
