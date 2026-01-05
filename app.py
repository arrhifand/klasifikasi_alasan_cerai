import streamlit as st
import joblib
import numpy as np
import io
from pathlib import Path
import pdfplumber
import pandas as pd
import re
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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
# CUSTOM CSS STYLING - LIGHT MODE
# ==========================
light_mode_css = """
<style>
/* ============================= */
/* Background utama */
/* ============================= */
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

/* ============================= */
/* Sidebar */
/* ============================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: #0f172a;
}

/* ============================= */
/* Header */
/* ============================= */
.header-container {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    padding: 1rem;
    border-radius: 10px;
}

/* ============================= */
/* Card */
/* ============================= */
.result-card {
    background-color: #ffffff;
    color: #0f172a;
    border-left: 5px solid #4f46e5;
}

/* ============================= */
/* Judul section */
/* ============================= */
.section-title {
    color: #4f46e5;
    border-bottom: 2px solid #4f46e5;
}

/* ============================= */
/* Button biasa */
/* ============================= */
.stButton > button {
    background-color: #4f46e5;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #4338ca;
    color: white;
}

/* ============================= */
/* FILE UPLOADER (Browse files) */
/* ============================= */

/* Container */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 10px;
    border: 2px dashed #4f46e5;
}

/* Sembunyikan input asli */
[data-testid="stFileUploader"] input[type="file"] {
    display: none;
}

/* Tombol Browse files */
[data-testid="stFileUploader"] label {
    background-color: #e0e7ff;
    color: #1e293b !important;   /* 🔴 WARNA TULISAN */
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    display: inline-block;
    transition: all 0.2s ease;
}

/* Hover Browse files */
[data-testid="stFileUploader"] label:hover {
    background-color: #c7d2fe;
    color: #0f172a !important;
}

/* Text bantuan */
[data-testid="stFileUploader"] small {
    color: #475569;
}

/* ============================= */
/* Browse files button (secondary) */
/* ============================= */
button[data-testid="stBaseButton-secondary"] {
    background-color: #e0e7ff;   /* background tombol */
    color: #1e293b !important;   /* WARNA TULISAN */
    font-weight: 600;
    border-radius: 8px;
    border: 1px solid #4f46e5;
    padding: 0.5rem 1.2rem;
}

/* Hover */
button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #c7d2fe;
    color: #0f172a !important;
}

[data-testid="stMarkdownContainer"] {
    color: #000000 !important;
}

div.st-emotion-cache-1q82h82,
div.exsm2dv3 {
    color: #000000 !important;
}

</style>
"""



# ==========================
# CUSTOM CSS STYLING - DARK MODE
# ==========================
dark_mode_css = """
<style>
/* ============================= */
/* Background utama */
/* ============================= */
.stApp {
    background-color: #0f172a;   /* gelap */
    color: #e5e7eb;              /* teks terang */
}

/* ============================= */
/* Sidebar */
/* ============================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    color: #e5e7eb;
}

/* ============================= */
/* Header */
/* ============================= */
.header-container {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    color: #ffffff;
    padding: 1rem;
    border-radius: 10px;
}

/* ============================= */
/* Card */
/* ============================= */
.result-card {
    background-color: #020617;
    color: #e5e7eb;
    border-left: 5px solid #7c3aed;
}

/* ============================= */
/* Judul section */
/* ============================= */
.section-title {
    color: #a855f7;
    border-bottom: 2px solid #7c3aed;
}

/* ============================= */
/* Button biasa */
/* ============================= */
.stButton > button {
    background-color: #7c3aed;
    color: #ffffff;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #6d28d9;
    color: #ffffff;
}

/* ============================= */
/* FILE UPLOADER (Browse files) */
/* ============================= */

/* Container */
[data-testid="stFileUploader"] {
    background-color: #020617;
    padding: 1rem;
    border-radius: 10px;
    border: 2px dashed #7c3aed;
}

/* Sembunyikan input asli */
[data-testid="stFileUploader"] input[type="file"] {
    display: none;
}

/* Tombol Browse files */
[data-testid="stFileUploader"] label {
    background-color: #1e1b4b;
    color: #e5e7eb !important;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    display: inline-block;
    transition: all 0.2s ease;
}

/* Hover Browse files */
[data-testid="stFileUploader"] label:hover {
    background-color: #312e81;
    color: #ffffff !important;
}

/* Text bantuan */
[data-testid="stFileUploader"] small {
    color: #c7d2fe;
}

/* ============================= */
/* Browse files button (secondary) */
/* ============================= */
button[data-testid="stBaseButton-secondary"] {
    background-color: #1e1b4b;
    color: #e5e7eb !important;
    font-weight: 600;
    border-radius: 8px;
    border: 1px solid #7c3aed;
    padding: 0.5rem 1.2rem;
}

/* Hover */
button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #312e81;
    color: #ffffff !important;
}
</style>
"""

# Apply theme CSS
if st.session_state.theme == 'light':
    st.markdown(light_mode_css, unsafe_allow_html=True)
else:
    st.markdown(dark_mode_css, unsafe_allow_html=True)

# ==========================
# LOAD MODEL DAN VECTORIZER
# ==========================

@st.cache_resource
def load_models():
    base_dir = Path(__file__).parent
    model_dir = base_dir / "save_models"
    svm_path = model_dir / "svm_linear_tfidf_model.joblib"
    logreg_path = model_dir / "log_reg_tfidf_model.joblib"
    vec_path = model_dir / "tfidf_vectorizer.joblib"

    try:
        svm_model = joblib.load(svm_path)
        logreg_model = joblib.load(logreg_path)
        vectorizer = joblib.load(vec_path)
        return svm_model, logreg_model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e.filename}. Pastikan folder 'save_models' dan file model ada di repo.")
        raise
svm_model, logreg_model, vectorizer = load_models()

# ===========================
# SIMPLE INDONESIAN STEMMER
# ===========================
def simple_stem(word):
    """Simple Indonesian stemmer tanpa dependency eksternal."""
    word = word.lower()
    
    # Hapus awalan umum
    prefixes = ['di', 'ke', 'me', 'be', 'ter', 'per']
    for prefix in prefixes:
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            word = word[len(prefix):]
            break
    
    # Hapus akhiran umum
    suffixes = ['kan', 'an', 'i', 'lah', 'nya', 'kah']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            word = word[:-len(suffix)]
            break
    
    return word


def preprocess_text(text: str) -> str:
    """Preprocess teks menjadi bentuk yang sesuai untuk vectorizer:
    - lowercase
    - tokenisasi sederhana
    - simple stemming per token
    """
    if not text:
        return ""
    txt = text.lower()
    tokens = re.findall(r"\b\w+\b", txt)
    stemmed = [simple_stem(t) for t in tokens]
    return " ".join(stemmed)

# Daftar kata kunci untuk setiap kategori alasan perceraian
KEYWORDS_BY_CATEGORY = {
    'KDRT': [simple_stem(word) for word in ['aniaya', 'pukul', 'kekerasan', 'banting', 'ancam', 'bentak', 'tampar', 'cekik', 'caci maki', 'kdrt']],
    'Perselingkuhan': [simple_stem(word) for word in ['selingkuh', 'orang ketiga', 'wil', 'pria idaman lain', 'wanita idaman lain', 'zina']],
    'Ekonomi': [simple_stem(word) for word in ['nafkah', 'uang', 'bekerja', 'gaji', 'usaha', 'ekonomi', 'hutang', 'modal', 'penghasilan', 'biaya']],
    'Pertengkaran terus-menerus': [simple_stem(word) for word in ['bertengkar', 'cekcok', 'ribut', 'berselisih', 'konflik', 'pertengkaran']],
    'Pisah rumah': [simple_stem(word) for word in ['pisah', 'tinggal beda', 'pulang', 'mengungsi', 'meninggalkan rumah', 'kembali ke orang tua', 'pisahan']],
    'Masalah moral': [simple_stem(word) for word in ['zina', 'judi', 'mabuk', 'narkoba', 'asusila', 'miras', 'maksiat']]
}

# ==========================
# FUNGSI PREDIKSI
# ==========================

def classify_text(text, model, vectorizer):
    # preprocess text the same way training likely did (tokenize + stem)
    processed = preprocess_text(text)
    X = vectorizer.transform([processed])
    prediction = model.predict(X)[0]
    probability = None

    # jika model support predict_proba (untuk LogisticRegression)
    if hasattr(model, "predict_proba"):
        probability = float(np.max(model.predict_proba(X)))
    # untuk SVM, gunakan decision_function untuk mendapatkan confidence
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        # extract scalar value if it's an array
        if isinstance(decision, np.ndarray):
            # untuk multi-class, ambil max; untuk binary, ambil nilai [0]
            if decision.ndim > 1:
                decision = np.max(np.abs(decision))
            else:
                decision = decision[0]
        # normalisasi decision_function ke range [0, 1] menggunakan sigmoid
        probability = float(1 / (1 + np.exp(-float(decision))))

    return prediction, probability


def format_structured_text(text: str) -> dict:
    """Bersihkan dan strukturkan teks.

    Mengembalikan dict berisi: sentences (list), summary (str), counts (dict).
    """
    # normalisasi whitespace
    txt = text.replace('\r', ' ').replace('\n', ' ').strip()
    txt = re.sub(r"\s+", ' ', txt)

    # pisah kalimat sederhana berdasarkan .!? diikuti spasi
    sentences = re.split(r'(?<=[\.\!\?])\s+', txt)
    # bersihkan dan hanya ambil yang bukan kosong
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    # pastikan tiap kalimat diawali huruf kapital
    sentences = [s[0].upper() + s[1:] if len(s) > 1 else s.upper() for s in sentences]

    # ringkasan sederhana: 1-3 kalimat pertama
    summary = ' '.join(sentences[:3]) if sentences else ''

    counts = {
        'sentences': len(sentences),
        'words': len(txt.split()),
        'characters': len(txt)
    }

    return {'sentences': sentences, 'summary': summary, 'counts': counts}


def extract_key_sentences(text: str, top_k=5):
    """Ekstrak kalimat penting berdasarkan kategori alasan perceraian.
    
    Menggunakan simple stemmer untuk matching yang lebih akurat.
    Mengembalikan dict berisi hasil ekstraksi per kategori.
    """
    if not text or len(text.strip()) == 0:
        return {}

    # tokenisasi sederhana dan stem
    words_raw = re.findall(r'\b\w+\b', text.lower())
    words_stemmed = [simple_stem(w) for w in words_raw]

    structured = format_structured_text(text)
    sentences = structured['sentences']
    lowered_sentences = [s.lower() for s in sentences]

    # ekstrak kalimat per kategori
    results = {}
    for category, keywords in KEYWORDS_BY_CATEGORY.items():
        matched_sents = []
        for i, s in enumerate(lowered_sentences):
            s_words_stemmed = [simple_stem(w) for w in re.findall(r'\b\w+\b', s)]
            for kw in keywords:
                if kw in s_words_stemmed:
                    if sentences[i] not in matched_sents:
                        matched_sents.append(sentences[i])
                    break
        
        results[category] = matched_sents[:top_k]

    return results


# ==========================
# EKSTRAK TEKS DARI FILE
# ==========================
def extract_text_from_uploaded(uploaded):
    """Ekstrak teks dari file TXT atau PDF."""
    if uploaded is None:
        return ""
    file_type = uploaded.type
    try:
        uploaded.seek(0)
    except Exception:
        pass

    # Text file
    if uploaded.name.lower().endswith('.txt') or file_type == 'text/plain':
        raw = uploaded.read()
        try:
            return raw.decode('utf-8')
        except Exception:
            return raw.decode('latin-1', errors='ignore')

    # PDF file
    if uploaded.name.lower().endswith('.pdf') or file_type == 'application/pdf':
        try:
            uploaded.seek(0)
            with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
                texts = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(texts)
        except Exception:
            return ""

    return ""


# ==========================
# STREAMLIT UI
# ==========================

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">⚖️ Sistem Klasifikasi Putusan Perceraian</h1>
        <p class="header-subtitle">Analisis otomatis alasan perceraian menggunakan Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar untuk settings
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    
    # Theme toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("☀️ Terang", use_container_width=True, key="light_btn"):
            st.session_state.theme = 'light'
            st.rerun()
    with col2:
        if st.button("🌙 Gelap", use_container_width=True, key="dark_btn"):
            st.session_state.theme = 'dark'
            st.rerun()
    
    st.markdown("---")
    
    model_choice = st.selectbox(
        "Pilih Model Klasifikasi",
        ["SVM", "Logistic Regression"],
        help="SVM: Support Vector Machine, LR: Logistic Regression"
    )
    st.markdown("---")
    st.markdown("### 📋 Tentang Aplikasi")
    st.info("""
    Aplikasi ini menggunakan Machine Learning untuk mengklasifikasi dan menganalisis putusan perceraian. 
    - **Input**: File TXT atau PDF
    - **Output**: Klasifikasi, alasan perceraian, dan statistik teks
    """)

# Main content
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Dokumen Putusan")
    uploaded_file = st.file_uploader(
        "Pilih file putusan (TXT atau PDF)",
        type=["txt", "pdf"],
        help="Upload dokumen putusan perceraian untuk dianalisis"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File '{uploaded_file.name}' berhasil diupload")

with col2:
    st.markdown("### 📊 Info Model")
    st.metric("Model Aktif", model_choice)
    st.metric("Fitur Ekstraksi", "6 Kategori")

st.markdown("---")

# Button untuk klasifikasi
if st.button("🔍 Mulai Klasifikasi", use_container_width=True, key="classify_btn"):
    if uploaded_file is None:
        st.error("❌ Silakan upload file putusan terlebih dahulu!")
    else:
        text_to_classify = extract_text_from_uploaded(uploaded_file)
        if not text_to_classify or len(text_to_classify.strip()) == 0:
            st.error("❌ Tidak dapat mengekstrak teks dari file. Pastikan file berupa TXT atau PDF yang berisi teks.")
        else:
            with st.spinner("⏳ Sedang menganalisis dokumen..."):
                # jalankan klasifikasi pada kedua model (SVM & Logistic Regression)
                pred_svm, prob_svm = classify_text(text_to_classify, svm_model, vectorizer)
                pred_lr, prob_lr = classify_text(text_to_classify, logreg_model, vectorizer)

                # Hasil Perbandingan Klasifikasi
                st.markdown('<div class="section-title">📌 Hasil Klasifikasi — Perbandingan Model</div>', unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    st.markdown(f"""
                        <div class="result-card">
                        <h3 style="margin-top: 0;">SVM (Support Vector Machine)</h3>
                        <h2 style="color: #667eea; margin: 0.5rem 0;">{pred_svm}</h2>
                        <p style="margin:0;">Confidence: <strong>{(prob_svm*100) if prob_svm is not None else 'N/A' :.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                        <div class="result-card">
                        <h3 style="margin-top: 0;">Logistic Regression</h3>
                        <h2 style="color: #667eea; margin: 0.5rem 0;">{pred_lr}</h2>
                        <p style="margin:0;">Confidence: <strong>{(prob_lr*100) if prob_lr is not None else 'N/A' :.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                # Agreement and recommendation
                agreement = (pred_svm == pred_lr)
                if agreement:
                    st.success(f"✅ Kedua model setuju pada label: {pred_svm}")
                    recommended = pred_svm
                else:
                    # pilih rekomendasi berdasarkan confidence (jika tersedia)
                    s_conf = prob_svm if prob_svm is not None else 0.0
                    l_conf = prob_lr if prob_lr is not None else 0.0
                    if s_conf >= l_conf:
                        recommended = pred_svm
                    else:
                        recommended = pred_lr
                    st.warning(f"⚠️ Perbedaan prediksi — SVM: {pred_svm} ({s_conf*100:.2f}%), LR: {pred_lr} ({l_conf*100:.2f}%). Rekomendasi: {recommended}")

                # Ekstrak kalimat penting berdasarkan kategori alasan
                key_sentences_by_category = extract_key_sentences(text_to_classify)

                st.markdown('<div class="section-title">🔎 Alasan Perceraian yang Terdeteksi</div>', unsafe_allow_html=True)
                
                found_any = False
                for category, sentences in key_sentences_by_category.items():
                    if sentences:
                        found_any = True
                        with st.expander(f"📍 **{category}** ({len(sentences)} kalimat)", expanded=(category == list(key_sentences_by_category.keys())[0])):
                            for i, s in enumerate(sentences, start=1):
                                st.markdown(f"**{i}.** {s}")

                if not found_any:
                    st.warning("⚠️ Tidak ditemukan kalimat yang cocok dengan kategori alasan perceraian terdeteksi")

                # Tampilkan ringkasan dan statistik teks
                structured = format_structured_text(text_to_classify)

                st.markdown('<div class="section-title">📄 Analisis Teks</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3, gap="large")
                with col1:
                    st.metric("📊 Jumlah Kalimat", structured['counts']['sentences'])
                with col2:
                    st.metric("📝 Jumlah Kata", structured['counts']['words'])
                with col3:
                    st.metric("🔤 Jumlah Karakter", structured['counts']['characters'])

                with st.expander("📋 Ringkasan Teks", expanded=True):
                    if structured['summary']:
                        st.write(structured['summary'])
                    else:
                        st.info("Tidak ada teks yang dapat diringkas")

st.markdown("""
    <div class="footer">
    <p>Dikembangkan oleh <strong>Arya</strong> | Sistem Klasifikasi Putusan Perceraian © 2025</p>
    <p style="font-size: 0.85rem; color: #aaa;">Powered by Streamlit, Scikit-learn, dan Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
