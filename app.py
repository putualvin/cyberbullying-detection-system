import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from scipy.sparse import hstack


# =========================================
# 1) PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Deteksi Cyberbullying",
    page_icon="🛡️",
    layout="centered"
)


# =========================================
# 2) CUSTOM CSS
# =========================================
st.markdown("""
<style>
    .main {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f172a, #111827);
        padding: 28px 30px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.20);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #d1d5db;
        line-height: 1.8;
    }

    .chip {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin: 0.25rem 0.35rem 0.25rem 0;
        border-radius: 999px;
        background: rgba(59,130,246,0.15);
        color: #bfdbfe;
        font-size: 0.86rem;
        border: 1px solid rgba(59,130,246,0.25);
    }

    .result-card {
        padding: 18px 20px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
        margin-top: 1rem;
    }

    .result-safe {
        color: #22c55e;
        font-weight: 700;
        font-size: 1.05rem;
    }

    .result-bully {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.05rem;
    }

    .footer-note {
        color: #9ca3af;
        font-size: 0.92rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================
# 3) HEADER
# =========================================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🛡️ Deteksi Cyberbullying</div>
    <div class="hero-subtitle">
        Aplikasi ini menggunakan <b>Machine Learning (LinearSVC)</b> dengan fitur
        <b>Word TF-IDF + Char TF-IDF</b> untuk mendeteksi cyberbullying pada teks.
        <br><br>
        <span class="chip">religion</span>
        <span class="chip">age</span>
        <span class="chip">gender</span>
        <span class="chip">ethnicity</span>
        <span class="chip">other_cyberbullying</span>
        <span class="chip">not_cyberbullying</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================
# 4) SETUP NLTK
# =========================================
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


setup_nltk()
stop_words = set(stopwords.words("english"))


# =========================================
# 5) SLANG DICTIONARY + PREPROCESSING
# =========================================
slangwords = {
    "kys": "kill yourself",
    "kms": "kill myself",
    "stfu": "shut the fuck up",
    "gtfo": "get the fuck out",
    "fck": "fuck",
    "fcking": "fucking",
    "fuk": "fuck",
    "bitchy": "bitch",
    "btch": "bitch",
    "noob": "newbie",
    "n00b": "newbie",
    "lame": "boring",
    "stpd": "stupid",
    "trash": "bad",
    "garbage": "bad",
    "dumbass": "stupid",
    "h8": "hate",
    "luzr": "loser",
    "u": "you",
    "ur": "your",
    "urs": "yours",
    "r": "are",
    "y": "why",
    "b": "be",
    "c": "see",
    "n": "and",
    "im": "i am",
    "ive": "i have",
    "idk": "i do not know",
    "idc": "i do not care",
    "dont": "do not",
    "cant": "can not",
    "wont": "will not",
    "aint": "is not",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "rt": "",
    "amp": "",
    "mkr": ""
}


def cleaning_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'\bRT\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fix_slangwords(text: str) -> str:
    words = text.split()
    fixed_words = [slangwords.get(word.lower(), word) for word in words]
    return " ".join(fixed_words)


def process_input(text: str, use_stopwords: bool = True) -> str:
    original_text = str(text)

    text = cleaning_text(original_text)
    text = text.lower()
    text = fix_slangwords(text)
    tokens = wordpunct_tokenize(text)

    if use_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    processed = " ".join(tokens).strip()

    if not processed:
        processed = cleaning_text(original_text).lower().strip()

    return processed


# =========================================
# 6) LOAD MODEL & VECTORIZERS
# =========================================
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)

    model_path = os.path.join(base_dir, "cyberbullying_model.joblib")
    word_vec_path = os.path.join(base_dir, "word_vectorizer.joblib")
    char_vec_path = os.path.join(base_dir, "char_vectorizer.joblib")

    missing_files = [
        path for path in [model_path, word_vec_path, char_vec_path]
        if not os.path.exists(path)
    ]

    if missing_files:
        return None, None, None, f"File tidak ditemukan: {missing_files}"

    try:
        loaded_model = joblib.load(model_path)
        word_vectorizer = joblib.load(word_vec_path)
        char_vectorizer = joblib.load(char_vec_path)

        if isinstance(loaded_model, dict):
            model = loaded_model.get("model")
            if model is None:
                return None, None, None, "Model bertipe dict, tetapi key 'model' tidak ada."
        else:
            model = loaded_model

        if not hasattr(model, "predict"):
            return None, None, None, "Objek model tidak valid karena tidak memiliki method predict()."

        if not hasattr(model, "decision_function"):
            return None, None, None, "Model tidak mendukung decision_function()."

        if not hasattr(model, "classes_"):
            return None, None, None, "Model tidak memiliki atribut classes_."

        return model, word_vectorizer, char_vectorizer, None

    except Exception as e:
        return None, None, None, str(e)


model, word_vectorizer, char_vectorizer, load_error = load_artifacts()

if load_error:
    st.error(f"⚠️ Gagal load model: {load_error}")
    st.stop()


# =========================================
# 7) FEATURE ENGINEERING
# =========================================
def vectorize_word_char(cleaned_text: str):
    X_word = word_vectorizer.transform([cleaned_text])
    X_char = char_vectorizer.transform([cleaned_text])
    return hstack([X_word, X_char])


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def get_class_percentages(model, X_vectorized):
    scores = model.decision_function(X_vectorized)
    scores = np.asarray(scores, dtype=float)

    if scores.ndim == 2:
        scores = scores[0]

    if scores.ndim == 0:
        scores = np.array([scores], dtype=float)

    classes = [str(c) for c in model.classes_]

    # Kasus binary LinearSVC sering hanya menghasilkan 1 skor
    if len(classes) == 2 and len(scores) == 1:
        s = float(scores[0])
        scores = np.array([-s, s], dtype=float)

    if len(scores) != len(classes):
        raise ValueError(
            f"Jumlah skor ({len(scores)}) tidak sama dengan jumlah kelas ({len(classes)})."
        )

    percentages = softmax(scores) * 100

    result_df = pd.DataFrame({
        "Kategori": classes,
        "Decision Score": scores,
        "Persentase": percentages
    }).sort_values("Persentase", ascending=False, ignore_index=True)

    return result_df


# =========================================
# 8) INPUT UI
# =========================================
st.subheader("Masukkan Teks")

input_text = st.text_area(
    "Tulis kalimat atau tweet di bawah ini:",
    height=140,
    placeholder="Contoh: you are stupid and ugly",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1.2, 1])

with col1:
    use_stopwords = st.toggle(
        "Gunakan stopword removal",
        value=True,
        help="Aktifkan jika training model memang menggunakan stopword removal."
    )

with col2:
    top_k = st.selectbox(
        "Tampilkan Top-K",
        options=[2, 3, 4, 5, 6],
        index=2
    )

analyze = st.button("🔍 Analisis Sekarang", use_container_width=True)


# =========================================
# 9) PREDICTION
# =========================================
if analyze:
    if not input_text.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
        st.stop()

    try:
        with st.spinner("Sedang menganalisis teks..."):
            cleaned_text = process_input(input_text, use_stopwords=use_stopwords)

            if not cleaned_text.strip():
                st.warning("Teks kosong setelah preprocessing. Silakan masukkan teks lain.")
                st.stop()

            X_vectorized = vectorize_word_char(cleaned_text)
            predicted_label = str(model.predict(X_vectorized)[0])
            result_df = get_class_percentages(model, X_vectorized)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Hasil Analisis")

        st.write("**Teks Asli:**", input_text)
        st.write("**Hasil Preprocessing:**", cleaned_text)
        st.write(f"**Kategori Prediksi:** `{predicted_label}`")

        top_percent = float(result_df.iloc[0]["Persentase"])
        st.write(f"**Tingkat Keyakinan Prediksi:** {top_percent:.2f}%")

        if predicted_label == "not_cyberbullying":
            st.markdown(
                '<p class="result-safe">✅ Teks terdeteksi aman / bukan cyberbullying.</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p class="result-bully">⚠️ Teks terdeteksi mengandung unsur cyberbullying.</p>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.subheader("Estimasi Persentase per Kategori")

        display_df = result_df.copy()
        display_df["Persentase"] = display_df["Persentase"].round(2)

        st.dataframe(
            display_df[["Kategori", "Persentase"]].head(top_k),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            display_df.set_index("Kategori")["Persentase"]
        )

        with st.expander("Lihat semua kategori"):
            st.dataframe(
                display_df[["Kategori", "Persentase"]],
                use_container_width=True,
                hide_index=True
            )

        st.caption("Persentase menunjukkan tingkat kecenderungan prediksi pada masing-masing kategori.")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")


# =========================================
# 10) FOOTER
# =========================================
st.markdown(
    "<p class='footer-note'>Dibuat untuk klasifikasi teks cyberbullying berbasis machine learning dengan model LinearSVC.</p>",
    unsafe_allow_html=True
)