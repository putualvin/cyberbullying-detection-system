# app.py
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize  # lebih stabil daripada word_tokenize
from scipy.sparse import hstack


# =========================================
# 1) KONFIGURASI HALAMAN
# =========================================
st.set_page_config(
    page_title="Deteksi Cyberbullying",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Deteksi Cyberbullying")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (LinearSVC)** dengan fitur **Word TF-IDF + Char TF-IDF** untuk mendeteksi cyberbullying.  
**Kategori:** *religion, age, gender, ethnicity, other_cyberbullying, not_cyberbullying*
""")


# =========================================
# 2) SETUP NLTK
# =========================================
@st.cache_resource
def setup_nltk():
    packages = ["stopwords"]
    for pkg in packages:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)

setup_nltk()


# =========================================
# 3) KAMUS SLANG + PREPROCESSING
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

stop_words = set(stopwords.words("english"))

def cleaningText(text: str) -> str:
    text = str(text)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)              # mention
    text = re.sub(r'\bRT\b', ' ', text)                      # RT
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)       # url
    text = re.sub(r'#(\w+)', r'\1', text)                    # #word -> word
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)                 # non huruf
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fix_slangwords(text: str) -> str:
    words = text.split()
    fixed_words = [slangwords.get(w.lower(), w) for w in words]
    return " ".join(fixed_words)

def process_input(text: str, use_stopwords: bool = True) -> str:
    """
    Samakan dengan Colab kamu:
    - Kalau di Colab stopwords dihapus, biarkan use_stopwords=True (default)
    - Kalau nanti kamu retrain tanpa stopwords, set False
    """
    text = cleaningText(text)
    text = text.lower()
    text = fix_slangwords(text)
    tokens = wordpunct_tokenize(text)

    if use_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    return " ".join(tokens)


# =========================================
# 4) LOAD MODEL (HANYA SEKALI)
# =========================================
@st.cache_resource
def load_bundle():
    base_dir = os.path.dirname(__file__)
    bundle_path = os.path.join(base_dir, "cyberbullying_model.joblib")

    if not os.path.exists(bundle_path):
        return None, None, None, f"File tidak ditemukan: {bundle_path}"

    try:
        bundle = joblib.load(bundle_path)

        model = bundle["model"]
        word_vectorizer = bundle["word_vectorizer"]
        char_vectorizer = bundle["char_vectorizer"]

        return model, word_vectorizer, char_vectorizer, None

    except Exception as e:
        return None, None, None, str(e)

model, word_vectorizer, char_vectorizer, load_error = load_bundle()

if load_error:
    st.error(f"⚠️ Gagal load model: {load_error}")
    st.stop()


# =========================================
# 5) UTILS: VECTORIZE + PSEUDO CONFIDENCE
# =========================================
def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def vectorize_word_char(cleaned_text: str):
    Xw = word_vectorizer.transform([cleaned_text])
    Xc = char_vectorizer.transform([cleaned_text])
    return hstack([Xw, Xc])

def pseudo_probs_from_decision(model, Xv):
    scores = model.decision_function(Xv)
    if hasattr(scores, "ndim") and scores.ndim > 1:
        scores = scores[0]

    scores = np.array(scores, dtype=float)

    # binary case
    if scores.ndim == 0:
        scores = np.array([scores])
    if scores.shape[0] == 1 and len(model.classes_) == 2:
        s = float(scores[0])
        scores = np.array([-s, s], dtype=float)

    return softmax(scores)


# =========================================
# 6) UI INPUT
# =========================================
input_text = st.text_area(
    "Masukkan kalimat/tweet di sini:",
    height=120,
    placeholder="Contoh: you are stupid and ugly"
)

# Karena Colab kamu membuang stopwords, defaultnya ON
use_stopwords = st.toggle("Gunakan stopword removal (sesuai training Colab)", value=True)

threshold = st.slider("Threshold confidence (pseudo)", 0, 100, 50, 1)
top_k = st.slider("Top-k prediksi", 2, 6, 3, 1)

if st.button("🔍 Analisis Tweet"):
    if not input_text.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
        st.stop()

    with st.spinner("Sedang menganalisis..."):
        clean_text = process_input(input_text, use_stopwords=use_stopwords)
        Xv = vectorize_word_char(clean_text)

        pred_label = model.predict(Xv)[0]

        probs = pseudo_probs_from_decision(model, Xv)
        conf = float(np.max(probs)) * 100

        classes = model.classes_
        if len(probs) != len(classes):
            probs = None

    st.write("---")
    st.subheader("Hasil Analisis")

    st.write("**Input:**", input_text)
    st.write("**Cleaned:**", clean_text)
    st.write(f"**Kategori:** `{pred_label}`")

    st.write(f"**Confidence (pseudo):** {conf:.2f}%")
    if conf < threshold:
        st.warning("Catatan: Model masih **UNCERTAIN** untuk input ini.")

    if str(pred_label) == "not_cyberbullying":
        st.success("✅ It's safe. Your sentence looks fine and not bullying.")
    else:
        st.error("⚠️ It's sensitive words. You better use more polite words.")

    if probs is not None:
        st.write("---")
        st.write("**Detail (pseudo) probabilitas per kategori:**")

        probs_df = pd.DataFrame({
            "Kategori": [str(c) for c in classes],
            "Probabilitas(%)": probs * 100
        }).sort_values("Probabilitas(%)", ascending=False)

        st.dataframe(probs_df.head(top_k), use_container_width=True)

        with st.expander("Lihat semua kategori"):
            st.dataframe(probs_df, use_container_width=True)