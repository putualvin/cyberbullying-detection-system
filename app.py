import streamlit as st
from src.preprocessing import process_input
from src.inference import predict_cyberbullying

st.set_page_config(page_title="Deteksi Cyberbullying", page_icon="🛡️")

st.title("🛡️ Deteksi Cyberbullying")

input_text = st.text_area("Masukkan teks tweet di sini:")

if st.button("Analisis"):
    if input_text:
        with st.spinner("Sedang menganalisis..."):
            # 1. Panggil fungsi dari preprocessing.py
            clean_text = process_input(input_text)
            
            # 2. Panggil fungsi dari inference.py
            pred_label, decision = predict_cyberbullying(clean_text)
            
        # 3. Tampilkan hasil di UI
        st.write("### Hasil Analisis")
        st.write(f"**Teks Bersih:** {clean_text}")
        st.write(f"**Kategori Prediksi:** `{pred_label}`")
    else:
        st.warning("Teks tidak boleh kosong!")