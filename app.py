import streamlit as st
from src.preprocessing import process_input
from src.inference import predict_cyberbullying

st.set_page_config(page_title="Cyberbullying Detection", page_icon="🛡️")

st.title("Cyberbullying Detection")

input_text = st.text_area("Insert text here:")

if st.button("Analyze"):
    if input_text:
        with st.spinner("Analyzing..."):
            
            clean_text = process_input(input_text)
            
            
            pred_label, decision = predict_cyberbullying(clean_text)
            
        # 3. Tampilkan hasil di UI
        st.write("### Results")
        st.write(f"**Clean text:** {clean_text}")
        st.write(f"**Prediction Category:** `{pred_label}`")
    else:
        st.warning("Text must not be empty!")