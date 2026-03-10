import joblib
from scipy.sparse import hstack
import numpy as np

# 1. Muat file "paket lengkap" milikmu
loaded_dict = joblib.load('models/cyberbullying_model.joblib')

# 2. Ekstrak ketiga komponen langsung dari dalam kamus menggunakan nama kunci yang benar
model = loaded_dict['model']
word_vec = loaded_dict['word_vectorizer']
char_vec = loaded_dict['char_vectorizer']

def predict_cyberbullying(clean_text):
    # Transformasi teks menjadi angka (Word + Char)
    X_word = word_vec.transform([clean_text])
    X_char = char_vec.transform([clean_text])
    
    # Gabungkan fitur
    X_combined = hstack([X_word, X_char])
    
    # Prediksi
    pred_label = model.predict(X_combined)[0]
    
    # Hitung pseudo-confidence menggunakan decision_function dari LinearSVC
    decision = model.decision_function(X_combined)
    confidence = float(np.max(decision)) if len(decision.shape) > 0 else 0.0
    
    return pred_label, confidence