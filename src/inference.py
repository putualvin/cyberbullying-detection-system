import joblib
from scipy.sparse import hstack

# Muat model dan vectorizer (Lakukan di luar fungsi agar tidak di-load berulang kali)
model = joblib.load('models/cyberbullying_model.joblib')
word_vec = joblib.load('models/word_vectorizer.joblib')
char_vec = joblib.load('models/char_vectorizer.joblib')

def predict_cyberbullying(clean_text):
    # 1. Ubah teks bersih jadi angka (HANYA gunakan .transform, BUKAN .fit_transform)
    X_word = word_vec.transform([clean_text])
    X_char = char_vec.transform([clean_text])
    
    # 2. Gabungkan fitur (sesuai cara kamu melatihnya dulu)
    X_combined = hstack([X_word, X_char])
    
    # 3. Prediksi dan ambil probabilitasnya
    pred_label = model.predict(X_combined)[0]
    
    # Jika menggunakan LinearSVC, kita gunakan decision_function sebagai pseudo-probabilitas
    decision = model.decision_function(X_combined) 
    
    return pred_label, decision