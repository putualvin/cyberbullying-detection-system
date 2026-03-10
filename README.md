# 🛡️ Cyberbullying Detection System

A web application for detecting **cyberbullying in text** using **Machine Learning**.
This system classifies text into several cyberbullying categories using **LinearSVC with Word TF-IDF and Character TF-IDF features**.

The application is deployed using **Streamlit** so users can easily input text and see prediction results.

---

## 📌 Features

* Detect cyberbullying in text
* Text preprocessing (cleaning, slang normalization, tokenization)
* Stopword removal option
* Multi-class prediction
* Confidence estimation per category
* Visualization of prediction results
* Interactive web interface using Streamlit

---

## 🧠 Cyberbullying Categories

The model classifies text into the following categories:

* **Religion**
* **Age**
* **Gender**
* **Ethnicity**
* **Other Cyberbullying**
* **Not Cyberbullying**

---

## 🏗️ Project Structure

```
cyberbullying-detection-system
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python version for deployment
│
├── cyberbullying_model.joblib # Trained ML model
├── word_vectorizer.joblib     # Word TF-IDF vectorizer
├── char_vectorizer.joblib     # Character TF-IDF vectorizer
│
├── models/                    # Additional model files
├── src/                       # Source code modules
└── README.md
```

---

## ⚙️ Technologies Used

* **Python**
* **Streamlit**
* **Scikit-learn**
* **NLTK**
* **Pandas**
* **NumPy**
* **TF-IDF Vectorization**
* **LinearSVC Classifier**

---

## 🔎 Machine Learning Model

The cyberbullying detection model uses:

**Algorithm**

* Linear Support Vector Classification (LinearSVC)

**Feature Extraction**

* Word-level TF-IDF
* Character-level TF-IDF

**Text Processing**

* Text cleaning
* Slang normalization
* Tokenization
* Stopword removal

---

## 🚀 How to Run Locally

1. Clone this repository

```bash
git clone https://github.com/your-username/cyberbullying-detection-system.git
cd cyberbullying-detection-system
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app

```bash
streamlit run app.py
```

4. Open in browser

```
http://localhost:8501
```

---

## 🌐 Deployment

This project is deployed using **Streamlit Community Cloud**.

Steps to deploy:

1. Push project to GitHub
2. Go to **Streamlit Community Cloud**
3. Connect repository
4. Select `app.py` as the main file
5. Deploy

---

## 📊 Example Usage

Input text example:

```
you are stupid and ugly
```

Output:

```
Prediction: Cyberbullying
Category: Other Cyberbullying
Confidence: 82%
```

The system will also display prediction probabilities for each category.

---

## 👨‍💻 Author

**Final Project DS Group 6**

Cyberbullying Detection System using Machine Learning and Streamlit.

---

## 📜 License

This project is created for educational purposes.
