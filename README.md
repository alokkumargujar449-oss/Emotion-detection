# 🧠 Emotion Detection from Text

An NLP web app that detects emotions (sadness, anger, love, joy, fear, surprise) from plain text, built with **Streamlit** and **scikit-learn**.

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📋 Features
- Real-time emotion prediction from user-typed text
- TF-IDF vectorization + Logistic Regression (~86% accuracy)
- Confidence score bar chart for all emotion classes
- Full NLP preprocessing pipeline

## 🗂️ Project Structure
```
├── app.py            # Streamlit application
├── train.txt         # Training dataset (text;emotion format)
├── requirements.txt  # Python dependencies
└── README.md
```

## ⚙️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/emotion-detection.git
cd emotion-detection
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, set **Main file path** to `app.py`.
4. Click **Deploy** — done!

## 🧪 Dataset
The model is trained on `train.txt` — a semicolon-separated file with columns `text;emotion`.

## 🛠️ Tech Stack
- Python, Streamlit
- scikit-learn (TF-IDF, Logistic Regression)
- NLTK (tokenization, stopwords)
- pandas, numpy
