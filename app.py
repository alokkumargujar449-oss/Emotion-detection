import streamlit as st
import pandas as pd
import numpy as np
import string
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Emotion label mapping (must match training order) ──────────────────────────
EMOTION_LABELS = {
    0: "😢 Sadness",
    1: "😠 Anger",
    2: "❤️ Love",
    3: "😊 Joy",
    4: "😨 Fear",
    5: "😲 Surprise",
}

EMOTION_COLORS = {
    0: "#6fa8dc",   # blue  – sadness
    1: "#e06666",   # red   – anger
    2: "#ff69b4",   # pink  – love
    3: "#ffd966",   # yellow– joy
    4: "#9fc5e8",   # light blue – fear
    5: "#93c47d",   # green – surprise
}

# ── Text preprocessing ──────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ''.join(ch for ch in text if ch.isascii())
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# ── Load / train model ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on train.txt …")
def load_model():
    df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])

    unique_emotions = df['emotion'].unique()
    emotion_map = {emo: idx for idx, emo in enumerate(unique_emotions)}
    df['emotion'] = df['emotion'].map(emotion_map)

    df['text'] = df['text'].apply(preprocess)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['emotion']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Build reverse label map (number → readable name)
    label_map = {v: k for k, v in emotion_map.items()}
    return model, vectorizer, label_map

# ── Streamlit UI ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emotion Detector", page_icon="🧠", layout="centered")

st.title("🧠 Emotion Detection from Text")
st.markdown("Enter any sentence and the model will predict the emotion behind it.")

try:
    model, vectorizer, label_map = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        "**train.txt not found.**  \n"
        "Make sure `train.txt` is in the same folder as `app.py` before running."
    )

if model_loaded:
    user_input = st.text_area("✏️ Your text", placeholder="Type something like: I feel so happy today!", height=120)

    if st.button("Predict Emotion", type="primary"):
        if user_input.strip():
            clean = preprocess(user_input)
            vec = vectorizer.transform([clean])
            pred_num = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]

            emotion_name = label_map.get(pred_num, str(pred_num))
            # Try to use our emoji labels if the emotion name matches
            emoji_map = {
                "sadness": "😢 Sadness", "anger": "😠 Anger",
                "love": "❤️ Love",    "joy": "😊 Joy",
                "fear": "😨 Fear",    "surprise": "😲 Surprise",
            }
            display_label = emoji_map.get(emotion_name.lower(), f"🔵 {emotion_name.title()}")

            st.markdown("---")
            st.subheader("Result")
            st.markdown(f"### {display_label}")

            # Confidence bar chart
            st.markdown("**Confidence scores:**")
            prob_df = pd.DataFrame({
                "Emotion": [label_map.get(i, str(i)).title() for i in range(len(proba))],
                "Confidence": proba
            }).sort_values("Confidence", ascending=False)
            st.bar_chart(prob_df.set_index("Emotion")["Confidence"])
        else:
            st.warning("Please enter some text first.")

    st.markdown("---")
    with st.expander("ℹ️ About this model"):
        st.markdown("""
        - **Algorithm:** Logistic Regression  
        - **Features:** TF-IDF (Term Frequency–Inverse Document Frequency)  
        - **Preprocessing:** Lowercasing → Punctuation removal → Number removal  
          → Emoji removal → Stopword removal  
        - **Training accuracy:** ~86%
        """)
