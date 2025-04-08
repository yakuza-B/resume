import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pdfplumber
import re
import joblib

# 1. Load Data & Train Model (or load pre-trained)
@st.cache_data
def load_data():
    return pd.read_csv("Resume.csv")  # Update path if needed

def train_and_save_model():
    df = load_data()
    X = df["Resume_str"]
    y = df["Category"]
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, y)
    
    # Save model and vectorizer
    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

# 2. PDF Text Extraction with Error Handling
def extract_text(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = " ".join(page.extract_text() or "" for page in pdf.pages)
        if not text.strip():
            raise ValueError("No text found in the PDF.")
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

# 3. Streamlit App
st.title("📄 Resume Classifier")

# Check if model files exist; otherwise, train and save them
try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.warning("Model files not found. Training model...")
    train_and_save_model()
    model, vectorizer = load_model()

# File Uploader
uploaded_file = st.file_uploader("Upload a resume (PDF)", type="pdf")

if uploaded_file:
    # Extract and clean text
    text = extract_text(uploaded_file)
    if text:
        text_clean = re.sub(r'[^\w\s]', ' ', text)  # Basic cleaning
        
        # Predict
        X_input = vectorizer.transform([text_clean])
        prediction = model.predict(X_input)[0]
        
        st.success(f"**Predicted Category:** {prediction}")
        
        # Show raw text (optional)
        with st.expander("View extracted text"):
            st.text(text[:1000] + "...")  # Show first 1000 chars
