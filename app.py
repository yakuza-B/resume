import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pdfplumber
import re

# 1. Load Data & Train Model (or load pre-trained)
@st.cache_data
def load_data():
    return pd.read_csv("Resume.csv")  # Update path if needed

@st.cache_resource
def train_model():
    df = load_data()
    X = df["Resume_str"]
    y = df["Category"]
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, y)
    
    return model, vectorizer

# 2. PDF Text Extraction
def extract_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages)

# 3. Streamlit App
st.title("📄 Resume Classifier")
uploaded_file = st.file_uploader("Upload a resume (PDF)", type="pdf")

if uploaded_file:
    # Load model (or train if not saved)
    model, vectorizer = train_model()
    
    # Extract and clean text
    text = extract_text(uploaded_file)
    text_clean = re.sub(r'[^\w\s]', ' ', text)  # Basic cleaning
    
    # Predict
    X_input = vectorizer.transform([text_clean])
    prediction = model.predict(X_input)[0]
    
    st.success(f"**Predicted Category:** {prediction}")
    
    # Show raw text (optional)
    with st.expander("View extracted text"):
        st.text(text[:1000] + "...")  # Show first 1000 chars
