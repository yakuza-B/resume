import streamlit as st
import joblib
import pdfplumber
import re

# Load pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("resume_model.joblib")  # (model, vectorizer)

def extract_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages)

# App UI
st.title("📄 Resume Classifier")
uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")

if uploaded_file:
    model, vectorizer = load_model()
    text = extract_text(uploaded_file)
    text_clean = re.sub(r'[^\w\s]', ' ', text)  # Basic cleaning
    
    # Predict
    prediction = model.predict(vectorizer.transform([text_clean]))[0]
    st.success(f"Predicted Category: {prediction}")
