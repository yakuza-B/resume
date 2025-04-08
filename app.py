import streamlit as st
import pickle
import pdfplumber
import re

# Load pre-trained model
@st.cache_resource
def load_model():
    with open("resume_model.pkl", "rb") as f:
        return pickle.load(f)  # Returns (model, vectorizer)

def extract_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages)

# App UI
st.title("📄 Resume Classifier")
uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")

if uploaded_file:
    try:
        model, vectorizer = load_model()
        text = extract_text(uploaded_file)
        text_clean = re.sub(r'[^\w\s]', ' ', text)  # Basic cleaning
        
        # Predict
        prediction = model.predict(vectorizer.transform([text_clean]))[0]
        st.success(f"Predicted Category: **{prediction}**")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
