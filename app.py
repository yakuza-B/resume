import streamlit as st
import pickle
import re

# Fallback text extraction (no pdfplumber needed)
def extract_text(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        st.error("Please upload a .txt file or install pdfplumber for PDF support")
        return ""

@st.cache_resource
def load_model():
    with open("resume_model.pkl", "rb") as f:
        return pickle.load(f)  # (model, vectorizer)

st.title("📄 Resume Classifier")
uploaded_file = st.file_uploader("Upload Resume (.txt)", type="txt")

if uploaded_file:
    model, vectorizer = load_model()
    text = extract_text(uploaded_file)
    
    if text:
        text_clean = re.sub(r'[^\w\s]', ' ', text)
        prediction = model.predict(vectorizer.transform([text_clean]))[0]
        st.success(f"Predicted: {prediction}")
