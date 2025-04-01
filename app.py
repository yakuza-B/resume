# app.py
import streamlit as st
from transformers import pipeline
import PyPDF2
import json

# --------- LOAD MODEL AND LABELS ---------
# Load the pre-trained text classification pipeline
@st.cache_resource
def load_model():
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased",  # Replace with your fine-tuned model path if available
        tokenizer="distilbert-base-uncased",
        return_all_scores=True
    )
    return classifier

classifier = load_model()

# Load labels
def load_labels():
    with open("labels.json", "r") as f:
        labels = json.load(f)
    return labels

labels = load_labels()

# --------- UTILITY FUNCTIONS ---------
# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Predict category using the text classifier
def predict_category(text):
    predictions = classifier(text)
    predicted_label_idx = max(predictions[0], key=lambda x: x["score"])["label"]
    predicted_label = labels[int(predicted_label_idx)]  # Map index to label
    return predicted_label

# --------- STREAMLIT APP ---------
st.title("Resume Category Predictor")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])

if uploaded_file is not None:
    # Display the uploaded file name
    st.write(f"Uploaded File: {uploaded_file.name}")
    
    # Extract text from the PDF
    resume_text = extract_text_from_pdf(uploaded_file)
    
    # Display the extracted text (optional)
    with st.expander("Preview Extracted Text"):
        st.text(resume_text[:500] + "...")  # Show only the first 500 characters
    
    # Predict the category
    if st.button("Predict Category"):
        predicted_category = predict_category(resume_text)
        st.success(f"Predicted Category: **{predicted_category}**")
