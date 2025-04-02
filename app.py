# app.py
import streamlit as st
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import json

# Load the trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = DistilBertForSequenceClassification.from_pretrained("models/resume_classifier")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

# Load labels
@st.cache_data
def load_labels():
    with open("outputs/labels.json", "r") as f:
        labels = json.load(f)
    return labels

# Perform inference on the input text
def predict_category(text, model, tokenizer, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return labels[prediction]

# Main Streamlit app
def main():
    # Title and description
    st.title("Resume Classification App")
    st.write("Upload a resume or paste its text to classify it into one of the predefined categories.")

    # Load model, tokenizer, and labels
    model, tokenizer = load_model_and_tokenizer()
    labels = load_labels()

    # Input options: File upload or text input
    input_option = st.radio("Choose Input Method:", ["Upload a File", "Paste Text"])

    if input_option == "Upload a File":
        uploaded_file = st.file_uploader("Upload a Resume (TXT or PDF)", type=["txt", "pdf"])
        if uploaded_file:
            # Read the file content
            if uploaded_file.type == "application/pdf":
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = "".join(page.extract_text() for page in pdf_reader.pages)
            else:
                text = uploaded_file.read().decode("utf-8")
            st.write("Uploaded Resume Text:")
            st.text_area("Resume Content", text, height=300)
    else:
        text = st.text_area("Paste Resume Text Here", height=300)

    # Predict button
    if st.button("Classify Resume"):
        if text.strip() == "":
            st.error("Please provide some text or upload a file.")
        else:
            # Perform prediction
            category = predict_category(text, model, tokenizer, labels)
            st.success(f"Predicted Category: **{category}**")

# Run the app
if __name__ == "__main__":
    main()
