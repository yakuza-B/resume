# --------- IMPORTS ---------
import streamlit as st
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from PyPDF2 import PdfReader
import json
import os

# --------- MODEL LOADING AND CONFIGURATION ---------
MODEL_NAME = "distilbert-base-uncased"
DATASET_PATH = "dataset.parquet"
LABELS_PATH = "labels.json"

# Load tokenizer and model
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(load_labels()))
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

# Load labels from JSON file
def load_labels():
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return labels

# Preprocess text for prediction
def preprocess_text(text, tokenizer, max_length=512):
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return tokens["input_ids"], tokens["attention_mask"]

# Predict job category
def predict_category(text, tokenizer, model, labels):
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).item()
    return labels[preds]

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --------- STREAMLIT APP ---------
def main():
    # Set page title and description
    st.set_page_config(page_title="Resume Classification App", page_icon="📋")
    st.title("Resume Classification App")
    st.markdown("Upload your resume (PDF) to predict the job category.")

    # Load tokenizer, model, and labels
    tokenizer, model = load_model_and_tokenizer()
    labels = load_labels()

    # File uploader
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    if uploaded_file is not None:
        # Display uploaded file name
        st.write(f"Uploaded file: **{uploaded_file.name}**")

        # Extract text from PDF
        try:
            with st.spinner("Extracting text from PDF..."):
                resume_text = extract_text_from_pdf(uploaded_file)
            st.success("Text extracted successfully!")
            st.text_area("Resume Text Preview", value=resume_text[:500] + "...", height=200)

            # Predict job category
            if st.button("Predict Job Category"):
                with st.spinner("Predicting job category..."):
                    predicted_category = predict_category(resume_text, tokenizer, model, labels)
                st.success(f"Predicted Job Category: **{predicted_category}**")
        except Exception as e:
            st.error(f"Error processing the PDF: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
