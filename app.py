# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer
import PyPDF2
import json
from transformers import DistilBertForSequenceClassification

# Load the tokenizer and model
MODEL = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "path_to_your_trained_model"  # Replace with the path to your saved model
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model

model = load_model()

# Load the labels
def load_labels():
    with open("labels.json", "r") as f:
        labels = json.load(f)
    return labels

labels = load_labels()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to preprocess and predict
def predict_category(text):
    # Tokenize the input text
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    # Move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
    
    # Map prediction to label
    predicted_label = labels[preds]
    return predicted_label

# Streamlit App
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
