import streamlit as st
from PyPDF2 import PdfReader
import re
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
from sklearn.metrics import classification_report
import json

# Load the tokenizer and model
MODEL = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Load the trained DistilBERT model (replace with your model path)
model_path = "path_to_your_trained_model"  # Update this path
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Load labels from JSON file
with open("labels.json", "r") as f:
    labels = json.loads(f.read())

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Custom Dataset for inference
class ResumeDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length
        )
        input_ids = torch.tensor(tokens["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokens["attention_mask"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

# Function to make predictions
def predict(text, model, tokenizer):
    # Preprocess the text
    processed_text = preprocess_text(text)

    # Create dataset and dataloader
    dataset = ResumeDataset([processed_text], tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Perform inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predicted_label = labels[preds.item()]
            return predicted_label

# Streamlit App
st.title("Resume Classification App")

# File uploader
uploaded_file = st.file_uploader("Upload a Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Display the uploaded file name
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Extract text from the PDF
    with st.spinner("Extracting text from PDF..."):
        try:
            raw_text = extract_text_from_pdf(uploaded_file)
            st.success("Text extraction complete!")
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            raw_text = None

    if raw_text:
        # Preprocess the text
        with st.spinner("Preprocessing text..."):
            processed_text = preprocess_text(raw_text)
            st.text_area("Extracted Text", processed_text, height=200)

        # Make predictions
        with st.spinner("Making predictions..."):
            prediction = predict(processed_text, model, tokenizer)
            st.success(f"Predicted Category: **{prediction}**")
else:
    st.info("Please upload a PDF file to proceed.")
