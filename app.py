# streamlit_app.py
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json
import os

# Load the labels
LABELS_PATH = "labels.json"
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)



# Load the Logistic Regression model
LOGISTIC_MODEL_PATH = "logistic_regression_model.pkl"  # Save your Logistic Regression model as a pickle file
vectorizer = TfidfVectorizer(max_features=5000)
clf = LogisticRegression(max_iter=1000)
clf = pd.read_pickle(LOGISTIC_MODEL_PATH)

# Function to preprocess text for DistilBERT
def preprocess_text_bert(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return tokens

# Function to predict using DistilBERT
def predict_with_bert(text):
    inputs = preprocess_text_bert(text)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
    return labels[preds.item()]

# Function to predict using Logistic Regression
def predict_with_logistic(text):
    text_tfidf = vectorizer.transform([text])
    pred = clf.predict(text_tfidf)
    return labels[pred[0]]

# Streamlit App
st.title("Resume Classification App")

# Sidebar for model selection
st.sidebar.header("Choose Model")
model_option = st.sidebar.selectbox(
    "Select Model",
    ("DistilBERT", "Logistic Regression")
)

# Text input for resume
resume_text = st.text_area("Paste your resume text here:", height=300)

# Prediction button
if st.button("Predict"):
    if resume_text.strip() == "":
        st.error("Please enter some text in the resume box.")
    else:
        st.info("Processing...")
        if model_option == "DistilBERT":
            prediction = predict_with_bert(resume_text)
        else:
            prediction = predict_with_logistic(resume_text)
        
        st.success(f"Predicted Category: **{prediction}**")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by [Your Name]")
