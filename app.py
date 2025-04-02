# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the Logistic Regression model and TF-IDF vectorizer
MODEL_PATH = "logistic_regression_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Load the model and vectorizer
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Function to predict using Logistic Regression
def predict_with_logistic(text):
    text_tfidf = vectorizer.transform([text])
    pred = clf.predict(text_tfidf)
    return pred[0]

# Streamlit App
st.title("Resume Classification App")

# Text input for resume
resume_text = st.text_area("Paste your resume text here:", height=300)

# Prediction button
if st.button("Predict"):
    if resume_text.strip() == "":
        st.error("Please enter some text in the resume box.")
    else:
        st.info("Processing...")
        prediction = predict_with_logistic(resume_text)
        st.success(f"Predicted Category: **{prediction}**")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by [Your Name]")
