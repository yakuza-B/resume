# app.py
import streamlit as st
import pandas as pd
import numpy as np
import spacy
from transformers import pipeline
import PyPDF2
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --------- LOAD MODELS AND DATA ---------
# Load SpaCy model
def load_spacy_model(model_name="en_core_web_md"):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"🔍 {model_name} not found. Downloading now...")
        import sys
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)

nlp_md = load_spacy_model()

# Load pre-trained DistilBERT model for inference
@st.cache_resource
def load_text_classifier():
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased",  # Replace with your fine-tuned model path if available
        tokenizer="distilbert-base-uncased",
        return_all_scores=True
    )
    return classifier

classifier = load_text_classifier()

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

# Visualize dataset statistics
def visualize_dataset_stats(df):
    _temp = df.Category.value_counts().sort_values(ascending=False)
    _tempGroupColor = []
    for i in _temp.index:
        if i in ["AUTOMOBILE", "AVIATION", "ENGINEERING", "INFORMATION-TECHNOLOGY"]:
            _tempGroupColor.append("red")
        elif i in ["BPO", "HR", "PUBLIC-RELATIONS", "CONSULTANT", "BANKING", "SALES", "ACCOUNTANT", "FINANCE", "BUSINESS-DEVELOPMENT"]:
            _tempGroupColor.append("green")
        else:
            _tempGroupColor.append("blue")
    _tempDF = pd.DataFrame({
        "JobTitle": _temp.index,
        "Count": _temp.values,
        "GroupColor": _tempGroupColor
    })
    return _tempDF

# --------- STREAMLIT APP ---------
st.title("Resume Category Predictor")

# Sidebar for navigation
menu = st.sidebar.selectbox("Menu", ["Upload Resume", "Dataset Statistics", "Model Performance"])

# Upload Resume Section
if menu == "Upload Resume":
    st.header("Upload a Resume")
    uploaded_file = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        st.write(f"Uploaded File: {uploaded_file.name}")
        
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")
        
        # Display extracted text
        with st.expander("Preview Extracted Text"):
            st.text(resume_text[:1000] + "...")  # Show only the first 1000 characters
        
        # Predict category
        if st.button("Predict Category"):
            predicted_category = predict_category(resume_text)
            st.success(f"Predicted Category: **{predicted_category}**")

# Dataset Statistics Section
elif menu == "Dataset Statistics":
    st.header("Dataset Statistics")
    df = pd.read_parquet("dataset.parquet")
    
    # Visualize job title frequency distribution
    stats_df = visualize_dataset_stats(df)
    fig_bar = px.bar(
        stats_df, y="JobTitle", x="Count", color="JobTitle",
        text="Count", orientation="h", height=600,
        title="Job Titles Frequency Distribution",
        color_discrete_sequence=stats_df["GroupColor"]
    )
    st.plotly_chart(fig_bar)
    
    fig_pie = px.pie(stats_df, names="JobTitle", values="Count", height=600, title="Job Titles Frequency Distribution")
    st.plotly_chart(fig_pie)

# Model Performance Section
elif menu == "Model Performance":
    st.header("Model Performance")
    true_labels = [0, 1, 2, 3, 4]  # Replace with actual test labels
    preds = [0, 1, 2, 3, 4]  # Replace with actual predictions
    
    # Classification Report
    report = classification_report(true_labels, preds, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df)
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    st.pyplot(plt)
