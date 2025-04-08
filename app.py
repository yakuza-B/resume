import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import pdfplumber  # For PDF text extraction
from io import BytesIO
import json
import re
import spacy
from spacy import displacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64

# Load the SpaCy model for NER visualization
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_md")
    except:
        st.warning("SpaCy model not found. Please install it first.")
        return None

nlp = load_spacy_model()

# Load the trained model and tokenizer
@st.cache_resource
def load_model():
    # Load label mapping
    with open("labels.json", "r") as f:
        labels = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels))
    
    # Load your trained weights here (adjust path as needed)
    model.load_state_dict(torch.load("resume_classifier.pth", map_location=torch.device('cpu')))
    model.eval()
    
    return model, tokenizer, labels

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to clean resume text
def clean_resume_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Function to predict category
def predict_category(text, model, tokenizer, labels):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
    predicted_idx = torch.argmax(logits, dim=1).item()
    
    return labels[predicted_idx], probabilities

# Function to display PDF
def display_pdf(uploaded_file):
    base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Main app function
def main():
    st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="wide")
    
    # Load model and tokenizer
    try:
        model, tokenizer, labels = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.title("Resume Classifier")
    st.sidebar.markdown("Upload a resume in PDF format to predict its job category.")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Resume Preview")
            display_pdf(uploaded_file)
        
        with col2:
            st.header("Analysis Results")
            
            # Extract and clean text
            with st.spinner("Extracting text from resume..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                cleaned_text = clean_resume_text(raw_text)
            
            # Display text summary
            with st.expander("View Extracted Text"):
                st.text_area("Cleaned Text", cleaned_text, height=300)
            
            # Predict category
            with st.spinner("Analyzing resume..."):
                predicted_category, probabilities = predict_category(cleaned_text, model, tokenizer, labels)
            
            # Display prediction
            st.success(f"**Predicted Category:** {predicted_category}")
            
            # Show confidence scores
            st.subheader("Confidence Scores")
            prob_df = pd.DataFrame({
                "Category": labels,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)
            
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))
            
            # Visualize probabilities
            fig, ax = plt.subplots(figsize=(10, 6))
            prob_df.plot.barh(x="Category", y="Probability", ax=ax, legend=False)
            ax.set_xlabel("Probability")
            ax.set_title("Category Probabilities")
            st.pyplot(fig)
            
            # NER Visualization
            if nlp is not None:
                st.subheader("Named Entity Recognition")
                doc = nlp(raw_text[:10000])  # Process first 10k chars for performance
                
                # Count entities
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                entity_counts = pd.DataFrame(entities, columns=["Entity", "Type"]).groupby("Type").count()
                
                if not entity_counts.empty:
                    st.write("**Entity Types Found:**")
                    st.dataframe(entity_counts)
                    
                    # Visualize entities
                    html = displacy.render(doc[:500], style="ent", page=True)  # First 500 chars
                    st.components.v1.html(html, height=300, scrolling=True)
                else:
                    st.warning("No named entities found in the resume.")
            
            # Word Cloud
            st.subheader("Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    else:
        st.info("Please upload a PDF resume to get started.")
        
        # Show sample categories
        st.markdown("### Supported Categories")
        st.write(", ".join(labels))

if __name__ == "__main__":
    main()
