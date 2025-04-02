# app.py
import streamlit as st
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch
import json
import os

# Load the trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "model"
    if not os.path.exists(model_path):
        st.error(f"Model directory '{model_path}' not found. Please ensure the model is saved in the 'model/' directory.")
        return None, None
    try:
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        st.success("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load labels
@st.cache_data
def load_labels():
    labels_path = "labels.json"
    if not os.path.exists(labels_path):
        st.error(f"Labels file '{labels_path}' not found. Please ensure the file is in the root directory.")
        return None
    try:
        with open(labels_path, "r") as f:
            labels = json.load(f)
        return labels
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return None

# Preprocess the input text
def preprocess_text(text, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    return inputs

# Predict the category of the resume
def predict(text, model, tokenizer, labels, threshold=0.5):
    inputs = preprocess_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()

    # Log raw predictions for debugging
    st.write("### Raw Model Predictions (Confidence Scores)")
    for i, prob in enumerate(probabilities):
        st.write(f"**{labels[i]}**: {prob:.2f}")

    # Identify all categories above the confidence threshold
    detected_categories = [
        (labels[i], probabilities[i]) for i in range(len(probabilities)) if probabilities[i] > threshold
    ]

    # If no category meets the threshold, return the highest probability category
    if not detected_categories:
        max_index = np.argmax(probabilities)
        detected_categories = [(labels[max_index], probabilities[max_index])]

    return detected_categories

# Streamlit app layout
def main():
    st.title("Resume Classification App 📝")
    st.write("Upload a resume or paste its text to classify it into one of the predefined categories.")

    # Load model, tokenizer, and labels
    model, tokenizer = load_model_and_tokenizer()
    labels = load_labels()
    if model is None or tokenizer is None or labels is None:
        return

    # Input options: File upload or text input
    input_option = st.radio("Choose Input Method:", ["Upload a File", "Paste Text"])

    text = ""
    if input_option == "Upload a File":
        uploaded_file = st.file_uploader("Upload a Resume (TXT)", type=["txt"])
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            st.text_area("Resume Content", text, height=300)
    else:
        text = st.text_area("Paste Resume Text Here", height=300)

    # Perform prediction
    if st.button("Classify Resume"):
        if text.strip() == "":
            st.error("Please provide some text or upload a file.")
        else:
            with st.spinner("Classifying..."):
                detected_categories = predict(text, model, tokenizer, labels)

                # Display detected categories and confidence levels
                st.success("### Detected Categories:")
                for category, confidence in detected_categories:
                    st.write(f"✅ **{category}** (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
