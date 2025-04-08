import streamlit as st
import pickle
import re
import base64
from io import BytesIO

# Try PDF parsers in order of reliability
try:
    import pdfplumber  # Best for text extraction
    PDF_SUPPORT = True
except ImportError:
    try:
        import PyPDF2  # Fallback option
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False

# Text extraction with fallbacks        
def extract_text(uploaded_file):
    if not PDF_SUPPORT:
        st.error("Dependencies missing! Add 'pdfplumber' or 'pypdf' to requirements.txt")
        return ""
    
    try:
        if uploaded_file.type == "application/pdf":
            text = ""
            if "pdfplumber" in globals():
                with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
                    text = " ".join(page.extract_text() for page in pdf.pages)
            elif "PyPDF2" in globals():
                reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                text = " ".join(page.extract_text() for page in reader.pages)
            return text
        else:
            return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""

@st.cache_resource
def load_model():
    with open("resume_model.pkl", "rb") as f:
        return pickle.load(f)  # (model, vectorizer)

# App UI
st.title("📄 Resume Classifier")
uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])

if uploaded_file:
    model, vectorizer = load_model()
    text = extract_text(uploaded_file)
    
    if text:
        text_clean = re.sub(r'[^\w\s]', ' ', text)
        prediction = model.predict(vectorizer.transform([text_clean]))[0]
        st.success(f"**Predicted Category:** {prediction}")

# Dependency warning
if not PDF_SUPPORT:
    st.warning("""
    ⚠️ For full PDF support:
    ```python
    # requirements.txt
    pdfplumber>=0.10.0
    ```
    """)
