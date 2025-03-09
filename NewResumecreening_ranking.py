import streamlit as st
import pandas as pd
import base64
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Function to add background image
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Call background function
add_bg_from_local("background.jpg")  # Make sure to place an image named 'background.jpg' in your folder

# Function to extract text from a PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(jobdescription, resumes):
    document = [jobdescription] + resumes
    vectorizer = TfidfVectorizer().fit_transform(document)
    vectors = vectorizer.toarray()
    jobdescription_vector = vectors[0]
    resumes_vector = vectors[1:]
    cosine_similarities = cosine_similarity([jobdescription_vector], resumes_vector).flatten()
    return cosine_similarities

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📄 AI Resume Screening & Ranking 🎯</h1>", unsafe_allow_html=True)

st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("📂 Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

st.header("📝 Job Description")
jobdescription = st.text_area("Enter the job description here")

if uploaded_files and jobdescription:
    st.header("📊 Ranking Resumes")
    resumes = []

    progress_bar = st.progress(0)  # Progress bar for better UX

    for i, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        resumes.append(text)
        progress_bar.progress((i + 1) / len(uploaded_files))

    scores = rank_resumes(jobdescription, resumes)
    result = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    result = result.sort_values(by="Score", ascending=False)

    st.write(result)

    st.success("✅ Resume ranking completed!")
