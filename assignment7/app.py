
import streamlit as st
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import pandas as pd

nltk.download('stopwords')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return ' '.join(tokens)

def read_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

st.title('Resume vs Job Description Matcher (PDF + Text + Informative Visuals)')

input_format = st.radio("Select Input Format:", ["Paste Text", "Upload PDF"])

if input_format == "Paste Text":
    resume = st.text_area("Paste your Resume text")
    job_desc = st.text_area("Paste Job Description")
else:
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume")
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="jd")
    resume = read_pdf(resume_file) if resume_file else ""
    job_desc = read_pdf(jd_file) if jd_file else ""

if st.button("Compare"):
    if resume.strip() and job_desc.strip():
        resume_p = preprocess(resume)
        jd_p = preprocess(job_desc)
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_p, jd_p])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]
        st.markdown(f"Matched Score: `{score:.2f}` (out of 1.00)")

        resume_set = set(resume_p.split())
        jd_set = set(jd_p.split())
        missing = jd_set - resume_set
        common = jd_set & resume_set

        st.markdown("Suggested Keywords to Add:")
        if missing:
            st.success(", ".join(sorted(missing)))
        else:
            st.info("Your resume already covers all key terms from the job description!")

        st.markdown("Job Skill Matched Percentage")
        skill_df = pd.DataFrame({
            'Skill': list(jd_set),
            'Present in Resume': [1 if word in resume_set else 0 for word in jd_set]
        })
        skill_df.sort_values('Present in Resume', ascending=False, inplace=True)
        skill_df['Status'] = skill_df['Present in Resume'].map({1: 'Matched', 0: 'Missing'})

        skill_counts = skill_df['Status'].value_counts()
        fig, ax = plt.subplots()
        skill_counts.plot(kind='bar', color=['green', 'red'], ax=ax)
        ax.set_title("Resume vs JD Skill Match")
        ax.set_ylabel("Number of Skills")
        ax.set_xlabel("Skill Status")
        st.pyplot(fig)

        st.markdown("Skill Distribution (Pie Chart)")
        pie_data = skill_df['Status'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=['#66bb6a', '#ef5350'])
        ax2.set_title("Skill Match Distribution")
        st.pyplot(fig2)

    else:
        st.warning("Please provide both Resume and Job Description.")
