import spacy
import streamlit as st
from spacy.matcher import Matcher
import pdfplumber
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])  # lighter pipeline
matcher = Matcher(nlp.vocab)
matcher.add('NAME', [[{'POS': 'PROPN'}, {'POS': 'PROPN'}]])

@st.cache_resource
def load_summarizer():
    tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6").to("cuda:0" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

@st.cache_data
def summarize_resume(text, tokenizer, model):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=130,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

class Resume:
    def __init__(self, text):
        self.raw_text = text
        self.name = self.extract_name()
        self.email = self.extract_email()
        self.phones = self.extract_phone_numbers()
        self.filtered_text = self.filter_keywords(self.extract_relevant_sections())
        self.summary = None

    def extract_name(self):
        nlp_text = nlp(self.raw_text)
        matches = matcher(nlp_text)
        for _, start, end in matches:
            return nlp_text[start:end].text
        return "not found"

    def extract_email(self):
        return re.findall(r'[\w\.-]+@[\w\.-]+', self.raw_text) or ["Not found"]

    def extract_phone_numbers(self):
        r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
        phone_numbers = r.findall(self.raw_text)
        return [re.sub(r'\D', '', number) for number in phone_numbers] or ["Not found"]

    def extract_relevant_sections(self):
        text = self.raw_text.lower()
        lines = text.split('\n')
        relevant, capture = [], False
        start_keywords = ['skills', 'experience', 'projects', 'responsibilities', 'summary', 'technical skills']
        end_keywords = ['education', 'certifications', 'references', 'hobbies']

        for line in lines:
            line_strip = line.strip()
            if any(k in line_strip for k in start_keywords):
                capture = True
            elif any(k in line_strip for k in end_keywords):
                capture = False
            if capture and line_strip:
                relevant.append(line_strip)
        return '\n'.join(relevant)

    def filter_keywords(self, text):
        doc = nlp(text)
        return ' '.join([
            token.lemma_.lower() for token in doc
            if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN']
        ])

    def generate_summary(self, tokenizer, model):
        if not self.filtered_text.strip():
            return "No relevant content found."
        self.summary = summarize_resume(self.filtered_text, tokenizer, model)
        return self.summary


class ResumeMatcher:
    def __init__(self, job_description):
        self.job_description = self.filter_keywords(job_description)
        self.resumes = []

    def add_resume(self, resume: Resume):
        score = self.calculate_similarity(resume.filtered_text)
        self.resumes.append((resume, score))

    def calculate_similarity(self, resume_text):
        texts = [self.job_description, resume_text]
        vectorizer = CountVectorizer().fit_transform(texts)
        vectors = vectorizer.toarray()
        return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    def get_top_resumes(self, top_n=5):
        sorted_resumes = sorted(self.resumes, key=lambda x: x[1], reverse=True)
        return sorted_resumes[:top_n]

    def filter_keywords(self, text):
        doc = nlp(text)
        return ' '.join([
            token.lemma_.lower() for token in doc
            if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN']
        ])

st.title("Candidate Recommendation Engine!")

job_desc = st.text_area("Enter your job description")
num_resumes = st.number_input("How many resumes will you provide?", min_value=1, step=1)
input_mode = st.radio("Choose an input method", ("pdfs", "text"))
resumes = []

if input_mode == "pdfs":
    uploaded_files = st.file_uploader(
        f"Upload up to {num_resumes} files", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) == num_resumes:
        for file in uploaded_files:
            with pdfplumber.open(file) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
                if text.strip():
                    resumes.append(Resume(text))
    elif uploaded_files:
        st.warning(f"You uploaded {len(uploaded_files)}. Please upload exactly {num_resumes}.")

elif input_mode == "text":
    st.markdown(f"Type {num_resumes} resume(s), separated by `---` (three dashes)")
    pasted_text = st.text_area("Text resume")
    if pasted_text:
        split_resumes = [res.strip() for res in pasted_text.split('---') if res.strip()]
        if len(split_resumes) == num_resumes:
            for res in split_resumes:
                resumes.append(Resume(res))
        else:
            st.warning(f"You entered {len(split_resumes)} resumes, but expected {num_resumes}.")

if job_desc and len(resumes) == num_resumes:
    matcher = ResumeMatcher(job_desc)
    for resume in resumes:
        matcher.add_resume(resume)

    st.subheader("Here are your top candidates for the role!")
    top_resumes = matcher.get_top_resumes(min(num_resumes, 5))

    tokenizer, model = load_summarizer()

    for i, (res, score) in enumerate(top_resumes, 1):
        if res.summary is None:
            res.summary = res.generate_summary(tokenizer, model)

        st.markdown(f"### {i}. {res.name}")
        st.write(f"**Similarity score:** {round(score * 100, 2)}%")
        st.write(f"**Email(s):** {', '.join(res.email)}")
        st.write(f"**Phone(s):** {', '.join(res.phones)}")
        st.markdown("**AI summary of their fit**")
        st.write(res.summary)
        st.markdown("---")

elif job_desc and len(resumes) != num_resumes:
    st.info("Waiting for the correct number of resumes before starting matching.")
elif not job_desc:
    st.info("Uh-oh! Please enter a job description")
