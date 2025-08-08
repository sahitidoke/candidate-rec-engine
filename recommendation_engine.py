import spacy
import streamlit as st
from spacy.matcher import Matcher
import pdfplumber
import re
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, BartForConditionalGeneration




nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

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
        matcher.add('NAME', [[{'POS': 'PROPN'}, {'POS': 'PROPN'}]])
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

    def generate_fit_summary(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_text = "summarize this person's skills: " + self.raw_text
        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(device)

        summary_ids = model.generate(
            inputs,
            max_length=500,
            min_length=60,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

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
            
    ''' here is my logic for cosine similarity!
    def cosine_sim(v1, v2):
        inter = set(v1.keys()) & set(v2.keys())
        num = sum([v1[x] * v2[x] for x in inter])
    
        sum1 = sum([v1[x]**2 for x in v1.keys()])
        sum2 = sum([v2[x]**2 for x in v2.keys()])
        den = math.sqrt(sum1) * math.sqrt(sum2)
    
        if not den:
            return 0.0
        else:
            return float(num) / den
    
    
    def vectorization(text):
        word = re.compile(r'\w+')
        words = word.findall(text)
        return Counter(words)
    '''
    def get_top_resumes(self, top_n=5):
        sorted_resumes = sorted(self.resumes, key=lambda x: x[1], reverse=True)
        return sorted_resumes[:top_n]

    def filter_keywords(self, text):
        doc = nlp(text)
        return ' '.join([
            token.lemma_.lower() for token in doc
            if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN']
        ])

st.title("candidate recommendation engine!")
job_desc = st.text_area("enter your job description")
num_resumes = st.number_input("how many resumes will you provide?", min_value=1, step=1)
input_mode = st.radio("choose an input method", ("pdfs", "text"))
resumes = []

if input_mode == "pdfs":
    uploaded_files = st.file_uploader(
        f"upload up to {num_resumes} files", 
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
        st.warning(f"you uploaded {len(uploaded_files)}. please upload exactly {num_resumes}.")

elif input_mode == "text":
    st.markdown(f"type {num_resumes} resume(s), separated by `---` (three dashes)")
    pasted_text = st.text_area("text resume")
    if pasted_text:
        split_resumes = [res.strip() for res in pasted_text.split('---') if res.strip()]
        if len(split_resumes) == num_resumes:
            for res in split_resumes:
                resumes.append(Resume(res))
        else:
            st.warning(f"you entered {len(split_resumes)} resumes, but expected {num_resumes}.")

if job_desc and len(resumes) == num_resumes:
    matcher = ResumeMatcher(job_desc)
    for resume in resumes:
        matcher.add_resume(resume)

    st.subheader("here are your top candidates for the role!")
    if (num_resumes<= 5):
        top_resumes = matcher.get_top_resumes(num_resumes)
    else:
        top_resumes =  matcher.get_top_resumes(5)

    for i, (res, score) in enumerate(top_resumes, 1):
        if res.summary is None:
            res.summary = res.generate_fit_summary()

        st.markdown(f"### {i}. {res.name}")
        st.write(f"**similarity score:** {round(score * 100, 2)}%")
        st.write(f"**email(s):** {', '.join(res.email)}")
        st.write(f"**phone(s):** {', '.join(res.phones)}")
        st.markdown("**AI summary of their fit**")
        st.write(res.summary)
        st.markdown("---")

elif job_desc and len(resumes) != num_resumes:
    st.info("waiting for the correct number of resumes before starting matching.")
elif not job_desc:
    st.info("uh-oh! please enter a job description")
