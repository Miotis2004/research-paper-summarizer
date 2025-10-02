# app.py
import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------
# Load Models
# -----------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

summarizer = load_summarizer()
embedder = load_embedder()

# -----------------------
# Helper Functions
# -----------------------
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks for embeddings."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_faiss_index(chunks):
    """Create FAISS index from text chunks."""
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def retrieve_chunks(question, chunks, index, top_k=3):
    """Retrieve most relevant chunks for a question."""
    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Research Paper Summarizer + Q&A", layout="wide")

st.title("📄 Research Paper Summarizer + Q&A")
st.write("Upload a research paper (PDF), get a structured summary, and ask questions.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    # Extract and display text preview
    text = extract_text_from_pdf(uploaded_file)
    st.success("PDF processed successfully.")
    st.write("### Document Preview")
    st.write(text[:1000] + "...")

    # Summarization
    st.subheader("📑 Summary")
    summary = summarizer(text[:2000], max_length=200, min_length=50, do_sample=False)
    st.write(summary[0]['summary_text'])

    # Build FAISS index for Q&A
    st.subheader("💬 Q&A with the Paper")
    chunks = chunk_text(text)
    index, embeddings = build_faiss_index(chunks)

    question = st.text_input("Ask a question about the paper:")
    if st.button("Get Answer") and question:
        relevant_chunks = retrieve_chunks(question, chunks, index)
        context = " ".join(relevant_chunks)

        # Simple QA using summarizer as a proxy (you can swap to an LLM later)
        answer = summarizer(context + " Question: " + question, max_length=150, min_length=30, do_sample=False)
        st.write("**Answer:**", answer[0]['summary_text'])
