import os

import faiss
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# Configuration
TOP_K = 1
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.1-flash-lite"

DATA_FOLDER = "data"


# Load Environment Variables

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in the .env file.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)


# Streamlit UI

st.set_page_config(
    page_title="Document RAG Chatbot",
    layout="wide"
)

st.title("Document RAG Chatbot")

st.write(
    "Ask questions about the documents stored in the data folder."
)


# Loading Embedding Model

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )


embedding_model = load_embedding_model()


# Loading Documents

@st.cache_resource
def load_documents():

    if not os.path.exists(DATA_FOLDER):
        st.error("The data folder was not found.")
        st.stop()

    documents = []

    for filename in sorted(os.listdir(DATA_FOLDER)):

        if filename.endswith(".txt"):

            filepath = os.path.join(DATA_FOLDER, filename)

            with open(filepath, "r", encoding="utf-8") as file:

                documents.append(
                    {
                        "filename": filename,
                        "text": file.read()
                    }
                )

    return documents


documents = load_documents()


# Chunking Documents

def chunk_documents(documents):

    chunks = []

    for document in documents:

        words = document["text"].split()

        start = 0

        while start < len(words):

            end = start + CHUNK_SIZE

            chunk = " ".join(words[start:end])

            chunks.append(
                {
                    "text": chunk,
                    "source": document["filename"]
                }
            )

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


chunks = chunk_documents(documents)


# Building FAISS Index

@st.cache_resource
def build_vector_store(chunks):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


index = build_vector_store(chunks)


# Sidebar for Streamlit UI

with st.sidebar:

    st.header("Application Information")

    st.write(f"Documents Loaded: {len(documents)}")
    st.write(f"Chunks Created: {len(chunks)}")
    st.write(f"Embedding Model: {EMBEDDING_MODEL}")
    st.write(f"Language Model: {LLM_MODEL}")


# User Question

question = st.text_input(
    "Enter your question"
)

if question.strip():

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        TOP_K
    )

    retrieved_chunks = [
        chunks[idx]
        for idx in indices[0]
    ]

    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not present in the context, respond exactly:

I don't know based on the provided documents.

Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""

    with st.spinner("Generating response..."):

        try:

            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )

            st.subheader("Answer")

            st.write(response.text)

            with st.expander("Retrieved Context"):

                for chunk in retrieved_chunks:

                    st.write(f"Source: {chunk['source']}")

                    st.write(chunk["text"])

                    st.divider()

        except Exception as error:

            st.error(f"Error: {error}")


# Footer

st.markdown("---")

st.caption(
    "Built with Streamlit, Sentence Transformers, FAISS, and Google Gemini."
)