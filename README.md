# Document Intelligence Assistant

## Overview

This project demonstrates a minimal Retrieval-Augmented Generation (RAG) system that answers user questions using local text documents.

The application loads text files, generates embeddings using Sentence Transformers, stores them in a FAISS vector index, retrieves the most relevant context, and uses Google Gemini to generate grounded responses.

## Features

- Load multiple text documents
- Word-based document chunking
- SentenceTransformer embeddings
- FAISS vector search
- Google Gemini 3.1 Flash Lite
- Streamlit web interface


## Architecture Flow

Text Documents
      ↓
Load Documents
      ↓
Chunk Documents
      ↓
SentenceTransformer
      ↓
FAISS
      ↓
Retrieve Context
      ↓
Gemini
      ↓
Answer

## Project Structure

```
RAG/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── data/
```

## Technologies

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Google GenAI SDK
- Gemini 3.1 Flash Lite

## Installation

```bash
pip install -r requirements.txt
```

## Configure

Create a `.env` file:

```
GOOGLE_API_KEY=YOUR_API_KEY
```

## Run

```bash
streamlit run app.py
```

## Sample Questions

- What is RAG?
- What is AWS?
- What is Artificial Intelligence?
- What is Amazon Bedrock?
