# RAG Document Question Answering System

A Retrieval-Augmented Generation (RAG) application built using Streamlit, FAISS, Sentence Transformers, and Ollama.

## Features

* Upload PDF, DOCX, TXT, and CSV files
* Automatic text extraction
* Text chunking with overlap
* Embedding generation using all-MiniLM-L6-v2
* FAISS vector database
* Semantic similarity search
* Local LLM inference using Llama 3 via Ollama
* Multi-document support
* Streamlit web interface

## Architecture

Document Upload
↓
Text Extraction
↓
Text Chunking
↓
Sentence Transformer Embeddings
↓
FAISS Vector Store
↓
Similarity Search
↓
Relevant Context Retrieval
↓
Llama 3 (Ollama)
↓
Answer Generation

## Tech Stack

* Python
* Streamlit
* FAISS
* Sentence Transformers
* Ollama
* Llama 3
* PyPDF
* Python-Docx
* Pandas

## Installation

```bash
git clone <repo_url>
cd rag-document-qa

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

## Setup Ollama

```bash
ollama pull llama3
```

Verify installation:

```bash
ollama list
```

## Run Application

```bash
streamlit run app.py
```

## Future Improvements

* Chat history
* Source citation tracking
* Hybrid search (BM25 + Vector Search)
* Multiple vector databases
* Conversation memory
* User authentication
