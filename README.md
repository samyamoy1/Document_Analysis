# Research Paper Assistant

A GenAI project that helps users understand academic research papers using a locally running Hugging Face LLM.

## Features
- Upload research paper PDFs
- Generate simple summaries
- Ask questions about papers
- Retrieve relevant passages using semantic search
- View source passages and page numbers

## Tech Stack
- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- PyMuPDF

## Models
- LLM: Qwen2.5-1.5B-Instruct
- Embeddings: all-MiniLM-L6-v2

## Run Locally
Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

## Limitations
AI-generated summaries and answers may contain errors.
Verify important information against the original paper.