# Research Paper Assistant

A Generative AI application that helps users understand academic research papers through automated summarization and question answering using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

## Features

- Upload research papers in PDF format.
- Generate simplified summaries of research papers.
- Ask questions about document content.
- Retrieve relevant passages using semantic search.
- View source passages and their page numbers.

## Tech Stack

- **Language:** Python
- **Interface:** Streamlit
- **LLM:** Qwen2.5-1.5B-Instruct
- **Embeddings:** all-MiniLM-L6-v2
- **Frameworks:** PyTorch, Hugging Face Transformers, Sentence Transformers
- **PDF Processing:** PyMuPDF

## Workflow

1. Extract text from uploaded PDF documents.
2. Split the text into smaller chunks.
3. Generate embeddings for semantic search.
4. Retrieve relevant passages based on the user's question.
5. Generate answers using the language model.
6. Display answers alongside relevant source passages.

The summarization pipeline processes document chunks in batches to generate a consolidated summary.

## Project Structure

```text
Document_Analysis/
├── app.py
├── extract_pdf.py
├── summarize_paper.py
├── build_embeddings.py
├── search_paper.py
├── ask_paper.py
├── test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/samyamoy1/Document_Analysis.git
cd Document_Analysis
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal. The required models may be downloaded during the first run.

## Limitations

- Generated summaries and answers may contain inaccuracies.
- Processing time depends on document length and available hardware.
- Important information should be verified against the original paper.

## Future Improvements

- Support multiple documents in a single session.
- Improve retrieval and answer accuracy.
- Add conversation history and follow-up questions.
- Deploy the application for online access.

## Author

**Samyamoy Das**

GitHub: [samyamoy1](https://github.com/samyamoy1)

---

This project explores Generative AI, semantic search, Large Language Models, and Retrieval-Augmented Generation.
