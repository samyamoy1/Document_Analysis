# Research Paper Assistant

A GenAI-powered application that helps users understand research papers through automated summarization and question answering.

## Features

- Upload research papers in PDF format.
- Generate structured summaries.
- Ask questions about the paper using RAG.
- Retrieve relevant passages using semantic search.
- Display source page references.
- Support GPU and CPU inference.

## Tech Stack

Python, Streamlit, PyTorch, Hugging Face Transformers, Qwen2.5-1.5B-Instruct, Sentence Transformers, and PyMuPDF.

## Installation

Clone the repository:

```bash
git clone https://github.com/samyamoy1/Document_Analysis.git
cd Document_Analysis
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and run the application:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL displayed in your terminal.

The first run may download the required models. An NVIDIA GPU is recommended for faster inference, but CPU execution is also supported.

## Limitations

Generated answers may contain inaccuracies. Always verify important information against the original paper. CPU inference may be slow for larger workloads.

## Author

Samyamoy Das

GitHub: [@samyamoy1](https://github.com/samyamoy1)

## License

A license has not yet been specified.