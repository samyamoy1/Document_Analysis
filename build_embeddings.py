
from pathlib import Path
import json

import numpy as np
import pymupdf
import torch
from sentence_transformers import SentenceTransformer

PDF_PATH = Path("sample_papers") / "Attention is all you need.pdf"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 150


def extract_chunks(pdf_path):
    """Extract PDF text and split it into chunks with page numbers."""
    chunks = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True).strip()

            # Skip pages with the diagram artefacts we discovered.
            if "<pad>" in text or "<EOS>" in text:
                continue

            words = text.split()

            for start in range(0, len(words), CHUNK_SIZE):
                chunk_text = " ".join(
                    words[start:start + CHUNK_SIZE]
                )

                if chunk_text.strip():
                    chunks.append({
                        "id": len(chunks),
                        "page": page_number,
                        "text": chunk_text
                    })

    return chunks


def main():
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    print("Extracting text...")
    chunks = extract_chunks(PDF_PATH)

    if not chunks:
        print("No text chunks found.")
        return

    print(f"Created {len(chunks)} chunks.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device}...")

    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Creating embeddings...")
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save("paper_embeddings.npy", embeddings)

    Path("paper_chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nDone!")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print("Saved: paper_embeddings.npy")
    print("Saved: paper_chunks.json")


if __name__ == "__main__":
    main()
