
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # Load the saved text chunks and their embeddings.
    with open("paper_chunks.json", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load("paper_embeddings.npy")

    if len(chunks) != len(embeddings):
        print("Error: chunk count and embedding count do not match.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    query = input("Ask a question about the paper: ").strip()

    if not query:
        print("Please enter a question.")
        return

    # Convert the question into an embedding.
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # Compare the question with every saved chunk.
    # Normalized vectors let us use the dot product as cosine similarity.
    scores = embeddings @ query_embedding

    # Find the three most similar chunks.
    top_indices = np.argsort(scores)[::-1][:3]

    print("\nMost relevant passages:\n")

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]

        print(f"--- Result {rank} ---")
        print(f"PDF page: {chunk['page']}")
        print(f"Similarity: {scores[index]:.3f}")
        print(chunk["text"])
        print()


if __name__ == "__main__":
    main()
