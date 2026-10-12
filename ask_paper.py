
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

CHUNKS_FILE = Path("paper_chunks.json")
EMBEDDINGS_FILE = Path("paper_embeddings.npy")
TOP_K = 3


def main():
    # Step 1: Load the saved paper chunks and embeddings.
    if not CHUNKS_FILE.exists() or not EMBEDDINGS_FILE.exists():
        print("Paper data not found.")
        print("Run build_embeddings.py first.")
        return

    with CHUNKS_FILE.open(encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(EMBEDDINGS_FILE)

    if len(chunks) != len(embeddings):
        print("Error: chunk count and embedding count do not match.")
        return

    # Step 2: Load the embedding model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device}...")

    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    # Step 3: Load the local Hugging Face LLM.
    print("Loading Qwen...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        dtype=torch.float16,
        device_map="auto"
    )

    # Step 4: Get the user's question.
    question = input("\nAsk about the research paper: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # Step 5: Convert the question into an embedding.
    query_vector = embedder.encode(
        question,
        normalize_embeddings=True
    )

    # Compare the question with the saved chunk embeddings.
    # The saved embeddings were also normalized.
    scores = embeddings @ query_vector

    # Retrieve the three most relevant chunks.
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    context_parts = []

    for index in top_indices:
        chunk = chunks[index]
        context_parts.append(
            f"[PDF page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # Step 6: Ask Qwen to answer using the retrieved passages.
    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic research assistant. "
                "Answer using only the supplied passages. "
                "Cite the provided PDF page numbers for important claims. "
                "Do not invent facts or citations. "
                "If the passages do not answer the question, "
                "say that the available context is insufficient."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Retrieved passages:\n{context}\n\n"
                "Answer in 2-4 sentences. Cite factual claims using "
                "the format [PDF page 2], using only page numbers from "
                "the supplied passages. Do not make unsupported claims." 
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    print("\nGenerating answer...")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    # Step 7: Display the generated answer.
    print("\n" + "=" * 50)
    print("ANSWER")
    print("=" * 50)
    print(answer)

    # Step 8: Display the source passages used for the answer.
    print("\n" + "=" * 50)
    print("SOURCES USED")
    print("=" * 50)

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]

        print(f"\nSource {rank}")
        print(f"PDF page: {chunk['page']}")
        print(f"Similarity score: {scores[index]:.3f}")
        print("Passage:")
        print(chunk["text"][:700])
        print("-" * 40)


if __name__ == "__main__":
    main()
