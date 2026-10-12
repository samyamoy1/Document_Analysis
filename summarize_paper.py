
from pathlib import Path

import torch
import pymupdf
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PDF_PATH = Path("sample_papers") / "Attention is all you need.pdf"


def extract_pdf_pages(pdf_path):
    """Extract PDF text while keeping page numbers."""
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            blocks = page.get_text("blocks", sort=True)

            raw_text = "\n".join(
                block[4].strip()
                for block in blocks
                if block[4].strip()
            )

            # Clean noisy text found inside some diagrams.
            if "<pad>" in raw_text or "<EOS>" in raw_text:
                clean_blocks = []

                for block in blocks:
                    text = block[4].strip()

                    if not text:
                        continue

                    if "<pad>" in text or "<EOS>" in text:
                        continue

                    if len(text.split()) < 5:
                        continue

                    clean_blocks.append(text)

                body_blocks = [
                    text for text in clean_blocks
                    if not text.startswith(("Figure ", "Table "))
                ]

                if body_blocks:
                    page_text = "\n\n".join(clean_blocks)
                else:
                    page_text = ""
            else:
                page_text = page.get_text("text", sort=True).strip()

            pages.append({
                "number": page_number,
                "text": page_text
            })

    return pages


def ask_llm(tokenizer, model, prompt, max_tokens=250):
    """Ask the local language model a question."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic research assistant. "
                "Use only the supplied text. Do not invent facts. "
                "If information is missing, say so."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()


def main():
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        print("Check the filename inside the sample_papers folder.")
        return

    print("Loading Hugging Face model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )

    print("Reading PDF...")
    pages = extract_pdf_pages(PDF_PATH)

    notes = []

    # Summarize three pages at a time.
    for start in range(0, len(pages), 3):
        group = [
            page for page in pages[start:start + 3]
            if page["text"].strip()
        ]

        if not group:
            print(
                f"Skipping empty pages "
                f"{start + 1}-{min(start + 3, len(pages))}"
            )
            continue

        page_text = "\n\n".join(
            f"[Page {page['number']}]\n{page['text']}"
            for page in group
        )

        page_label = f"{group[0]['number']}-{group[-1]['number']}"

        print(f"Summarizing pages {page_label}...")

        prompt = f"""
Summarize these academic paper pages in simple language.

Cover:
1. Main ideas
2. Methods or techniques
3. Results explicitly mentioned
4. Limitations, only if explicitly mentioned

Use only the supplied text.
Do not invent facts or numerical results.
Include page numbers when possible.
Keep the notes concise.

PAPER TEXT:
{page_text}
"""

        note = ask_llm(
            tokenizer,
            model,
            prompt,
            max_tokens=250
        )

        notes.append(
            f"Notes from pages {page_label}:\n{note}"
        )

    if not notes:
        print("No readable text was found in this PDF.")
        return

    # Save the notes from each group.
    Path("intermediate_notes.txt").write_text(
        "\n\n".join(notes),
        encoding="utf-8"
    )

    print("Creating final summary...")

    final_prompt = """
Create a concise research-paper summary using ONLY the notes below.

Use these headings:
1. Research Problem
2. Proposed Method
3. Main Findings
4. Limitations
5. Conclusion

Use simple language.
Do not invent facts or numerical results.
If a section lacks information, say "Not specified in the notes".
Mention page numbers only when they are available in the notes.

NOTES:
""" + "\n\n".join(notes)

    summary = ask_llm(
        tokenizer,
        model,
        final_prompt,
        max_tokens=400
    )

    print("\n" + "=" * 40)
    print("RESEARCH PAPER SUMMARY")
    print("=" * 40)
    print(summary)

    Path("paper_summary.txt").write_text(
        summary,
        encoding="utf-8"
    )

    print("\nSummary saved to paper_summary.txt")
    print("Intermediate notes saved to intermediate_notes.txt")


if __name__ == "__main__":
    main()
