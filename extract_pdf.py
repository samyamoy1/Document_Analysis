
from pathlib import Path
import pymupdf

PDF_DIR = Path("sample_papers")


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract text from a PDF while preserving page numbers."""
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in the sample_papers folder.")
        print("Place your research papers there and try again.")
        return

    for pdf_path in pdf_files:
        print(f"\n{'=' * 50}")
        print(f"Paper: {pdf_path.name}")

        pages = extract_pdf_pages(pdf_path)
        full_text = "\n".join(page["text"] for page in pages)

        print(f"Total pages: {len(pages)}")
        print(f"Extracted characters: {len(full_text)}")

        if not full_text.strip():
            print("Warning: No text extracted. This may be a scanned PDF.")
            continue

        print("\nFirst page preview:")
        print(pages[0]["text"][:600])


if __name__ == "__main__":
    main()
