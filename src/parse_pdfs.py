"""
Step 2: Parse PDFs into clean text for the RAG project.

Reads every PDF in data/pdfs/, extracts text page-by-page using PyMuPDF,
cleans common PDF noise (headers/footers/page numbers), strips the
References section, and saves one JSON file per paper into data/parsed_text/.

Run: python 02_parse_pdfs.py
"""

import fitz  # PyMuPDF
import json
import re
from pathlib import Path

# ---------- CONFIG ----------
DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
METADATA_PATH = DATA_DIR / "metadata.json"
PARSED_DIR = DATA_DIR / "parsed_text"
MIN_LINE_LENGTH = 40          # lines shorter than this are likely headers/footers/junk
# -----------------------------

PARSED_DIR.mkdir(parents=True, exist_ok=True)

# Matches common "References" / "Bibliography" section headers (case-insensitive,
# allowing for numbering like "7 References" or "References and Notes")
REFERENCES_PATTERN = re.compile(
    r"^\s*\d*\.?\s*(references|bibliography|works cited)\s*$",
    re.IGNORECASE,
)

# Matches likely arXiv header/footer noise, e.g. "arXiv:2310.11511v2 [cs.CL] 3 Jan 2024"
ARXIV_FOOTER_PATTERN = re.compile(r"arxiv:\d{4}\.\d{4,5}", re.IGNORECASE)


def extract_raw_pages(pdf_path: Path) -> list[str]:
    """Extract raw text from each page of a PDF."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # "text" mode (not "blocks"/"dict") already reads in natural
            # reading order for most single/multi-column academic PDFs in
            # recent PyMuPDF versions, since it sorts blocks top-to-bottom,
            # left-to-right per column.
            text = page.get_text("text")
            pages.append(text)
    return pages


def clean_page_text(page_text: str) -> str:
    """Remove short junk lines (headers, footers, page numbers, arXiv stamps)."""
    cleaned_lines = []
    for line in page_text.split("\n"):
        stripped = line.strip()

        if not stripped:
            continue
        if ARXIV_FOOTER_PATTERN.search(stripped):
            continue
        if stripped.isdigit():  # bare page numbers
            continue
        if len(stripped) < MIN_LINE_LENGTH and not stripped.endswith((".", ":", "?")):
            # Short lines that don't look like the end of a sentence are
            # usually headers/footers/section numbers, not real content.
            # We keep short lines that look like genuine sentence endings.
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def truncate_at_references(full_text: str) -> str:
    """Cut off everything from the References/Bibliography section onward."""
    lines = full_text.split("\n")
    for i, line in enumerate(lines):
        if REFERENCES_PATTERN.match(line):
            return "\n".join(lines[:i])
    return full_text  # no references section found, keep everything


def parse_single_pdf(pdf_path: Path) -> dict:
    """Parse one PDF into cleaned full text, returning stats for sanity checks."""
    raw_pages = extract_raw_pages(pdf_path)
    cleaned_pages = [clean_page_text(p) for p in raw_pages]
    full_text = "\n".join(cleaned_pages)
    full_text = truncate_at_references(full_text)

    return {
        "num_pages": len(raw_pages),
        "raw_char_count": sum(len(p) for p in raw_pages),
        "cleaned_char_count": len(full_text),
        "text": full_text,
    }


def main():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    print(f"Found {len(metadata)} papers in metadata.json")

    success_count = 0
    failed = []

    for entry in metadata:
        arxiv_id = entry["arxiv_id"]
        out_path = PARSED_DIR / f"{arxiv_id.replace('/', '_')}.json"

        if out_path.exists():
            success_count += 1
            continue  # already parsed, skip (resumable)

        pdf_path = Path(entry["pdf_path"])
        if not pdf_path.exists():
            failed.append((arxiv_id, "PDF file not found"))
            continue

        try:
            parsed = parse_single_pdf(pdf_path)

            # sanity check: a real paper should have a reasonable amount of
            # extracted text. If it's tiny, something went wrong (e.g. a
            # scanned/image-only PDF with no extractable text layer).
            if parsed["cleaned_char_count"] < 1000:
                failed.append((arxiv_id, f"Only {parsed['cleaned_char_count']} chars extracted — likely scanned/broken PDF"))
                continue

            output = {
                "arxiv_id": arxiv_id,
                "title": entry["title"],
                "num_pages": parsed["num_pages"],
                "raw_char_count": parsed["raw_char_count"],
                "cleaned_char_count": parsed["cleaned_char_count"],
                "text": parsed["text"],
            }

            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)

            success_count += 1

            if success_count % 50 == 0:
                print(f"Parsed {success_count} papers...")

        except Exception as e:
            failed.append((arxiv_id, str(e)))
            continue

    print(f"\nDone. Successfully parsed: {success_count}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nFirst 10 failures:")
        for arxiv_id, reason in failed[:10]:
            print(f"  - {arxiv_id}: {reason}")

        # save full failure log for inspection
        with open(DATA_DIR / "parse_failures.json", "w") as f:
            json.dump(
                [{"arxiv_id": a, "reason": r} for a, r in failed], f, indent=2
            )
        print(f"\nFull failure list saved to: {DATA_DIR / 'parse_failures.json'}")


if __name__ == "__main__":
    main()