"""
Step 3: Chunk parsed paper text into retrievable segments.

Two strategies are implemented side by side:
  1. Fixed-size  : 512-token chunks with 50-token overlap (using tiktoken)
  2. Semantic    : split on paragraph / section boundaries, merge short segments

Both output to data/chunks_fixed.json and data/chunks_semantic.json.

Run: python 03_chunk.py
"""

import json
import re
import tiktoken
from pathlib import Path

# ---------- CONFIG ----------
DATA_DIR        = Path("data")
PARSED_DIR      = DATA_DIR / "parsed_text"
FIXED_OUT       = DATA_DIR / "chunks_fixed.json"
SEMANTIC_OUT    = DATA_DIR / "chunks_semantic.json"

# Fixed-size params
CHUNK_SIZE      = 512    # tokens per chunk
CHUNK_OVERLAP   = 50     # overlap between consecutive chunks

# Semantic params
MIN_CHUNK_CHARS = 200    # merge segments shorter than this with the next one
MAX_CHUNK_CHARS = 3000   # split segments longer than this (safety cap)
# ----------------------------

# ── Tokenizer ───────────────────────────────────────────────────────────────

def get_tokenizer():
    """Load tiktoken tokenizer (cl100k_base = same as GPT-4 / text-embedding-3)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return enc


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def decode_tokens(token_ids: list, enc) -> str:
    return enc.decode(token_ids)


# ── Fixed-size chunker ───────────────────────────────────────────────────────

def fixed_size_chunks(text: str, enc, chunk_size: int, overlap: int) -> list[str]:
    """
    Tokenize the full text, then slide a window of `chunk_size` tokens
    with `overlap` tokens carried over from the previous chunk.
    Returns a list of decoded text strings.
    """
    token_ids = enc.encode(text, disallowed_special=())
    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_tokens = token_ids[start:end]
        chunks.append(decode_tokens(chunk_tokens, enc))
        if end == len(token_ids):
            break
        start += chunk_size - overlap  # slide forward, keeping overlap

    return chunks


# ── Semantic chunker ─────────────────────────────────────────────────────────

# Patterns that identify section/paragraph boundaries in academic PDFs
SECTION_HEADER = re.compile(
    r"^(\d+\.?\s+[A-Z][A-Za-z\s]{2,50}|"   # "1. Introduction" / "2 Related Work"
    r"[A-Z][A-Z\s]{4,50}|"                  # "INTRODUCTION" (all-caps headers)
    r"Abstract|Introduction|"
    r"Related Work|Background|"
    r"Methodology|Method|Methods|"
    r"Experiments?|Results?|"
    r"Discussion|Conclusion|"
    r"Limitations?|Future Work|Appendix)$",
    re.MULTILINE,
)


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text on blank lines (paragraph boundaries) and section headers.
    Returns a list of raw paragraph strings, stripped of whitespace.
    """
    # normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # split on 2+ consecutive newlines (blank line = paragraph break)
    raw_paragraphs = re.split(r"\n{2,}", text)

    segments = []
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        # If a section header sits inside a paragraph block, split there too
        parts = SECTION_HEADER.split(para)
        for part in parts:
            part = part.strip()
            if part:
                segments.append(part)

    return segments


def semantic_chunks(text: str, min_chars: int, max_chars: int) -> list[str]:
    """
    1. Split text into paragraphs / section-boundary segments.
    2. Merge consecutive segments that are too short (< min_chars).
    3. Hard-split any segment that is too long (> max_chars) at sentence boundaries.
    Returns a list of chunk strings.
    """
    segments = split_into_paragraphs(text)

    # ── Step 1: merge short segments into the next one ──────────────────────
    merged = []
    buffer = ""

    for seg in segments:
        if buffer:
            buffer += "\n" + seg
        else:
            buffer = seg

        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""

    if buffer:  # flush remaining
        merged.append(buffer)

    # ── Step 2: hard-split any chunk that exceeded max_chars ─────────────────
    final_chunks = []
    for chunk in merged:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # split at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            sub_chunk = ""
            for sent in sentences:
                if len(sub_chunk) + len(sent) + 1 > max_chars and sub_chunk:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = sent
                else:
                    sub_chunk += (" " if sub_chunk else "") + sent
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())

    return final_chunks


# ── Build chunk records ──────────────────────────────────────────────────────

def build_chunk_records(
    parsed_files: list[Path],
    strategy: str,
    enc=None,
) -> list[dict]:
    """
    Iterate over all parsed paper JSON files and produce a flat list of
    chunk records. Each record carries the chunk text plus provenance
    metadata so we can cite sources after retrieval.

    strategy: "fixed" | "semantic"
    """
    all_chunks = []
    chunk_id = 0

    for path in parsed_files:
        with open(path) as f:
            paper = json.load(f)

        text = paper["text"]
        if not text.strip():
            continue

        if strategy == "fixed":
            chunks = fixed_size_chunks(text, enc, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            chunks = semantic_chunks(text, MIN_CHUNK_CHARS, MAX_CHUNK_CHARS)

        for i, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            all_chunks.append({
                "chunk_id":   chunk_id,
                "arxiv_id":   paper["arxiv_id"],
                "title":      paper["title"],
                "chunk_index": i,            # position within this paper
                "total_chunks": len(chunks), # how many chunks this paper produced
                "text":       chunk_text,
                "char_count": len(chunk_text),
            })
            chunk_id += 1

    return all_chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    print(f"Found {len(parsed_files)} parsed papers.\n")

    # ── Fixed-size ──────────────────────────────────────────────────────────
    print("Building fixed-size chunks (512 tokens, 50 overlap)...")
    enc = get_tokenizer()
    fixed = build_chunk_records(parsed_files, strategy="fixed", enc=enc)
    with open(FIXED_OUT, "w") as f:
        json.dump(fixed, f, indent=2)

    avg_fixed = sum(c["char_count"] for c in fixed) / len(fixed)
    print(f"  Total chunks : {len(fixed):,}")
    print(f"  Avg chars    : {avg_fixed:.0f}")
    print(f"  Saved to     : {FIXED_OUT}\n")

    # ── Semantic ────────────────────────────────────────────────────────────
    print("Building semantic chunks (paragraph-boundary split)...")
    semantic = build_chunk_records(parsed_files, strategy="semantic")
    with open(SEMANTIC_OUT, "w") as f:
        json.dump(semantic, f, indent=2)

    avg_semantic = sum(c["char_count"] for c in semantic) / len(semantic)
    print(f"  Total chunks : {len(semantic):,}")
    print(f"  Avg chars    : {avg_semantic:.0f}")
    print(f"  Saved to     : {SEMANTIC_OUT}\n")

    # ── Comparison summary ──────────────────────────────────────────────────
    print("── Summary ─────────────────────────────────")
    print(f"  Papers processed     : {len(parsed_files)}")
    print(f"  Fixed chunks total   : {len(fixed):,}  (avg {avg_fixed:.0f} chars)")
    print(f"  Semantic chunks total: {len(semantic):,}  (avg {avg_semantic:.0f} chars)")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()