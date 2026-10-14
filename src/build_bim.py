"""
Week 2, Step 1: Build and save BM25 sparse index over fixed chunks.

BM25 is a keyword-based ranking algorithm used alongside dense retrieval
in the hybrid search strategy. It excels at exact term matching (e.g.
"RAPTOR", "HyDE", "ColBERT") where dense embeddings can dilute specificity.

Saves two files:
  data/bm25_index.pkl      — the BM25Okapi index object
  data/bm25_corpus.pkl     — tokenised corpus (needed for score computation)

Run: python 06_build_bm25.py
"""

import json
import pickle
import re
import time
from pathlib import Path
from rank_bm25 import BM25Okapi

# ---------- CONFIG ----------
DATA_DIR        = Path("data")
FIXED_CHUNKS    = DATA_DIR / "chunks_fixed.json"   # BM25 runs on fixed chunks
BM25_INDEX_OUT  = DATA_DIR / "bm25_index.pkl"
BM25_CORPUS_OUT = DATA_DIR / "bm25_corpus.pkl"
# ----------------------------


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases and splits on non-alphanumeric characters.
    Keeps tokens of length >= 2 to filter single-char noise.
    """
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


def build_bm25_index(chunks_path: Path):
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path) as f:
        chunks = json.load(f)
    print(f"  {len(chunks):,} chunks loaded")

    print("\nTokenizing chunks...")
    t0 = time.time()
    corpus = []
    for i, chunk in enumerate(chunks):
        tokens = tokenize(chunk["text"])
        corpus.append(tokens)
        if (i + 1) % 5000 == 0:
            print(f"  Tokenized {i+1:,}/{len(chunks):,}...")

    print(f"  Done in {time.time()-t0:.1f}s")

    print("\nBuilding BM25 index...")
    t0 = time.time()
    bm25 = BM25Okapi(corpus)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\nSaving index and corpus...")
    with open(BM25_INDEX_OUT, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_CORPUS_OUT, "wb") as f:
        pickle.dump(corpus, f)

    print(f"  BM25 index  → {BM25_INDEX_OUT}")
    print(f"  BM25 corpus → {BM25_CORPUS_OUT}")

    return bm25, chunks


def sanity_check(bm25: BM25Okapi, chunks: list[dict]):
    """Run a test query and print top-5 BM25 results."""
    query = "retrieval augmented generation"
    query_tokens = tokenize(query)

    print(f"\nSanity check — query: '{query}'")
    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    for rank, idx in enumerate(top_indices):
        print(f"\n  [{rank+1}] score={scores[idx]:.4f}")
        print(f"       title : {chunks[idx]['title'][:80]}")
        print(f"       chunk : {chunks[idx]['text'][:120]}...")


def main():
    bm25, chunks = build_bm25_index(FIXED_CHUNKS)
    sanity_check(bm25, chunks)

    print("\n── Summary ──────────────────────────────────")
    print(f"  Chunks indexed : {len(chunks):,}")
    print(f"  Avg tokens/chunk: {sum(len(tokenize(c['text'])) for c in chunks[:500]) // 500}")
    print(f"  Index saved to : {DATA_DIR}/")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    main()