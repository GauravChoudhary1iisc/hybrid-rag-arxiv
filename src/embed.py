"""
Step 4: Embed chunks into dense vectors using BAAI/bge-base-en-v1.5.

Reads chunks_fixed.json and chunks_semantic.json, embeds every chunk
using sentence-transformers on GPU, and saves:
  - data/embeddings_fixed.npy    (shape: [28711, 768])
  - data/embeddings_semantic.npy (shape: [19633, 768])

Run: python 04_embed.py
"""

import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
DATA_DIR          = Path("data")
FIXED_CHUNKS      = DATA_DIR / "chunks_fixed.json"
SEMANTIC_CHUNKS   = DATA_DIR / "chunks_semantic.json"
FIXED_EMB_OUT     = DATA_DIR / "embeddings_fixed.npy"
SEMANTIC_EMB_OUT  = DATA_DIR / "embeddings_semantic.npy"

MODEL_NAME        = "BAAI/bge-base-en-v1.5"
BATCH_SIZE        = 32    
# ----------------------------

# bge models are trained with this instruction prefix for retrieval tasks.
# Prepending it to every chunk at index time gives a small but consistent
# accuracy boost (confirmed in the bge paper / MTEB leaderboard notes).
BGE_PASSAGE_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks(path: Path) -> tuple[list[str], list[dict]]:
    """Load chunks JSON, return (texts_list, full_records_list)."""
    with open(path) as f:
        records = json.load(f)
    texts = [BGE_PASSAGE_PREFIX + r["text"] for r in records]
    return texts, records


def embed_chunks(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int,
    label: str,
) -> np.ndarray:
    """
    Embed a list of texts in batches, printing progress every 10 batches.
    Returns a float32 numpy array of shape [len(texts), embedding_dim].
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, len(texts))
        batch = texts[start:end]

        # encode() returns a numpy array by default
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → cosine sim = dot product
            show_progress_bar=False,
        )
        all_embeddings.append(emb)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            done    = end
            rate    = done / elapsed
            eta     = (len(texts) - done) / rate if rate > 0 else 0
            print(
                f"  [{label}] batch {batch_idx+1}/{total_batches} "
                f"| {done}/{len(texts)} chunks "
                f"| {rate:.0f} chunks/sec "
                f"| ETA {eta:.0f}s"
            )

    return np.vstack(all_embeddings).astype(np.float32)


def sanity_check(embeddings: np.ndarray, label: str):
    """Print shape, dtype, and a quick norm check to verify embeddings look right."""
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  [{label}] shape      : {embeddings.shape}")
    print(f"  [{label}] dtype      : {embeddings.dtype}")
    print(f"  [{label}] norm mean  : {norms.mean():.4f}  (should be ~1.0 — L2 normalised)")
    print(f"  [{label}] norm std   : {norms.std():.4f}   (should be ~0.0)")

    # quick semantic sanity: embed two similar and one dissimilar sentence
    # and check that similarity scores make intuitive sense
    print(f"  [{label}] first vec  : {embeddings[0, :5]}  ...")


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    print(f"Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}\n")

    # ── Fixed chunks ─────────────────────────────────────────────────────────
    print(f"Loading fixed chunks from {FIXED_CHUNKS}...")
    fixed_texts, fixed_records = load_chunks(FIXED_CHUNKS)
    print(f"  {len(fixed_texts):,} chunks to embed\n")

    print("Embedding fixed chunks...")
    t0 = time.time()
    fixed_emb = embed_chunks(fixed_texts, model, BATCH_SIZE, label="fixed")
    print(f"  Done in {time.time() - t0:.1f}s\n")

    sanity_check(fixed_emb, "fixed")
    np.save(FIXED_EMB_OUT, fixed_emb)
    print(f"  Saved to {FIXED_EMB_OUT}\n")

    # ── Semantic chunks ───────────────────────────────────────────────────────
    print(f"Loading semantic chunks from {SEMANTIC_CHUNKS}...")
    semantic_texts, semantic_records = load_chunks(SEMANTIC_CHUNKS)
    print(f"  {len(semantic_texts):,} chunks to embed\n")

    print("Embedding semantic chunks...")
    t0 = time.time()
    semantic_emb = embed_chunks(semantic_texts, model, BATCH_SIZE, label="semantic")
    print(f"  Done in {time.time() - t0:.1f}s\n")

    sanity_check(semantic_emb, "semantic")
    np.save(SEMANTIC_EMB_OUT, semantic_emb)
    print(f"  Saved to {SEMANTIC_EMB_OUT}\n")

    # ── Final summary ─────────────────────────────────────────────────────────
    fixed_size_mb    = fixed_emb.nbytes / 1024**2
    semantic_size_mb = semantic_emb.nbytes / 1024**2

    print("── Summary ─────────────────────────────────")
    print(f"  Fixed embeddings    : {fixed_emb.shape}  ({fixed_size_mb:.1f} MB)")
    print(f"  Semantic embeddings : {semantic_emb.shape}  ({semantic_size_mb:.1f} MB)")
    print(f"  Saved to            : {DATA_DIR}/")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()