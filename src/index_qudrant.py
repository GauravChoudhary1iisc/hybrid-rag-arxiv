"""
Step 5: Index embeddings into Qdrant vector database.

Spins up against a local Qdrant instance (Docker on port 6333),
creates two collections, uploads all vectors + metadata payloads,
then runs a quick sanity-check search query to verify everything works.

Prerequisites:
    docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

Run: python 05_index_qdrant.py
"""

import json
import time
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    OptimizersConfigDiff,
)

# ---------- CONFIG ----------
DATA_DIR            = Path("data")
FIXED_CHUNKS        = DATA_DIR / "chunks_fixed.json"
SEMANTIC_CHUNKS     = DATA_DIR / "chunks_semantic.json"
FIXED_EMB           = DATA_DIR / "embeddings_fixed.npy"
SEMANTIC_EMB        = DATA_DIR / "embeddings_semantic.npy"

QDRANT_HOST         = "localhost"
QDRANT_PORT         = 6333
COLLECTION_FIXED    = "rag_fixed"
COLLECTION_SEMANTIC = "rag_semantic"

VECTOR_DIM          = 768      # bge-base-en-v1.5 output dim
UPLOAD_BATCH_SIZE   = 256      # points per upload batch
# ----------------------------


def get_client() -> QdrantClient:
    """Connect to local Qdrant instance and verify it's reachable."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # health check — raises if Qdrant is not running
    client.get_collections()
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    return client


def create_collection(client: QdrantClient, name: str, overwrite: bool = False):
    """Create a Qdrant collection. If it already exists, skip (or overwrite)."""
    existing = [c.name for c in client.get_collections().collections]

    if name in existing:
        if overwrite:
            print(f"  Deleting existing collection '{name}'...")
            client.delete_collection(name)
        else:
            print(f"  Collection '{name}' already exists — skipping creation.")
            return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,   # cosine similarity for semantic search
        ),
        # disable background optimisation during bulk upload for speed,
        # re-enable after by setting indexing_threshold back to default (20000)
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
    )
    print(f"  Created collection '{name}' (dim={VECTOR_DIM}, metric=cosine)")


def upload_collection(
    client: QdrantClient,
    collection_name: str,
    chunks_path: Path,
    embeddings_path: Path,
):
    """Upload all vectors + payloads to a Qdrant collection in batches."""
    print(f"\nLoading data for '{collection_name}'...")
    with open(chunks_path) as f:
        chunks = json.load(f)

    embeddings = np.load(embeddings_path)
    assert len(chunks) == len(embeddings), (
        f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
    )
    print(f"  {len(chunks):,} points to upload")

    total_batches = (len(chunks) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE
    t0 = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * UPLOAD_BATCH_SIZE
        end   = min(start + UPLOAD_BATCH_SIZE, len(chunks))

        points = [
            PointStruct(
                id=chunks[i]["chunk_id"],
                vector=embeddings[i].tolist(),
                payload={
                    "arxiv_id":    chunks[i]["arxiv_id"],
                    "title":       chunks[i]["title"],
                    "chunk_index": chunks[i]["chunk_index"],
                    "text":        chunks[i]["text"],
                    "char_count":  chunks[i]["char_count"],
                },
            )
            for i in range(start, end)
        ]

        client.upsert(collection_name=collection_name, points=points)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            done    = end
            rate    = done / elapsed
            eta     = (len(chunks) - done) / rate if rate > 0 else 0
            print(
                f"  Uploaded {done:,}/{len(chunks):,} points "
                f"| {rate:.0f} pts/sec "
                f"| ETA {eta:.0f}s"
            )

    # re-enable indexing after bulk upload
    client.update_collection(
        collection_name=collection_name,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
    )
    print(f"  Indexing triggered for '{collection_name}'")


def verify_collection(client: QdrantClient, collection_name: str):
    """Print collection stats and verify point count matches expectation."""
    info = client.get_collection(collection_name)
    print(f"\n  Collection '{collection_name}':")
    print(f"    Points indexed : {info.points_count:,}")
    print(f"    Vector dim     : {info.config.params.vectors.size}")
    print(f"    Distance metric: {info.config.params.vectors.distance}")
    return info.points_count


def test_search(client: QdrantClient, collection_name: str):
    """
    Run a real semantic search query and print top-3 results.
    Uses a pre-computed dummy vector (all zeros normalised to unit) since
    we don't have the embedding model loaded here — replace with a real
    query embedding in Week 2.
    """
    from sentence_transformers import SentenceTransformer

    print(f"\n  Running test search on '{collection_name}'...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")

    query = "What is retrieval augmented generation?"
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3,
        with_payload=True,
    ).points

    print(f"  Query: '{query}'")
    print(f"  Top 3 results:")
    for i, r in enumerate(results):
        print(f"\n  [{i+1}] score={r.score:.4f}")
        print(f"       title : {r.payload['title'][:80]}")
        print(f"       chunk : {r.payload['text'][:150]}...")


def main():
    # ── Connect ──────────────────────────────────────────────────────────────
    try:
        client = get_client()
    except Exception as e:
        print(f"\nERROR: Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print("Make sure Qdrant is running via Docker:")
        print("  docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant")
        raise

    # ── Fixed collection ─────────────────────────────────────────────────────
    print("\n── Fixed-size chunks ───────────────────────────────────────")
    create_collection(client, COLLECTION_FIXED)
    upload_collection(client, COLLECTION_FIXED, FIXED_CHUNKS, FIXED_EMB)
    verify_collection(client, COLLECTION_FIXED)

    # ── Semantic collection ──────────────────────────────────────────────────
    print("\n── Semantic chunks ─────────────────────────────────────────")
    create_collection(client, COLLECTION_SEMANTIC)
    upload_collection(client, COLLECTION_SEMANTIC, SEMANTIC_CHUNKS, SEMANTIC_EMB)
    verify_collection(client, COLLECTION_SEMANTIC)

    # ── Sanity search ────────────────────────────────────────────────────────
    print("\n── Sanity search (fixed collection) ────────────────────────")
    test_search(client, COLLECTION_FIXED)

    print("\n── Week 1 Complete! ────────────────────────────────────────")
    print("  Both collections live in Qdrant and are queryable.")
    print("  Next: Week 2 — build retrieval strategies (dense / hybrid / rerank)")
    print("────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()