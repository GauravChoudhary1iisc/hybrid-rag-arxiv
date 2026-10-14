"""
Week 2, Step 2: Three retrieval strategies over Qdrant + BM25.

Strategies (all return a list of chunk dicts ranked by relevance):
  1. dense_retrieve      — pure vector search via Qdrant
  2. hybrid_retrieve     — dense + BM25 fused with Reciprocal Rank Fusion (RRF)
  3. hybrid_rerank       — hybrid top-50 re-scored by a cross-encoder, top-K returned

Usage:
    from retrievers import Retriever
    r = Retriever()
    results = r.dense_retrieve("What is RAG?", top_k=5)
    results = r.hybrid_retrieve("What is RAG?", top_k=5)
    results = r.hybrid_rerank("What is RAG?", top_k=5)
"""

import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------- CONFIG ----------
DATA_DIR            = Path("data")
FIXED_CHUNKS_PATH   = DATA_DIR / "chunks_fixed.json"
BM25_INDEX_PATH     = DATA_DIR / "bm25_index.pkl"
BM25_CORPUS_PATH    = DATA_DIR / "bm25_corpus.pkl"

QDRANT_HOST         = "localhost"
QDRANT_PORT         = 6333
COLLECTION_FIXED    = "rag_fixed"

EMBED_MODEL         = "BAAI/bge-base-en-v1.5"
RERANK_MODEL        = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BGE_QUERY_PREFIX    = "Represent this sentence for searching relevant passages: "

RRF_K               = 60      # RRF constant (standard value from the RRF paper)
HYBRID_FETCH        = 50      # how many candidates hybrid retrieves before reranking
# ----------------------------


def tokenize(text: str) -> list[str]:
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


class Retriever:
    """
    Loads all retrieval components once at init, then exposes three
    strategy methods that share the same embedding model and indices.
    """

    def __init__(self, device: str = "cuda"):
        print("Loading retriever components...")

        # ── Qdrant client ────────────────────────────────────────────────────
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # ── Embedding model ──────────────────────────────────────────────────
        print("  Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)

        # ── Cross-encoder reranker ───────────────────────────────────────────
        print("  Loading cross-encoder reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL, device=device)

        # ── BM25 index ───────────────────────────────────────────────────────
        print("  Loading BM25 index...")
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(BM25_CORPUS_PATH, "rb") as f:
            self.bm25_corpus = pickle.load(f)

        # ── Chunk records (for payload lookup) ───────────────────────────────
        print("  Loading chunk records...")
        with open(FIXED_CHUNKS_PATH) as f:
            self.chunks = json.load(f)

        print("Retriever ready.\n")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query with the BGE query prefix and return as a list."""
        vec = self.embedder.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    def _qdrant_search(self, query_vec: list[float], top_k: int) -> list[ScoredPoint]:
        """Run a dense vector search against Qdrant."""
        return self.qdrant.query_points(
            collection_name=COLLECTION_FIXED,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        ).points

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, bm25_score) pairs for the top-K BM25 results."""
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _reciprocal_rank_fusion(
        self,
        dense_ids: list[int],
        bm25_ids: list[int],
        k: int = RRF_K,
    ) -> list[tuple[int, float]]:
        """
        Combine two ranked lists using Reciprocal Rank Fusion.
        RRF score = sum(1 / (k + rank)) across all lists.
        Returns (chunk_id, rrf_score) sorted descending.
        """
        scores: dict[int, float] = {}
        for rank, chunk_id in enumerate(dense_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, chunk_id in enumerate(bm25_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _chunk_to_dict(self, chunk_id: int, score: float) -> dict:
        """Look up a chunk record by its chunk_id and attach the retrieval score."""
        chunk = self.chunks[chunk_id].copy()
        chunk["retrieval_score"] = score
        return chunk

    # ── Strategy 1: Dense retrieval ──────────────────────────────────────────

    def dense_retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Pure dense vector search. Fast, no BM25 needed."""
        query_vec = self._embed_query(query)
        hits = self._qdrant_search(query_vec, top_k)
        return [
            {**hit.payload, "retrieval_score": hit.score, "chunk_id": hit.id}
            for hit in hits
        ]

    # ── Strategy 2: Hybrid retrieval (dense + BM25 via RRF) ─────────────────

    def hybrid_retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Hybrid search: retrieve top-K from both dense and BM25 independently,
        then fuse the two ranked lists using Reciprocal Rank Fusion.
        """
        fetch = max(top_k * 4, 50)  # fetch more candidates before fusion

        # dense ranked list
        query_vec = self._embed_query(query)
        dense_hits = self._qdrant_search(query_vec, fetch)
        dense_ids = [hit.id for hit in dense_hits]

        # BM25 ranked list
        bm25_hits = self._bm25_search(query, fetch)
        bm25_ids = [idx for idx, _ in bm25_hits]

        # fuse
        fused = self._reciprocal_rank_fusion(dense_ids, bm25_ids)[:top_k]
        return [self._chunk_to_dict(chunk_id, score) for chunk_id, score in fused]

    # ── Strategy 3: Hybrid + rerank ──────────────────────────────────────────

    def hybrid_rerank(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Hybrid retrieval followed by cross-encoder reranking.
        1. Hybrid retrieves HYBRID_FETCH candidates (default 50).
        2. Cross-encoder scores all (query, chunk) pairs jointly.
        3. Return top_k by cross-encoder score.
        """
        # step 1: hybrid retrieves a large candidate pool
        candidates = self.hybrid_retrieve(query, top_k=HYBRID_FETCH)

        # step 2: cross-encoder scores each candidate against the query
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs)

        # step 3: sort by cross-encoder score, return top_k
        ranked = sorted(
            zip(ce_scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        results = []
        for score, chunk in ranked[:top_k]:
            chunk = chunk.copy()
            chunk["rerank_score"] = float(score)
            results.append(chunk)
        return results


# ── Quick test ───────────────────────────────────────────────────────────────

def print_results(label: str, results: list[dict], elapsed: float):
    print(f"\n{'─'*55}")
    print(f"  Strategy : {label}  ({elapsed*1000:.0f}ms)")
    print(f"{'─'*55}")
    for i, r in enumerate(results):
        score_key = "rerank_score" if "rerank_score" in r else "retrieval_score"
        print(f"\n  [{i+1}] score={r[score_key]:.4f}")
        print(f"       title : {r['title'][:75]}")
        print(f"       chunk : {r['text'][:130]}...")


if __name__ == "__main__":
    retriever = Retriever(device="cuda")
    query = "How does RAG handle multi-hop reasoning?"

    print(f"\nQuery: '{query}'\n")

    # Dense
    t0 = time.time()
    r1 = retriever.dense_retrieve(query, top_k=5)
    print_results("Dense only", r1, time.time() - t0)

    # Hybrid
    t0 = time.time()
    r2 = retriever.hybrid_retrieve(query, top_k=5)
    print_results("Hybrid (dense + BM25 RRF)", r2, time.time() - t0)

    # Hybrid + Rerank
    t0 = time.time()
    r3 = retriever.hybrid_rerank(query, top_k=5)
    print_results("Hybrid + Rerank", r3, time.time() - t0)

    # Latency comparison
    print(f"\n{'─'*55}")
    print("  Note: Dense is fastest. Rerank adds cross-encoder cost.")
    print(f"{'─'*55}\n")