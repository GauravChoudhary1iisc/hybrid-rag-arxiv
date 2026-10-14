"""
Week 2, Step 3: Answer generation using retrieved chunks + Anthropic API.

Takes a query + list of retrieved chunks, builds a prompt with context,
calls claude-haiku-4-5 (fast + cheap), and returns a cited answer.

Usage:
    from generator import Generator
    g = Generator()
    answer = g.generate("What is RAG?", chunks)
    print(answer["answer"])
    print(answer["sources"])
"""

import os
from pathlib import Path
import requests

# ---------- CONFIG ----------
OLLAMA_MODEL       = "qwen2.5:7b"
OLLAMA_URL         = "http://localhost:11434/api/generate"
MAX_TOKENS         = 1024
MAX_CONTEXT_CHUNKS = 5
MAX_CHUNK_CHARS    = 800
# ----------------------------

SYSTEM_PROMPT = """You are a precise research assistant answering questions about \
retrieval-augmented generation (RAG) and related NLP topics. 

You are given a question and a set of numbered context passages retrieved from \
academic papers. Answer the question using ONLY the provided context. 

Rules:
- Cite sources inline using [1], [2], etc. matching the passage numbers
- If the context does not contain enough information, say so clearly
- Be concise but complete — aim for 3-6 sentences
- Do not hallucinate or add information not present in the context"""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build the user prompt with retrieved context passages."""
    context_parts = []
    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS]):
        text = chunk["text"][:MAX_CHUNK_CHARS]
        title = chunk.get("title", "Unknown")[:80]
        context_parts.append(
            f"[{i+1}] From: \"{title}\"\n{text}"
        )

    context_block = "\n\n".join(context_parts)

    return f"""Context passages:

{context_block}

---
Question: {query}

Answer (cite sources inline as [1], [2], etc.):"""


class Generator:
    def __init__(self):
        # verify Ollama is running
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(OLLAMA_MODEL.split(":")[0] in m for m in models):
                print(f"Warning: {OLLAMA_MODEL} not found. Run: ollama pull {OLLAMA_MODEL}")
            else:
                print(f"Generator ready (model: {OLLAMA_MODEL})")
        except Exception:
            raise RuntimeError(
                "Ollama not running. Start it with: ollama serve"
            )

    def generate(self, query: str, chunks: list[dict]) -> dict:
        prompt = build_prompt(query, chunks)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {"num_predict": MAX_TOKENS, "temperature": 0.1},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        answer_text = data.get("response", "").strip()

        return {
            "query":   query,
            "answer":  answer_text,
            "sources": [
                {
                    "rank":     i + 1,
                    "title":    c.get("title", ""),
                    "arxiv_id": c.get("arxiv_id", ""),
                    "score":    c.get("rerank_score", c.get("retrieval_score", 0.0)),
                }
                for i, c in enumerate(chunks[:MAX_CONTEXT_CHUNKS])
            ],
            "model": OLLAMA_MODEL,
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
        }
# ── Quick end-to-end test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, ".")
    from retreiver import Retriever

    # initialise both components
    retriever = Retriever(device="cuda")
    generator = Generator()

    queries = [
        "How does RAG handle multi-hop reasoning?",
        "What are the main failure modes of naive RAG systems?",
        "How does hybrid retrieval improve over dense-only search?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        # use hybrid+rerank (best quality)
        t0 = time.time()
        chunks = retriever.hybrid_rerank(query, top_k=5)
        retrieval_time = time.time() - t0

        t0 = time.time()
        result = generator.generate(query, chunks)
        generation_time = time.time() - t0

        print(f"\nAnswer ({generation_time:.1f}s generation, {retrieval_time:.1f}s retrieval):")
        print(f"\n{result['answer']}")

        print(f"\nSources:")
        for s in result["sources"]:
            print(f"  [{s['rank']}] {s['title'][:70]}")

        print(f"\nTokens used: {result['usage']['prompt_tokens']} in / {result['usage']['output_tokens']} out")