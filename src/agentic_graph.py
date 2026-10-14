"""
Project 1: Multi-step agentic RAG with LangGraph.

Graph:
    START → query_classifier ─┬─(needs retrieval)→ retrieve → grade_relevance ─┬─(relevant)→ generate → END
                              │                       ▲                         │
                              │                       └──(not relevant,         │
                              │                           retry_count < 2,      │
                              │                           query rewritten)──────┘
                              └─(no retrieval)──────────────────────────────────→ generate

Reuses the existing components (no rebuilding):
    - Retriever.hybrid_rerank   (Qdrant dense + BM25 RRF + cross-encoder)
    - Generator.generate        (Ollama qwen2.5:7b, cited answers)

Usage:
    from agentic_graph import build_graph
    app = build_graph(retriever, generator)
    final_state = app.invoke(initial_state("What does the paper say about X?"))
"""

import re
from typing import TypedDict

import requests
from langgraph.graph import StateGraph, START, END

from generator import OLLAMA_MODEL, OLLAMA_URL

# ---------- CONFIG ----------
TOP_K           = 5
MAX_RETRIES     = 2       # max loop-backs → at most 3 retrievals per query
GRADE_CHUNKS    = 3       # how many top chunks the grader reads
GRADE_CHARS     = 600
# ----------------------------


class RAGState(TypedDict):
    # required fields
    question: str                 # current (possibly rewritten) query
    retrieved_chunks: list[dict]
    relevance_ok: bool
    retry_count: int
    answer: str
    # bookkeeping for logging / evaluation
    original_question: str
    needs_retrieval: bool
    attempts: list[dict]          # one entry per retrieval: query, chunks, grade
    sources: list[dict]


def initial_state(question: str) -> RAGState:
    return {
        "question": question,
        "retrieved_chunks": [],
        "relevance_ok": False,
        "retry_count": 0,
        "answer": "",
        "original_question": question,
        "needs_retrieval": True,
        "attempts": [],
        "sources": [],
    }


def ollama(prompt: str, max_tokens: int = 128) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ── Node 1: query classifier (heuristic, no LLM call) ────────────────────────

_ARITHMETIC = re.compile(r"^\s*(what\s+is\s+)?[\d\s\.\+\-\*/x×÷\^\(\)=]+\??\s*$", re.I)
_SMALLTALK  = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening))\b", re.I)


def query_classifier(state: RAGState) -> dict:
    q = state["question"]
    needs = not (_ARITHMETIC.match(q) or _SMALLTALK.match(q))
    return {"needs_retrieval": needs}


# ── Node 2: retrieve (rewrites the query first when looping back) ────────────

REWRITE_PROMPT = """You are helping a search engine over academic papers about \
retrieval-augmented generation and NLP. The query below returned passages that \
did not answer it. Rewrite it into a better search query: keep every specific \
entity, method name and dataset name, add likely synonyms or related technical \
terms, and remove filler words. {hint}

Original question: {original}
Previous query: {previous}

Return ONLY the rewritten query on one line."""


def make_retrieve(retriever):
    def retrieve(state: RAGState) -> dict:
        query = state["question"]
        if state["retry_count"] > 0:
            hint = ("Try a substantially different phrasing than before."
                    if state["retry_count"] > 1 else "")
            rewritten = ollama(REWRITE_PROMPT.format(
                hint=hint, original=state["original_question"], previous=query))
            rewritten = rewritten.splitlines()[0].strip().strip('"') if rewritten else ""
            query = rewritten or query

        chunks = retriever.hybrid_rerank(query, top_k=TOP_K)
        attempts = state["attempts"] + [{"query": query, "chunks": chunks}]
        return {"question": query, "retrieved_chunks": chunks, "attempts": attempts}
    return retrieve


# ── Node 3: grade relevance (LLM yes/no) ─────────────────────────────────────

GRADE_PROMPT = """You are grading a search result. Decide whether the passages \
below contain the specific information needed to answer the question. \
Passages that are merely on the same general topic do NOT count.

Question: {question}

Passages:
{passages}

Answer with exactly one word: yes or no."""


def grade_relevance(state: RAGState) -> dict:
    passages = "\n\n".join(
        f"[{i+1}] {c['text'][:GRADE_CHARS]}"
        for i, c in enumerate(state["retrieved_chunks"][:GRADE_CHUNKS])
    )
    # always grade against the user's original question, not the rewrite
    verdict = ollama(GRADE_PROMPT.format(
        question=state["original_question"], passages=passages), max_tokens=3)
    ok = verdict.lower().startswith("yes")

    attempts = list(state["attempts"])
    attempts[-1] = {**attempts[-1], "relevance_ok": ok, "grader_raw": verdict}
    update = {"relevance_ok": ok, "attempts": attempts}
    if not ok:
        update["retry_count"] = state["retry_count"] + 1
    return update


# ── Node 4: generate ─────────────────────────────────────────────────────────

def rerank_against(retriever, question: str, chunks: list[dict]) -> list[dict]:
    """Re-score chunks with the cross-encoder against the ORIGINAL question."""
    scores = retriever.reranker.predict([(question, c["text"]) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [{**c, "rerank_score": float(s)} for s, c in ranked]


def select_chunks(retriever, state: RAGState, mode: str) -> list[dict]:
    """
    Choose the context for generation when the graph retrieved more than once.
      last  — the final attempt's chunks (original behaviour)
      best  — the attempt whose chunks score highest (mean cross-encoder score
              vs the original question)
      merge — union of all attempts, deduplicated, reranked vs the original
              question, top-K
    """
    attempts = state["attempts"]
    if mode == "last" or len(attempts) <= 1:
        return state["retrieved_chunks"]
    q = state["original_question"]
    if mode == "best":
        rescored = [rerank_against(retriever, q, a["chunks"]) for a in attempts]
        return max(rescored, key=lambda cs: sum(c["rerank_score"] for c in cs) / len(cs))
    if mode == "merge":
        pool = {}
        for a in attempts:
            for c in a["chunks"]:
                pool.setdefault(c["chunk_id"], c)
        return rerank_against(retriever, q, list(pool.values()))[:TOP_K]
    raise ValueError(f"unknown context mode: {mode}")


def make_generate(generator, retriever, mode: str):
    def generate(state: RAGState) -> dict:
        q = state["original_question"]
        if not state["needs_retrieval"]:
            return {"answer": ollama(f"Answer briefly: {q}", max_tokens=256), "sources": []}
        chunks = select_chunks(retriever, state, mode)
        result = generator.generate(q, chunks)
        return {"answer": result["answer"], "sources": result["sources"],
                "retrieved_chunks": chunks}
    return generate


# ── Conditional edges ────────────────────────────────────────────────────────

def route_after_classifier(state: RAGState) -> str:
    return "retrieve" if state["needs_retrieval"] else "generate"


def route_after_grade(state: RAGState) -> str:
    if state["relevance_ok"]:
        return "generate"
    # retry_count was already incremented for this failure;
    # loop back while fewer than MAX_RETRIES rewrites have been made
    if state["retry_count"] <= MAX_RETRIES:
        return "retrieve"
    return "generate"   # give up, answer with the last retrieval


def build_graph(retriever, generator, context_mode: str = "last"):
    g = StateGraph(RAGState)
    g.add_node("query_classifier", query_classifier)
    g.add_node("retrieve", make_retrieve(retriever))
    g.add_node("grade_relevance", grade_relevance)
    g.add_node("generate", make_generate(generator, retriever, context_mode))

    g.add_edge(START, "query_classifier")
    g.add_conditional_edges("query_classifier", route_after_classifier,
                            {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve", "grade_relevance")
    g.add_conditional_edges("grade_relevance", route_after_grade,
                            {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("generate", END)
    return g.compile()


if __name__ == "__main__":
    from retreiver import Retriever
    from generator import Generator

    app = build_graph(Retriever(device="cuda"), Generator())
    for q in ["What is 2+2?", "How does RAG handle multi-hop reasoning?"]:
        s = app.invoke(initial_state(q))
        print(f"\nQ: {q}\n  retrieval={s['needs_retrieval']}  "
              f"retrievals={len(s['attempts'])}  queries={[a['query'] for a in s['attempts']]}")
        print(f"  A: {s['answer'][:300]}")
