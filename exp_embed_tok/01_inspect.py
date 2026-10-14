"""Step 0: inspect existing project artefacts (read-only)."""
import json
import random
from pathlib import Path

from common import SRC, OUT, load_chunks, load_qa, save_json
from qdrant_client import QdrantClient

info = {}
files = {
    "chunks_fixed": "data/chunks_fixed.json",
    "chunks_semantic": "data/chunks_semantic.json",
    "bm25_index": "data/bm25_index.pkl",
    "bm25_corpus": "data/bm25_corpus.pkl",
    "index_qdrant (named index_qudrant.py)": "index_qudrant.py",
    "build_bm25 (named build_bim.py)": "build_bim.py",
    "retrievers (named retreiver.py)": "retreiver.py",
    "evaluate": "evaluate.py",
    "generator": "generator.py",
    "eval_agentic (88-question LangGraph eval)": "eval_agentic.py",
    "qa_set": "data/eval_qa_set.json",
}
print("== Files (relative to src/)")
for k, p in files.items():
    path = SRC / p
    print(f"  {k:45s} {p:30s} exists={path.exists()} size={path.stat().st_size if path.exists() else '-'}")
    info[k] = {"path": p, "exists": path.exists()}

chunks = load_chunks()
print(f"\n== chunks_fixed: {len(chunks):,} chunks; keys={list(chunks[0].keys())}")
print(f"   chunk_id == index for all: {all(c['chunk_id'] == i for i, c in enumerate(chunks))}")
print(f"   distinct papers: {len(set(c['arxiv_id'] for c in chunks))}")
info["n_chunks"] = len(chunks)

print("\n== Qdrant collections")
q = QdrantClient(host="localhost", port=6333)
info["collections"] = {}
for c in q.get_collections().collections:
    ci = q.get_collection(c.name)
    v = ci.config.params.vectors
    print(f"  {c.name}: points={ci.points_count} dim={v.size} distance={v.distance}")
    info["collections"][c.name] = {"points": ci.points_count, "dim": v.size, "distance": str(v.distance)}

qa = load_qa()
print(f"\n== QA set data/eval_qa_set.json: {len(qa)} questions; keys={sorted(set(k for x in qa for k in x))}")
for key in ["ground_truth", "arxiv_id", "chunk_id"]:
    n = sum(1 for x in qa if x.get(key) not in (None, ""))
    print(f"   questions with {key}: {n}/{len(qa)}")
    info[f"qa_with_{key}"] = n
print(f"   distinct source papers: {len(set(x['arxiv_id'] for x in qa))}")
src_ok = sum(chunks[x["chunk_id"]]["arxiv_id"] == x["arxiv_id"] for x in qa)
print(f"   source chunk_id belongs to recorded arxiv_id: {src_ok}/{len(qa)}")

# the 30-question set is not a file: evaluate.py samples it from the 88
random.seed(42)
s30 = random.sample(qa, 30)
print(f"\n== 30-question set: evaluate.py uses random.sample(seed=42, n=30) from the same 88 "
      f"(first ids: {[x['id'] for x in s30[:5]]})")
print("   eval_agentic.py imports QA_PATH from evaluate.py and runs all 88 -> 88 set is the LangGraph set.")
print("\n== DECISION: main evaluation set = all 88 questions (ground truth present).")
info["eval_set"] = "src/data/eval_qa_set.json (88)"
save_json(info, OUT / "01_inspect.json")
