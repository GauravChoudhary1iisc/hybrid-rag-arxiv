"""Shared helpers for exp_embed_tok scripts (paths, seeding, metrics, bootstrap)."""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
SRC = EXP.parent / "src"
OUT = EXP / "out"
OUT.mkdir(exist_ok=True)

# existing modules use relative "data/..." paths, so run from src/
os.chdir(SRC)
sys.path.insert(0, str(SRC))

SEED = 42
N_BOOT = 1000
CHUNKS_PATH = SRC / "data" / "chunks_fixed.json"
QA_PATH = SRC / "data" / "eval_qa_set.json"

BGE_PREFIX = "Represent this sentence for searching relevant passages: "
MODELS = {
    "bge":    {"name": "BAAI/bge-base-en-v1.5", "passage_prefix": BGE_PREFIX, "query_prefix": BGE_PREFIX},
    "minilm": {"name": "sentence-transformers/all-MiniLM-L6-v2", "passage_prefix": "", "query_prefix": ""},
    "e5":     {"name": "intfloat/e5-base-v2", "passage_prefix": "passage: ", "query_prefix": "query: "},
}
BATCH_SIZE = 32


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_chunks():
    with open(CHUNKS_PATH) as f:
        return json.load(f)


def load_qa():
    with open(QA_PATH) as f:
        return json.load(f)


def retrieval_metrics(qa, retrieved):
    """retrieved: list of chunk dicts (with text, arxiv_id, chunk_id), ranked."""
    from evaluate import context_recall
    ids = [r["arxiv_id"] for r in retrieved]
    mrr = 0.0
    for rank, a in enumerate(ids, 1):
        if a == qa["arxiv_id"]:
            mrr = 1.0 / rank
            break
    return {
        "context_recall": context_recall(qa["ground_truth"], [r["text"] for r in retrieved]),
        "hit5_paper": float(qa["arxiv_id"] in ids),
        "mrr5_paper": mrr,
        "hit5_chunk": float(qa["chunk_id"] in [int(r["chunk_id"]) for r in retrieved]),
    }


def boot_ci(values, seed=SEED, n=N_BOOT):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return {"mean": float(v.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "n": int(len(v))}


def paired_diff(a, b, seed=SEED, n=N_BOOT):
    """Bootstrap CI of mean(a - b) over paired questions."""
    d = boot_ci(np.asarray(a, float) - np.asarray(b, float), seed, n)
    d["ci_excludes_zero"] = bool(d["ci_low"] > 0 or d["ci_high"] < 0)
    return d


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
