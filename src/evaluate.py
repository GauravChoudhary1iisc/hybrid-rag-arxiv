"""
Week 3, Step 2: Lightweight evaluation harness (no RAGAS / no LLM-as-judge).

Evaluates three retrieval strategies using deterministic metrics:

  1. Context Recall     — did retrieved chunks contain the ground truth answer?
                          (keyword overlap: what % of GT key terms appear in context)

  2. Answer Faithfulness — is the generated answer grounded in retrieved context?
                           (what % of answer key terms appear in retrieved chunks)

  3. Answer F1          — token-level F1 between generated answer and ground truth
                           (measures generation quality)

All metrics are 0-1, higher is better, computed instantly with no LLM calls.

Run: python 10_evaluate.py
"""

import json
import re
import csv
import sys
import time
from pathlib import Path
from collections import Counter

# ---------- CONFIG ----------
DATA_DIR        = Path("data")
EVAL_DIR        = Path("eval")
RESULTS_DIR     = EVAL_DIR / "results"
QA_PATH         = DATA_DIR / "eval_qa_set.json"
TABLE_OUT       = EVAL_DIR / "comparison_table.csv"

OLLAMA_MODEL    = "qwen2.5:7b"
OLLAMA_URL      = "http://localhost:11434"

EVAL_SAMPLE     = 30      # 30 samples — runs in minutes not hours
RANDOM_SEED     = 42
STRATEGIES      = ["dense", "hybrid", "hybrid_rerank"]

# stopwords to ignore in keyword matching
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up",
    "about", "into", "through", "during", "it", "its", "this", "that",
    "these", "those", "and", "or", "but", "if", "as", "what", "which",
    "who", "how", "when", "where", "why", "not", "no", "so", "than",
    "then", "there", "their", "they", "we", "i", "he", "she", "you",
}
# ----------------------------

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Metric functions ─────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def context_recall(ground_truth: str, contexts: list[str]) -> float:
    """
    What fraction of key terms in the ground truth appear in the retrieved context?
    Score = |GT_tokens ∩ context_tokens| / |GT_tokens|
    Range: 0 (no overlap) to 1 (all GT terms found in context)
    """
    gt_tokens   = set(tokenize(ground_truth))
    ctx_text    = " ".join(contexts)
    ctx_tokens  = set(tokenize(ctx_text))
    if not gt_tokens:
        return 0.0
    overlap = gt_tokens & ctx_tokens
    return len(overlap) / len(gt_tokens)


def answer_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    What fraction of key terms in the generated answer appear in the retrieved context?
    Score = |answer_tokens ∩ context_tokens| / |answer_tokens|
    High score = answer is grounded in context (not hallucinated)
    Range: 0 (fully hallucinated) to 1 (fully grounded)
    """
    ans_tokens  = set(tokenize(answer))
    ctx_text    = " ".join(contexts)
    ctx_tokens  = set(tokenize(ctx_text))
    if not ans_tokens:
        return 0.0
    overlap = ans_tokens & ctx_tokens
    return len(overlap) / len(ans_tokens)


def answer_f1(answer: str, ground_truth: str) -> float:
    """
    Token-level F1 between generated answer and ground truth.
    F1 = 2 * precision * recall / (precision + recall)
    Standard metric used in QA benchmarks like SQuAD.
    """
    ans_tokens = Counter(tokenize(answer))
    gt_tokens  = Counter(tokenize(ground_truth))

    if not ans_tokens or not gt_tokens:
        return 0.0

    # token overlap (shared token counts)
    overlap = sum((ans_tokens & gt_tokens).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(ans_tokens.values())
    recall    = overlap / sum(gt_tokens.values())
    f1        = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


# ── Evaluation pipeline ───────────────────────────────────────────────────────

def load_qa_set(path: Path, n: int, seed: int) -> list[dict]:
    import random
    random.seed(seed)
    with open(path) as f:
        qa_set = json.load(f)
    sample = random.sample(qa_set, min(n, len(qa_set)))
    print(f"  Loaded {len(sample)} QA pairs (from {len(qa_set)} total)")
    return sample


def run_and_evaluate(
    qa_set: list[dict],
    strategy: str,
    retriever,
    generator,
) -> tuple[list[dict], dict]:
    """
    Run retrieval + generation + metric scoring for every QA pair.
    Returns (detailed_records, aggregate_scores).
    """
    records = []

    for i, qa in enumerate(qa_set):
        print(f"  [{i+1}/{len(qa_set)}] {qa['question'][:60]}...")

        # retrieve
        if strategy == "dense":
            chunks = retriever.dense_retrieve(qa["question"], top_k=5)
        elif strategy == "hybrid":
            chunks = retriever.hybrid_retrieve(qa["question"], top_k=5)
        else:
            chunks = retriever.hybrid_rerank(qa["question"], top_k=5)

        # generate
        result = generator.generate(qa["question"], chunks)
        answer   = result["answer"]
        contexts = [c["text"] for c in chunks]

        # score
        cr  = context_recall(qa["ground_truth"], contexts)
        af  = answer_faithfulness(answer, contexts)
        f1  = answer_f1(answer, qa["ground_truth"])

        records.append({
            "question":          qa["question"],
            "ground_truth":      qa["ground_truth"],
            "answer":            answer,
            "contexts":          contexts,
            "context_recall":    cr,
            "answer_faithfulness": af,
            "answer_f1":         f1,
        })

        print(f"      ctx_recall={cr:.3f}  faithfulness={af:.3f}  f1={f1:.3f}")

    # aggregate
    n = len(records)
    scores = {
        "strategy":            strategy,
        "context_recall":      round(sum(r["context_recall"]    for r in records) / n, 4),
        "answer_faithfulness": round(sum(r["answer_faithfulness"] for r in records) / n, 4),
        "answer_f1":           round(sum(r["answer_f1"]          for r in records) / n, 4),
        "n_samples":           n,
    }

    return records, scores


def save_strategy_results(strategy: str, records: list[dict], scores: dict):
    out_path = RESULTS_DIR / f"{strategy}_scores.json"
    with open(out_path, "w") as f:
        json.dump({"scores": scores, "records": records}, f, indent=2)
    print(f"  Saved to {out_path}")


def print_table(all_scores: list[dict]):
    print(f"\n{'─'*62}")
    print(f"{'Strategy':<22} {'Ctx Recall':>12} {'Faithfulness':>14} {'Answer F1':>12}")
    print(f"{'─'*62}")
    for s in all_scores:
        print(
            f"{s['strategy']:<22} "
            f"{s['context_recall']:>12.4f} "
            f"{s['answer_faithfulness']:>14.4f} "
            f"{s['answer_f1']:>12.4f}"
        )
    print(f"{'─'*62}")
    print()
    print("Metric definitions:")
    print("  Context Recall    : % of ground-truth key terms found in retrieved chunks")
    print("  Faithfulness      : % of answer key terms grounded in retrieved context")
    print("  Answer F1         : token-level F1 between generated answer and ground truth")


def save_csv(all_scores: list[dict]):
    fieldnames = ["strategy", "context_recall", "answer_faithfulness",
                  "answer_f1", "n_samples"]
    with open(TABLE_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_scores)
    print(f"\nComparison table saved to: {TABLE_OUT}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, ".")
    from retreiver import Retriever
    from generator import Generator

    print("Initialising retriever and generator...")
    retriever = Retriever(device="cuda")
    generator = Generator()

    print("\nLoading QA test set...")
    qa_set = load_qa_set(QA_PATH, EVAL_SAMPLE, RANDOM_SEED)

    all_scores = []

    for strategy in STRATEGIES:
        out_path = RESULTS_DIR / f"{strategy}_scores.json"

        if out_path.exists():
            print(f"\nLoading cached results for '{strategy}'...")
            with open(out_path) as f:
                data = json.load(f)
            all_scores.append(data["scores"])
            print(f"  context_recall={data['scores']['context_recall']}  "
                  f"faithfulness={data['scores']['answer_faithfulness']}  "
                  f"f1={data['scores']['answer_f1']}")
            continue

        print(f"\n{'='*55}")
        print(f"Strategy: {strategy.upper()}  ({EVAL_SAMPLE} samples)")
        print(f"{'='*55}\n")

        t0 = time.time()
        records, scores = run_and_evaluate(qa_set, strategy, retriever, generator)
        elapsed = time.time() - t0

        save_strategy_results(strategy, records, scores)
        all_scores.append(scores)

        print(f"\n  {strategy} results ({elapsed:.0f}s):")
        print(f"    context_recall     : {scores['context_recall']:.4f}")
        print(f"    answer_faithfulness: {scores['answer_faithfulness']:.4f}")
        print(f"    answer_f1          : {scores['answer_f1']:.4f}")

    print(f"\n{'='*55}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*55}")
    print_table(all_scores)
    save_csv(all_scores)


if __name__ == "__main__":
    main()