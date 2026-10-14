"""
Project 1, Steps 5–8: run the LangGraph agentic RAG over the full 88-pair QA set
and measure whether the grade → rewrite → re-retrieve loop actually helps.

For each query we log the path taken (skipped / single / looped xN), and score:
  - first-pass: attempt[0] chunks (== the old single-pass hybrid_rerank pipeline)
                and an answer generated from them
  - final:      the chunks/answer the graph actually ended with
For queries that never looped, first-pass == final (same chunks, same answer).

Outputs (eval/agentic/):
  path_log.jsonl        one record per query (resumable cache)
  metrics_table.csv     before/after table
  summary.md            human-readable result

Run (from src/):  python eval_agentic.py [last|best|merge]
  last  → eval/agentic/        (answer from the final retrieval)
  best  → eval/agentic_best/   (answer from the best-scoring retrieval)
  merge → eval/agentic_merge/  (answer from all retrievals merged + reranked)
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from evaluate import context_recall, answer_f1, answer_faithfulness, QA_PATH
from agentic_graph import build_graph, initial_state
from retreiver import Retriever
from generator import Generator

MODE      = sys.argv[1] if len(sys.argv) > 1 else "last"   # last | best | merge
OUT_DIR   = Path("eval/agentic" if MODE == "last" else f"eval/agentic_{MODE}")
LOG_PATH  = OUT_DIR / "path_log.jsonl"
TABLE_OUT = OUT_DIR / "metrics_table.csv"
SUMMARY   = OUT_DIR / "summary.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_QUESTIONS = ["What is 2+2?", "what is 12 * 7", "Hello, how are you?"]


def score(gt: str, chunks: list[dict], answer: str) -> dict:
    ctx = [c["text"] for c in chunks]
    return {
        "context_recall": round(context_recall(gt, ctx), 4),
        "answer_f1": answer_f1(answer, gt),
        "faithfulness": round(answer_faithfulness(answer, ctx), 4),
    }


def run_query(app, generator, qa: dict) -> dict:
    t0 = time.time()
    s = app.invoke(initial_state(qa["question"]))
    n_retr = len(s["attempts"])
    path = "skipped" if not s["needs_retrieval"] else ("single" if n_retr == 1 else "looped")

    final = score(qa["ground_truth"], s["retrieved_chunks"], s["answer"])
    if path == "looped":
        first_chunks = s["attempts"][0]["chunks"]
        first_answer = generator.generate(qa["question"], first_chunks)["answer"]
        first = score(qa["ground_truth"], first_chunks, first_answer)
    else:
        first_answer, first = s["answer"], final

    return {
        "id": qa["id"],
        "question": qa["question"],
        "ground_truth": qa["ground_truth"],
        "path": path,
        "n_retrievals": n_retr,
        "n_loops": max(n_retr - 1, 0),
        "final_relevance_ok": s["relevance_ok"],
        "attempts": [
            {"query": a["query"], "relevance_ok": a.get("relevance_ok"),
             "grader_raw": a.get("grader_raw"),
             "chunk_ids": [c.get("chunk_id") for c in a["chunks"]],
             "top_rerank_score": a["chunks"][0].get("rerank_score") if a["chunks"] else None,
             "context_recall": round(context_recall(qa["ground_truth"],
                                                    [c["text"] for c in a["chunks"]]), 4)}
            for a in s["attempts"]
        ],
        "gold_chunk_in_first": qa["chunk_id"] in [c.get("chunk_id") for c in s["attempts"][0]["chunks"]],
        "gold_chunk_in_final": qa["chunk_id"] in [c.get("chunk_id") for c in s["retrieved_chunks"]],
        "first_pass": {**first, "answer": first_answer},
        "final": {**final, "answer": s["answer"]},
        "seconds": round(time.time() - t0, 1),
    }


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else float("nan")


def main():
    retriever = Retriever(device="cuda")
    generator = Generator()
    app = build_graph(retriever, generator, context_mode=MODE)

    # sanity check of the skip branch (not part of the QA metrics)
    print("Classifier smoke test:")
    for q in SMOKE_QUESTIONS:
        s = app.invoke(initial_state(q))
        print(f"  {q!r:28} retrieval={s['needs_retrieval']}  answer={s['answer'][:60]!r}")

    with open(QA_PATH) as f:
        qa_set = json.load(f)

    done = {}
    if LOG_PATH.exists():
        for line in LOG_PATH.open():
            r = json.loads(line)
            done[r["id"]] = r
    print(f"\n{len(done)} cached, {len(qa_set) - len(done)} to run")

    with LOG_PATH.open("a") as log:
        for i, qa in enumerate(qa_set):
            if qa["id"] in done:
                continue
            r = run_query(app, generator, qa)
            done[qa["id"]] = r
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(f"[{i+1}/{len(qa_set)}] {r['path']:7} loops={r['n_loops']} "
                  f"recall {r['first_pass']['context_recall']:.2f}→{r['final']['context_recall']:.2f} "
                  f"f1 {r['first_pass']['answer_f1']:.2f}→{r['final']['answer_f1']:.2f} ({r['seconds']}s)")

    report(list(done.values()))


def report(records: list[dict]):
    n = len(records)
    single = [r for r in records if r["path"] == "single"]
    looped = [r for r in records if r["path"] == "looped"]
    skipped = [r for r in records if r["path"] == "skipped"]

    rows = []
    def row(group, recs, stage):
        rows.append({
            "group": group, "stage": stage, "n": len(recs),
            "context_recall": mean(r[stage]["context_recall"] for r in recs),
            "answer_f1": mean(r[stage]["answer_f1"] for r in recs),
            "faithfulness": mean(r[stage]["faithfulness"] for r in recs),
            "gold_chunk_hit": mean(r["gold_chunk_in_first" if stage == "first_pass"
                                     else "gold_chunk_in_final"] for r in recs),
        })
    row("all", records, "first_pass")     # == single-pass baseline
    row("all", records, "final")          # == agentic pipeline
    row("never_looped", single, "final")
    row("looped", looped, "first_pass")
    row("looped", looped, "final")

    with TABLE_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def delta(key):
        better = sum(r["final"][key] > r["first_pass"][key] + 1e-9 for r in looped)
        worse  = sum(r["final"][key] < r["first_pass"][key] - 1e-9 for r in looped)
        return better, worse, len(looped) - better - worse

    rb, rw, rs = delta("context_recall")
    fb, fw, fs = delta("answer_f1")
    loop_hist = {k: sum(r["n_loops"] == k for r in records) for k in range(3)}
    gave_up = sum(r["path"] == "looped" and not r["final_relevance_ok"] for r in looped)
    rescued = sum(r["path"] == "looped" and r["final_relevance_ok"] for r in looped)

    lines = [
        f"# Agentic RAG (LangGraph) — loop-back evaluation (context mode: {MODE})",
        "",
        f"QA pairs: {n}  |  retrieval skipped: {len(skipped)}  |  single retrieval: {len(single)}  "
        f"|  looped: {len(looped)} ({100*len(looped)/n:.1f}%)",
        f"Loop counts: 0 loops={loop_hist[0]}, 1 loop={loop_hist[1]}, 2 loops={loop_hist[2]}  "
        f"|  grader accepted after rewrite: {rescued}, hit retry cap: {gave_up}",
        "",
        "| group | stage | n | context recall | answer F1 | faithfulness | gold-chunk hit |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r['group']} | {r['stage']} | {r['n']} | {r['context_recall']} | {r['answer_f1']} "
          f"| {r['faithfulness']} | {r['gold_chunk_hit']} |" for r in rows],
        "",
        "Among looped queries (final vs first-pass):",
        f"- context recall: improved {rb}, worse {rw}, unchanged {rs}",
        f"- answer F1:      improved {fb}, worse {fw}, unchanged {fs}",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved: {LOG_PATH}, {TABLE_OUT}, {SUMMARY}")


if __name__ == "__main__":
    main()
