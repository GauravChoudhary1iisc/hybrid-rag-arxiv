"""
Export the per-query routing log (path taken by each query through the LangGraph)
from eval/agentic*/path_log.jsonl into readable CSVs.

Run (from src/):  python export_routing.py
"""

import csv
import json
from pathlib import Path

MODES = {"last": "eval/agentic", "best": "eval/agentic_best", "merge": "eval/agentic_merge"}
ROUTE = {0: "single-pass", 1: "looped once", 2: "looped twice"}

for mode, d in MODES.items():
    records = sorted((json.loads(l) for l in open(Path(d) / "path_log.jsonl")), key=lambda r: r["id"])
    out = Path(d) / "routing_per_query.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "route", "n_retrievals", "grader_verdicts",
                    "queries_tried", "recall_first", "recall_final", "f1_first", "f1_final"])
        for r in records:
            route = "skipped retrieval" if r["path"] == "skipped" else ROUTE[r["n_loops"]]
            w.writerow([
                r["id"], r["question"], route, r["n_retrievals"],
                " | ".join(str(a["grader_raw"]) for a in r["attempts"]),
                " || ".join(a["query"] for a in r["attempts"]),
                r["first_pass"]["context_recall"], r["final"]["context_recall"],
                r["first_pass"]["answer_f1"], r["final"]["answer_f1"],
            ])
    counts = {}
    for r in records:
        k = "skipped retrieval" if r["path"] == "skipped" else ROUTE[r["n_loops"]]
        counts[k] = counts.get(k, 0) + 1
    print(f"{mode:5} → {out}  {counts}")
