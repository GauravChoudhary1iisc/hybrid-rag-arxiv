"""Aggregate out/*.json (produced by scripts 01-06 in this run) into results.json,
results.csv (per-question) and SUMMARY.md (tables). No numbers are typed by hand."""
import csv
import json
import sys

from common import EXP, OUT, save_json

R = {n: json.load(open(OUT / f"{n}.json")) for n in
     ["01_inspect", "02_embed_index", "03_retrieval_eval", "04_tokenizer_stats", "05_bm25_tokenization", "06_generation"]}
save_json(R, EXP / "results.json")

# per-question CSV
rows = []
for src, key in [("03_retrieval_eval", "step1"), ("05_bm25_tokenization", "step2iii_bm25")]:
    for cfg, qs in R[src]["per_question"].items():
        for q in qs:
            rows.append({"step": key, "config": cfg, "qid": q["qid"], "context_recall": q["context_recall"],
                         "hit5_paper": q["hit5_paper"], "mrr5_paper": q["mrr5_paper"], "hit5_chunk": q["hit5_chunk"],
                         "answer_f1": "", "answer_faithfulness": "",
                         "retrieved_chunk_ids": " ".join(map(str, q["retrieved_chunk_ids"]))})
for cfg, qs in R["06_generation"]["per_question"].items():
    for q in qs:
        rows.append({"step": "step3_generation", "config": cfg, "qid": q["qid"], "context_recall": q["context_recall"],
                     "hit5_paper": "", "mrr5_paper": "", "hit5_chunk": "", "answer_f1": q["answer_f1"],
                     "answer_faithfulness": q["answer_faithfulness"], "retrieved_chunk_ids": ""})
with open(EXP / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def ci(s, p=4):
    return f"{s['mean']:.{p}f} [{s['ci_low']:.{p}f}, {s['ci_high']:.{p}f}]"


def dci(d):
    return f"{d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]" + (" *" if d["ci_excludes_zero"] else "")


M = ["context_recall", "hit5_paper", "mrr5_paper", "hit5_chunk"]
e, r3, t4, b5, g6 = (R[k] for k in ["02_embed_index", "03_retrieval_eval", "04_tokenizer_stats",
                                     "05_bm25_tokenization", "06_generation"])
L = ["# SUMMARY — embedding & tokenizer comparison", "",
     f"Questions: {r3['summary']['bge|full']['context_recall']['n']} (src/data/eval_qa_set.json). "
     "All values: mean [bootstrap 95% CI, 1000 resamples, seed 42]. `*` = paired CI excludes 0.", "",
     "## Step 1 — models", "", "| model | dim | max_seq | encode s | chunks/s | points | raw vectors MB | Qdrant disk MB |",
     "|---|---|---|---|---|---|---|---|"]
for k, m in e["models"].items():
    L.append(f"| {m['model']} | {m['dim']} | {m['max_seq_length']} | {m['encode_seconds']} | {m['chunks_per_second']} | "
             f"{m['points_count']} | {m['raw_vector_mb']} | {m['qdrant_disk_mb']} |")
L += ["", "## Step 1 — retrieval (top-5)", "", "| config | " + " | ".join(M) + " |", "|---" * (len(M) + 1) + "|"]
for c in [k for k in r3["summary"] if "|" in k and not k.startswith("repro")]:
    L.append(f"| {c} | " + " | ".join(ci(r3["summary"][c][m]) for m in M) + " |")
L += ["", f"Reproducibility: identical top-5 lists, exp_bge vs existing rag_fixed: dense "
      f"{r3['summary']['repro_identical_top5|dense']}/88, full {r3['summary']['repro_identical_top5|full']}/88.",
      "", "### Paired difference vs bge-base (same pipeline)", "", "| diff | " + " | ".join(M) + " |", "|---" * (len(M) + 1) + "|"]
for k in ["minilm", "e5"]:
    for p in ["dense", "full"]:
        L.append(f"| {k} − bge, {p} | " + " | ".join(dci(r3["paired_vs_bge"][f"{k}-bge|{p}|{m}"]) for m in M) + " |")
L += ["", "## Step 2 (i) — truncation (passage string as embedded, special tokens included)", "",
      "| model | max_seq_length | mean tok | median | p95 | max | % > max (n) | % > max without prefix |", "|---|---|---|---|---|---|---|---|"]
for k, t in t4["truncation"].items():
    w = t["with_prefix"]
    L.append(f"| {t['model']} | {t['max_seq_length']} | {w['mean']} | {w['median']} | {w['p95']} | {w['max']} | "
             f"{w['pct_over_max']}% ({w['n_over_max']}) | {t['no_prefix']['pct_over_max']}% |")
L += ["", "## Step 2 (ii) — fragmentation", "", "| tokenizer | tokens / whitespace-word | distinct words ≥4 tokens | % word occurrences ≥4 tokens |",
      "|---|---|---|---|"]
for k, f in t4["fragmentation"].items():
    L.append(f"| {f['hf_id']} | {f['tokens_per_word']} | {f['distinct_words_4plus_tokens']} | {f['pct_word_occurrences_4plus']}% |")
for k, f in t4["fragmentation"].items():
    L += ["", f"Top-20 frequent words with ≥4 tokens — {f['hf_id']}: " +
          ", ".join(f"`{d['word']}` ({d['n_tokens']} tok, ×{d['freq']})" for d in f["top20_4plus"])]
L += ["", "## Step 2 (iii) — BM25-only, top-5", "", "| tokenizer | " + " | ".join(M) + " |", "|---" * (len(M) + 1) + "|"]
for k in ["regex", "regex_porter", "wordpiece"]:
    L.append(f"| {k} | " + " | ".join(ci(b5["summary"][k][m]) for m in M) + " |")
for k in ["regex_porter", "wordpiece"]:
    L.append(f"| {k} − regex | " + " | ".join(dci(b5["paired_vs_regex"][f"{k}-regex|{m}"]) for m in M) + " |")
L += ["", f"Sanity: in-memory regex BM25 top-5 identical to existing pickle for "
      f"{b5['summary']['sanity_regex_matches_existing_pickle_top5']}/88 questions.",
      "", "## Step 3 — generation (qwen2.5:7b, existing generator)", "",
      f"Current = `{g6['current']}`; compared = `{g6['best']}` ({g6['selection_note']}). Questions: {g6['questions']} (n={g6['n']}).", "",
      "| config | answer_f1 | answer_faithfulness | context_recall |", "|---|---|---|---|"]
for c in [g6["current"], g6["best"]]:
    L.append(f"| {c} | " + " | ".join(ci(g6["summary"][c][m]) for m in ["answer_f1", "answer_faithfulness", "context_recall"]) + " |")
L.append(f"| {g6['best']} − current | " + " | ".join(dci(g6["paired_best_minus_current"][m]) for m in ["answer_f1", "answer_faithfulness", "context_recall"]) + " |")
L += ["", "## Limits", "",
      f"- {g6['n']} questions for generation, 88 for retrieval; one QA set, synthetic questions each generated from one chunk.",
      "- context_recall, answer_f1 and faithfulness are word-overlap metrics (src/evaluate.py), not semantic judgements.",
      f"- Hardware: {e['gpu']}, {e['gpu_vram_gb']} GB VRAM; timings are single runs on this machine.",
      f"- Step 3 subset: {g6['questions']}.",
      "- Ollama generation is not seeded in the existing generator (used unchanged), so answers may vary between runs.", ""]
with open(EXP / "SUMMARY.md", "w") as f:
    f.write("\n".join(L))
print("\n".join(L))
print("\nwrote results.json, results.csv, SUMMARY.md")
