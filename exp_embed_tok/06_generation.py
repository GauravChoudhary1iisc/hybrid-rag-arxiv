"""Step 3: generate answers (existing Generator, unchanged) for the current config and the
best config chosen by the PROTOCOL.md rule, using the top-5 chunks saved by 03."""
import json
import random
import time

from common import OUT, load_chunks, load_qa, seed_all, save_json, boot_ci, paired_diff
from evaluate import answer_f1, answer_faithfulness, context_recall
from generator import Generator

seed_all()
chunks, qa = load_chunks(), load_qa()
r3 = json.load(open(OUT / "03_retrieval_eval.json"))

# selection rule (PROTOCOL.md): order bge, minilm, e5; full before dense; ties -> earlier
order = [f"{m}|{p}" for m in ["bge", "minilm", "e5"] for p in ["full", "dense"]]
scores = {c: r3["summary"][c]["context_recall"]["mean"] for c in order}
ranked = sorted(order, key=lambda c: (-scores[c], order.index(c)))
best = ranked[0]
note = "best by context recall"
if best == "bge|full":
    best = ranked[1]
    note = "rank-1 was bge|full (= current system); using rank-2 per protocol"
current = "bge_ragfixed|full"
print("context-recall ranking: " + ", ".join(f"{c}={scores[c]:.4f}" for c in ranked))
print(f"current = {current}; comparison = {best} ({note})")

gen = Generator()
configs = [current, best]
ret = {c: {row["qid"]: row["retrieved_chunk_ids"] for row in r3["per_question"][c]} for c in configs}
questions = qa
subset_note = "all 88 questions"
per_q = {c: [] for c in configs}
t_start = time.time()
n_done = 0
i = 0
while i < len(questions):
    x = questions[i]
    for c in configs:
        ctx = [chunks[cid] for cid in ret[c][x["id"]]]
        t0 = time.time()
        ans = gen.generate(x["question"], ctx)["answer"]
        dt = time.time() - t0
        texts = [k["text"] for k in ctx]
        per_q[c].append({"qid": x["id"], "answer": ans, "gen_seconds": round(dt, 2),
                         "answer_f1": answer_f1(ans, x["ground_truth"]),
                         "answer_faithfulness": answer_faithfulness(ans, texts),
                         "context_recall": context_recall(x["ground_truth"], texts)})
        n_done += 1
    print(f"[{i+1}/{len(questions)}] f1 current={per_q[current][-1]['answer_f1']:.3f} "
          f"best={per_q[best][-1]['answer_f1']:.3f}  ({per_q[current][-1]['gen_seconds']}s, {per_q[best][-1]['gen_seconds']}s)")
    i += 1
    if i == 10 and len(questions) == 88:
        mean_s = (time.time() - t_start) / n_done
        print(f"mean generation time after 10 questions: {mean_s:.1f}s/answer")
        if mean_s > 60:
            sub_ids = {q["id"] for q in random.Random(42).sample(qa, 30)}
            questions = [q for q in qa if q["id"] in sub_ids]
            per_q = {c: [] for c in configs}
            i, n_done, t_start = 0, 0, time.time()
            subset_note = "fixed random subset of 30 (random.Random(42).sample) - generation too slow"
            print("FALLBACK: " + subset_note + "; restarting on the subset")

summary = {c: {m: boot_ci([r[m] for r in per_q[c]]) for m in ["answer_f1", "answer_faithfulness", "context_recall"]}
           for c in configs}
diff = {m: paired_diff([r[m] for r in per_q[best]], [r[m] for r in per_q[current]])
        for m in ["answer_f1", "answer_faithfulness", "context_recall"]}
for c in configs:
    print(f"{c:20s} " + "  ".join(f"{m}={summary[c][m]['mean']:.4f} [{summary[c][m]['ci_low']:.4f},{summary[c][m]['ci_high']:.4f}]"
                                   for m in summary[c]))
for m, d in diff.items():
    print(f"diff {best} - current {m:20s} {d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
          + (" *CI excludes 0" if d["ci_excludes_zero"] else ""))
save_json({"current": current, "best": best, "selection_note": note, "context_recall_ranking": {c: scores[c] for c in ranked},
           "questions": subset_note, "n": len(questions), "summary": summary, "paired_best_minus_current": diff,
           "per_question": per_q, "total_minutes": round((time.time() - t_start) / 60, 1),
           "config": {"generator": "src/generator.py unchanged (qwen2.5:7b, temperature 0.1, num_predict 1024, no seed)"}},
          OUT / "06_generation.json")
print("saved out/06_generation.json")
