"""Step 2 (iii): BM25-only retrieval with three tokenizers, rebuilt in memory only."""
import pickle
import time

import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

from common import OUT, load_chunks, load_qa, seed_all, save_json, retrieval_metrics, boot_ci, paired_diff
from build_bim import tokenize as regex_tokenize  # existing tokenizer, unchanged (needs common's sys.path)

METRICS = ["context_recall", "hit5_paper", "mrr5_paper", "hit5_chunk"]
seed_all()
chunks, qa = load_chunks(), load_qa()
stem = PorterStemmer()
wp = AutoTokenizer.from_pretrained("bert-base-uncased")
stem_cache = {}


def porter(text):
    out = []
    for t in regex_tokenize(text):
        if t not in stem_cache:
            stem_cache[t] = stem.stem(t)
        out.append(stem_cache[t])
    return out


def wordpiece(text):
    return wp.tokenize(text)


tokenizers = {"regex": regex_tokenize, "regex_porter": porter, "wordpiece": wordpiece}
per_q, summary = {}, {}
for name, fn in tokenizers.items():
    t0 = time.time()
    bm = BM25Okapi([fn(c["text"]) for c in chunks])
    build_s = time.time() - t0
    rows = []
    for x in qa:
        sc = bm.get_scores(fn(x["question"]))
        top = np.argsort(sc)[::-1][:5]  # same ranking code as retreiver._bm25_search
        m = retrieval_metrics(x, [chunks[int(i)] for i in top])
        m["qid"], m["retrieved_chunk_ids"] = x["id"], [int(i) for i in top]
        rows.append(m)
    per_q[name] = rows
    summary[name] = {mt: boot_ci([r[mt] for r in rows]) for mt in METRICS}
    summary[name]["build_seconds"] = round(build_s, 1)
    print(f"{name:13s} build {build_s:.0f}s  " + "  ".join(f"{mt}={summary[name][mt]['mean']:.4f}" for mt in METRICS))
    if name == "regex":  # sanity: matches the existing pickle?
        with open("data/bm25_index.pkl", "rb") as f:
            old = pickle.load(f)
        same = sum(list(np.argsort(old.get_scores(regex_tokenize(x["question"])))[::-1][:5]) == r["retrieved_chunk_ids"]
                   for x, r in zip(qa, rows))
        summary["sanity_regex_matches_existing_pickle_top5"] = int(same)
        print(f"  sanity: in-memory regex top-5 == existing pickle top-5 for {same}/{len(qa)} questions")
        del old

diffs = {}
for name in ["regex_porter", "wordpiece"]:
    for mt in METRICS:
        d = paired_diff([r[mt] for r in per_q[name]], [r[mt] for r in per_q["regex"]])
        diffs[f"{name}-regex|{mt}"] = d
        print(f"diff {name}-regex {mt:15s} {d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
              + (" *CI excludes 0" if d["ci_excludes_zero"] else ""))
save_json({"summary": summary, "paired_vs_regex": diffs, "per_question": per_q,
           "config": {"bm25": "rank_bm25.BM25Okapi defaults", "top_k": 5, "wordpiece": "bert-base-uncased",
                      "stemmer": "nltk PorterStemmer"}}, OUT / "05_bm25_tokenization.json")
print("saved out/05_bm25_tokenization.json")
