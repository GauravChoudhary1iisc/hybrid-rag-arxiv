"""Step 1b: dense top-5 and full pipeline (hybrid RRF + cross-encoder rerank, top-5) for each
embedding model, using the existing Retriever code with only embedder/prefix/collection swapped."""
import time

import torch
from sentence_transformers import SentenceTransformer

from common import MODELS, OUT, load_qa, seed_all, save_json, retrieval_metrics, boot_ci, paired_diff
from retreiver import Retriever

METRICS = ["context_recall", "hit5_paper", "mrr5_paper", "hit5_chunk"]


class ExpRetriever(Retriever):
    def configure(self, embedder, query_prefix, collection):
        self.embedder, self.query_prefix, self.collection = embedder, query_prefix, collection

    def _embed_query(self, query):
        return self.embedder.encode(self.query_prefix + query, normalize_embeddings=True,
                                    convert_to_numpy=True).tolist()

    def _qdrant_search(self, query_vec, top_k):
        return self.qdrant.query_points(collection_name=self.collection, query=query_vec,
                                        limit=top_k, with_payload=True).points


seed_all()
qa = load_qa()
r = ExpRetriever(device="cuda")  # loads existing bge, reranker, BM25 pickle (read-only)
bge_loaded = r.embedder

# (key, model key, collection); rag_fixed = existing index = current system / reproducibility check
runs = [("bge_ragfixed", "bge", "rag_fixed")] + [(k, k, f"exp_{k}") for k in MODELS]
per_q, summary = {}, {}
for run_key, mkey, coll in runs:
    cfg = MODELS[mkey]
    emb = bge_loaded if mkey == "bge" else SentenceTransformer(cfg["name"], device="cuda")
    r.configure(emb, cfg["query_prefix"], coll)
    for pipe in ["dense", "full"]:
        name = f"{run_key}|{pipe}"
        rows, t0 = [], time.time()
        for x in qa:
            res = r.dense_retrieve(x["question"], 5) if pipe == "dense" else r.hybrid_rerank(x["question"], 5)
            m = retrieval_metrics(x, res)
            m["qid"] = x["id"]
            m["retrieved_chunk_ids"] = [int(c["chunk_id"]) for c in res]
            rows.append(m)
        el = time.time() - t0
        per_q[name] = rows
        summary[name] = {mt: boot_ci([row[mt] for row in rows]) for mt in METRICS}
        summary[name]["seconds_total"] = round(el, 1)
        print(f"{name:22s} " + "  ".join(f"{mt}={summary[name][mt]['mean']:.4f}" for mt in METRICS)
              + f"  ({el:.0f}s)")
    if mkey != "bge":
        del emb
        torch.cuda.empty_cache()

# reproducibility check: exp_bge vs existing rag_fixed
for pipe in ["dense", "full"]:
    a, b = per_q[f"bge_ragfixed|{pipe}"], per_q[f"bge|{pipe}"]
    same = sum(x["retrieved_chunk_ids"] == y["retrieved_chunk_ids"] for x, y in zip(a, b))
    print(f"repro check {pipe}: identical top-5 lists exp_bge vs rag_fixed: {same}/{len(a)}")
    summary[f"repro_identical_top5|{pipe}"] = same

# paired diffs vs bge (exp_bge, same pipeline)
diffs = {}
for k in MODELS:
    if k == "bge":
        continue
    for pipe in ["dense", "full"]:
        for mt in METRICS:
            d = paired_diff([x[mt] for x in per_q[f"{k}|{pipe}"]], [x[mt] for x in per_q[f"bge|{pipe}"]])
            diffs[f"{k}-bge|{pipe}|{mt}"] = d
            print(f"diff {k}-bge {pipe:5s} {mt:15s} {d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
                  + (" *CI excludes 0" if d["ci_excludes_zero"] else ""))

save_json({"summary": summary, "paired_vs_bge": diffs, "per_question": per_q,
           "config": {"top_k": 5, "rrf_k": 60, "hybrid_fetch": 50, "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                      "models": MODELS}}, OUT / "03_retrieval_eval.json")
print("saved out/03_retrieval_eval.json")
