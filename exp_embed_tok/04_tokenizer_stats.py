"""Step 2 (i) truncation per embedding-model tokenizer, (ii) fragmentation for
bert-base-uncased / gpt2 / Qwen2.5 tokenizers."""
import re
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from common import MODELS, OUT, load_chunks, seed_all, save_json

seed_all()
chunks = load_chunks()
texts = [c["text"] for c in chunks]
res = {"truncation": {}, "fragmentation": {}}


def stats(lens):
    a = np.asarray(lens)
    return {"mean": round(float(a.mean()), 1), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": int(a.max())}


print("== (i) truncation")
for key, cfg in MODELS.items():
    st = SentenceTransformer(cfg["name"], device="cpu")
    tok, max_len = st.tokenizer, st.max_seq_length
    out = {"model": cfg["name"], "max_seq_length": max_len, "tokenizer": type(tok).__name__}
    for variant, pref in [("with_prefix", cfg["passage_prefix"]), ("no_prefix", "")]:
        lens = [len(ids) for ids in tok([pref + t for t in texts], add_special_tokens=True,
                                         truncation=False)["input_ids"]]
        over = sum(l > max_len for l in lens)
        out[variant] = {**stats(lens), "n_over_max": over, "pct_over_max": round(100 * over / len(lens), 2)}
    res["truncation"][key] = out
    w = out["with_prefix"]
    print(f"  {key:6s} max={max_len} tokens/chunk mean={w['mean']} median={w['median']} p95={w['p95']} "
          f"-> {w['pct_over_max']}% truncated ({w['n_over_max']}); no prefix {out['no_prefix']['pct_over_max']}%")

print("\n== (ii) fragmentation")
total_words = sum(len(t.split()) for t in texts)
word_counts = Counter()
for t in texts:
    for w in t.split():
        w = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", w)
        if w and re.search(r"[A-Za-z]", w):
            word_counts[w] += 1
vocab = list(word_counts)
print(f"  whitespace words: {total_words:,}; distinct cleaned words: {len(vocab):,}")
for name, hf in [("bert_wordpiece", "bert-base-uncased"), ("gpt2_bpe", "gpt2"),
                 ("qwen2.5", "Qwen/Qwen2.5-7B-Instruct")]:
    tok = AutoTokenizer.from_pretrained(hf)
    ntok = sum(len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"])
    lead = "" if name == "bert_wordpiece" else " "
    wl = tok([lead + w for w in vocab], add_special_tokens=False)["input_ids"]
    split = [(vocab[i], len(ids), word_counts[vocab[i]]) for i, ids in enumerate(wl) if len(ids) >= 4]
    split.sort(key=lambda x: (-x[2], x[0]))
    top = [{"word": w, "n_tokens": n, "freq": f, "pieces": tok.convert_ids_to_tokens(
        tok(lead + w, add_special_tokens=False)["input_ids"])} for w, n, f in split[:20]]
    res["fragmentation"][name] = {"hf_id": hf, "total_tokens": ntok, "total_words": total_words,
                                  "tokens_per_word": round(ntok / total_words, 4),
                                  "distinct_words_4plus_tokens": len(split),
                                  "pct_word_occurrences_4plus": round(100 * sum(f for _, _, f in split) / sum(word_counts.values()), 2),
                                  "top20_4plus": top}
    print(f"  {name:15s} tokens/word={ntok/total_words:.4f}  distinct words with >=4 tokens: {len(split):,}")
    print("     top20: " + ", ".join(f"{d['word']}({d['n_tokens']},{d['freq']})" for d in top))

save_json(res, OUT / "04_tokenizer_stats.json")
print("saved out/04_tokenizer_stats.json")
