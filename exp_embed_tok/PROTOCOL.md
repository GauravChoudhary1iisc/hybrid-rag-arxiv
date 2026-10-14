# PROTOCOL — embedding-model & tokenizer comparison

Written 2026-10-14, after Step 0 inspection and BEFORE any experiment was run.
It is not to be changed after results are seen. Any deviation is listed in SUMMARY.md.

## Fixed inputs (identical for every configuration)
- Chunks: `src/data/chunks_fixed.json` (28,711 fixed-size chunks; `chunk_id == list index`).
- Evaluation set: `src/data/eval_qa_set.json`, **all 88 questions** (each has `ground_truth`,
  `arxiv_id` and the source `chunk_id`). The 30-question set used by `src/evaluate.py` is a
  `random.sample(seed=42)` subset of the same file, so the 88 set is the main set.
  No parameter is tuned on these questions; there is no tuning at all.
- BM25: existing `src/data/bm25_index.pkl` (read-only) for the hybrid pipeline.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`. RRF k=60, hybrid fetch 50 per list,
  rerank pool 50, final top_k=5. All of this is reused from `src/retreiver.py` by subclassing
  (only the embedder, query prefix and collection name are swapped).
- Generator: `src/generator.py` `Generator.generate`, unchanged (qwen2.5:7b via Ollama,
  temperature 0.1, num_predict 1024, same prompt, 5 chunks × 800 chars).
- Seed 42 for python `random`, numpy, torch and the bootstrap.

## Step 1 — embedding models
| key | model | passage text embedded | query text embedded | max_seq_length |
|---|---|---|---|---|
| bge | BAAI/bge-base-en-v1.5 | `"Represent this sentence for searching relevant passages: " + text` (exactly as existing `src/embed.py`) | same prefix + query (as `src/retreiver.py`) | model default (512) |
| minilm | sentence-transformers/all-MiniLM-L6-v2 | text | query | model default (256) |
| e5 | intfloat/e5-base-v2 | `"passage: " + text` | `"query: " + query` | model default (512) |

- All: `normalize_embeddings=True`, fp32, batch_size 32, cuda. Collections `exp_bge`, `exp_minilm`,
  `exp_e5`, cosine distance, dim from the model. Existing `rag_fixed` is not touched; it is also
  queried (bge, read-only) as a reproducibility check, and bge+full on `rag_fixed` is the
  "current configuration".
- Recorded: dim, encode wall time (cuda-synchronised, includes tokenisation), chunks/s, Qdrant
  points_count, raw vector size (N×dim×4 bytes), on-disk collection size if readable.
- Pipelines per model: `dense` (top 5) and `full` = hybrid RRF + cross-encoder rerank (top 5).

## Metrics (per question, top 5 retrieved chunks)
- `context_recall`: imported unchanged from `src/evaluate.py`.
- `hit5_paper`: 1 if any retrieved chunk's `arxiv_id` equals the question's `arxiv_id`.
- `mrr5_paper`: 1/rank of the first retrieved chunk from the source paper, else 0.
- `hit5_chunk`: 1 if the exact source `chunk_id` is retrieved (extra, stricter).
- `answer_f1`: from `src/evaluate.py`, Step 3 only.
- Uncertainty: percentile bootstrap 95% CI, 1000 resamples over questions, `numpy.random.default_rng(42)`.
  Paired differences vs the reference (bge for Step 1, regex for Step 2iii, current for Step 3):
  bootstrap of the per-question difference.
- Wording rule: "improvement/worse" only if the paired 95% CI excludes 0; otherwise "directional"
  or "no meaningful difference".

## Step 2 — tokenizers
- (i) Truncation: each Step-1 model's tokenizer on exactly the passage string it embedded
  (prefix included), special tokens included; % of chunks with token count > max_seq_length;
  mean / median / p95 tokens per chunk. Also reported without the prefix.
- (ii) Fragmentation: `bert-base-uncased`, `gpt2`, `Qwen/Qwen2.5-7B-Instruct` tokenizers.
  tokens per whitespace-word = Σ tokens(chunk, no special tokens) / Σ whitespace words, over all chunks.
  Top-20 split words: word = whitespace token stripped of leading/trailing non-alphanumerics,
  must contain a letter; each distinct word tokenised alone (GPT-2/Qwen with a leading space, as
  it occurs mid-text); words with ≥4 tokens ranked by corpus frequency.
- (iii) BM25-only, rebuilt in memory (BM25Okapi defaults): `regex` (current tokenize from
  `src/build_bim.py`), `regex+porter` (nltk PorterStemmer on each regex token), `wordpiece`
  (bert-base-uncased subword tokens, no special tokens). Top 5; context_recall, hit5_paper,
  mrr5_paper, hit5_chunk with CIs; paired vs regex. Sanity check: in-memory regex BM25 must give
  the same top-5 as the existing pickle.

## Step 3 — selection rule and generation
- Candidates: the 6 Step-1 configs {bge, minilm, e5} × {dense, full}.
- Current configuration = bge + full on the existing `rag_fixed` collection.
- Best = highest mean context_recall among the candidates (ties → the earlier in order
  bge, minilm, e5; full before dense). If the best is bge+full (i.e. the current system),
  the comparison uses the highest-ranked candidate that is not bge+full.
- Generate for all 88 questions for both configs, using the top-5 chunks saved in Step 1.
  If the mean generation time after the first 10 questions exceeds 60 s, fall back to a
  fixed random subset of 30 questions (`random.Random(42).sample`) for both configs.
- Report answer_f1 (+ answer_faithfulness, context_recall for reference) with CI and paired diff.
- Ollama sampling: the existing generator sends no seed; it is used unchanged, so answers
  are not guaranteed bit-reproducible.
