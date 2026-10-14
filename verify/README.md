# Agentic RAG with LangGraph: verification package

Eval set: all 88 QA pairs from `data/eval_qa_set.json`.

The graph was run three times. The grader and the retry loop are identical in all three runs;
they differ only in which retrieved chunks the final answer is generated from:

| Run folder | Chunks used for the final answer |
|---|---|
| `agentic` | the last retrieval |
| `agentic_best` | the retrieval that scores highest against the original question |
| `agentic_merge` | every retrieval's chunks merged, then reranked against the original question |

## 1. LangGraph code: `1_code/`
- `agentic_graph.py`
  - State definition: `RAGState` (line 38). It holds `question`, `retrieved_chunks`, `relevance_ok`, `retry_count` and `answer`, plus fields used for logging.
  - The four nodes: `query_classifier`, `retrieve`, `grade_relevance` and `generate`.
  - Graph wiring: `build_graph`. The conditional edges are at lines 226 and 229.
  - The retry loop: `route_after_grade` (line 208) sends the run back to `retrieve` while `retry_count <= MAX_RETRIES` (2).
- `eval_agentic.py`: the evaluation script.
- `export_routing.py`: builds the per-query routing tables used in section 2.

## 2. Per-query routing: `2_routing_log/<run>/`
- `routing_per_query.csv`: one row per query. Columns:
  - the route (skipped retrieval / single-pass / looped once / looped twice)
  - the grader's verdict for each retrieval
  - every query tried, including the rewrites
  - context recall and F1 for the first and the final retrieval
- `path_log.jsonl`: the raw record for each query, including the chunk IDs from every retrieval.

Route counts (the same in all three runs): 57 single-pass, 5 looped once, 26 looped twice, 0 skipped.
None of the eval questions should skip retrieval. The skip branch is shown working in the first
lines of `5_terminal_logs/agentic_run.log` ("What is 2+2?", "what is 12 * 7" and "Hello, how are you?"
all ran with `retrieval=False`).

## 3. Before/after on looped queries: `3_before_after/<run>/`
"Before" is the first retrieval and "after" is the final one. Each folder's `summary.md` gives
the numbers and how many queries improved or got worse; `metrics_table.csv` has the same data.

| Looped queries (n=31) | Context recall | Answer F1 |
|---|---|---|
| first retrieval (before) | 0.786 | ~0.32 |
| final, last retrieval | 0.696 | 0.231 |
| final, best retrieval | 0.786 | 0.322 |
| final, merged retrievals | 0.813 | 0.347 |

## 4. Aggregate vs. the original single-pass RAG: `4_aggregate/`
- `comparison_table.csv`: the original single-pass project's results. Hybrid + rerank gave 0.851 context recall and 0.365 F1, measured on 30 of the QA pairs.
- `agentic_comparison.md`: all three runs compared with single-pass, on all 88 pairs:

| Pipeline (88 pairs) | Context recall | Answer F1 |
|---|---|---|
| single-pass hybrid + rerank | 0.848 | 0.36 |
| LangGraph, last retrieval | 0.817 | 0.328 |
| LangGraph, best retrieval | 0.848 | 0.366 |
| LangGraph, merged retrievals | 0.858 | 0.371 |

## 5. Terminal output: `5_terminal_logs/`
The unedited stdout and stderr of the three runs. Each contains the model-loading output and one
progress line per query (88 in each), for example:
`[70/88] looped  loops=2 recall 0.70→0.90 f1 0.15→0.25 (9.8s)`.

## Result
- Re-retrieval fired on 35% of queries.
- Answering from the last retrieval made results worse.
- Merging all retrievals improved context recall on 5 of the 31 looped queries and made none worse.
- Overall gain from merging: about +0.01 context recall and +0.01 F1. The F1 gain is within run-to-run generation noise.
