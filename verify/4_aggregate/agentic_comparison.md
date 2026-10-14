# Agentic RAG loop-back: which context reaches the generator

88 QA pairs, hybrid_rerank top-5, Qwen2.5-7B grader/rewriter/generator, max 2 retries.
The grader is identical across modes: 31/88 queries (35.2%) looped
(5 accepted after 1 rewrite, 26 hit the retry cap).

## Looped queries only (n=31)

| mode | context recall | answer F1 | gold-chunk hit | recall improved / worse | F1 improved / worse |
|---|---|---|---|---|---|
| first pass (single-pass baseline) | 0.786 | 0.316–0.322 | 0.61 | — | — |
| last attempt | 0.696 | 0.231 | 0.39 | 3 / 10 | 7 / 19 |
| best attempt | 0.786 | 0.322 | 0.61 | 1 / 0 | 13 / 6 |
| merge all attempts + rerank | **0.813** | **0.347** | **0.65** | **5 / 0** | 12 / 9 |

## All 88 queries

| pipeline | context recall | answer F1 | gold-chunk hit |
|---|---|---|---|
| single-pass hybrid_rerank | 0.848 | 0.359–0.364 | 0.76 |
| agentic, last attempt | 0.817 | 0.328 | 0.68 |
| agentic, best attempt | 0.848 | 0.366 | 0.76 |
| agentic, merge | **0.858** | **0.371** | **0.77** |

(Baseline F1 varies slightly between runs because generation uses temperature 0.1.)

## Conclusion

- The grader is a useful detector: queries it flagged had lower first-pass recall (0.79 vs 0.88).
- Rewritten queries alone hurt: replacing the original retrieval drifted away from the
  source paper's wording (gold-chunk hit 61% → 39%).
- Rewrites rarely beat the first retrieval as a whole set (best-attempt chose the first pass in 30/31).
- But they do surface a few useful extra chunks: merging all attempts and reranking against the
  original question improved recall on 5/31 looped queries and worsened none
  (+0.027 recall, +0.03 F1 on looped queries; about +0.01 overall).
- The F1 gain is small and within generation noise (12 improved vs 9 worse). The recall gain is
  real, but partly expected from the design: the merged pool always contains the first-pass chunks,
  so the reranker can only swap in a new chunk when it scores higher.

Per-query logs: `agentic/`, `agentic_best/`, `agentic_merge/` → `path_log.jsonl`, `metrics_table.csv`, `summary.md`.
