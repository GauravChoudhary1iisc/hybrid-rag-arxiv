# Agentic RAG (LangGraph) — loop-back evaluation (context mode: best)

QA pairs: 88  |  retrieval skipped: 0  |  single retrieval: 57  |  looped: 31 (35.2%)
Loop counts: 0 loops=57, 1 loop=5, 2 loops=26  |  grader accepted after rewrite: 5, hit retry cap: 26

| group | stage | n | context recall | answer F1 | faithfulness | gold-chunk hit |
|---|---|---|---|---|---|---|
| all | first_pass | 88 | 0.8481 | 0.3638 | 0.826 | 0.7614 |
| all | final | 88 | 0.8483 | 0.366 | 0.8242 | 0.7614 |
| never_looped | final | 57 | 0.882 | 0.3897 | 0.8461 | 0.8421 |
| looped | first_pass | 31 | 0.7858 | 0.3163 | 0.7889 | 0.6129 |
| looped | final | 31 | 0.7864 | 0.3223 | 0.7839 | 0.6129 |

Among looped queries (final vs first-pass):
- context recall: improved 1, worse 0, unchanged 30
- answer F1:      improved 13, worse 6, unchanged 12
