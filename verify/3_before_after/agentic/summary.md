# Agentic RAG (LangGraph) — loop-back evaluation

QA pairs: 88  |  retrieval skipped: 0  |  single retrieval: 57  |  looped: 31 (35.2%)
Loop counts: 0 loops=57, 1 loop=5, 2 loops=26  |  grader accepted after rewrite: 5, hit retry cap: 26

| group | stage | n | context recall | answer F1 | faithfulness | gold-chunk hit |
|---|---|---|---|---|---|---|
| all | first_pass | 88 | 0.8481 | 0.3594 | 0.8209 | 0.7614 |
| all | final | 88 | 0.8166 | 0.3284 | 0.8028 | 0.6818 |
| never_looped | final | 57 | 0.882 | 0.3814 | 0.845 | 0.8421 |
| looped | first_pass | 31 | 0.7858 | 0.3191 | 0.7765 | 0.6129 |
| looped | final | 31 | 0.6964 | 0.2311 | 0.7251 | 0.3871 |

Among looped queries (final vs first-pass):
- context recall: improved 3, worse 10, unchanged 18
- answer F1:      improved 7, worse 19, unchanged 5
