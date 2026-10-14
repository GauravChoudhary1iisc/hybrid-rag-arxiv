# Agentic RAG (LangGraph) — loop-back evaluation (context mode: merge)

QA pairs: 88  |  retrieval skipped: 0  |  single retrieval: 57  |  looped: 31 (35.2%)
Loop counts: 0 loops=57, 1 loop=5, 2 loops=26  |  grader accepted after rewrite: 5, hit retry cap: 26

| group | stage | n | context recall | answer F1 | faithfulness | gold-chunk hit |
|---|---|---|---|---|---|---|
| all | first_pass | 88 | 0.8481 | 0.3625 | 0.8262 | 0.7614 |
| all | final | 88 | 0.8575 | 0.3714 | 0.8229 | 0.7727 |
| never_looped | final | 57 | 0.882 | 0.3847 | 0.8494 | 0.8421 |
| looped | first_pass | 31 | 0.7858 | 0.3218 | 0.7835 | 0.6129 |
| looped | final | 31 | 0.8125 | 0.3469 | 0.774 | 0.6452 |

Among looped queries (final vs first-pass):
- context recall: improved 5, worse 0, unchanged 26
- answer F1:      improved 12, worse 9, unchanged 10
