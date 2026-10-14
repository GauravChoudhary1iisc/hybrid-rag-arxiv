# LangGraph agentic RAG project: interview answer

## Short version (about 30 seconds)

I extended my RAG system over arXiv papers into an agentic pipeline using LangGraph. After
retrieval, an LLM grades whether the chunks actually answer the question. If they don't, it
rewrites the query and retrieves again, up to two retries. The goal was to test whether that
loop really improves answers. On 88 QA pairs the grader was a good detector of weak retrievals.
But simply replacing the retrieval with the rewrite made things worse: recall on looped queries
dropped from 0.79 to 0.70. I fixed that by merging the chunks from all attempts and reranking
them against the original question. That raised recall on looped queries to 0.81, with zero
regressions, though the overall gain was only about one point.

## Full version (about 2 minutes)

**Context.** I had already built a RAG system over about 28,000 chunks from arXiv papers on RAG
and NLP. It uses hybrid retrieval: dense BGE embeddings in Qdrant plus BM25, fused with
reciprocal rank fusion, then a cross-encoder reranker. Generation runs locally with Qwen2.5-7B
on Ollama. The single-pass pipeline scored about 0.85 context recall and 0.36 answer F1.

**What I built.** I rebuilt it as a LangGraph state machine with four nodes:
- A **query classifier** that skips retrieval for things like arithmetic or greetings.
- **Retrieve**, which reuses my existing hybrid-plus-rerank retriever.
- **Grade relevance**, where the LLM gives a strict yes/no on whether the chunks contain the
  answer, always judged against the original question.
- **Generate**, which produces the cited answer.

The key part is the conditional edge after grading: if the grade is "no," the graph loops back
to retrieve with an LLM-rewritten query, capped at two retries. That loop is what makes it a
graph rather than a linear pipeline.

**How I evaluated it.** I ran all 88 QA pairs and logged each query's route. For every query
that looped, I compared the first retrieval with the final one. The first retrieval is exactly
the old single-pass pipeline, so that's a clean before/after comparison. I also generated an
answer from the first-pass chunks, so the F1 comparison is fair too.

**What I found.**
- The loop fired on 35% of queries. The grader was a good detector: queries it flagged had
  first-pass recall of 0.79, vs. 0.88 for the rest.
- But the naive version **hurt**. Recall on looped queries fell from 0.79 to 0.70, and the
  chance of retrieving the source chunk fell from 61% to 39%. The rewrites moved away from
  the paper's exact terminology, which BM25 and the reranker rely on.
- So I tested two fixes. Picking the best single attempt basically always chose the first
  retrieval, so rewrites alone weren't better. But **merging the chunks from all attempts
  and reranking against the original question** worked. It improved recall on 5 of 31
  looped queries and made none worse.
- Final numbers on all 88 queries: recall 0.848 → 0.858, F1 0.36 → 0.37.

**The takeaway.** The agentic loop adds extra LLM calls on a third of queries for about a
one-point gain. The more useful lessons were that a retrieval grader is a good detector,
query rewriting is a weak fixer, and when you retry, you should add to the context rather
than replace it. I'd rather report that honestly than claim agents automatically make RAG better.

## Likely follow-up questions

**Why LangGraph instead of a simple loop in Python?**
The control flow is explicit: typed state, named nodes and conditional edges. That makes the
routing easy to log, test and extend. For example, adding a web-search fallback node is one
new edge. For four nodes a plain while-loop would also work. What LangGraph adds is
structure and observability, not new capability.

**Why did query rewriting hurt?**
The QA questions were generated from specific chunks, so they already use the paper's wording.
The LLM rewrites added synonyms and generic phrasing, which hurt BM25's exact-term matching
and moved the dense embedding away from the source chunk.

**How do you know the grader works?**
Queries it rejected had clearly lower first-pass recall (0.79 vs. 0.88) and a lower
source-chunk hit rate (61% vs. 84%). One caveat: 26 of the 31 looped queries hit the retry cap,
so it may be too strict, or those chunks really weren't in the index.

**What are the limitations of your metrics?**
Context recall and F1 are keyword-overlap metrics, deterministic and cheap but not semantic.
Generation runs at temperature 0.1, so F1 varies a bit between runs, and the F1 gain is within
that noise. The recall improvement is the more reliable signal.

**What would you do next?**
- Use the grader's "no" to trigger a different retrieval strategy (HyDE, or decomposing the
  question into sub-questions) instead of a paraphrase.
- Use the reranker's top score as a cheaper relevance signal than an LLM call.
- Evaluate with an LLM judge or human labels instead of keyword overlap.
- Test on multi-hop questions, where iterative retrieval should help more than on
  single-chunk questions.

## Numbers to remember
- 88 QA pairs; loop fired on 31 (35%): 5 looped once, 26 looped twice.
- Single-pass: 0.848 recall / 0.36 F1.
- Replace with the last retrieval: 0.817 / 0.33 (worse).
- Merge all retrievals and rerank: 0.858 / 0.37 (best).
- Looped queries: recall 0.786 → 0.813, 5 improved, 0 worse.
