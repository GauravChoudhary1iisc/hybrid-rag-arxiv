# Hybrid RAG project: interview answer

## Short version (about 30 seconds)

I built an end-to-end RAG system over about 1,000 recent arXiv papers on retrieval-augmented
generation, running fully locally. It covers PDF parsing and chunking, a hybrid retriever that
combines dense embeddings in Qdrant with BM25 and then reranks with a cross-encoder, and cited
answers from Qwen2.5-7B through Ollama. I built my own evaluation set and measured each
retrieval stage. Moving from dense-only to hybrid to hybrid with reranking raised context recall
from 0.75 to 0.82 to 0.85, and answer F1 from 0.30 to 0.36. I also put a Streamlit UI on top.

## Full version (about 2 minutes)

**Data pipeline.**
- I scraped about 1,000 cs.CL papers matching "retrieval augmented generation" from the arXiv API.
- I parsed them with PyMuPDF, cleaning out headers, footers, arXiv stamps and reference sections,
  since those add noise to retrieval.
- I implemented two chunking strategies: fixed-size (512 tokens with 50-token overlap,
  about 28,700 chunks) and semantic, split on paragraph and section boundaries (about 19,600 chunks).

**Indexing.**
- I embedded the chunks with BAAI/bge-base-en-v1.5 (768 dimensions) on GPU and stored them in
  Qdrant running in Docker.
- In parallel I built a BM25 index over the same chunks. Dense embeddings are good at meaning
  but weak on exact terms like "RAPTOR", "HyDE" or "ColBERT", which is exactly what BM25 is good at.

**Retrieval.** I implemented three strategies so I could measure each stage:
1. **Dense only**: vector search in Qdrant.
2. **Hybrid**: dense and BM25 results fused with Reciprocal Rank Fusion (k=60), which combines
   rankings without having to calibrate two different score scales.
3. **Hybrid + rerank**: take the top 50 hybrid candidates, re-score each against the question with
   a cross-encoder (ms-marco-MiniLM-L-6-v2), and keep the top 5.

**Generation.** Qwen2.5-7B runs locally via Ollama with a strict prompt: answer only from the
numbered passages, cite them inline as [1], [2], and say so when the context isn't enough.

**Evaluation.**
- There was no labelled dataset, so I generated QA pairs from sampled chunks with the LLM and
  filtered out low-value questions like author names and citations. That left 88 pairs, each
  linked to its source chunk.
- I used three deterministic metrics, with no LLM-as-judge, so they're cheap and reproducible:
  - **Context recall**: the share of the ground-truth answer's key terms that appear in the retrieved chunks.
  - **Faithfulness**: how much of the answer is grounded in the retrieved context.
  - **Token-level F1** against the ground truth.

**Results** (30-query sample):

| Strategy | Context recall | Faithfulness | Answer F1 |
|---|---|---|---|
| Dense | 0.754 | 0.785 | 0.303 |
| Hybrid (RRF) | 0.822 | 0.809 | 0.328 |
| Hybrid + rerank | **0.851** | **0.813** | **0.365** |

Each stage added measurable value. Better retrieval also improved the answers, not just the
recall numbers. Finally, I wrapped it in a Streamlit app.

## Likely follow-up questions

**Why hybrid search?**
Dense and sparse retrieval fail differently. Embeddings handle paraphrases; BM25 handles exact
method names and acronyms, which are everywhere in research papers. Adding BM25 gave the
biggest single jump in recall: +0.07.

**Why RRF instead of adding the scores together?**
BM25 scores and cosine similarities are on different, unbounded scales. RRF only uses rank
positions, so it needs no normalization or weight tuning and holds up well.

**Why rerank, and what does it cost?**
A bi-encoder embeds the query and the chunk separately. A cross-encoder reads them together,
so it's more accurate but too slow to run over the whole corpus. Retrieving 50 candidates
cheaply and reranking only those gets most of the accuracy at a fraction of the cost. It adds
latency on every query, which is the trade-off.

**Why fixed-size chunks if you also built semantic chunking?**
The final evaluation and the BM25 index use the fixed-size chunks. Semantic chunking is
indexed in Qdrant as a second collection, but I haven't benchmarked it head-to-head.
That's a clear next step.

**What are the limitations of your evaluation?**
- The questions were generated from single chunks by the same model family that answers them,
  so they're easier than real user questions and don't test multi-hop reasoning.
- The metrics are keyword-overlap based, so they miss paraphrased correct answers.
- The strategy comparison used 30 of the 88 pairs. The later agentic project re-ran single-pass
  retrieval on all 88 and got consistent numbers: 0.848 recall, 0.36 F1.

**Why run everything locally?**
No API cost, the data stays private, and it fits on an 8 GB RTX 3050: a 7B model through Ollama
plus small embedding and reranker models. The trade-off is answer quality compared with
larger hosted models.

**What would you improve?**
- Benchmark semantic vs. fixed chunking.
- Evaluate with an LLM judge or human labels instead of keyword overlap.
- Add multi-hop questions to the evaluation set.
- Add metadata filters such as paper or date.
- Check embedding prefixes: BGE's retrieval instruction is meant for queries only, and my indexing
  script also added it to passages. Re-embedding without it could improve dense recall.

**What came next?** I extended this into an agentic version with LangGraph. See
`INTERVIEW_LANGGRAPH.md`.

## Numbers to remember
- About 1,000 arXiv papers; 28,711 fixed-size chunks (512 tokens, 50 overlap); 768-dim BGE embeddings.
- Retrieve top 50 hybrid candidates, rerank, keep top 5; RRF k=60.
- Context recall: 0.75 dense → 0.82 hybrid → 0.85 rerank.
- Answer F1: 0.30 → 0.33 → 0.36.
- 88 generated QA pairs; the strategy comparison used 30.
