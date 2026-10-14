"""
Week 4: Streamlit UI for the Hybrid RAG system.

Wires together retrievers.py and generator.py into a web interface.

Run:
    cd ~/Desktop/rag_project/src
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG — arXiv ML Papers",
    page_icon="🔍",
    layout="wide",
)

# ── Load models once (cached across reruns) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval models...")
def load_retriever():
    from retreiver import Retriever
    return Retriever(device="cuda")


@st.cache_resource(show_spinner="Loading generator...")
def load_generator():
    from generator import Generator
    return Generator()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    strategy = st.selectbox(
        "Retrieval strategy",
        options=["hybrid_rerank", "hybrid", "dense"],
        format_func=lambda x: {
            "hybrid_rerank": "Hybrid + Rerank (best)",
            "hybrid":        "Hybrid (BM25 + dense)",
            "dense":         "Dense only (fastest)",
        }[x],
    )

    top_k = st.slider("Top-K chunks", min_value=3, max_value=10, value=5)

    st.divider()
    st.markdown("**📊 Eval results** (30 samples)")
    st.markdown("""
| Strategy | Ctx Recall | Faithful | F1 |
|---|---|---|---|
| Dense | 0.754 | 0.785 | 0.303 |
| Hybrid | 0.822 | 0.809 | 0.328 |
| **Hybrid+RR** | **0.851** | **0.813** | **0.365** |
""")

    st.divider()
    st.markdown("**🗂️ Corpus**")
    st.markdown("999 arXiv papers on RAG  \n28,711 chunks indexed  \nBAAI/bge-base-en-v1.5")

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🔍 Hybrid RAG over arXiv ML Papers")
st.caption("Ask anything about retrieval-augmented generation. Sources are cited inline.")

query = st.text_area(
    "Your question",
    placeholder="e.g. How does hybrid retrieval improve over dense-only search?",
    height=80,
)

ask_btn = st.button("Ask", type="primary", use_container_width=False)

if ask_btn and query.strip():
    retriever = load_retriever()
    generator = load_generator()

    # ── Retrieve ──────────────────────────────────────────────────────────────
    with st.spinner("Retrieving relevant chunks..."):
        t0 = time.time()
        if strategy == "dense":
            chunks = retriever.dense_retrieve(query, top_k=top_k)
        elif strategy == "hybrid":
            chunks = retriever.hybrid_retrieve(query, top_k=top_k)
        else:
            chunks = retriever.hybrid_rerank(query, top_k=top_k)
        retrieval_time = time.time() - t0

    # ── Generate ──────────────────────────────────────────────────────────────
    with st.spinner("Generating answer..."):
        t0 = time.time()
        result = generator.generate(query, chunks)
        generation_time = time.time() - t0

    # ── Display ───────────────────────────────────────────────────────────────
    strategy_label = {
        "dense":         "Dense only",
        "hybrid":        "Hybrid (BM25 + dense, RRF)",
        "hybrid_rerank": "Hybrid + cross-encoder rerank",
    }[strategy]

    st.success(
        f"**{strategy_label}** · "
        f"retrieval {retrieval_time:.2f}s · "
        f"generation {generation_time:.1f}s"
    )

    st.markdown("### Answer")
    st.markdown(result["answer"])

    st.markdown("### Retrieved sources")
    for i, chunk in enumerate(chunks[:top_k]):
        score_key = "rerank_score" if "rerank_score" in chunk else "retrieval_score"
        score = chunk.get(score_key, 0.0)
        with st.expander(
            f"[{i+1}] {chunk['title'][:80]}  —  score: {score:.4f}"
        ):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**arXiv ID:** `{chunk['arxiv_id']}`")
                st.markdown(f"**Chunk:** {chunk['chunk_index']+1} / {chunk['total_chunks']}")
                st.markdown(f"**Chars:** {chunk['char_count']}")
            with col2:
                st.markdown("**Chunk text:**")
                st.text(chunk["text"][:600] + ("..." if len(chunk["text"]) > 600 else ""))

elif ask_btn and not query.strip():
    st.warning("Please enter a question first.")

else:
    # default state — show example questions
    st.markdown("#### Example questions")
    examples = [
        "How does RAG handle multi-hop reasoning?",
        "What are the main failure modes of naive RAG systems?",
        "How does hybrid retrieval improve over dense-only search?",
        "What is Reciprocal Rank Fusion and how does it work?",
        "How do cross-encoders differ from bi-encoders for reranking?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["_example"] = ex
            st.rerun()

# handle example button clicks
if "_example" in st.session_state:
    ex = st.session_state.pop("_example")
    st.query_params["q"] = ex
    st.rerun()