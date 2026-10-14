"""
Week 3, Step 1: Generate a QA test set from paper chunks using Ollama.

Samples diverse chunks from chunks_fixed.json, sends each to Ollama
asking it to generate a question + ground-truth answer, then saves
the result to data/eval_qa_set.json.

Run: python 09_build_qa_set.py
"""

import json
import random
import time
import requests
from pathlib import Path

# ---------- CONFIG ----------
DATA_DIR        = Path("data")
CHUNKS_PATH     = DATA_DIR / "chunks_fixed.json"
QA_OUTPUT       = DATA_DIR / "eval_qa_set.json"

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "qwen2.5:7b"

TARGET_QA_PAIRS = 100
MIN_CHUNK_CHARS = 400
MAX_CHUNK_CHARS = 1200
SEED            = 42
# ----------------------------

random.seed(SEED)

QA_PROMPT = """You are creating evaluation questions for a RAG system about retrieval-augmented generation.

Given the passage below, generate ONE question that tests understanding of a RAG concept, method, result, or system design decision.

STRICT RULES — your question MUST NOT:
- Ask for paper titles, author names, or publication venues
- Ask about citations, references, or bibliography entries
- Ask about URLs, DOIs, or page numbers
- Ask "what is the title of..." or "who wrote..." or "which conference..."
- Reference specific years, volume numbers, or journal names

Your question MUST:
- Ask about a technical concept, method, finding, or design decision
- Be answerable from the passage content alone
- Start with: How, What, Why, When, Which (but not "Which paper/author/conference")

Passage:
\"\"\"
{chunk_text}
\"\"\"

If this passage is ONLY citations/references with no technical content, respond with exactly:
{{"skip": true}}

Otherwise respond in this EXACT JSON format:
{{
  "question": "your technical question here",
  "ground_truth": "the answer based only on the passage"
}}"""


def call_ollama(prompt: str, timeout: int = 60) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 300},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"    Ollama error: {e}")
        return None


def parse_qa_response(response: str) -> dict | None:
    if not response:
        return None
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        parsed = json.loads(text)

        # model decided to skip this chunk
        if parsed.get("skip"):
            return None

        if "question" in parsed and "ground_truth" in parsed:
            q = parsed["question"].strip()
            a = parsed["ground_truth"].strip()

            # length checks
            if len(q) < 20 or len(a) < 30:
                return None

            # reject reference-style questions
            q_lower = q.lower()
            bad_patterns = [
                "what is the title",
                "who wrote",
                "which paper",
                "which conference",
                "which journal",
                "which author",
                "what is the name of the paper",
                "what year was",
                "what is the doi",
                "what is the url",
                "published in",
                "et al.",
                "passage",
                "reference",
            ]
            if any(p in q_lower for p in bad_patterns):
                return None

            return {"question": q, "ground_truth": a}

    except json.JSONDecodeError:
        return None
    return None


def sample_diverse_chunks(chunks: list[dict], n: int) -> list[dict]:
    # keywords that indicate a chunk is from a references/bibliography section
    REF_SIGNALS = [
        "et al.", "doi:", "arxiv:", "proceedings of", "conference on",
        "journal of", "volume", "pp.", "isbn", "http://", "https://",
        "in press", "preprint", "workshop on",
    ]

    def is_reference_chunk(text: str) -> bool:
        text_lower = text.lower()
        # count how many reference signals appear
        signal_count = sum(1 for s in REF_SIGNALS if s in text_lower)
        # if more than 3 signals per 500 chars, it's likely a references section
        density = signal_count / (len(text) / 500)
        return density > 2.5

    eligible = [
        c for c in chunks
        if MIN_CHUNK_CHARS <= c["char_count"] <= MAX_CHUNK_CHARS
        and not is_reference_chunk(c["text"])
    ]
    print(f"  Eligible chunks (length filter): {len(eligible):,} of {len(chunks):,}")

    by_paper: dict[str, list] = {}
    for c in eligible:
        by_paper.setdefault(c["arxiv_id"], []).append(c)

    sampled = []
    for paper_chunks in by_paper.values():
        k = min(2, len(paper_chunks))
        sampled.extend(random.sample(paper_chunks, k))

    random.shuffle(sampled)
    return sampled[:n * 3]


def main():
    print(f"Loading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"  {len(chunks):,} chunks loaded")

    existing_qa = []
    if QA_OUTPUT.exists():
        with open(QA_OUTPUT) as f:
            existing_qa = json.load(f)
        print(f"  Resuming: {len(existing_qa)} pairs already generated")

    already_questions = {qa["question"] for qa in existing_qa}
    qa_set = existing_qa.copy()

    if len(qa_set) >= TARGET_QA_PAIRS:
        print(f"\nAlready have {len(qa_set)} pairs. Nothing to do.")
        return

    print(f"\nSampling diverse chunks...")
    candidates = sample_diverse_chunks(chunks, TARGET_QA_PAIRS)
    print(f"  {len(candidates)} candidate chunks selected\n")

    success = len(qa_set)
    failed  = 0

    for chunk in candidates:
        if success >= TARGET_QA_PAIRS:
            break

        prompt = QA_PROMPT.format(chunk_text=chunk["text"][:MAX_CHUNK_CHARS])

        print(f"[{success+1}/{TARGET_QA_PAIRS}] {chunk['title'][:55]}...")
        t0 = time.time()
        response = call_ollama(prompt)
        elapsed = time.time() - t0

        parsed = parse_qa_response(response)
        if parsed is None:
            failed += 1
            print(f"    Failed to parse (attempt {failed})")
            continue

        if parsed["question"] in already_questions:
            print(f"    Duplicate, skipping")
            continue

        record = {
            "id":                success,
            "question":          parsed["question"],
            "ground_truth":      parsed["ground_truth"],
            "reference_context": chunk["text"][:MAX_CHUNK_CHARS],
            "arxiv_id":          chunk["arxiv_id"],
            "title":             chunk["title"],
            "chunk_id":          chunk["chunk_id"],
        }
        qa_set.append(record)
        already_questions.add(parsed["question"])
        success += 1

        print(f"    Q: {parsed['question'][:80]}")
        print(f"    {elapsed:.1f}s")

        if success % 10 == 0:
            with open(QA_OUTPUT, "w") as f:
                json.dump(qa_set, f, indent=2)
            print(f"\n  Checkpoint saved ({success} pairs)\n")

    with open(QA_OUTPUT, "w") as f:
        json.dump(qa_set, f, indent=2)

    print(f"\n── Summary ──────────────────────────────────")
    print(f"  QA pairs generated : {len(qa_set)}")
    print(f"  Parse failures     : {failed}")
    print(f"  Saved to           : {QA_OUTPUT}")
    print(f"────────────────────────────────────────────")

    print("\nSample QA pairs:")
    for qa in random.sample(qa_set, min(3, len(qa_set))):
        print(f"\n  Q: {qa['question']}")
        print(f"  A: {qa['ground_truth'][:120]}...")
        print(f"  From: {qa['title'][:60]}")


if __name__ == "__main__":
    main()