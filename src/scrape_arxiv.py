"""
Step 1: Scrape arXiv papers for the RAG project.

Downloads PDFs + metadata for a target subfield from arXiv.
Run: python 01_scrape_arxiv.py
"""

import arxiv
import json
import time
import os
import requests
from pathlib import Path


# CONFIG
QUERY = "retrieval augmented generation"   # change this to your chosen subfield
MAX_PAPERS = 1000                          # target number of papers
CATEGORY_FILTER = "cs.CL"                  # cs.CL = NLP, cs.LG = ML, cs.AI = AI
DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
METADATA_PATH = DATA_DIR / "metadata.json"


PDF_DIR.mkdir(parents=True, exist_ok=True)


def search_arxiv(query: str, max_results: int, category: str):
    """Search arXiv and yield paper results matching the query + category."""
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)

    full_query = f'all:"{query}" AND cat:{category}'
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    return client.results(search)


def download_papers():
    metadata = []
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        already_have = {m["arxiv_id"] for m in metadata}
        print(f"Resuming. Already have {len(already_have)} papers.")
    else:
        already_have = set()

    results = search_arxiv(QUERY, MAX_PAPERS, CATEGORY_FILTER)

    count = len(already_have)
    for paper in results:
        if count >= MAX_PAPERS:
            break

        arxiv_id = paper.get_short_id()
        if arxiv_id in already_have:
            continue

        pdf_filename = f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_path = PDF_DIR / pdf_filename

        try:
            if not pdf_path.exists():
                if paper.pdf_url is None:
                    print(f"No PDF URL for {arxiv_id}, skipping.")
                    continue
                resp = requests.get(paper.pdf_url, timeout=30)
                resp.raise_for_status()
                with open(pdf_path, "wb") as f:
                    f.write(resp.content)
                time.sleep(1)  # be polite to arXiv's servers

            entry = {
                "arxiv_id": arxiv_id,
                "title": paper.title.strip().replace("\n", " "),
                "abstract": paper.summary.strip().replace("\n", " "),
                "authors": [a.name for a in paper.authors],
                "published": str(paper.published.date()),
                "categories": paper.categories,
                "pdf_path": str(pdf_path),
                "url": paper.entry_id,
            }
            metadata.append(entry)
            already_have.add(arxiv_id)
            count += 1

            if count % 25 == 0:
                print(f"Downloaded {count} papers...")
                # checkpoint save every 25 papers
                with open(METADATA_PATH, "w") as f:
                    json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"Failed on {arxiv_id}: {e}")
            continue

    # final save
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Total papers downloaded: {len(metadata)}")
    print(f"PDFs saved to: {PDF_DIR}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    download_papers()