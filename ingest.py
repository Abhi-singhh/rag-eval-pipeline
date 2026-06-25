"""
ingest.py
---------
Week 1: Fetch arXiv abstracts -> clean -> chunk -> embed -> store in ChromaDB.

Run:
    python ingest.py
"""

import re
import json
import time
import hashlib
from pathlib import Path

import arxiv
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm


# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_PATH   = "./data/chroma_db"
COLLECTION    = "arxiv_ml"
RAW_DUMP_PATH = "./data/raw_abstracts.json"

# Highly targeted queries — each one should return almost exclusively ML papers.
# Using ti: (title) and abs: (abstract) field filters for precision.
SEARCH_QUERIES = [
    'ti:"retrieval augmented generation"',
    'ti:"large language model" AND abs:"evaluation"',
    'ti:"RAG" AND abs:"retrieval"',
    'abs:"retrieval augmented generation" AND abs:"hallucination"',
    'abs:"RAG" AND abs:"faithfulness" AND abs:"context"',
    'ti:"language model" AND abs:"factual accuracy"',
    'abs:"LLM evaluation" AND abs:"benchmark"',
    'abs:"retrieval" AND abs:"generation" AND abs:"pipeline"',
    'ti:"AI safety" OR ti:"responsible AI" AND abs:"evaluation"',
    'abs:"model evaluation" AND abs:"language model" AND abs:"drift"',
]
RESULTS_PER_QUERY = 50   # 10 x 50 = 500 attempts, expect ~300 after dedup

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
EMBED_MODEL   = "all-MiniLM-L6-v2"

# Hard blacklist on title only — catches stray physics/bio papers
TITLE_BLACKLIST = [
    "quarkonium", "gluon", "quark", "qcd", "woven", "textile",
    "protein", "molecular", "seismic", "photonic", "nanoparticle",
    "polymer", "combustion", "turbulence", "crystalline", "haptic",
    "emergent mechanics", "fluid", "thermodynamic", "genomic",
    "astronomical", "geophysical",
]

# Must contain at least one of these in title+abstract to be kept
ML_REQUIRED = [
    "language model", "llm", "retrieval", "transformer", "neural network",
    "machine learning", "deep learning", "embedding", "fine-tun",
    "hallucination", "prompt", "generation", "nlp", "bert", "gpt",
    "rag", "question answering", "benchmark", "evaluation metric",
    "in-context", "few-shot", "zero-shot", "chain-of-thought",
]


# ── Filter ────────────────────────────────────────────────────────────────────

def is_ml_paper(title: str, abstract: str) -> bool:
    title_lower = title.lower()
    combined    = (title + " " + abstract).lower()

    # Hard blacklist on title
    for term in TITLE_BLACKLIST:
        if term in title_lower:
            return False

    # Must have at least one ML signal
    return any(kw in combined for kw in ML_REQUIRED)


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_abstracts(queries: list, max_per_query: int) -> list:
    client  = arxiv.Client()
    seen_ids = set()
    papers   = []

    for query in queries:
        print(f"\n-> Fetching: '{query[:60]}...' " if len(query) > 60 else f"\n-> Fetching: '{query}'")
        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        try:
            results = list(client.results(search))
        except Exception as e:
            print(f"  Warning: query failed ({e}), skipping.")
            continue

        added = 0
        for paper in results:
            paper_id = paper.entry_id.split("/")[-1]
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)

            if not is_ml_paper(paper.title, paper.summary):
                continue

            papers.append({
                "id":       paper_id,
                "title":    paper.title.strip(),
                "abstract": paper.summary.strip(),
                "authors":  [a.name for a in paper.authors[:3]],
                "year":     paper.published.year,
                "url":      paper.entry_id,
            })
            added += 1

        print(f"  +{added} papers (total: {len(papers)})")
        time.sleep(2)

    print(f"\nTotal unique ML papers: {len(papers)}")
    return papers


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\$[^$]*\$", "[MATH]", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


# ── Chunk ─────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]


# ── Embed + Store ─────────────────────────────────────────────────────────────

def build_chroma_collection(papers: list) -> chromadb.Collection:
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    try:
        db.delete_collection(COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}' for fresh ingest.")
    except Exception:
        pass

    collection = db.create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    all_docs, all_ids, all_meta = [], [], []

    for paper in papers:
        full_text = f"{paper['title']}. {clean_text(paper['abstract'])}"
        chunks    = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{paper['id']}_{i}".encode()).hexdigest()
            all_docs.append(chunk)
            all_ids.append(chunk_id)
            all_meta.append({
                "paper_id":    paper["id"],
                "title":       paper["title"],
                "authors":     ", ".join(paper["authors"]),
                "year":        str(paper["year"]),
                "url":         paper["url"],
                "chunk_index": i,
            })

    batch_size = 100
    print(f"\n-> Embedding and storing {len(all_docs)} chunks...")
    for i in tqdm(range(0, len(all_docs), batch_size), desc="Upserting batches"):
        collection.upsert(
            documents=all_docs[i:i+batch_size],
            ids=all_ids[i:i+batch_size],
            metadatas=all_meta[i:i+batch_size],
        )

    return collection


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(collection: chromadb.Collection):
    test_queries = [
        "challenges in building reliable RAG systems",
        "how do language models handle factual accuracy",
        "retrieval augmented generation evaluation methods",
        "hallucination in large language models",
    ]

    print("\n-> Sanity-check retrieval (want distances < 0.50)...")
    ok_count = 0
    total    = 0

    for q in test_queries:
        results = collection.query(query_texts=[q], n_results=3)
        print(f"\n  Q: '{q}'")
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            flag = "OK  " if dist < 0.50 else "WARN"
            print(f"    [{flag} {round(dist, 3)}] {meta['title'][:65]}")
            if dist < 0.50:
                ok_count += 1
            total += 1

    pct = round(ok_count / total * 100)
    print(f"\n  {ok_count}/{total} results under 0.50 distance ({pct}%)")

    if pct >= 60:
        print("  Collection looks good — proceed to pipeline.py")
    else:
        print("  Collection still noisy. Check raw_abstracts.json for bad papers.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RAG Eval Pipeline -- Ingestion")
    print("=" * 60)

    papers = fetch_abstracts(SEARCH_QUERIES, RESULTS_PER_QUERY)

    if len(papers) < 50:
        print(f"\nWarning: only {len(papers)} papers fetched.")
        print("arXiv may be rate-limiting. Wait 60s and re-run.")
        return

    Path("./data").mkdir(exist_ok=True)
    with open(RAW_DUMP_PATH, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Saved {len(papers)} papers to {RAW_DUMP_PATH}")

    collection = build_chroma_collection(papers)
    print(f"\nTotal chunks stored: {collection.count()}")

    sanity_check(collection)

    print("\n" + "=" * 60)
    print("Done. Now run: python pipeline.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
