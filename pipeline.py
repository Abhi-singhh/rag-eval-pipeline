"""
pipeline.py
-----------
Week 2: Query → Retrieve → Generate.

Given a user question:
  1. Embeds the question with the same model used at ingest time
  2. Retrieves the top-k most relevant chunks from ChromaDB
  3. Injects them into a prompt template
  4. Sends the prompt to Ollama (Llama 3) and streams the answer
  5. Returns a structured result dict ready for the evaluator (Week 3)

Run:
    python pipeline.py
    python pipeline.py --query "What are the main challenges in RAG systems?"
    python pipeline.py --query "your question" --top_k 5
"""

import argparse
import json
import time
from pathlib import Path

import chromadb
import requests
from chromadb.utils import embedding_functions


# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_PATH  = "./data/chroma_db"
COLLECTION   = "arxiv_ml"
EMBED_MODEL  = "all-MiniLM-L6-v2"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

DEFAULT_TOP_K = 3   # number of chunks to retrieve

# Prompt template — instructs the LLM to stay grounded in context only.
# This is the key responsible-AI design decision: we explicitly tell the model
# NOT to use outside knowledge, so faithfulness scores are meaningful.
PROMPT_TEMPLATE = """You are a research assistant answering questions about AI and machine learning papers.

Use ONLY the context passages below to answer the question. 
If the context does not contain enough information to answer, say "I don't have enough context to answer this."
Do not use any knowledge outside of the provided context.

Context:
{context}

Question: {question}

Answer:"""


# ── Retrieval ─────────────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    """Load the existing ChromaDB collection built by ingest.py."""
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return db.get_collection(name=COLLECTION, embedding_function=embed_fn)


def retrieve(collection: chromadb.Collection, query: str, top_k: int) -> list[dict]:
    """
    Embed the query and find the top-k most similar chunks.

    Returns a list of dicts, each with:
        text      — the chunk text
        metadata  — paper title, authors, year, url
        distance  — cosine distance (lower = more similar)
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "metadata": meta,
            "distance": round(dist, 4),
        })

    # Relevance guard: warn if all chunks are poor matches
    best_distance = min(c["distance"] for c in chunks) if chunks else 1.0
    if best_distance > 0.55:
        print(f"\n  Warning: best retrieval distance is {best_distance:.3f} (>0.55).")
        print("  The collection may not contain relevant documents for this query.")

    return chunks


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Inject retrieved chunks into the prompt template.

    Each chunk is labelled with its source paper so the LLM can reference it,
    and so we can trace which sources contributed to the answer.
    """
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        source_line = f"[Source {i+1}] {meta['title']} ({meta['year']})"
        context_parts.append(f"{source_line}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)


# ── Generation ────────────────────────────────────────────────────────────────

def generate(prompt: str, stream: bool = True) -> tuple[str, float]:
    """
    Send the prompt to Ollama and return (answer_text, latency_ms).

    Uses streaming so you see the answer appear token by token in the terminal.
    Falls back to non-streaming for programmatic use (e.g. batch evaluation).
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
    }

    start = time.time()

    if stream:
        print("\nAnswer (streaming):\n" + "-" * 40)
        full_answer = ""
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk_data = json.loads(line)
                    token = chunk_data.get("response", "")
                    print(token, end="", flush=True)
                    full_answer += token
                    if chunk_data.get("done"):
                        break
        print("\n" + "-" * 40)
    else:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        full_answer = resp.json().get("response", "")

    latency_ms = round((time.time() - start) * 1000)
    return full_answer.strip(), latency_ms


# ── Main pipeline function ────────────────────────────────────────────────────

def run_query(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    stream: bool = True,
    collection: chromadb.Collection = None,
) -> dict:
    """
    Full RAG pipeline: query → retrieve → generate.

    Returns a structured result dict that evaluate.py (Week 3) will consume:
    {
        query:           the user's question
        retrieved_chunks: list of {text, metadata, distance}
        contexts:        just the text strings (RAGAS expects this format)
        prompt:          the full prompt sent to the LLM
        answer:          the LLM's response
        latency_ms:      end-to-end time in milliseconds
    }
    """
    if collection is None:
        collection = get_collection()

    print(f"\nQuery: {query}")
    print(f"Retrieving top {top_k} chunks from ChromaDB...")

    # Step 1: Retrieve
    t_retrieve_start = time.time()
    chunks = retrieve(collection, query, top_k)
    retrieve_ms = round((time.time() - t_retrieve_start) * 1000)

    print(f"\nRetrieved {len(chunks)} chunks ({retrieve_ms}ms):")
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        print(f"  [{i+1}] {meta['title']} ({meta['year']}) — distance: {c['distance']}")

    # Step 2: Build prompt
    prompt = build_prompt(query, chunks)

    # Step 3: Generate
    answer, latency_ms = generate(prompt, stream=stream)

    result = {
        "query":            query,
        "retrieved_chunks": chunks,
        "contexts":         [c["text"] for c in chunks],   # RAGAS format
        "prompt":           prompt,
        "answer":           answer,
        "retrieve_ms":      retrieve_ms,
        "latency_ms":       latency_ms,
    }

    print(f"\nLatency: {latency_ms}ms total ({retrieve_ms}ms retrieval)")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG query pipeline")
    parser.add_argument(
        "--query",
        type=str,
        default="What are the main challenges in building reliable RAG systems?",
        help="Question to ask the pipeline",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve (default: 3)",
    )
    args = parser.parse_args()

    # Check Ollama is running
    try:
        resp = requests.get("http://localhost:11434", timeout=3)
    except requests.exceptions.ConnectionError:
        print("\n❌ Ollama is not running.")
        print("   Start it with:  ollama serve")
        print("   Then re-run this script.")
        return

    collection = get_collection()
    result = run_query(args.query, top_k=args.top_k, collection=collection)

    # Show sources used
    print("\nSources used:")
    for i, chunk in enumerate(result["retrieved_chunks"]):
        meta = chunk["metadata"]
        print(f"  [{i+1}] {meta['title']}")
        print(f"       {meta['url']}")


if __name__ == "__main__":
    main()