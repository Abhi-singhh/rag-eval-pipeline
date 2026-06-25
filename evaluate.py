"""
evaluate.py
-----------
Week 3: Evaluation harness — score every RAG answer.

We implement faithfulness, answer relevancy, and context recall
directly using Ollama calls — no RAGAS async dependency issues.

The scoring logic mirrors what RAGAS does internally:
  - Faithfulness:      LLM checks each claim in the answer against context
  - Answer relevancy:  LLM checks if the answer actually addresses the question
  - Context recall:    LLM checks if context contains info from ground truth

Run:
    python evaluate.py              # single query demo
    python evaluate.py --batch      # all 20 test questions
    python evaluate.py --batch --top_k 5   # drift experiment
"""

import os
import re
import json
import argparse
import datetime
from pathlib import Path

import requests

from pipeline import run_query, get_collection


# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
LOG_PATH     = "./logs/eval_log.jsonl"


# ── Ollama helper ─────────────────────────────────────────────────────────────

def ollama(prompt: str, max_retries: int = 2) -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"  Ollama retry {attempt+1}: {e}")


def extract_score(text: str) -> float:
    """
    Parse a score from LLM output.
    Looks for patterns like: 0.8, 0.75, 1.0, yes, no
    Returns float in [0, 1].
    """
    text = text.lower().strip()

    # Direct float match
    match = re.search(r"\b(0\.\d+|1\.0+|0|1)\b", text)
    if match:
        return min(1.0, max(0.0, float(match.group(1))))

    # Yes/no
    if text.startswith("yes"):
        return 1.0
    if text.startswith("no"):
        return 0.0

    return 0.5   # uncertain


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: are the claims in the answer supported by the context?

    Method: extract key claims from the answer, then for each claim ask
    the LLM whether the context supports it. Score = supported / total.
    """
    context_str = "\n\n".join(contexts)

    # Step 1: extract claims
    claim_prompt = f"""Extract the key factual claims from this answer as a numbered list.
Each claim should be one sentence. List only claims, nothing else.

Answer: {answer}

Numbered list of claims:"""

    claims_raw = ollama(claim_prompt)

    # Parse numbered list
    claims = []
    for line in claims_raw.split("\n"):
        line = line.strip()
        match = re.match(r"^\d+[\.\)]\s*(.+)", line)
        if match:
            claims.append(match.group(1).strip())

    if not claims:
        return 0.5   # couldn't parse claims

    # Step 2: verify each claim against context
    supported = 0
    for claim in claims:
        verify_prompt = f"""Context:
{context_str}

Claim: {claim}

Is this claim directly supported by the context above?
Answer with only a number between 0 and 1, where 1 = fully supported, 0 = not supported."""

        result = ollama(verify_prompt)
        supported += extract_score(result)

    return round(supported / len(claims), 3)


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer relevancy: does the answer actually address the question?

    Method: ask the LLM to rate how well the answer addresses the question.
    """
    prompt = f"""Question: {question}

Answer: {answer}

How well does this answer address the question?
Score from 0 to 1 where:
  1.0 = directly and completely answers the question
  0.5 = partially answers the question
  0.0 = does not answer the question at all

Respond with only a number between 0 and 1."""

    result = ollama(prompt)
    return round(extract_score(result), 3)


def score_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """
    Context recall: does the retrieved context contain the information
    needed to produce the ground truth answer?

    Method: break ground truth into key points, check each against context.
    """
    context_str = "\n\n".join(contexts)

    prompt = f"""Ground truth answer: {ground_truth}

Retrieved context:
{context_str}

How much of the ground truth answer can be inferred from the retrieved context?
Score from 0 to 1 where:
  1.0 = all key information in the ground truth is present in the context
  0.5 = some key information is present
  0.0 = the context contains none of the relevant information

Respond with only a number between 0 and 1."""

    result = ollama(prompt)
    return round(extract_score(result), 3)


# ── Logging ───────────────────────────────────────────────────────────────────

def log_result(record: dict):
    Path("./logs").mkdir(exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Test set ──────────────────────────────────────────────────────────────────

TEST_SET = [
    {
        "question": "What are the main challenges in building reliable RAG systems?",
        "ground_truth": "Key challenges include chunking strategy selection, knowledge conflicts between retrieved context and model parameters, ensuring factual accuracy of generated answers, and defending against retrieval manipulation attacks."
    },
    {
        "question": "How is faithfulness measured in retrieval augmented generation?",
        "ground_truth": "Faithfulness measures whether each claim in the generated answer is supported by the retrieved context, computed by checking if statements can be inferred from the provided passages."
    },
    {
        "question": "What methods reduce hallucination in large language models?",
        "ground_truth": "Methods include retrieval augmented generation, factual confidence prediction, self-consistency checking, chain-of-thought prompting, and fine-tuning on factual datasets."
    },
    {
        "question": "How does chunking strategy affect RAG performance?",
        "ground_truth": "Chunking strategy affects retrieval effectiveness and computational cost. Smaller chunks improve precision but may lose context, while larger chunks retain context but reduce retrieval specificity."
    },
    {
        "question": "What evaluation benchmarks exist for RAG systems?",
        "ground_truth": "Benchmarks for RAG evaluation include frameworks measuring retrieval quality, answer faithfulness, and answer relevance across diverse question types and domains."
    },
    {
        "question": "How do knowledge conflicts affect LLM-based RAG systems?",
        "ground_truth": "Knowledge conflicts arise when retrieved context contradicts the model's parametric knowledge, causing the model to ignore retrieved evidence or produce inconsistent answers."
    },
    {
        "question": "What is context recall in RAG evaluation?",
        "ground_truth": "Context recall measures whether retrieved chunks contain the information needed to answer the question, comparing retrieved context against a ground truth reference answer."
    },
    {
        "question": "How do retrieval attacks threaten RAG system reliability?",
        "ground_truth": "Retrieval attacks inject adversarial documents into the knowledge base that manipulate LLM outputs, exploiting the RAG system's tendency to rely on retrieved context."
    },
    {
        "question": "What role does factual confidence prediction play in RAG?",
        "ground_truth": "Factual confidence prediction estimates the reliability of generated statements, enabling RAG systems to flag uncertain answers and provide certified responses with quantified confidence."
    },
    {
        "question": "How is answer relevancy different from faithfulness in RAG metrics?",
        "ground_truth": "Faithfulness checks if the answer is grounded in context, while answer relevancy checks if the answer addresses what the user asked, regardless of factual support."
    },
    {
        "question": "What are the computational costs of different chunking methods?",
        "ground_truth": "Fixed-size chunking is cheapest, semantic chunking is more expensive but improves retrieval quality, and hierarchical chunking offers a balance between cost and effectiveness."
    },
    {
        "question": "How does RAG compare to fine-tuning for knowledge injection?",
        "ground_truth": "RAG is more flexible than fine-tuning since knowledge can be updated by changing the vector store, while fine-tuning requires retraining and risks catastrophic forgetting."
    },
    {
        "question": "What metrics evaluate LLM factual accuracy?",
        "ground_truth": "Metrics include exact match, F1 score, ROUGE, BERTScore, and model-based evaluators that assess whether generated text aligns with verified factual sources."
    },
    {
        "question": "How does context window size affect RAG generation quality?",
        "ground_truth": "Larger context windows allow more retrieved chunks, improving completeness, but may introduce noise and reduce focus on the most relevant passages."
    },
    {
        "question": "What is the role of embedding models in RAG retrieval?",
        "ground_truth": "Embedding models convert text to dense vectors capturing semantic meaning, enabling similarity search to retrieve chunks most relevant to a given query."
    },
    {
        "question": "How do LLMs perform on multi-hop reasoning questions in RAG?",
        "ground_truth": "Multi-hop questions require synthesising information across multiple retrieved passages, challenging RAG systems to surface all relevant chunks and combine them correctly."
    },
    {
        "question": "What is responsible AI evaluation in the context of LLMs?",
        "ground_truth": "Responsible AI evaluation assesses LLMs for safety, fairness, bias, toxicity, and alignment with human values, beyond accuracy to include societal impact."
    },
    {
        "question": "How does self-correction improve LLM reliability?",
        "ground_truth": "Self-correction allows LLMs to review and revise outputs, most effective when combined with external feedback or retrieval rather than applied in isolation."
    },
    {
        "question": "What are systematic evaluation frameworks for RAG?",
        "ground_truth": "Systematic RAG evaluation frameworks test retrieval quality, generation faithfulness, answer relevance, and end-to-end performance with automated scoring pipelines."
    },
    {
        "question": "How does passage reranking improve RAG answer quality?",
        "ground_truth": "Reranking re-orders retrieved passages by relevance before passing to the generator, ensuring the most pertinent context appears first and improving faithfulness scores."
    },
]


# ── Single evaluation ─────────────────────────────────────────────────────────

def evaluate_single(
    question: str,
    ground_truth: str,
    collection,
    top_k: int = 3,
    verbose: bool = True,
) -> dict:

    result = run_query(question, top_k=top_k, stream=verbose, collection=collection)

    if verbose:
        print("\nScoring (3 metrics × ~15s each)...")

    f_score  = score_faithfulness(result["answer"], result["contexts"])
    if verbose:
        print(f"  Faithfulness:     {f_score:.3f} ✓")

    ar_score = score_answer_relevancy(question, result["answer"])
    if verbose:
        print(f"  Answer relevancy: {ar_score:.3f} ✓")

    cr_score = score_context_recall(ground_truth, result["contexts"])
    if verbose:
        print(f"  Context recall:   {cr_score:.3f} ✓")

    record = {
        "timestamp":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question":          question,
        "answer":            result["answer"],
        "ground_truth":      ground_truth,
        "contexts":          result["contexts"],
        "faithfulness":      f_score,
        "answer_relevancy":  ar_score,
        "context_recall":    cr_score,
        "latency_ms":        result["latency_ms"],
        "retrieve_ms":       result["retrieve_ms"],
        "top_k":             top_k,
        "sources": [
            {
                "title":    c["metadata"]["title"],
                "year":     c["metadata"]["year"],
                "url":      c["metadata"]["url"],
                "distance": c["distance"],
            }
            for c in result["retrieved_chunks"]
        ],
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"EVALUATION SCORES")
        print(f"{'='*50}")
        print(f"  Faithfulness:     {f_score:.3f}  (grounded in context?)")
        print(f"  Answer relevancy: {ar_score:.3f}  (answers the question?)")
        print(f"  Context recall:   {cr_score:.3f}  (right chunks retrieved?)")
        print(f"  Latency:          {result['latency_ms']}ms")
        print(f"{'='*50}")

    return record


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_batch(collection, top_k: int = 3):
    print(f"\nBatch: {len(TEST_SET)} questions, top_k={top_k}")
    print("Approx time: 15-25 min\n")

    results = []
    for i, item in enumerate(TEST_SET):
        print(f"[{i+1:02d}/{len(TEST_SET)}] {item['question'][:65]}...")
        try:
            record = evaluate_single(
                question=item["question"],
                ground_truth=item["ground_truth"],
                collection=collection,
                top_k=top_k,
                verbose=False,
            )
            log_result(record)
            results.append(record)
            print(f"       F={record['faithfulness']:.3f}  "
                  f"AR={record['answer_relevancy']:.3f}  "
                  f"CR={record['context_recall']:.3f}  "
                  f"({record['latency_ms']}ms)")
        except Exception as e:
            print(f"       Error: {e}")

    valid = [r for r in results if r["faithfulness"] is not None]
    if not valid:
        return

    avg_f  = round(sum(r["faithfulness"]       for r in valid) / len(valid), 3)
    avg_ar = round(sum(r["answer_relevancy"]    for r in valid) / len(valid), 3)
    avg_cr = round(sum(r["context_recall"]      for r in valid) / len(valid), 3)
    avg_ms = round(sum(r["latency_ms"]          for r in valid) / len(valid))

    print(f"\n{'='*50}")
    print(f"BATCH SUMMARY  top_k={top_k}  n={len(valid)}")
    print(f"{'='*50}")
    print(f"  Avg Faithfulness:      {avg_f}")
    print(f"  Avg Answer Relevancy:  {avg_ar}")
    print(f"  Avg Context Recall:    {avg_cr}")
    print(f"  Avg Latency:           {avg_ms}ms")
    print(f"{'='*50}")

    Path("./logs").mkdir(exist_ok=True)
    summary = {
        "timestamp":             datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_type":              "batch",
        "top_k":                 top_k,
        "n_questions":           len(valid),
        "avg_faithfulness":      avg_f,
        "avg_answer_relevancy":  avg_ar,
        "avg_context_recall":    avg_cr,
        "avg_latency_ms":        avg_ms,
    }
    with open("./logs/run_summaries.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"  Saved to ./logs/run_summaries.jsonl")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",  action="store_true",
                        help="Run all 20 test questions")
    parser.add_argument("--top_k",  type=int, default=3,
                        help="Chunks to retrieve (default 3, use 5 for drift experiment)")
    args = parser.parse_args()

    # Check Ollama
    try:
        requests.get("http://localhost:11434", timeout=3)
    except requests.exceptions.ConnectionError:
        print("Ollama not running. Start it with: ollama serve")
        return

    collection = get_collection()

    if args.batch:
        evaluate_batch(collection, top_k=args.top_k)
    else:
        item = TEST_SET[0]
        record = evaluate_single(
            question=item["question"],
            ground_truth=item["ground_truth"],
            collection=collection,
            top_k=args.top_k,
            verbose=True,
        )
        log_result(record)
        print(f"\nLogged to {LOG_PATH}")


if __name__ == "__main__":
    main()