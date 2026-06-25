# Observable RAG Evaluation Pipeline

> An end-to-end Retrieval-Augmented Generation system with a **custom evaluation harness** and **real-time observability dashboard** — built to surface answer quality, faithfulness drift, and retrieval performance across a corpus of 402 arXiv ML papers.

---

## Why this project exists

Most RAG systems stop at generation: a query goes in, an answer comes out. This project adds the layer that production AI platforms actually need — **automated evaluation of every answer** across three dimensions, logged to a structured observability dashboard with drift detection.

This mirrors the infrastructure Autodesk's AI platform team builds: shared evaluation harnesses, observability tooling, and responsible AI measurement systems that make ML outputs reliable at scale.

---

## Architecture

```
User query
    │
    ▼
Embedding model ── all-MiniLM-L6-v2 (local, no API cost)
    │  converts query → vector
    ▼
ChromaDB vector store ── 1,533 chunks, cosine similarity
    │  retrieves top-k most relevant passages
    ▼
Prompt manager ── context injection + grounded generation constraint
    │  LLM instructed to answer from context only
    ▼
Ollama / Llama 3 ── local LLM, zero API cost
    │  generates answer grounded in retrieved context
    ▼
Evaluation harness ── custom scoring via direct Ollama calls
    ├── Faithfulness:      claims grounded in context? (0–1)
    ├── Answer relevancy:  answers the question asked? (0–1)
    └── Context recall:    right chunks retrieved? (0–1)
    │
    ▼
Structured JSONL logger ──► Streamlit observability dashboard
                               score trends · drift detection · latency
```

---

## Results

### Baseline run — top_k=3, 20 questions

| Metric | Score | Interpretation |
|---|---|---|
| Avg Faithfulness | 0.184 | Llama 3 8B hedges language; scorer correctly penalises indirect claims |
| Avg Answer Relevancy | 0.465 | Model answers the question ~50% of the time with 3 retrieved chunks |
| Avg Context Recall | 0.642 | Retriever surfaces relevant chunks for 64% of questions |
| Avg Retrieval latency | 55ms | Fast — embedding + cosine search over 1,533 chunks |
| Avg Total latency | 6.9s | Retrieval: 55ms · Generation: ~6.8s (local Llama 3) |

### Drift experiment — top_k=3 vs top_k=5

| Metric | top_k=3 | top_k=5 | Δ | Interpretation |
|---|---|---|---|---|
| Faithfulness | 0.184 | 0.179 | −0.005 | Stable — extra context doesn't cause hallucination |
| Answer Relevancy | 0.465 | 0.610 | **+0.145** | More context → model answers more confidently |
| Context Recall | 0.642 | 0.630 | −0.012 | Stable — retriever quality consistent |
| Avg Latency | 6.9s | 8.1s | +1.2s | Measurable cost of additional context |

**Finding:** Increasing retrieved context from 3 to 5 chunks improved answer relevancy by +14.5% while faithfulness and recall held stable. The +1.2s latency cost is observable and quantified. This is exactly the kind of configuration trade-off an AI platform evaluation harness is designed to detect automatically.

---

## Observability dashboard

The Streamlit dashboard (`dashboard.py`) provides:

- **Overall metrics** — queries evaluated, avg faithfulness, answer relevancy, context recall, latency
- **Score trends** — line chart across all runs showing per-query score variation
- **Drift detection** — automatic top_k comparison table and bar chart; flags score changes > 0.05
- **Per-question breakdown** — colour-coded faithfulness table (green ≥ 0.7, amber ≥ 0.4, red < 0.4)
- **Latency distribution** — stacked bar chart splitting retrieval vs generation time per query
- **Live evaluation log** — last 10 runs with timestamps, scores, and top_k setting

All data is read from `./logs/eval_log.jsonl` — a structured append-only log updated after every query.

---

## Evaluation harness design

**Faithfulness** — extracts factual claims from the generated answer using Llama 3, then verifies each claim against the retrieved context. Score = fraction of claims directly supported. Low scores indicate hallucination or over-hedged language.

**Answer relevancy** — asks Llama 3 to rate 0–1 how directly the answer addresses the original question. Catches responses that are contextually accurate but off-topic.

**Context recall** — asks Llama 3 to rate 0–1 how much of the ground truth answer's key information appears in the retrieved chunks. Diagnoses retriever failures independently of generation quality.

**Why not RAGAS?** RAGAS 0.1.x has an asyncio incompatibility with Python 3.13. Rather than downgrade the Python runtime, we implemented equivalent scoring logic using direct synchronous Ollama calls — same measurement methodology, no external dependency issues. This is also more transparent: every scoring prompt is visible in `evaluate.py`.

---

## Project structure

```
rag-eval-pipeline/
├── ingest.py       # Fetch arXiv papers → chunk → embed → store in ChromaDB
├── pipeline.py     # Query → retrieve → generate (full RAG pipeline)
├── evaluate.py     # Evaluation harness: faithfulness, relevancy, recall
├── dashboard.py    # Streamlit observability dashboard with drift detection
├── data/
│   ├── chroma_db/          ← persistent vector store (1,533 chunks)
│   └── raw_abstracts.json  ← 402 arXiv ML papers
└── logs/
    ├── eval_log.jsonl       ← per-query evaluation records
    └── run_summaries.jsonl  ← batch run summaries for drift charts
```

---

## Stack

| Component | Tool | Reason |
|---|---|---|
| Vector store | ChromaDB | Local, persistent, cosine similarity, no infra needed |
| Embedding model | all-MiniLM-L6-v2 | Fast, free, runs fully locally |
| LLM | Ollama / Llama 3 | Zero API cost, fully local inference |
| Orchestration | LangChain | Pipeline composition and retrieval |
| Evaluation | Custom Ollama scoring | Python 3.13 compatible, fully transparent |
| Observability | Streamlit | Real-time dashboard, no deployment needed |
| Dataset | arXiv ML abstracts | 402 papers across RAG, LLM evaluation, AI safety |

---

## Setup and run

**1. Install dependencies**

```bash
python -m venv venv
source venv/bin/activate
pip install chromadb langchain langchain-community sentence-transformers \
    arxiv streamlit pandas datasets tqdm requests langchain-ollama
```

**2. Install Ollama and pull Llama 3**

```bash
# macOS
brew install ollama
ollama pull llama3
ollama serve        # keep running in a separate terminal tab
```

**3. Build the vector store**

```bash
python ingest.py
# Fetches 402 arXiv ML papers, builds 1,533-chunk ChromaDB collection (~5 min)
```

**4. Run a single query**

```bash
python pipeline.py --query "What are the main challenges in RAG systems?"
python pipeline.py --query "How does chunking affect retrieval quality?" --top_k 5
```

**5. Run evaluation**

```bash
python evaluate.py                       # single query demo with scores
python evaluate.py --batch               # all 20 test questions (top_k=3)
python evaluate.py --batch --top_k 5    # drift experiment
```

**6. Launch observability dashboard**

```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

---

## Key design decisions

**Grounded generation prompt** — the system prompt instructs the LLM to answer only from retrieved context and explicitly say "I don't have enough context" if it can't. This makes faithfulness scores meaningful: a low score genuinely indicates the model deviated from evidence, not just that the answer was wrong.

**Drift detection via top_k variation** — running the same 20-question test set with different retrieval depths (top_k=3 vs top_k=5) simulates the kind of configuration change that silently degrades AI system quality in production. The observability dashboard flags score changes above 0.05 automatically with directional analysis.

**JSONL structured logging** — every query is logged with timestamp, all three scores, latency breakdown (retrieval ms vs total ms), source paper attributions, and top_k setting. This makes the dashboard filterable by run configuration and enables historical comparison across experiments.

**Title + abstract field-scoped arXiv queries** — initial ingestion returned physics and materials science papers due to arXiv's broad category matching. Switching to `ti:"retrieval augmented generation"` and `abs:"faithfulness" AND abs:"context"` field-scoped queries, combined with a domain blacklist, produced a clean ML-only corpus with retrieval distances under 0.40 for target queries.

---

## Observations and limitations

- Faithfulness scores are low (avg 0.18) because Llama 3 8B generates hedged language ("it can be inferred", "suggests that") rather than direct claims — the scorer correctly penalises this. A larger model (Llama 3 70B, GPT-4) or a more directive prompt would raise this significantly.
- Context recall of 0.64 is meaningful given the corpus size and lightweight local embedding model. Highly specific queries (hallucination detection, chunking strategies) achieved distances under 0.25 — near-perfect retrieval.
- The custom scorer introduces variance since it uses the same model for generation and evaluation. In a production system you would use a separate, stronger evaluation model (the two-model pattern).
- The +14.5% answer relevancy improvement from top_k=3→5 is the most actionable finding: for this corpus and model, serving more context meaningfully improves output quality at an acceptable latency cost.

---

*Master of Engineering in Applied Data Science · University of Victoria*
*Built as a portfolio project demonstrating RAG architecture, evaluation harness design, observability tooling, and drift detection — core capabilities for AI platform engineering roles.*
