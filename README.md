# Observable RAG Evaluation Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) system with a custom evaluation harness and real-time observability dashboard — built over 402 arXiv ML papers.

## Dashboard
![Overview](screenshots/dashboard1.png)
![Drift Detection](screenshots/dashboard2.png)
![Latency](screenshots/dashboard3.png)
![Log](screenshots/dashboard4.png)

## Results

| Metric | top_k=3 | top_k=5 | Change |
|---|---|---|---|
| Faithfulness | 0.184 | 0.179 | stable |
| Answer Relevancy | 0.465 | 0.610 | +14.5% |
| Context Recall | 0.642 | 0.630 | stable |
| Avg Latency | 6.9s | 8.1s | +1.2s |

## Stack
Python · LangChain · ChromaDB · Ollama (Llama 3) · Streamlit · arXiv API

## Run it
```bash
python ingest.py          # build ChromaDB from arXiv papers
python pipeline.py        # single RAG query
python evaluate.py --batch         # score 20 questions
python evaluate.py --batch --top_k 5   # drift experiment
streamlit run dashboard.py             # observability dashboard
```

Built for Autodesk AI/ML Internship · MEng Applied Data Science · University of Victoria
