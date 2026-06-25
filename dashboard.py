"""
dashboard.py
------------
Week 4: Streamlit observability dashboard.

Shows:
  - Live score trends across all logged runs
  - Per-question breakdown (faithfulness, answer relevancy, context recall)
  - Drift detection: score comparison between top_k=3 and top_k=5 runs
  - Latency distribution
  - Source attribution table

Run:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Eval Pipeline — Observability Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Load logs ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_eval_log() -> pd.DataFrame:
    path = Path("./logs/eval_log.jsonl")
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["run_index"] = range(1, len(df) + 1)
    return df


@st.cache_data(ttl=10)
def load_summaries() -> pd.DataFrame:
    path = Path("./logs/run_summaries.jsonl")
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📊 RAG Evaluation Pipeline — Observability Dashboard")
st.caption(
    "Observable ML pipeline: arXiv corpus · Llama 3 · ChromaDB · "
    "Custom evaluation harness · Drift detection"
)

df      = load_eval_log()
summary = load_summaries()

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

if df.empty:
    st.warning(
        "No evaluation logs found yet. "
        "Run `python evaluate.py` or `python evaluate.py --batch` first."
    )
    st.stop()

# ── Top-level metrics ─────────────────────────────────────────────────────────

st.subheader("Overall performance")

col1, col2, col3, col4, col5 = st.columns(5)

avg_f  = df["faithfulness"].mean()
avg_ar = df["answer_relevancy"].mean()
avg_cr = df["context_recall"].mean()
avg_ms = df["latency_ms"].mean()
n_runs = len(df)

col1.metric("Queries evaluated",  n_runs)
col2.metric("Avg faithfulness",   f"{avg_f:.3f}",  help="Grounded in context? (0–1)")
col3.metric("Avg answer relevancy", f"{avg_ar:.3f}", help="Addresses the question? (0–1)")
col4.metric("Avg context recall", f"{avg_cr:.3f}",  help="Right chunks retrieved? (0–1)")
col5.metric("Avg latency",        f"{avg_ms/1000:.1f}s")

st.divider()

# ── Score trends ──────────────────────────────────────────────────────────────

st.subheader("Score trends across runs")
st.caption("Each point is one evaluated query. Watch for downward drift as conditions change.")

trend_df = df[["run_index", "faithfulness", "answer_relevancy", "context_recall"]].copy()
trend_df = trend_df.rename(columns={
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_recall":    "Context Recall",
    "run_index":         "Run",
})
trend_df = trend_df.set_index("Run")
st.line_chart(trend_df, height=280)

st.divider()

# ── Drift detection ───────────────────────────────────────────────────────────

st.subheader("🔍 Drift detection — top_k comparison")
st.caption(
    "Compares runs with top_k=3 vs top_k=5. "
    "Score changes reveal how retrieval depth affects answer quality."
)

if "top_k" in df.columns and df["top_k"].nunique() > 1:
    drift_df = (
        df.groupby("top_k")[["faithfulness", "answer_relevancy", "context_recall"]]
        .mean()
        .round(3)
        .reset_index()
    )
    drift_df.columns = ["top_k", "Faithfulness", "Answer Relevancy", "Context Recall"]
    drift_df["top_k"] = drift_df["top_k"].apply(lambda x: f"top_k={x}")
    drift_df = drift_df.set_index("top_k")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.dataframe(drift_df, use_container_width=True)
    with col_b:
        st.bar_chart(drift_df, height=220)

    # Flag significant drift
    if len(drift_df) == 2:
        vals = drift_df["Faithfulness"].values
        delta = abs(vals[1] - vals[0])
        if delta > 0.05:
            direction = "improved" if vals[1] > vals[0] else "degraded"
            st.warning(
                f"⚠️ Drift detected: faithfulness {direction} by {delta:.3f} "
                f"when changing top_k. This is what an AI platform observability "
                f"system is designed to catch."
            )
        else:
            st.success(f"✅ Faithfulness stable across top_k settings (Δ={delta:.3f})")
else:
    st.info(
        "Run the drift experiment to populate this section:\n\n"
        "```bash\npython evaluate.py --batch --top_k 5\n```"
    )

st.divider()

# ── Per-question breakdown ────────────────────────────────────────────────────

st.subheader("Per-question breakdown")

question_df = df[[
    "question", "faithfulness", "answer_relevancy",
    "context_recall", "latency_ms", "top_k"
]].copy()
question_df["question"] = question_df["question"].str[:80] + "..."
question_df["latency_s"] = (question_df["latency_ms"] / 1000).round(1)
question_df = question_df.drop(columns=["latency_ms"])
question_df.columns = [
    "Question", "Faithfulness", "Answer Relevancy",
    "Context Recall", "top_k", "Latency (s)"
]

# Colour-code by faithfulness
def colour_faithfulness(val):
    if isinstance(val, float):
        if val >= 0.7:
            return "background-color: #c6efce; color: #276221"
        elif val >= 0.4:
            return "background-color: #ffeb9c; color: #9c6500"
        else:
            return "background-color: #ffc7ce; color: #9c0006"
    return ""

st.dataframe(
    question_df.style.map(colour_faithfulness, subset=["Faithfulness"]),
    width="stretch",
    height=350,
)

st.divider()

# ── Latency distribution ──────────────────────────────────────────────────────

st.subheader("Latency distribution")

col_lat1, col_lat2 = st.columns(2)

with col_lat1:
    latency_df = df[["run_index", "latency_ms", "retrieve_ms"]].copy()
    latency_df["generate_ms"] = latency_df["latency_ms"] - latency_df["retrieve_ms"]
    latency_df = latency_df.rename(columns={
        "retrieve_ms":  "Retrieval (ms)",
        "generate_ms":  "Generation (ms)",
        "run_index":    "Run",
    }).set_index("Run")[["Retrieval (ms)", "Generation (ms)"]]
    st.bar_chart(latency_df, height=220)
    st.caption("Retrieval is fast (<200ms). Generation dominates total latency.")

with col_lat2:
    st.metric("Min latency",  f"{df['latency_ms'].min()/1000:.1f}s")
    st.metric("Max latency",  f"{df['latency_ms'].max()/1000:.1f}s")
    st.metric("Median latency", f"{df['latency_ms'].median()/1000:.1f}s")
    st.metric("Retrieval avg", f"{df['retrieve_ms'].mean():.0f}ms")

st.divider()

# ── Recent runs log ───────────────────────────────────────────────────────────

st.subheader("Recent evaluation log")

recent = df.tail(10)[[
    "timestamp", "question", "faithfulness",
    "answer_relevancy", "context_recall", "top_k"
]].copy()
recent["timestamp"] = recent["timestamp"].dt.strftime("%H:%M:%S")
recent["question"]  = recent["question"].str[:60] + "..."
recent.columns = [
    "Time", "Question", "Faithfulness",
    "Ans. Relevancy", "Context Recall", "top_k"
]
st.dataframe(recent, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "RAG Evaluation Pipeline · Abhishek Kumar Singh · "
    "Master of Engineering in Applied Data Science, University of Victoria · "
    "Built for Autodesk AI/ML Internship application"
)