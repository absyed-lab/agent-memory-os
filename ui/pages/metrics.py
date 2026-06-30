import streamlit as st
import json
import os
import pandas as pd

st.set_page_config(page_title="Memory OS Metrics", page_icon="📊", layout="wide")

st.markdown("# 📊 Agent Memory OS — Metrics Dashboard")
st.markdown("*Real-time performance tracking across all conversations*")
st.markdown("---")

METRICS_PATH = "./data/metrics.json"

def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return []

metrics = load_metrics()

if not metrics:
    st.info("No metrics yet. Go to the main chat page and have a conversation first.")
    st.stop()

df = pd.DataFrame(metrics)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["turn_label"] = df.index + 1

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Turns", len(df))
with col2:
    st.metric("Avg Response Time", f"{df['latency_seconds'].mean():.2f}s")
with col3:
    st.metric("Total Tokens Used", f"{df['tokens_used'].sum():,}")
with col4:
    st.metric("Unique Customers", df["customer_id"].nunique())

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Response Latency Over Turns")
    st.line_chart(df.set_index("turn_label")["latency_seconds"])
    st.markdown("### Memory Facts Growth")
    st.line_chart(df.set_index("turn_label")["memory_facts"])

with col_right:
    st.markdown("### Token Usage Per Turn")
    st.bar_chart(df.set_index("turn_label")["tokens_used"])
    st.markdown("### Episodes Accumulated")
    st.line_chart(df.set_index("turn_label")["episodes"])

st.markdown("---")
st.markdown("### Raw Metrics Log")
st.dataframe(
    df[["timestamp", "customer_id", "turn", "latency_seconds", "tokens_used", "memory_facts", "episodes", "backend"]],
    use_container_width=True
)

if st.button("Clear Metrics"):
    if os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)
    st.rerun()
