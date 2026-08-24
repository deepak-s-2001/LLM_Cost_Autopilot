import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.config import load_env
from app.costs import stats

load_env()

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")
st.title("LLM Cost Autopilot — Dashboard")
st.caption("Chat now lives in the standalone frontend (`frontend/index.html`, served by the API). This page is analytics only.")

savings = stats.total_savings()

if savings["request_count"] == 0:
    st.info("No requests logged yet — use the chat frontend to generate data.")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cost saved vs. all-flagship baseline", f"${savings['savings_usd']:.4f}", f"{savings['savings_pct']:.1f}%")
    col2.metric("Actual spend (incl. escalations)", f"${savings['actual_total_usd']:.4f}")
    col3.metric("Escalation rerun cost", f"${savings['escalation_cost_usd']:.4f}")
    col4.metric("Requests routed", savings["request_count"])

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Routing distribution")
        distribution = stats.routing_distribution()
        fig = px.pie(names=list(distribution.keys()), values=list(distribution.values()))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Quality score distribution")
        scores = stats.quality_score_distribution()
        if scores:
            fig = px.histogram(x=scores, nbins=10, range_x=[0, 5], labels={"x": "quality score"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No verification scores yet — the async verifier worker hasn't processed any jobs.")

    st.subheader("Escalation rate over time")
    rate_data = stats.escalation_rate_over_time()
    if rate_data:
        rate_df = pd.DataFrame(rate_data)
        rate_df["date"] = pd.to_datetime(rate_df["date"])
        fig = px.line(rate_df, x="date", y="escalation_rate", markers=True)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No escalation data yet.")
