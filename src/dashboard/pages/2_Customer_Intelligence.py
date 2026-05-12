"""Customer intelligence dashboard page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.theme import inject_global_styles, render_page_header, render_panel_heading, style_figure
from src.dashboard.utils import download_frame, get_customer_features, get_service

inject_global_styles()
render_page_header("Customer Intelligence", "Segment customers, inspect value patterns, and review churn-risk accounts.")

service = get_service()
customers = get_customer_features()

segmentation = service.segment_customers()
profile = pd.DataFrame(segmentation["segment_profile"])
assignments = pd.DataFrame(segmentation["assignments"])
joined = customers.merge(assignments, on="customer_id", how="left")
joined["rfm_bubble_size"] = joined["rfm_score"].fillna(1).clip(lower=1)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Customers", f"{len(customers):,}")
metric2.metric("Avg Monetary", f"${customers['monetary'].mean():,.0f}")
metric3.metric("Avg RFM", f"{customers['rfm_score'].mean():.1f}")
metric4.metric("Segments", f"{profile['segment_id'].nunique():,}")

render_panel_heading("Segments", "See how customer value and frequency cluster together.")
col1, col2 = st.columns(2)
with col1:
    scatter_fig = px.scatter(
        joined,
        x="frequency",
        y="monetary",
        color="segment_id",
        size="rfm_bubble_size",
        hover_data=["customer_id", "recency"],
        title="Customer Value Distribution by Segment",
        labels={
            "frequency": "Purchase Frequency",
            "monetary": "Customer Spend ($)",
            "segment_id": "Customer Segment",
            "rfm_bubble_size": "RFM Score",
            "recency": "Days Since Last Purchase",
        },
        size_max=30,
        color_discrete_sequence=["#21D4FD", "#FFB703", "#FF5D8F", "#7B61FF", "#2DE2A7", "#FF7F50"],
    )
    st.plotly_chart(
        style_figure(scatter_fig),
        use_container_width=True,
    )
with col2:
    segment_fig = px.bar(
        profile,
        x="segment_id",
        y="customers",
        color="segment_id",
        title="Customer Count by Segment",
        labels={
            "segment_id": "Customer Segment",
            "customers": "Number of Customers",
        },
        color_discrete_sequence=["#FF5D8F", "#21D4FD", "#2DE2A7", "#FFB703", "#7B61FF", "#FF7F50"],
    )
    st.plotly_chart(
        style_figure(segment_fig),
        use_container_width=True,
    )

render_panel_heading("Retention Radar", "Review the highest-risk sample accounts and export the list.")
top_risk = []
for customer_id in customers["customer_id"].head(15):
    prediction = service.predict_churn(customer_id=customer_id)
    top_risk.append(prediction)
top_risk_frame = pd.DataFrame(top_risk).sort_values("churn_probability", ascending=False)
st.dataframe(top_risk_frame, use_container_width=True, hide_index=True)
download_frame(top_risk_frame, "customer_intelligence")

render_panel_heading("Segment Profile", "Summary view for targeting and lifecycle planning.")
st.dataframe(profile, use_container_width=True, hide_index=True)
