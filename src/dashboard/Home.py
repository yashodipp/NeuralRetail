"""Executive overview dashboard page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.theme import inject_global_styles, render_note_box, render_page_header, render_panel_heading, style_figure
from src.dashboard.utils import apply_global_filters, download_frame, get_service, get_transactions

inject_global_styles()
render_page_header("Executive Overview", "A clean control room for revenue, pricing response, and filtered retail performance.")

service = get_service()
transactions = get_transactions()
transactions["date"] = pd.to_datetime(transactions["date"])

render_panel_heading("Filters", "Focus the view by SKU and region.")
filtered, selected_sku, selected_region = apply_global_filters(transactions)

render_panel_heading("Snapshot", "Core KPIs for the current retail slice.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Revenue", f"${filtered['revenue'].sum():,.0f}")
kpi2.metric("Units Sold", f"{filtered['quantity'].sum():,.0f}")
kpi3.metric("Customers", f"{filtered['customer_id'].nunique():,}")
kpi4.metric("Avg Discount", f"{filtered['discount_pct'].mean():.1%}")

render_panel_heading("Revenue and Pricing", "Track revenue over time and test live price movement.")
chart_col, simulator_col = st.columns([2, 1])
with chart_col:
    revenue_trend = (
        filtered.groupby("date", as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
        .sort_values("date")
    )
    revenue_fig = px.area(
        revenue_trend,
        x="date",
        y="revenue",
        title="Daily Sales Revenue Trend",
        markers=True,
        labels={
            "date": "Date",
            "revenue": "Sales Revenue ($)",
        },
        color_discrete_sequence=["#21D4FD"],
    )
    st.plotly_chart(
        style_figure(revenue_fig),
        use_container_width=True,
    )

with simulator_col:
    render_note_box(
        "Price What-If Lab",
        "Push a candidate price into the elasticity model and see how expected demand and revenue react in real time.",
        tone="coral",
    )
    sku_options = sorted(transactions["sku"].unique().tolist())
    simulator_sku = st.selectbox("Simulator SKU", sku_options, index=0, key="sim_sku")
    candidate_price = st.slider("Candidate Price", min_value=5.0, max_value=60.0, value=24.0, step=0.5)
    scenario = service.price_what_if(simulator_sku, candidate_price)
    st.metric("Expected Demand", f"{scenario['expected_demand']:.1f}")
    st.metric("Expected Revenue", f"${scenario['expected_revenue']:.2f}")
    st.metric("Elasticity", f"{scenario['price_elasticity']:.2f}")

regional = filtered.groupby("region", as_index=False).agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
col1, col2 = st.columns(2)
with col1:
    regional_fig = px.bar(
        regional,
        x="region",
        y="revenue",
        title="Regional Revenue Comparison",
        color="region",
        labels={
            "region": "Sales Region",
            "revenue": "Revenue ($)",
        },
        color_discrete_sequence=["#21D4FD", "#FFB703", "#FF5D8F", "#2DE2A7"],
    )
    st.plotly_chart(
        style_figure(regional_fig),
        use_container_width=True,
    )
with col2:
    sku_mix = filtered.groupby("sku", as_index=False).agg(quantity=("quantity", "sum"))
    sku_mix_fig = px.pie(
        sku_mix,
        values="quantity",
        names="sku",
        title="Sales Volume Share by SKU",
        color_discrete_sequence=["#FF5D8F", "#21D4FD", "#FFB703", "#7B61FF"],
    )
    sku_mix_fig.update_traces(hole=0.58, textposition="inside", textinfo="percent+label")
    st.plotly_chart(
        style_figure(sku_mix_fig),
        use_container_width=True,
    )

render_panel_heading("Records", "Review and export the filtered transaction set.")
download_frame(filtered, f"executive_overview_{selected_sku}_{selected_region}")
st.dataframe(filtered, use_container_width=True, hide_index=True)
