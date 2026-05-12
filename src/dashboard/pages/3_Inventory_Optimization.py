"""Inventory optimization dashboard page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.theme import inject_global_styles, render_note_box, render_page_header, render_panel_heading, style_figure
from src.dashboard.utils import download_frame, get_service

inject_global_styles()
render_page_header("Inventory Optimization", "Tune replenishment assumptions and review reorder recommendations.")

service = get_service()

render_panel_heading("Controls", "Adjust lead time and cost assumptions.")
control1, control2, control3 = st.columns(3)
lead_time = control1.slider("Lead Time (days)", min_value=1, max_value=30, value=7)
ordering_cost = control2.number_input("Ordering Cost", min_value=10.0, value=150.0, step=10.0)
holding_cost = control3.number_input("Holding Cost", min_value=1.0, value=8.0, step=1.0)

recommendations = pd.DataFrame(
    service.inventory_reorder(
        sku=None,
        lead_time_days=lead_time,
        ordering_cost=ordering_cost,
        holding_cost=holding_cost,
    )["recommendations"]
)
reorder_now = recommendations.query("recommendation == 'Reorder now'")

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Reorder Now", f"{len(reorder_now):,}")
metric2.metric("Total Suggested Qty", f"{recommendations['recommended_order_qty'].sum():,.0f}")
metric3.metric("Avg Safety Stock", f"{recommendations['safety_stock'].mean():.1f}")
metric4.metric("Avg Reorder Point", f"{recommendations['reorder_point'].mean():.1f}")

render_panel_heading("Reorder Surface", "Compare reorder points against current stock.")
col1, col2 = st.columns(2)
with col1:
    order_fig = px.bar(
        recommendations,
        x="sku",
        y="recommended_order_qty",
        color="recommendation",
        title="Recommended Reorder Quantity by SKU",
        labels={
            "sku": "Product SKU",
            "recommended_order_qty": "Units to Reorder",
            "recommendation": "Reorder Status",
        },
        color_discrete_sequence=["#FF5D8F", "#2DE2A7"],
    )
    st.plotly_chart(
        style_figure(order_fig),
        use_container_width=True,
    )
with col2:
    inventory_fig = px.scatter(
        recommendations,
        x="reorder_point",
        y="current_inventory",
        color="abc_class",
        symbol="xyz_class",
        size="safety_stock",
        title="Current Inventory vs Reorder Point",
        labels={
            "reorder_point": "Reorder Point (Units)",
            "current_inventory": "Current Inventory (Units)",
            "abc_class": "ABC Category",
            "xyz_class": "XYZ Category",
            "safety_stock": "Safety Stock",
        },
        color_discrete_sequence=["#21D4FD", "#FFB703", "#FF5D8F"],
    )
    st.plotly_chart(
        style_figure(inventory_fig),
        use_container_width=True,
    )

render_note_box(
    "Planner Hint",
    "SKUs marked 'Reorder now' are below the modeled reorder point after safety stock is applied.",
    tone="teal",
)
download_frame(recommendations, "inventory_recommendations")
st.dataframe(recommendations, use_container_width=True, hide_index=True)
