"""Demand forecasting dashboard page."""

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
from src.dashboard.utils import download_frame, get_service, get_time_series_features

inject_global_styles()
render_page_header("Demand Forecasting", "Project demand by SKU and region with a cleaner planning view.")

service = get_service()
features = get_time_series_features()
features["date"] = pd.to_datetime(features["date"])

render_panel_heading("Controls", "Choose the valid SKU-region pair and horizon.")
control_col1, control_col2, control_col3 = st.columns(3)
sku = control_col1.selectbox("SKU", sorted(features["sku"].unique().tolist()))
valid_regions = sorted(features.loc[features["sku"] == sku, "region"].dropna().unique().tolist())
region = control_col2.selectbox("Region", valid_regions)
horizon_days = control_col3.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=1)

history = features.query("sku == @sku and region == @region").sort_values("date")
if history.empty:
    st.warning(f"No historical records are available for {sku} in {region}.")
    st.stop()

forecast_payload = service.forecast_demand(sku, region, horizon_days)
forecast = pd.DataFrame(forecast_payload["forecast"])
forecast["date"] = pd.to_datetime(forecast["date"])
avg_next_week = forecast["daily_forecast"].head(7).mean()
total_month = forecast["monthly_forecast"].tail(1).iloc[0]

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Last Observed Daily Demand", f"{history['quantity'].tail(1).iloc[0]:.1f}")
metric2.metric("Avg Next 7-Day Forecast", f"{avg_next_week:.1f}")
metric3.metric("Forecast Accuracy (MAPE)", f"{forecast_payload['metrics']['mape']:.1f}%")

chart_data = history[["date", "quantity"]].rename(columns={"quantity": "value"})
chart_data["series"] = "Historical Demand"
forecast_data = forecast[["date", "daily_forecast"]].rename(columns={"daily_forecast": "value"})
forecast_data["series"] = "Forecasted Demand"
combined = pd.concat([chart_data, forecast_data], ignore_index=True)

render_panel_heading("Demand Outlook", "Compare history against the forecast path.")
forecast_fig = px.line(
    combined,
    x="date",
    y="value",
    color="series",
    markers=True,
    title=f"Demand Forecast for {sku} in {region}",
    labels={
        "date": "Date",
        "value": "Demand Units",
        "series": "Data Type",
    },
    color_discrete_sequence=["#21D4FD", "#FF5D8F"],
)
st.plotly_chart(
    style_figure(forecast_fig),
    use_container_width=True,
)

render_panel_heading("Forecast Detail", f"Projected monthly volume: {total_month:.1f} units.")
download_frame(forecast, f"demand_forecast_{sku}_{region}")
st.dataframe(forecast, use_container_width=True, hide_index=True)

render_panel_heading("Price Simulator", "Check how demand and revenue respond to price changes.")
render_note_box(
    "Elasticity Context",
    "The pricing model estimates how demand and revenue move when price changes.",
    tone="amber",
)
candidate_price = st.number_input("Candidate Price", min_value=1.0, value=float(history["avg_price"].tail(7).mean()), step=0.5)
scenario = service.price_what_if(sku, candidate_price)
col1, col2, col3 = st.columns(3)
col1.metric("Projected Demand", f"{scenario['expected_demand']:.2f}")
col2.metric("Projected Revenue", f"${scenario['expected_revenue']:.2f}")
col3.metric("Price Elasticity", f"{scenario['price_elasticity']:.2f}")
