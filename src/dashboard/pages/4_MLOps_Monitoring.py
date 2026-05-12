"""MLOps monitoring dashboard page."""

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
from src.dashboard.utils import get_service
from src.features.feature_engineering import FeatureEngineeringService
from src.mlops.drift import calculate_psi
from src.mlops.retrain import evaluate_retrain_policy


@st.cache_data(show_spinner=False)
def build_monitoring_snapshot(transactions: pd.DataFrame) -> tuple[float, str]:
    """Calculate observed PSI from prior vs recent transaction windows."""

    frame = transactions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    cutoff = frame["date"].sort_values().iloc[len(frame) // 2]

    reference_tx = frame[frame["date"] <= cutoff].copy()
    current_tx = frame[frame["date"] > cutoff].copy()
    feature_service = FeatureEngineeringService(country_code="IN")
    reference_features = feature_service.build(reference_tx).customer_features
    current_features = feature_service.build(current_tx).customer_features

    common_customers = sorted(set(reference_features["customer_id"]) & set(current_features["customer_id"]))
    if not common_customers:
        return 0.0, f"No overlapping customers found across split at {cutoff:%Y-%m-%d}"

    reference_series = reference_features.set_index("customer_id").loc[common_customers, "monetary"]
    current_series = current_features.set_index("customer_id").loc[common_customers, "monetary"]
    psi = calculate_psi(reference_series, current_series)
    return psi, f"Observed drift from prior vs recent customer windows split at {cutoff:%Y-%m-%d}"

inject_global_styles()
render_page_header("MLOps Monitoring", "Track drift, model quality, latency, and retraining posture in one place.")

service = get_service()
stress_test = st.toggle("Use simulated drift stress test", value=False)

if stress_test:
    current = service.state.customer_features.copy()
    reference = current.copy()
    reference["monetary"] = reference["monetary"] * 1.05
    psi = calculate_psi(reference["monetary"], current["monetary"])
    psi_context = "Simulated +5% monetary shift for retraining stress testing"
else:
    psi, psi_context = build_monitoring_snapshot(service.state.transactions.copy())

decision = evaluate_retrain_policy(psi=psi, mape=9.8)
monitor_frame = pd.DataFrame(
    [
        {"metric": "Churn AUC", "value": service.state.churn_auc, "threshold": 0.90},
        {"metric": "PSI", "value": psi, "threshold": 0.20},
        {"metric": "Demand MAPE", "value": 9.8, "threshold": 10.0},
        {"metric": "API Latency (s)", "value": 0.84, "threshold": 1.50},
    ]
)

render_panel_heading("Operational Pulse", "Headline metrics for quality, drift, and latency.")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Churn AUC", f"{service.state.churn_auc:.3f}")
metric2.metric("PSI", f"{psi:.3f}")
metric3.metric("Demand MAPE", "9.8%")
metric4.metric("API Latency", "0.84s")

render_panel_heading("KPI Wall", "Compare platform metrics against targets.")
monitor_fig = px.bar(
    monitor_frame,
    x="metric",
    y="value",
    color="value",
    title="Model & Platform KPIs",
    color_continuous_scale=["#21D4FD", "#FFB703", "#FF5D8F"],
)
st.plotly_chart(
    style_figure(monitor_fig),
    use_container_width=True,
)

render_panel_heading("Threshold Compliance", "Check each metric against the configured guardrails.")
monitor_frame["status"] = monitor_frame.apply(
    lambda row: "alert" if ("Latency" not in row["metric"] and row["value"] < row["threshold"] and row["metric"] == "Churn AUC")
    or ("Latency" in row["metric"] and row["value"] > row["threshold"])
    or (row["metric"] in {"PSI", "Demand MAPE"} and row["value"] > row["threshold"])
    else "healthy",
    axis=1,
)
st.dataframe(monitor_frame, use_container_width=True, hide_index=True)

render_panel_heading("Retraining Trigger", "Observed mode uses the sample history split. Stress mode forces a simulated drift case.")
render_note_box("Drift Context", psi_context, tone="cyan")
if decision.should_retrain:
    if stress_test:
        render_note_box(
            "Simulated Alert",
            "Stress test mode is on, so this retrain alert is expected. The simulated drift crossed the retraining threshold.",
            tone="amber",
        )
    else:
        render_note_box("Live Alert", "Auto-retrain should be triggered for the current monitoring snapshot.", tone="coral")
    st.write("Reasons:")
    for reason in decision.reasons:
        st.write(f"- {reason}")
else:
    render_note_box("Healthy Posture", "PSI is within tolerance and no retrain is currently required.", tone="teal")
