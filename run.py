"""Named Streamlit entrypoint for the NeuralRetail dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="NeuralRetail", page_icon=":bar_chart:", layout="wide")

navigation = st.navigation(
    [
        st.Page(
            ROOT / "src" / "dashboard" / "Home.py",
            title="Main Dashboard",
            default=True,
        ),
        st.Page(
            ROOT / "src" / "dashboard" / "pages" / "1_Demand_Forecasting.py",
            title="Demand Forecasting",
        ),
        st.Page(
            ROOT / "src" / "dashboard" / "pages" / "2_Customer_Intelligence.py",
            title="Customer Intelligence",
        ),
        st.Page(
            ROOT / "src" / "dashboard" / "pages" / "3_Inventory_Optimization.py",
            title="Inventory Optimization",
        ),
        st.Page(
            ROOT / "src" / "dashboard" / "pages" / "4_MLOps_Monitoring.py",
            title="MLOps Monitoring",
        ),
    ]
)

navigation.run()
