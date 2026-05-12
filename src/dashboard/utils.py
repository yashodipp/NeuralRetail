"""Dashboard helpers for loading data, filtering, and exporting."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.api.services import get_platform_service_singleton


@st.cache_resource(show_spinner=False)
def get_service():
    """Return the shared platform service."""

    return get_platform_service_singleton()


@st.cache_data(show_spinner=False)
def get_transactions() -> pd.DataFrame:
    """Load transaction data for dashboard pages."""

    return get_service().state.transactions.copy()


@st.cache_data(show_spinner=False)
def get_time_series_features() -> pd.DataFrame:
    """Load time-series feature data."""

    return get_service().state.time_series_features.copy()


@st.cache_data(show_spinner=False)
def get_customer_features() -> pd.DataFrame:
    """Load customer feature data."""

    return get_service().state.customer_features.copy()


def apply_global_filters(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Render common filters and return filtered data."""

    sku_options = ["All", *sorted(frame["sku"].dropna().unique().tolist())] if "sku" in frame else ["All"]
    region_options = ["All", *sorted(frame["region"].dropna().unique().tolist())] if "region" in frame else ["All"]

    col1, col2 = st.columns(2)
    selected_sku = col1.selectbox("SKU", sku_options)
    selected_region = col2.selectbox("Region", region_options)

    filtered = frame.copy()
    if selected_sku != "All" and "sku" in filtered:
        filtered = filtered.query("sku == @selected_sku")
    if selected_region != "All" and "region" in filtered:
        filtered = filtered.query("region == @selected_region")
    return filtered, selected_sku, selected_region


def export_excel(frame: pd.DataFrame) -> bytes:
    """Convert a DataFrame to an in-memory Excel file."""

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="NeuralRetail")
    return buffer.getvalue()


def export_pdf(title: str, frame: pd.DataFrame) -> bytes:
    """Render a tabular PDF summary for download."""

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, height - 40, title)
    pdf.setFont("Helvetica", 8)

    y = height - 70
    for row in frame.head(25).astype(str).itertuples(index=False):
        pdf.drawString(40, y, " | ".join(row)[:110])
        y -= 12
        if y < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 8)
            y = height - 40
    pdf.save()
    return buffer.getvalue()


def download_frame(frame: pd.DataFrame, stem: str) -> None:
    """Render Excel and PDF export buttons."""

    col1, col2 = st.columns(2)
    col1.download_button(
        label="Export Excel",
        data=export_excel(frame),
        file_name=f"{stem}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    col2.download_button(
        label="Export PDF",
        data=export_pdf(stem, frame),
        file_name=f"{stem}.pdf",
        mime="application/pdf",
    )
