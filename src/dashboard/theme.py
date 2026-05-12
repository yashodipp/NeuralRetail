"""Shared theming primitives for the Streamlit dashboard."""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st


CHART_COLORS = ["#21D4FD", "#FFB703", "#FF5D8F", "#7B61FF", "#2DE2A7", "#FF7F50"]


def inject_global_styles() -> None:
    """Inject a colorful, frontend-like CSS theme into the page."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

        :root {
            --nr-bg-top: #08111f;
            --nr-bg-bottom: #150f29;
            --nr-panel: rgba(10, 18, 35, 0.76);
            --nr-panel-strong: rgba(15, 24, 46, 0.94);
            --nr-card-line: rgba(255, 255, 255, 0.10);
            --nr-text: #EFF5FF;
            --nr-muted: #94A8C0;
            --nr-cyan: #21D4FD;
            --nr-teal: #2DE2A7;
            --nr-amber: #FFB703;
            --nr-coral: #FF5D8F;
            --nr-violet: #7B61FF;
            --nr-orange: #FF7F50;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 16%, rgba(33, 212, 253, 0.16), transparent 24%),
                radial-gradient(circle at 88% 14%, rgba(255, 93, 143, 0.16), transparent 26%),
                radial-gradient(circle at 80% 80%, rgba(45, 226, 167, 0.10), transparent 24%),
                linear-gradient(145deg, var(--nr-bg-top) 0%, #111b39 52%, var(--nr-bg-bottom) 100%);
            color: var(--nr-text);
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        html, body, p, div, label, span, button, input, textarea {
            font-family: 'Manrope', sans-serif;
        }

        h1, h2, h3, h4, h5 {
            font-family: 'Sora', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 19, 34, 0.96), rgba(19, 12, 35, 0.94));
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            font-family: 'Manrope', sans-serif !important;
        }

        .nr-page-header {
            margin: 0 0 1.1rem 0;
            padding: 0.2rem 0 0.15rem 0;
        }

        .nr-page-header h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.08;
            letter-spacing: -0.05em;
            color: var(--nr-text);
        }

        .nr-page-header p {
            margin: 0.35rem 0 0 0;
            color: var(--nr-muted);
            font-size: 0.98rem;
            line-height: 1.6;
            max-width: 820px;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(15, 24, 45, 0.92), rgba(9, 17, 34, 0.88));
            border: 1px solid var(--nr-card-line);
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 24px 60px rgba(3, 7, 18, 0.35);
        }

        [data-testid="stMetricLabel"] {
            color: var(--nr-muted);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.76rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: var(--nr-text);
            font-family: 'Sora', sans-serif;
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }

        [data-testid="stMetricDelta"] {
            color: var(--nr-teal);
        }

        .nr-panel-heading {
            margin: 0.55rem 0 0.8rem 0;
            padding: 0.9rem 1rem;
            border-left: 3px solid rgba(33, 212, 253, 0.75);
            border-radius: 18px;
            background: linear-gradient(145deg, rgba(14, 24, 46, 0.80), rgba(9, 17, 34, 0.62));
            box-shadow: 0 14px 34px rgba(3, 7, 18, 0.14);
        }

        .nr-panel-heading h3 {
            margin: 0;
            color: var(--nr-text);
            font-family: 'Sora', sans-serif;
            font-size: 1rem;
            letter-spacing: -0.03em;
        }

        .nr-panel-heading p {
            margin: 0.35rem 0 0 0;
            color: var(--nr-muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .nr-note-box {
            border-radius: 20px;
            padding: 1rem 1.05rem;
            border: 1px solid var(--nr-card-line);
            background: linear-gradient(160deg, rgba(14, 24, 47, 0.92), rgba(9, 17, 34, 0.82));
            box-shadow: 0 22px 45px rgba(4, 9, 21, 0.25);
        }

        .nr-note-box h4 {
            margin: 0;
            font-family: 'Sora', sans-serif;
            font-size: 0.95rem;
            letter-spacing: -0.02em;
        }

        .nr-note-box p {
            margin: 0.35rem 0 0 0;
            color: var(--nr-muted);
            line-height: 1.55;
            font-size: 0.92rem;
        }

        .nr-note-cyan { box-shadow: inset 0 0 0 1px rgba(33, 212, 253, 0.18), 0 22px 45px rgba(4, 9, 21, 0.25); }
        .nr-note-coral { box-shadow: inset 0 0 0 1px rgba(255, 93, 143, 0.18), 0 22px 45px rgba(4, 9, 21, 0.25); }
        .nr-note-amber { box-shadow: inset 0 0 0 1px rgba(255, 183, 3, 0.18), 0 22px 45px rgba(4, 9, 21, 0.25); }
        .nr-note-teal { box-shadow: inset 0 0 0 1px rgba(45, 226, 167, 0.18), 0 22px 45px rgba(4, 9, 21, 0.25); }

        div[data-baseweb="select"] > div,
        [data-testid="stNumberInputContainer"] > div,
        [data-testid="stTextInputRootElement"] > div,
        [data-testid="stDateInputField"] > div {
            background: rgba(14, 24, 46, 0.92) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 16px !important;
            color: var(--nr-text) !important;
            box-shadow: 0 16px 32px rgba(4, 9, 21, 0.18);
        }

        label[data-testid="stWidgetLabel"] p {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--nr-muted) !important;
        }

        .stSlider [data-baseweb="slider"] {
            padding-top: 0.85rem;
            padding-bottom: 0.5rem;
        }

        .stSlider [role="slider"] {
            background: linear-gradient(135deg, var(--nr-coral), var(--nr-amber)) !important;
            box-shadow: 0 0 0 6px rgba(255, 93, 143, 0.16);
        }

        .stButton > button,
        .stDownloadButton > button {
            border: none !important;
            border-radius: 999px !important;
            background: linear-gradient(90deg, var(--nr-cyan), var(--nr-violet)) !important;
            color: #07111f !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em !important;
            padding: 0.65rem 1rem !important;
            box-shadow: 0 16px 32px rgba(33, 212, 253, 0.22) !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.04);
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--nr-card-line);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 18px 38px rgba(4, 9, 21, 0.18);
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--nr-card-line);
            border-radius: 20px;
            background: rgba(14, 24, 46, 0.86);
        }

        [data-testid="stAlert"] {
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .stMarkdown a {
            color: var(--nr-cyan);
        }

        [data-testid="stCaptionContainer"] {
            color: var(--nr-muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str) -> None:
    """Render a concise page header."""

    st.markdown(
        f"""
        <div class="nr-page-header">
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_heading(title: str, caption: str) -> None:
    """Render a reusable panel title block."""

    st.markdown(
        f"""
        <div class="nr-panel-heading">
            <h3>{escape(title)}</h3>
            <p>{escape(caption)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_note_box(title: str, body: str, tone: str = "cyan") -> None:
    """Render a high-color HTML note box."""

    st.markdown(
        f"""
        <div class="nr-note-box nr-note-{escape(tone)}">
            <h4>{escape(title)}</h4>
            <p>{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the dashboard visual system to Plotly figures."""

    fig.update_layout(
        title=title,
        title_font=dict(size=19, color="#EEF6FF", family="Sora, sans-serif"),
        font=dict(family="Manrope, sans-serif", color="#E8F3FF", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8, 17, 33, 0.02)",
        colorway=CHART_COLORS,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=18, r=18, t=66, b=18),
        hoverlabel=dict(
            bgcolor="rgba(9, 17, 34, 0.96)",
            bordercolor="rgba(255,255,255,0.16)",
            font=dict(color="#EEF6FF"),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig
