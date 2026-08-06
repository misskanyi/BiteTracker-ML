"""Shared Streamlit page configuration, styling, and layout helpers for
BiteTracker's green/white/gold visual identity.

Every page calls `configure_page()` first (sets page config + injects CSS),
then uses `page_header()` for a consistent title/subtitle/badge block and
`kpi_row()`/`kpi_card()` for styled metric tiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Allow `from config...` / `from src...` imports when Streamlit runs a page
# directly (its CWD/sys.path handling differs from a normal script run).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Brand palette -----------------------------------------------------------
GREEN = "#2E7D32"
GREEN_DARK = "#1B5E20"
GREEN_LIGHT = "#E8F3E9"
GOLD = "#C9A227"
GOLD_LIGHT = "#F6EFD8"
CREAM = "#FBF8F1"
WHITE = "#FFFFFF"
INK = "#1B2B1E"
BORDER = "#E0DACD"

# Plotly template shared by every chart in the app, so colours/fonts match
# the rest of the UI instead of Plotly's default blue/orange palette.
PLOTLY_COLORWAY = [GREEN, GOLD, "#5C8A5F", "#B98900", "#8FBF91", "#8C6D1F"]

CUSTOM_CSS = f"""
<style>
    /* ---- Base layout & typography ---- */
    .stApp {{
        background-color: {CREAM};
    }}
    html, body, [class*="css"] {{
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}
    h1, h2, h3, h4 {{
        color: {GREEN_DARK};
        font-weight: 700;
    }}
    p, li, label {{
        color: {INK};
    }}
    hr {{
        border-color: {BORDER};
    }}

    /* ---- Sidebar navigation ---- */
    [data-testid="stSidebar"] {{
        background-color: {GREEN_DARK};
    }}
    [data-testid="stSidebar"] * {{
        color: #F5F0E1 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        font-weight: 500;
    }}
    [data-testid="stSidebarNav"] a {{
        border-radius: 6px;
        margin: 2px 6px;
    }}
    [data-testid="stSidebarNav"] a:hover {{
        background-color: rgba(255, 255, 255, 0.08);
    }}

    /* ---- Badges ---- */
    .bite-badge {{
        display: inline-block;
        background-color: {GOLD};
        color: {GREEN_DARK};
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}

    /* ---- Page header block ---- */
    .bite-header {{
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid {GOLD};
    }}
    .bite-header h1 {{
        margin-bottom: 0.15rem;
        font-size: 2rem;
    }}
    .bite-header .bite-subtitle {{
        color: #4A5A4C;
        font-size: 1.02rem;
        margin-top: 0;
    }}

    /* ---- Generic content card ---- */
    .bite-card {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-left: 4px solid {GREEN};
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}

    /* ---- KPI metric tiles ---- */
    div[data-testid="stMetric"] {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-top: 3px solid {GOLD};
        border-radius: 10px;
        padding: 1rem 1rem 0.85rem 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}
    div[data-testid="stMetric"] label {{
        color: #4A5A4C !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.03em;
    }}
    div[data-testid="stMetricValue"] {{
        color: {GREEN_DARK} !important;
        font-weight: 700;
    }}

    /* ---- Tables / dataframes ---- */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* ---- Tabs ---- */
    button[data-baseweb="tab"] {{
        font-weight: 600;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {GREEN_DARK} !important;
        border-bottom-color: {GOLD} !important;
    }}

    /* ---- Buttons ---- */
    .stButton > button, .stDownloadButton > button {{
        background-color: {GREEN};
        color: {WHITE};
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {GREEN_DARK};
        color: {WHITE};
    }}

    /* ---- Alerts (info/success/warning) keep readable on cream bg ---- */
    div[data-testid="stAlertContainer"] {{
        border-radius: 8px;
    }}
</style>
"""


def configure_page(title: str, icon: str = "🍽️") -> None:
    """Set Streamlit page config and inject the shared CSS. Call first, before any other st.* call."""
    st.set_page_config(page_title=f"BiteTracker — {title}", page_icon=icon, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = "", show_badge: bool = True) -> None:
    """Render a consistent page title + subtitle + synthetic-data badge block."""
    badge_html = '<span class="bite-badge">Synthetic Development Data</span>' if show_badge else ""
    subtitle_html = f'<p class="bite-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="bite-header">
            {badge_html}
            <h1>{icon} {title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def synthetic_data_badge() -> None:
    """Standalone badge for pages that render their own header (kept for backward compatibility)."""
    st.markdown('<span class="bite-badge">Synthetic Development Data</span>', unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of KPI metrics. Each item is (label, value, help_text_or_None)."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def style_chart(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply BiteTracker's shared Plotly styling (colourway, font, title placement)."""
    fig.update_layout(
        template="plotly_white",
        colorway=PLOTLY_COLORWAY,
        font=dict(family="Segoe UI, Helvetica Neue, Arial, sans-serif", color=INK, size=13),
        title=title or fig.layout.title.text,
        title_x=0.02,
        title_font=dict(color=GREEN_DARK, size=17),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=70, l=50, r=30, b=50),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
    )
    return fig
