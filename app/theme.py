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
# Warm, organic palette inspired by Food4Education's editorial identity:
# cream backdrop, deep forest green, and a citrus gold/orange accent.
GREEN = "#2E7D32"
GREEN_DARK = "#1B5E20"
GREEN_LIGHT = "#E8F3E9"
GOLD = "#C9A227"
GOLD_LIGHT = "#F6EFD8"
ORANGE = "#E8973A"
CREAM = "#F3EFDD"
CREAM_DARK = "#EAE4C9"
WHITE = "#FFFFFF"
INK = "#1B2B1E"
BORDER = "#E0DACD"

# Plotly template shared by every chart in the app, so colours/fonts match
# the rest of the UI instead of Plotly's default blue/orange palette.
PLOTLY_COLORWAY = [GREEN, GOLD, ORANGE, "#5C8A5F", "#8FBF91", "#8C6D1F"]

CUSTOM_CSS = f"""
<style>
    @keyframes bite-fade-up {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes bite-fade-in {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    @keyframes bite-blob-float {{
        0%, 100% {{ transform: translate(0, 0) rotate(0deg); }}
        50% {{ transform: translate(6px, -10px) rotate(4deg); }}
    }}
    @keyframes bite-pulse {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(233, 151, 58, 0.35); }}
        70% {{ box-shadow: 0 0 0 8px rgba(233, 151, 58, 0); }}
    }}

    /* ---- Base layout & typography ---- */
    .stApp {{
        background:
            radial-gradient(circle at 92% -4%, {GOLD_LIGHT} 0%, transparent 38%),
            radial-gradient(circle at -6% 8%, {GREEN_LIGHT} 0%, transparent 32%),
            {CREAM};
    }}
    html, body, [class*="css"] {{
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
        animation: bite-fade-in 0.5s ease both;
    }}
    h1, h2, h3, h4 {{
        color: {GREEN_DARK};
        font-weight: 800;
        letter-spacing: -0.01em;
    }}
    p, li, label {{
        color: {INK};
    }}
    hr {{
        border-color: {BORDER};
    }}

    /* ---- Sidebar navigation ---- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {GREEN_DARK} 0%, #16491a 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: #F5F0E1 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        font-weight: 500;
    }}
    [data-testid="stSidebarNav"] a {{
        border-radius: 999px;
        margin: 2px 10px;
        transition: background-color 0.18s ease, transform 0.18s ease;
    }}
    [data-testid="stSidebarNav"] a:hover {{
        background-color: rgba(255, 255, 255, 0.1);
        transform: translateX(3px);
    }}

    /* ---- Badges ---- */
    .bite-badge {{
        display: inline-block;
        background-color: {ORANGE};
        color: {WHITE};
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 12px;
        animation: bite-fade-up 0.45s ease both;
    }}

    /* ---- Page header block (organic blob accent, like Food4Education) ---- */
    .bite-header {{
        position: relative;
        padding: 1.6rem 1.8rem 1.4rem 1.8rem;
        margin-bottom: 1.4rem;
        background: linear-gradient(135deg, {WHITE} 0%, {GOLD_LIGHT} 130%);
        border-radius: 28px 60px 28px 60px / 60px 28px 60px 28px;
        border: 1px solid {BORDER};
        box-shadow: 0 6px 18px rgba(27, 43, 30, 0.06);
        overflow: hidden;
        animation: bite-fade-up 0.5s ease both;
    }}
    .bite-header::after {{
        content: "";
        position: absolute;
        top: -30px;
        right: -30px;
        width: 110px;
        height: 110px;
        background: {GOLD};
        opacity: 0.16;
        border-radius: 42% 58% 63% 37% / 41% 44% 56% 59%;
        animation: bite-blob-float 7s ease-in-out infinite;
    }}
    .bite-header h1 {{
        margin-bottom: 0.15rem;
        font-size: 2.2rem;
        position: relative;
    }}
    .bite-header .bite-subtitle {{
        color: #4A5A4C;
        font-size: 1.04rem;
        margin-top: 0;
        position: relative;
    }}

    /* ---- Generic content card ---- */
    .bite-card {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-left: 4px solid {GREEN};
        border-radius: 10px 18px 10px 18px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: bite-fade-up 0.5s ease both;
    }}
    .bite-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 22px rgba(27, 43, 30, 0.1);
    }}

    /* ---- KPI metric tiles ---- */
    div[data-testid="stMetric"] {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-top: 3px solid {GOLD};
        border-radius: 10px 18px 10px 18px;
        padding: 1rem 1rem 0.85rem 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-top-color 0.2s ease;
        animation: bite-fade-up 0.45s ease both;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(27, 43, 30, 0.12);
        border-top-color: {ORANGE};
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
        font-weight: 800;
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
        transition: color 0.15s ease;
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
        border-radius: 999px;
        font-weight: 700;
        padding: 0.55rem 1.6rem;
        transition: background-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {GREEN_DARK};
        color: {WHITE};
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(46, 125, 50, 0.28);
    }}
    .stButton > button:active, .stDownloadButton > button:active {{
        transform: translateY(0);
    }}

    /* ---- Alerts (info/success/warning) keep readable on cream bg ---- */
    div[data-testid="stAlertContainer"] {{
        border-radius: 12px;
        animation: bite-fade-up 0.4s ease both;
    }}

    /* ---- Full-bleed colour bands (breaks out of the centred column,
       like Food4Education's alternating dark-green / gold sections) ---- */
    .bite-band {{
        position: relative;
        width: 100vw;
        margin-left: calc(50% - 50vw);
        margin-top: 1.6rem;
        margin-bottom: 1.6rem;
        padding: 2.6rem calc(50vw - 50% + 1.6rem);
        overflow: hidden;
        animation: bite-fade-up 0.55s ease both;
    }}
    .bite-band-dark {{ background: linear-gradient(120deg, {GREEN_DARK} 0%, #123c15 100%); color: {CREAM}; }}
    .bite-band-dark h2, .bite-band-dark h3, .bite-band-dark p, .bite-band-dark li {{ color: {CREAM} !important; }}
    .bite-band-gold {{ background: linear-gradient(120deg, {GOLD} 0%, {ORANGE} 100%); color: {GREEN_DARK}; }}
    .bite-band-gold h2, .bite-band-gold h3 {{ color: {GREEN_DARK} !important; }}
    .bite-band-cream {{ background: {CREAM_DARK}; color: {INK}; }}
    .bite-band .bite-blob {{
        position: absolute;
        border-radius: 42% 58% 63% 37% / 41% 44% 56% 59%;
        opacity: 0.18;
        animation: bite-blob-float 8s ease-in-out infinite;
    }}
    .bite-band-dark .bite-blob {{ background: {GOLD}; }}
    .bite-band-gold .bite-blob {{ background: {GREEN_DARK}; opacity: 0.12; }}
    .bite-band-content {{ position: relative; max-width: 1200px; margin: 0 auto; }}
    .bite-band h2 {{ font-size: 1.9rem; margin-bottom: 0.6rem; }}
    .bite-band-eyebrow {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        opacity: 0.85;
    }}

    /* ---- Hero (Home page splash, mirrors Food4Education's opening
       "Hungry kids can't learn or grow." headline treatment) ---- */
    .bite-hero {{
        position: relative;
        padding: 3rem 2.4rem;
        margin-bottom: 1.8rem;
        border-radius: 32px 72px 32px 72px / 72px 32px 72px 32px;
        background: linear-gradient(135deg, {WHITE} 0%, {GOLD_LIGHT} 75%, {GOLD_LIGHT} 100%);
        border: 1px solid {BORDER};
        box-shadow: 0 12px 30px rgba(27, 43, 30, 0.08);
        overflow: hidden;
        animation: bite-fade-up 0.6s ease both;
    }}
    .bite-hero::before {{
        content: "";
        position: absolute;
        top: -50px;
        right: -40px;
        width: 190px;
        height: 190px;
        background: {ORANGE};
        opacity: 0.16;
        border-radius: 42% 58% 63% 37% / 41% 44% 56% 59%;
        animation: bite-blob-float 9s ease-in-out infinite;
    }}
    .bite-hero::after {{
        content: "";
        position: absolute;
        bottom: -60px;
        left: -50px;
        width: 160px;
        height: 160px;
        background: {GREEN};
        opacity: 0.12;
        border-radius: 58% 42% 37% 63% / 44% 41% 59% 56%;
        animation: bite-blob-float 11s ease-in-out infinite reverse;
    }}
    .bite-hero-title {{
        position: relative;
        font-size: 3.1rem;
        line-height: 1.05;
        font-weight: 900;
        color: {GREEN_DARK};
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.02em;
    }}
    .bite-hero-title .accent {{ color: {ORANGE}; }}
    .bite-hero-subtitle {{
        position: relative;
        font-size: 1.15rem;
        color: #3E4E40;
        max-width: 720px;
        margin: 0;
    }}

    /* ---- Big stat strip (Food4Education-style giant orange numbers) ---- */
    .bite-stat-strip {{
        display: flex;
        flex-wrap: wrap;
        gap: 1.6rem;
        margin: 1.2rem 0;
    }}
    .bite-stat {{
        flex: 1 1 150px;
        text-align: center;
        animation: bite-fade-up 0.5s ease both;
    }}
    .bite-stat-number {{
        font-size: 2.6rem;
        font-weight: 900;
        color: {ORANGE};
        line-height: 1;
    }}
    .bite-stat-label {{
        font-size: 0.85rem;
        font-weight: 600;
        color: inherit;
        opacity: 0.85;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}

    /* ---- Pull-quote block ---- */
    .bite-quote {{
        position: relative;
        padding: 1.6rem 2.2rem;
        margin: 1.4rem 0;
        background: {WHITE};
        border-radius: 18px;
        border: 1px solid {BORDER};
        box-shadow: 0 4px 14px rgba(27, 43, 30, 0.05);
        animation: bite-fade-up 0.55s ease both;
    }}
    .bite-quote .bite-quote-mark {{
        font-size: 3rem;
        color: {ORANGE};
        font-weight: 900;
        line-height: 0.4;
        display: block;
        margin-bottom: 0.6rem;
    }}
    .bite-quote p {{
        font-size: 1.12rem;
        font-style: italic;
        color: {GREEN_DARK};
        margin-bottom: 0.6rem;
    }}
    .bite-quote .bite-quote-author {{
        font-weight: 700;
        color: {INK};
        opacity: 0.85;
        font-size: 0.9rem;
    }}

    /* ---- Step / process pills (Design > Operate > Replicate style) ---- */
    .bite-step-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1rem 0;
    }}
    .bite-step {{
        flex: 1 1 220px;
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: bite-fade-up 0.5s ease both;
    }}
    .bite-step:hover {{ transform: translateY(-4px); box-shadow: 0 12px 24px rgba(27, 43, 30, 0.1); }}
    .bite-step-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: {GREEN_DARK};
        color: {WHITE};
        font-weight: 800;
        margin-bottom: 0.6rem;
    }}
</style>
"""


def configure_page(title: str, icon: str = "🍽️") -> None:
    """Set Streamlit page config and inject the shared CSS. Call first, before any other st.* call."""
    st.set_page_config(page_title=f"BiteTracker — {title}", page_icon=icon, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = "", show_badge: bool = True, badge_text: str = "Synthetic Development Data") -> None:
    """Render a consistent page title + subtitle + dataset-source badge block.

    `badge_text` labels which dataset is active. Pages that can operate on
    either the synthetic dataset or a user upload should pass "User-Uploaded
    Data" once an upload is selected, so the two are never mixed silently.
    """
    badge_html = f'<span class="bite-badge">{badge_text}</span>' if show_badge else ""
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


def hero(title: str, subtitle: str = "", icon: str = "", badge_text: str = "") -> None:
    """Big Food4Education-style splash hero for a landing page (Home).

    Larger and more decorative than `page_header` — use once per app, at the
    very top of Home.py.
    """
    badge_html = f'<span class="bite-badge">{badge_text}</span><br>' if badge_text else ""
    subtitle_html = f'<p class="bite-hero-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="bite-hero">
            {badge_html}
            <h1 class="bite-hero-title">{icon} {title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def band(content_html: str, variant: str = "dark", eyebrow: str = "") -> None:
    """Full-bleed colour section, breaking out of the centred column.

    `variant` is one of "dark" (deep green), "gold", or "cream". `content_html`
    is raw HTML rendered inside the band (headings/paragraphs/lists) — build it
    with normal markdown-to-html the same way `bite-card` blocks already are,
    since Streamlit widgets can't be nested inside an injected HTML div.
    """
    eyebrow_html = f'<span class="bite-band-eyebrow">{eyebrow}</span>' if eyebrow else ""
    markup = f"""
        <div class="bite-band bite-band-{variant}">
            <div class="bite-blob" style="width:160px;height:160px;top:-40px;right:6%;"></div>
            <div class="bite-blob" style="width:110px;height:110px;bottom:-30px;left:4%;"></div>
            <div class="bite-band-content">
                {eyebrow_html}
                {content_html}
            </div>
        </div>
        """
    # st.markdown parses Markdown first: a blank line ends the HTML block, and
    # any line after it indented 4+ spaces renders as a code block (the stray
    # "</div>" boxes). Unindent every line and drop blank ones so the whole
    # band stays a single HTML block.
    markup = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(markup, unsafe_allow_html=True)


def stat_strip(items: list[tuple[str, str]]) -> None:
    """Row of big bold numbers with labels underneath (e.g. F4E's "5,000+ employees").

    Each item is (number_text, label). Purely decorative HTML — for metrics
    that need Streamlit's built-in help tooltips/formatting, use `kpi_row` instead.
    """
    stats_html = "".join(
        f'<div class="bite-stat"><div class="bite-stat-number">{number}</div>'
        f'<div class="bite-stat-label">{label}</div></div>'
        for number, label in items
    )
    st.markdown(f'<div class="bite-stat-strip">{stats_html}</div>', unsafe_allow_html=True)


def quote_block(text: str, author: str = "", role: str = "") -> None:
    """Pull-quote card with a large decorative quotation mark."""
    byline = f"{author} — {role}" if author and role else author or role
    byline_html = f'<div class="bite-quote-author">{byline}</div>' if byline else ""
    st.markdown(
        f"""
        <div class="bite-quote">
            <span class="bite-quote-mark">&ldquo;</span>
            <p>{text}</p>
            {byline_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_row(steps: list[tuple[str, str]]) -> None:
    """Numbered step cards in a row (e.g. F4E's Design / Operate / Replicate)."""
    steps_html = "".join(
        f'<div class="bite-step"><span class="bite-step-num">{i}</span>'
        f"<h4 style='margin:0 0 0.4rem 0;'>{title}</h4><p style='margin:0;font-size:0.92rem;'>{desc}</p></div>"
        for i, (title, desc) in enumerate(steps, start=1)
    )
    st.markdown(f'<div class="bite-step-row">{steps_html}</div>', unsafe_allow_html=True)


def synthetic_data_badge() -> None:
    """Standalone badge for pages that render their own header (kept for backward compatibility)."""
    st.markdown('<span class="bite-badge">Synthetic Development Data</span>', unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of KPI metrics. Each item is (label, value, help_text_or_None)."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def quality_score_badge(score: int) -> str:
    """Colour-coded label for a 0-100 data quality score."""
    if score >= 80:
        return f"🟢 {score}/100 — Good"
    if score >= 50:
        return f"🟡 {score}/100 — Fair"
    return f"🔴 {score}/100 — Poor"


def empty_state(message: str) -> None:
    """Consistent placeholder for pages/sections with no data yet."""
    st.markdown(f'<div class="bite-card" style="text-align:center;color:{INK};opacity:0.7;">{message}</div>', unsafe_allow_html=True)


_RISK_COLORS = {"Low": GREEN, "Moderate": GOLD, "High": "#B3261E"}


def decision_support_card(result) -> None:
    """Render a `src.decision_support.DecisionSupportResult` as a consistent result block.

    Shared by every page that produces a single-point prediction (Forecast,
    Specific Date Prediction, Scenario Simulator) so the decision-support
    framing — predicted vs recommended vs buffer vs risk — never diverges
    between pages.
    """
    risk_color = _RISK_COLORS.get(result.risk_level, INK)
    cols = st.columns(4)
    cols[0].metric("Predicted Demand", f"{result.predicted_meals:.0f} meals")
    cols[1].metric("Recommended Preparation", f"{result.recommended_meals:.0f} meals")
    cols[2].metric("Safety Buffer", f"{result.buffer_meals:.0f} meals ({result.buffer_pct:.1f}%)")
    cols[3].markdown(
        f'<div style="padding-top:0.6rem;"><span style="color:{INK};opacity:0.7;font-size:0.8rem;">Risk Level</span>'
        f'<br><span style="color:{risk_color};font-size:1.4rem;font-weight:700;">{result.risk_level}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="bite-card">
<b>Forecast horizon:</b> {result.horizon_label} &nbsp;·&nbsp; <b>Model:</b> {result.model_label}<br>
<b>Uncertainty:</b> {result.uncertainty}
</div>
""",
        unsafe_allow_html=True,
    )

    if result.assumptions:
        with st.expander("Assumptions used in this prediction"):
            for a in result.assumptions:
                st.markdown(f"- {a}")

    for w in result.warnings:
        st.warning(w)

    st.caption(
        "This is a decision-support estimate, not a guaranteed outcome. Use it to inform meal "
        "preparation planning alongside local judgement."
    )


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
