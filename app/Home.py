"""BiteTracker — Home page.

Run with: streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_APP_DIR.parent))

import streamlit as st
from theme import band, configure_page, hero, kpi_row, quote_block, step_row

from data_access import artifacts_ready

configure_page("Home", icon="🍽️")

hero(
    "BiteTracker",
    subtitle=(
        "Proactive School Meal Planning: <b>Time Series Forecasting</b> for Food Demand "
        "Prediction — BiteTracker is a machine-learning decision-support platform that predicts "
        "future school meal demand, so schools, county governments, NGOs, and donors can plan "
        "procurement and budgets <b>before</b> a shortage or a wasteful surplus happens, not after."
    ),
    icon="🍽️",
    badge_text="Synthetic Development Data",
)

# ---- Problem / Stakeholders, as alternating full-bleed bands (F4E layout) ----
band(
    """
    <h2>School feeding programmes routinely face two opposite failures.</h2>
    <p style="max-width:760px;font-size:1.05rem;">
    <b>Food waste</b> — preparing more meals than students who show up can eat, driven by
    rainfall, holidays, exams, and other hard-to-anticipate attendance swings.<br><br>
    <b>Meal shortages</b> — under-preparing on days attendance spikes (school events,
    post-holiday return), leaving students without a meal.<br><br>
    Both come from the same root cause: <b>meal preparation decisions are reactive</b>, based on
    yesterday's numbers or gut feel, not a forecast of tomorrow's actual demand.
    </p>
    """,
    variant="dark",
    eyebrow="🎯 Problem Statement",
)

band(
    """
    <h2>Built for the people who plan the plate.</h2>
    <ul style="font-size:1.02rem;max-width:760px;">
    <li><b>Schools</b> — kitchen/catering staff deciding how much to prepare each day.</li>
    <li><b>County governments</b> — allocating feeding-programme budgets across schools.</li>
    <li><b>NGOs &amp; development partners</b> — planning food supply and logistics.</li>
    <li><b>Donors</b> — evaluating programme efficiency and impact.</li>
    </ul>
    """,
    variant="gold",
    eyebrow="🤝 Key Stakeholders",
)

# ---- System purpose, as a Design > Operate > Replicate-style step row ----
st.markdown("### ⚙️ System Purpose")
st.markdown(
    """
BiteTracker compares four regression approaches — <b>Linear Regression, Decision Tree,
Random Forest, and XGBoost</b> — trained on historical attendance, weather, and calendar
data, selects the best-performing model by measured test-set error (MAE/RMSE/R², not
assumption), and uses it to forecast next-day, next-7-day, and next-30-day meal demand
per school.
""",
    unsafe_allow_html=True,
)

step_row(
    [
        ("Predict", "Forecast next-day, next-7-day, and next-30-day meal demand per school from the winning model."),
        ("Plan", "Turn a demand forecast into a recommended prep quantity, a safety buffer, and a risk level."),
        ("Compare & Allocate", "Compare demand, attendance, waste, and cost across schools, then allocate budget and resources."),
    ]
)

st.markdown(
    """
Explore the pages in the sidebar:

- <b>Dashboard</b> — current state at a glance.
- <b>Forecast</b> — generate a demand estimate for a chosen school and horizon.
- <b>Specific Date Prediction</b> — an estimate for one exact future date.
- <b>Scenario Simulator</b> — compare a baseline forecast against a what-if scenario.
- <b>Forecast Calendar</b> — a date-based planning table with per-day risk status.
- <b>School Comparison</b> — compare demand, attendance, waste, and cost across schools.
- <b>Resource Allocation</b> — forecast demand vs planned capacity, per school.
- <b>Budget Planning</b> — translate demand forecasts into estimated procurement cost.
- <b>Upload School Data</b> — bring your own historical dataset for validation.
- <b>Data Explorer</b> — browse, filter, and download the underlying dataset.
- <b>Model Performance</b> — how the four models compare, and why the winner was selected.
- <b>About</b> — methodology, ethics, and limitations.
""",
    unsafe_allow_html=True,
)

quote_block(
    "This is a decision-support estimate, not a guaranteed outcome — use it to inform meal "
    "preparation planning alongside local judgement.",
    author="BiteTracker",
    role="Decision Support Framing",
)

st.divider()
st.markdown("### 🩺 Pipeline Status")

status = artifacts_ready()
labels = {
    "processed_data": "Processed Dataset",
    "baseline_metrics": "Baseline Models",
    "tuned_metrics": "Tuned Models",
    "final_model": "Final Best Model",
    "final_model_metadata": "Final Model Metadata",
}
kpi_row([(labels[key], "Ready ✅" if ready else "Missing ❌", None) for key, ready in status.items()])

if not all(status.values()):
    st.warning(
        "Some pipeline artifacts are missing. Run, in order: "
        "`python run_pipeline.py`, `python run_training.py`, `python run_tuning.py`."
    )

st.caption(
    "All figures in this app are computed from a SYNTHETIC dataset generated for development "
    "(see About page) — not real school records."
)
