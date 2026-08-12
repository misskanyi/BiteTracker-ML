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
from theme import configure_page, page_header

from data_access import artifacts_ready

configure_page("Home", icon="🍽️")
page_header(
    "BiteTracker",
    subtitle="Proactive School Meal Planning: Time Series Forecasting for Food Demand Prediction",
    icon="🍽️",
)

st.markdown(
    """
BiteTracker is a machine-learning decision-support platform that predicts **future school meal
demand**, so schools, county governments, NGOs, and donors can plan procurement and budgets
*before* a shortage or a wasteful surplus happens — not after.
"""
)

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("#### 🎯 Problem Statement")
    st.markdown(
        """
<div class="bite-card">
School feeding programmes routinely face two opposite failures:

- <b>Food waste</b> — preparing more meals than students who show up can eat, driven by
  rainfall, holidays, exams, and other hard-to-anticipate attendance swings.
- <b>Meal shortages</b> — under-preparing on days attendance spikes (school events,
  post-holiday return), leaving students without a meal.

Both come from the same root cause: <b>meal preparation decisions are reactive</b>, based on
yesterday's numbers or gut feel, not a forecast of tomorrow's actual demand.
</div>
""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("#### 🤝 Key Stakeholders")
    st.markdown(
        """
<div class="bite-card">

- <b>Schools</b> — kitchen/catering staff deciding how much to prepare each day.
- <b>County governments</b> — allocating feeding-programme budgets across schools.
- <b>NGOs & development partners</b> — planning food supply and logistics.
- <b>Donors</b> — evaluating programme efficiency and impact.

</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("#### ⚙️ System Purpose")
st.markdown(
    """
<div class="bite-card">
BiteTracker compares four regression approaches — <b>Linear Regression, Decision Tree,
Random Forest, and XGBoost</b> — trained on historical attendance, weather, and calendar
data, selects the best-performing model by measured test-set error (MAE/RMSE/R², not
assumption), and uses it to forecast next-day, next-7-day, and next-30-day meal demand
per school. Explore the pages in the sidebar:

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
</div>
""",
    unsafe_allow_html=True,
)

st.divider()
st.markdown("#### 🩺 Pipeline Status")

status = artifacts_ready()
cols = st.columns(len(status))
labels = {
    "processed_data": "Processed Dataset",
    "baseline_metrics": "Baseline Models",
    "tuned_metrics": "Tuned Models",
    "final_model": "Final Best Model",
    "final_model_metadata": "Final Model Metadata",
}
for col, (key, ready) in zip(cols, status.items()):
    col.metric(labels[key], "Ready ✅" if ready else "Missing ❌")

if not all(status.values()):
    st.warning(
        "Some pipeline artifacts are missing. Run, in order: "
        "`python run_pipeline.py`, `python run_training.py`, `python run_tuning.py`."
    )

st.caption(
    "All figures in this app are computed from a SYNTHETIC dataset generated for development "
    "(see About page) — not real school records."
)
