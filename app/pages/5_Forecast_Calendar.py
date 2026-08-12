"""BiteTracker — Forecast Calendar: date-based planning table with per-day risk status."""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_APP_DIR.parent))

import streamlit as st
from theme import configure_page, decision_support_card, page_header

from config import settings
from data_access import artifacts_ready, load_final_model, load_processed_data
from src.budgeting import estimate_meal_cost
from src.decision_support import build_decision_support_result, classify_risk, compute_safety_buffer
from src.forecasting import forecast_school_demand, get_school_history

configure_page("Forecast Calendar", icon="🗓️")
page_header(
    "Forecast Calendar",
    subtitle="Date-based planning view: predicted demand, preparation, and risk for upcoming school days.",
    icon="🗓️",
)

status = artifacts_ready()
if not (status["processed_data"] and status["final_model"]):
    st.error("Processed data or final model missing. Run `python run_pipeline.py` then `python run_tuning.py`.")
    st.stop()

df = load_processed_data()
model = load_final_model()

schools = df[["school_id", "school_name"]].drop_duplicates().sort_values("school_name")
school_options = dict(zip(schools["school_name"], schools["school_id"]))

col1, col2 = st.columns(2)
with col1:
    school_name = st.selectbox("School", list(school_options.keys()))
    school_id = school_options[school_name]
with col2:
    n_days = st.selectbox("Planning horizon", [14, 30], index=0)

history = get_school_history(df, school_id)
avg_cost_per_meal = float(history["average_cost_per_meal"].tail(14).mean()) if "average_cost_per_meal" in history else 0.0
recent_std = float(history["meals_served"].tail(14).std() or 0.0)

with st.spinner("Building forecast calendar..."):
    forecast_df = forecast_school_demand(df, school_id, horizon_days=n_days, model=model)

rows = []
for i, r in forecast_df.iterrows():
    horizon_days = i + 1
    buffer = compute_safety_buffer(r["predicted_meals_served"], recent_std, horizon_days=horizon_days)
    risk = classify_risk(buffer["buffer_pct"])
    recommended = round(r["predicted_meals_served"]) + buffer["buffer_meals"]
    cost = estimate_meal_cost(r["predicted_meals_served"], avg_cost_per_meal) if avg_cost_per_meal else {"daily_cost": 0.0}
    rows.append({
        "Date": r["date"], "Predicted Demand": round(r["predicted_meals_served"]),
        "Recommended Preparation": recommended, "Risk": risk,
        "Estimated Cost (KSh)": round(cost["daily_cost"]),
    })

import pandas as pd
calendar_df = pd.DataFrame(rows)

risk_icon = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
display_df = calendar_df.copy()
display_df["Risk"] = display_df["Risk"].map(lambda r: f"{risk_icon.get(r, '')} {r}")

st.markdown(f"### {school_name} — Next {n_days} School Days")
st.dataframe(display_df, use_container_width=True, hide_index=True)
st.caption(
    "Risk reflects forecast uncertainty (demand volatility + horizon), not a confirmed operational "
    "problem. See Scenario Simulator or Specific Date Prediction for a deeper look at any single day."
)

st.markdown("### Inspect a Specific Day")
selected_date = st.selectbox("Select a date", calendar_df["Date"].tolist())
selected_row = calendar_df[calendar_df["Date"] == selected_date].iloc[0]
selected_idx = calendar_df[calendar_df["Date"] == selected_date].index[0]
selected_forecast_row = forecast_df.iloc[selected_idx]

ds_result = build_decision_support_result(
    predicted_meals=selected_forecast_row["predicted_meals_served"],
    recent_std=recent_std,
    horizon_days=selected_idx + 1,
    horizon_label=f"{selected_date:%d %b %Y} ({selected_idx + 1} school day(s) ahead)",
    assumptions=[
        f"Attendance basis: {selected_forecast_row['attendance_assumption']}",
        f"Rainfall basis: {selected_forecast_row['rainfall_assumption']}",
        f"Temperature basis: {selected_forecast_row['temperature_assumption']}",
    ],
)
decision_support_card(ds_result)
