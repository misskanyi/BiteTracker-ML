"""BiteTracker — Scenario Simulator: compare a baseline forecast against a what-if scenario."""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_APP_DIR.parent))

import plotly.graph_objects as go
import streamlit as st
from theme import configure_page, page_header, style_chart

from config import settings
from data_access import artifacts_ready, load_final_model, load_processed_data
from src.forecasting import _next_school_days, forecast_school_demand, get_school_history, predict_specific_date
from src.synthetic_data import _is_exam_period, _is_holiday, _school_term

configure_page("Scenario Simulator", icon="🧪")
page_header(
    "What-If Scenario Simulator",
    subtitle="Compare the baseline forecast against an alternative planning scenario for one school.",
    icon="🧪",
)

status = artifacts_ready()
if not (status["processed_data"] and status["final_model"]):
    st.error("Processed data or final model missing. Run `python run_pipeline.py` then `python run_tuning.py`.")
    st.stop()

df = load_processed_data()
model = load_final_model()

schools = df[["school_id", "school_name"]].drop_duplicates().sort_values("school_name")
school_options = dict(zip(schools["school_name"], schools["school_id"]))

school_name = st.selectbox("School", list(school_options.keys()))
school_id = school_options[school_name]

history = get_school_history(df, school_id)
last_date = history[settings.DATE_COLUMN].max().date()
next_day = _next_school_days(last_date, 1)[0]

recent_attendance = float(history["students_present"].tail(14).mean())
recent_rainfall = float(history["rainfall_mm"].tail(14).mean())
recent_temp = float(history["temperature_c"].tail(14).mean())

baseline_term = _school_term(next_day)
baseline_holiday = _is_holiday(next_day)
baseline_exam = (not baseline_holiday) and _is_exam_period(next_day, baseline_term)

# --- Baseline forecast (recent-average assumptions, calendar-derived context) ---
baseline_forecast = forecast_school_demand(df, school_id, horizon_days=1, model=model)
baseline_meals = float(baseline_forecast["predicted_meals_served"].iloc[0])

st.markdown(f"### Baseline Forecast — {next_day:%d %b %Y}")
st.metric("Baseline Forecast", f"{baseline_meals:.0f} meals", help="Next school day, using recent 14-day averages and calendar-derived term/holiday/exam status.")

st.markdown("### Build a Scenario")
st.caption("Adjust conditions relative to the baseline above; leave a control unchanged to keep the baseline value.")

col1, col2, col3 = st.columns(3)
with col1:
    attendance_change_pct = st.slider("Expected attendance change (%)", -50, 50, 0, step=5)
with col2:
    scenario_rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=round(recent_rainfall, 1), step=1.0)
with col3:
    scenario_temp = st.number_input("Temperature (°C)", min_value=-10.0, value=round(recent_temp, 1), step=0.5)

col4, col5, col6 = st.columns(3)
with col4:
    scenario_holiday = st.checkbox("Holiday", value=baseline_holiday)
with col5:
    scenario_exam = st.checkbox("Exam period", value=baseline_exam, disabled=scenario_holiday)
with col6:
    scenario_event = st.checkbox("School event", value=False)

scenario_attendance = max(0.0, recent_attendance * (1 + attendance_change_pct / 100))
st.caption(f"Scenario attendance: {scenario_attendance:.0f} students present ({attendance_change_pct:+d}% vs recent average of {recent_attendance:.0f}).")

col_a, col_b = st.columns([1, 5])
with col_a:
    run_scenario = st.button("Run Scenario", type="primary")
with col_b:
    reset = st.button("Reset Scenario")

if reset:
    st.rerun()

if run_scenario:
    scenario_result = predict_specific_date(
        df, school_id, next_day,
        expected_attendance=scenario_attendance,
        expected_rainfall_mm=scenario_rainfall,
        expected_temperature_c=scenario_temp,
        school_term=baseline_term,
        is_holiday=scenario_holiday,
        is_exam_period=scenario_exam and not scenario_holiday,
        is_school_event=scenario_event,
        model=model,
    )
    scenario_meals = scenario_result["predicted_meals_served"]
    difference = scenario_meals - baseline_meals

    st.markdown("### Baseline vs Scenario")
    kcol1, kcol2, kcol3 = st.columns(3)
    kcol1.metric("Baseline Forecast", f"{baseline_meals:.0f} meals")
    kcol2.metric("Scenario Forecast", f"{scenario_meals:.0f} meals")
    kcol3.metric("Difference", f"{difference:+.0f} meals", delta=f"{difference:+.0f}")

    direction = "increases" if difference > 0 else "decreases" if difference < 0 else "stays roughly the same"
    st.info(
        f"Under this scenario, predicted demand {direction} by approximately **{abs(difference):.0f} meals** "
        f"compared to the baseline forecast for {next_day:%d %b %Y}. This reflects the model's learned "
        "association between these conditions and past demand — it is not a claim that any single input "
        "*caused* the change on its own."
    )

    fig = go.Figure(go.Bar(
        x=["Baseline", "Scenario"], y=[baseline_meals, scenario_meals],
        marker_color=["#2E7D32", "#C9A227"], text=[f"{baseline_meals:.0f}", f"{scenario_meals:.0f}"],
        textposition="outside",
    ))
    style_chart(fig, f"{school_name} — Baseline vs Scenario ({next_day:%d %b %Y})")
    fig.update_layout(yaxis_title="Meals Served")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Scenario forecast is an estimate for decision-support purposes, not a guaranteed outcome."
    )
