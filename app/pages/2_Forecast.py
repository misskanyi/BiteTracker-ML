"""BiteTracker — Forecast page: generate a meal-demand estimate for a school."""

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
from data_access import artifacts_ready, load_processed_data, load_final_model
from src.forecasting import forecast_school_demand

configure_page("Forecast", icon="🔮")
page_header("Forecast", subtitle="Generate a meal-demand estimate for a chosen school and horizon.", icon="🔮")

status = artifacts_ready()
if not (status["processed_data"] and status["final_model"]):
    st.error("Processed data or final model missing. Run `python run_pipeline.py` then `python run_tuning.py`.")
    st.stop()

df = load_processed_data()
model = load_final_model()

st.markdown(
    """
<div class="bite-card">
⚠️ Predictions below are <b>estimates</b>, not guarantees. Multi-day forecasts feed each
day's prediction back in to compute the next day's trend, so uncertainty compounds
the further out you look — treat a 30-day forecast as directional, not precise.
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

schools = df[["school_id", "school_name"]].drop_duplicates().sort_values("school_name")
school_options = dict(zip(schools["school_name"], schools["school_id"]))

col1, col2 = st.columns(2)
with col1:
    school_name = st.selectbox("Select school", list(school_options.keys()))
    school_id = school_options[school_name]
with col2:
    horizon_label = st.selectbox("Forecast horizon", ["Next school day", "Next 7 school days", "Next 30 calendar days (school days only)"])
    horizon_days = {"Next school day": 1, "Next 7 school days": 7, "Next 30 calendar days (school days only)": 30}[horizon_label]

history = df[df["school_id"] == school_id].sort_values(settings.DATE_COLUMN)
recent_attendance = history["students_present"].tail(14).mean()
recent_rainfall = history["rainfall_mm"].tail(14).mean()
recent_temp = history["temperature_c"].tail(14).mean()

st.markdown("### Assumptions for Day 1 of the Forecast")
st.caption(
    "Provide your best estimate, or leave as the recent 14-day average shown below "
    "(clearly labelled in the results either way)."
)

col3, col4, col5 = st.columns(3)
with col3:
    use_custom_attendance = st.checkbox("Provide expected attendance", value=False)
    expected_attendance = (
        st.number_input("Expected students present", min_value=0.0, value=float(round(recent_attendance)), step=1.0)
        if use_custom_attendance else None
    )
with col4:
    use_custom_rainfall = st.checkbox("Provide expected rainfall (mm)", value=False)
    expected_rainfall = (
        st.number_input("Expected rainfall (mm)", min_value=0.0, value=float(round(recent_rainfall, 1)), step=0.5)
        if use_custom_rainfall else None
    )
with col5:
    use_custom_temp = st.checkbox("Provide expected temperature (°C)", value=False)
    expected_temp = (
        st.number_input("Expected temperature (°C)", min_value=-10.0, value=float(round(recent_temp, 1)), step=0.5)
        if use_custom_temp else None
    )

if not use_custom_attendance:
    st.caption(f"Attendance assumption: recent 14-day average ≈ {recent_attendance:.0f} students present.")
if not use_custom_rainfall:
    st.caption(f"Rainfall assumption: recent 14-day average ≈ {recent_rainfall:.1f} mm.")
if not use_custom_temp:
    st.caption(f"Temperature assumption: recent 14-day average ≈ {recent_temp:.1f} °C.")

if st.button("Generate Forecast", type="primary"):
    with st.spinner("Generating forecast..."):
        forecast_df = forecast_school_demand(
            df, school_id, horizon_days,
            expected_attendance=expected_attendance,
            expected_rainfall_mm=expected_rainfall,
            expected_temperature_c=expected_temp,
            model=model,
        )

    st.markdown(f"### Forecast for {school_name}")
    display_df = forecast_df.copy()
    display_df["predicted_meals_served"] = display_df["predicted_meals_served"].round(0)
    st.dataframe(
        display_df.rename(columns={
            "date": "Date", "predicted_meals_served": "Predicted Meals Served (estimate)",
            "attendance_assumption": "Attendance Basis", "rainfall_assumption": "Rainfall Basis",
            "temperature_assumption": "Temperature Basis",
        }),
        use_container_width=True, hide_index=True,
    )

    recent_history = history.tail(21)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_history[settings.DATE_COLUMN], y=recent_history["meals_served"],
        mode="lines+markers", name="Recent Actual", line=dict(color="#2E7D32"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["predicted_meals_served"],
        mode="lines+markers", name="Forecast (estimate)", line=dict(color="#C9A227", dash="dash"),
    ))
    style_chart(fig, f"{school_name} — Recent Demand vs Forecast")
    fig.update_layout(xaxis_title="Date", yaxis_title="Meals Served")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Estimates only. Day 1 uses the assumptions above; subsequent days fall back to "
        "recent historical averages for attendance/weather since those cannot be known "
        "in advance — see the Basis columns above for exactly which assumption was used per day."
    )
