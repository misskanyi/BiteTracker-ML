"""BiteTracker — Specific Date Prediction: meal-demand estimate for one chosen future date."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_APP_DIR.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from theme import configure_page, decision_support_card, page_header, style_chart

from config import settings
from data_access import artifacts_ready, load_final_model, load_processed_data
from src.decision_support import build_decision_support_result
from src.feature_engineering import FEATURE_COLUMNS
from src.forecasting import predict_specific_date

configure_page("Specific Date Prediction", icon="📅")
page_header(
    "Specific Date Prediction",
    subtitle="Estimate meal demand for one school on one exact future date.",
    icon="📅",
)

status = artifacts_ready()
if not (status["processed_data"] and status["final_model"]):
    st.error("Processed data or final model missing. Run `python run_pipeline.py` then `python run_tuning.py`.")
    st.stop()

df = load_processed_data()
model = load_final_model()

schools = df[["school_id", "school_name"]].drop_duplicates().sort_values("school_name")
school_options = dict(zip(schools["school_name"], schools["school_id"]))

st.markdown("### 1. Choose School and Date")
col1, col2 = st.columns(2)
with col1:
    school_name = st.selectbox("School", list(school_options.keys()))
    school_id = school_options[school_name]
with col2:
    history = df[df["school_id"] == school_id].sort_values(settings.DATE_COLUMN)
    last_date = history[settings.DATE_COLUMN].max().date()
    target_date = st.date_input(
        "Forecast date",
        value=last_date + timedelta(days=7),
        min_value=last_date + timedelta(days=1),
    )

if target_date.weekday() >= 5:
    st.warning("Selected date is a weekend — the pipeline only models school days (Mon–Fri). Predictions may not be meaningful.")

recent_attendance = history["students_present"].tail(14).mean()
recent_rainfall = history["rainfall_mm"].tail(14).mean()
recent_temp = history["temperature_c"].tail(14).mean()

st.markdown("### 2. Expected Conditions")
st.caption(
    "🟢 auto-derived from history/date · 🔵 you provide · 🟡 assumed default if left unchanged"
)

col3, col4, col5 = st.columns(3)
with col3:
    expected_attendance = st.number_input(
        "🔵 Expected attendance (students present)", min_value=0.0,
        value=float(round(recent_attendance)), step=1.0,
    )
with col4:
    expected_rainfall = st.number_input(
        "🔵 Expected rainfall (mm)", min_value=0.0, value=float(round(recent_rainfall, 1)), step=0.5,
    )
with col5:
    expected_temp = st.number_input(
        "🔵 Expected temperature (°C)", min_value=-10.0, value=float(round(recent_temp, 1)), step=0.5,
    )

col6, col7, col8, col9 = st.columns(4)
with col6:
    school_term = st.selectbox("🔵 School term", ["Term 1", "Term 2", "Term 3"])
with col7:
    is_holiday = st.checkbox("🔵 Holiday", value=False)
with col8:
    is_exam_period = st.checkbox("🔵 Exam period", value=False, disabled=is_holiday)
with col9:
    is_school_event = st.checkbox("🔵 School event", value=False)

st.caption(
    f"🟢 Auto-derived: day of week = **{target_date.strftime('%A')}**, month = **{target_date.strftime('%B')}**, "
    f"and lag/rolling demand history computed from this school's own past records."
)

if st.button("Generate Prediction", type="primary"):
    try:
        result = predict_specific_date(
            df, school_id, target_date,
            expected_attendance=expected_attendance,
            expected_rainfall_mm=expected_rainfall,
            expected_temperature_c=expected_temp,
            school_term=school_term,
            is_holiday=is_holiday,
            is_exam_period=is_exam_period and not is_holiday,
            is_school_event=is_school_event,
            model=model,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    horizon_label = f"{result['days_ahead']} school day(s) ahead of latest history ({target_date:%d %b %Y})"
    ds_result = build_decision_support_result(
        predicted_meals=result["predicted_meals_served"],
        recent_std=result["recent_std_7"],
        horizon_days=result["days_ahead"],
        horizon_label=horizon_label,
        assumptions=[
            f"Attendance: {expected_attendance:.0f} students (user-provided)",
            f"Rainfall: {expected_rainfall:.1f} mm (user-provided)",
            f"Temperature: {expected_temp:.1f} °C (user-provided)",
            f"School term: {school_term}, Holiday: {is_holiday}, Exam period: {is_exam_period and not is_holiday}, "
            f"School event: {is_school_event} (all user-provided)",
        ],
        warnings=(
            ["This forecast date is more than 30 days beyond the school's historical data — "
             "treat the result as strictly directional."]
            if result["days_ahead"] > 30 else []
        ),
    )

    st.markdown(f"## Prediction for {school_name} — {target_date:%d %b %Y}")
    decision_support_card(ds_result)

    st.markdown("#### What Influenced This Prediction?")
    st.caption(
        "Feature importance reflects overall model behaviour across all training data, not an "
        "exact causal breakdown of this single prediction."
    )
    if hasattr(model, "feature_importances_"):
        importance_df = (
            pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=False)
            .head(5)
        )
        fig_imp = go.Figure(go.Bar(x=importance_df["Importance"], y=importance_df["Feature"], orientation="h"))
        style_chart(fig_imp, "Top 5 Influencing Factors (Model-Wide Importance)")
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Relative Importance")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model type.")

    st.markdown("#### Recent Demand vs Predicted Demand")
    recent_history = result["recent_history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_history[settings.DATE_COLUMN], y=recent_history["meals_served"],
        mode="lines+markers", name="Recent Actual", line=dict(color="#2E7D32"),
    ))
    fig.add_trace(go.Scatter(
        x=[target_date], y=[result["predicted_meals_served"]],
        mode="markers", name="Specific-Date Prediction",
        marker=dict(color="#C9A227", size=14, symbol="star"),
    ))
    style_chart(fig, f"{school_name} — Recent Demand vs Prediction for {target_date:%d %b %Y}")
    fig.update_layout(xaxis_title="Date", yaxis_title="Meals Served")
    st.plotly_chart(fig, use_container_width=True)
