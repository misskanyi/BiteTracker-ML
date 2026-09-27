"""BiteTracker — Planning: predict/compare demand for one chosen date against
its auto-generated baseline, and browse a multi-day forecast calendar.

Two tabs instead of three: "Predict & Compare" folds together what used to be
two separate tools (a specific-date prediction with custom inputs, and a
scenario comparison against a baseline) — every custom prediction is always
shown next to its baseline, so there's one workflow instead of two nearly
identical ones. "Forecast Calendar" stays separate because it's a different
kind of tool: no manual inputs, an auto-generated multi-day table to browse.
"""

from __future__ import annotations

import sys
from datetime import timedelta
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
from src.budgeting import estimate_meal_cost
from src.decision_support import build_decision_support_result, classify_risk, compute_safety_buffer
from src.feature_engineering import FEATURE_COLUMNS
from src.forecasting import forecast_school_demand, get_school_history, predict_specific_date
from src.synthetic_data import _is_exam_period, _is_holiday, _school_term

configure_page("Planning", icon="🗓️")
page_header(
    "Planning",
    subtitle="Predict and compare demand for one date against its baseline, or browse a forecast calendar.",
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

school_name = st.selectbox("School", list(school_options.keys()))
school_id = school_options[school_name]
history = get_school_history(df, school_id)

tab_predict, tab_calendar = st.tabs(["🔮 Predict & Compare", "🗓️ Forecast Calendar"])

# ====================================================== Predict & Compare ===
with tab_predict:
    st.caption(
        "Pick a date and describe expected conditions. Your prediction is always shown next to "
        "the model's auto-generated baseline for that same date, so you can see exactly what "
        "your assumptions changed."
    )

    last_date = history[settings.DATE_COLUMN].max().date()
    target_date = st.date_input(
        "Forecast date",
        value=last_date + timedelta(days=1),
        min_value=last_date + timedelta(days=1),
        key="pc_target_date",
    )

    if target_date.weekday() >= 5:
        st.warning("Selected date is a weekend — the pipeline only models school days (Mon–Fri). Predictions may not be meaningful.")

    days_ahead = (target_date - last_date).days

    recent_attendance = history["students_present"].tail(14).mean()
    recent_rainfall = history["rainfall_mm"].tail(14).mean()
    recent_temp = history["temperature_c"].tail(14).mean()

    baseline_term = _school_term(target_date)
    baseline_holiday = _is_holiday(target_date)
    baseline_exam = (not baseline_holiday) and _is_exam_period(target_date, baseline_term)

    with st.spinner("Computing baseline..."):
        baseline_forecast_df = forecast_school_demand(df, school_id, horizon_days=days_ahead, model=model)
    baseline_meals = float(baseline_forecast_df["predicted_meals_served"].iloc[-1])

    st.markdown("##### Baseline (recent 14-day averages + calendar-derived term/holiday/exam status)")
    st.metric(f"Baseline Forecast — {target_date:%d %b %Y}", f"{baseline_meals:.0f} meals")

    st.markdown("##### Your Conditions")
    st.caption("🟢 auto-derived from history/date · 🔵 you provide · pre-filled with the baseline's own values")

    col3, col4, col5 = st.columns(3)
    with col3:
        expected_attendance = st.number_input(
            "🔵 Expected attendance (students present)", min_value=0.0,
            value=float(round(recent_attendance)), step=1.0, key="pc_attendance",
        )
    with col4:
        expected_rainfall = st.number_input(
            "🔵 Expected rainfall (mm)", min_value=0.0, value=float(round(recent_rainfall, 1)), step=0.5, key="pc_rainfall",
        )
    with col5:
        expected_temp = st.number_input(
            "🔵 Expected temperature (°C)", min_value=-10.0, value=float(round(recent_temp, 1)), step=0.5, key="pc_temp",
        )

    col6, col7, col8, col9 = st.columns(4)
    with col6:
        school_term = st.selectbox("🔵 School term", ["Term 1", "Term 2", "Term 3"],
                                    index=["Term 1", "Term 2", "Term 3"].index(baseline_term), key="pc_term")
    with col7:
        is_holiday = st.checkbox("🔵 Holiday", value=baseline_holiday, key="pc_holiday")
    with col8:
        is_exam_period = st.checkbox("🔵 Exam period", value=baseline_exam, disabled=is_holiday, key="pc_exam")
    with col9:
        is_school_event = st.checkbox("🔵 School event", value=False, key="pc_event")

    st.caption(
        f"🟢 Auto-derived: day of week = **{target_date.strftime('%A')}**, month = **{target_date.strftime('%B')}**, "
        f"and lag/rolling demand history computed from this school's own past records."
    )

    if st.button("Generate Prediction", type="primary", key="pc_generate"):
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

        predicted_meals = result["predicted_meals_served"]
        difference = predicted_meals - baseline_meals

        st.markdown(f"##### Baseline vs Your Prediction — {target_date:%d %b %Y}")
        kcol1, kcol2, kcol3 = st.columns(3)
        kcol1.metric("Baseline Forecast", f"{baseline_meals:.0f} meals")
        kcol2.metric("Your Prediction", f"{predicted_meals:.0f} meals")
        kcol3.metric("Difference", f"{difference:+.0f} meals", delta=f"{difference:+.0f}")

        direction = "increases" if difference > 0 else "decreases" if difference < 0 else "stays roughly the same"
        st.info(
            f"Given these conditions, predicted demand {direction} by approximately **{abs(difference):.0f} meals** "
            f"compared to the baseline forecast for {target_date:%d %b %Y}. This reflects the model's learned "
            "association between these conditions and past demand — it is not a claim that any single input "
            "*caused* the change on its own."
        )

        fig_bar = go.Figure(go.Bar(
            x=["Baseline", "Your Prediction"], y=[baseline_meals, predicted_meals],
            marker_color=["#2E7D32", "#C9A227"], text=[f"{baseline_meals:.0f}", f"{predicted_meals:.0f}"],
            textposition="outside",
        ))
        style_chart(fig_bar, f"{school_name} — Baseline vs Your Prediction ({target_date:%d %b %Y})")
        fig_bar.update_layout(yaxis_title="Meals Served")
        st.plotly_chart(fig_bar, use_container_width=True)

        horizon_label = f"{result['days_ahead']} school day(s) ahead of latest history ({target_date:%d %b %Y})"
        ds_result = build_decision_support_result(
            predicted_meals=predicted_meals,
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
        decision_support_card(ds_result)

        st.markdown("##### What Influenced This Prediction?")
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

        st.markdown("##### Recent Demand vs Predicted Demand")
        recent_history = result["recent_history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_history[settings.DATE_COLUMN], y=recent_history["meals_served"],
            mode="lines+markers", name="Recent Actual", line=dict(color="#2E7D32"),
        ))
        fig.add_trace(go.Scatter(
            x=[target_date], y=[predicted_meals],
            mode="markers", name="Your Prediction",
            marker=dict(color="#C9A227", size=14, symbol="star"),
        ))
        style_chart(fig, f"{school_name} — Recent Demand vs Prediction for {target_date:%d %b %Y}")
        fig.update_layout(xaxis_title="Date", yaxis_title="Meals Served")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "This is a decision-support estimate, not a guaranteed outcome. Use it to inform meal "
            "preparation planning alongside local judgement."
        )

# ============================================================== Calendar ===
with tab_calendar:
    st.caption("Date-based planning view: predicted demand, preparation, and risk for upcoming school days.")

    n_days = st.selectbox("Planning horizon", [14, 30], index=0, key="cal_horizon")

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

    calendar_df = pd.DataFrame(rows)

    risk_icon = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
    display_df = calendar_df.copy()
    display_df["Risk"] = display_df["Risk"].map(lambda r: f"{risk_icon.get(r, '')} {r}")

    st.markdown(f"##### {school_name} — Next {n_days} School Days")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(
        "Risk reflects forecast uncertainty (demand volatility + horizon), not a confirmed operational "
        "problem. See the Predict & Compare tab for a deeper look at any single day."
    )

    st.markdown("##### Inspect a Specific Day")
    selected_date = st.selectbox("Select a date", calendar_df["Date"].tolist(), key="cal_selected_date")
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
