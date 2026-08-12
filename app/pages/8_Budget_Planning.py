"""BiteTracker — Budget Planning: translate meal-demand forecasts into cost projections."""

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
from src.budgeting import estimate_meal_cost, historical_expenditure
from src.forecasting import forecast_school_demand, get_school_history

configure_page("Budget Planning", icon="💰")
page_header(
    "Budget Planning",
    subtitle="Translate meal-demand forecasts into estimated procurement cost.",
    icon="💰",
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
if "average_cost_per_meal" not in history.columns:
    st.warning("This dataset has no `average_cost_per_meal` column — cost estimates are unavailable.")
    st.stop()

recent_cost = float(history["average_cost_per_meal"].tail(14).mean())
cost_per_meal = st.number_input("Average cost per meal (KSh)", min_value=0.0, value=round(recent_cost, 2), step=1.0)

forecast_df = forecast_school_demand(df, school_id, horizon_days=1, model=model)
predicted_meals = float(forecast_df["predicted_meals_served"].iloc[0])

cost = estimate_meal_cost(predicted_meals, cost_per_meal)

st.markdown(f"### Estimated Requirement — {school_name}")
c1, c2 = st.columns(2)
c1.metric("Predicted Demand", f"{predicted_meals:.0f} meals")
c2.metric("Average Cost per Meal", f"KSh {cost_per_meal:,.2f}")

c3, c4, c5 = st.columns(3)
c3.metric("Estimated Daily Requirement", f"KSh {cost['daily_cost']:,.0f}")
c4.metric("Estimated Weekly Requirement", f"KSh {cost['weekly_cost']:,.0f}")
c5.metric("Estimated Monthly Requirement", f"KSh {cost['monthly_cost']:,.0f}")

st.caption(
    "Weekly/monthly figures assume the same daily demand repeats across 5 school days/week and "
    "~22 school days/month — a simple extrapolation, not a separate multi-day re-forecast."
)

st.markdown("### Historical vs Forecast Expenditure")
hist_expenditure = historical_expenditure(history.tail(30), date_column=settings.DATE_COLUMN)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_expenditure[settings.DATE_COLUMN], y=hist_expenditure["expenditure"],
    mode="lines+markers", name="Historical Daily Expenditure",
))
fig.add_trace(go.Scatter(
    x=forecast_df["date"], y=[cost["daily_cost"]] * len(forecast_df),
    mode="markers", name="Forecast Daily Expenditure", marker=dict(color="#C9A227", size=12, symbol="star"),
))
style_chart(fig, f"{school_name} — Historical vs Forecast Daily Expenditure (KSh)")
fig.update_layout(yaxis_title="KSh")
st.plotly_chart(fig, use_container_width=True)

st.caption("Estimate for planning purposes — actual procurement cost may vary with market prices and supplier terms.")
