"""BiteTracker — School Comparison: side-by-side metrics across multiple schools."""

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
from theme import configure_page, empty_state, page_header, style_chart

from data_access import artifacts_ready, load_final_model, load_processed_data
from src.resource_allocation import compare_schools

configure_page("School Comparison", icon="📊")
page_header(
    "School Comparison",
    subtitle="Compare forecast demand, attendance, waste, and cost across schools.",
    icon="📊",
)

status = artifacts_ready()
if not (status["processed_data"] and status["final_model"]):
    st.error("Processed data or final model missing. Run `python run_pipeline.py` then `python run_tuning.py`.")
    st.stop()

df = load_processed_data()
model = load_final_model()

schools = df[["school_id", "school_name"]].drop_duplicates().sort_values("school_name")
school_options = dict(zip(schools["school_name"], schools["school_id"]))

selected_names = st.multiselect(
    "Select schools to compare", list(school_options.keys()), default=list(school_options.keys())[:2]
)

if len(selected_names) < 2:
    empty_state("Select at least two schools above to compare them.")
    st.stop()

selected_ids = [school_options[n] for n in selected_names]
comparison = compare_schools(df, selected_ids, model=model)

st.markdown("### Comparison Table")
st.dataframe(
    comparison.rename(columns={
        "school_name": "School", "predicted_demand": "Predicted Demand",
        "avg_attendance": "Avg Attendance", "attendance_rate": "Attendance Rate (%)",
        "avg_meals_served": "Avg Meals Served", "avg_leftovers": "Avg Leftovers",
        "estimated_waste_pct": "Est. Waste (%)", "avg_cost_per_meal": "Avg Cost/Meal (KSh)",
        "expected_daily_cost": "Expected Daily Cost (KSh)", "risk_level": "Forecast Risk",
    }).drop(columns=["school_id"]),
    use_container_width=True, hide_index=True,
)

st.markdown("### Predicted Demand vs Expected Daily Cost")
fig1 = go.Figure()
fig1.add_trace(go.Bar(x=comparison["school_name"], y=comparison["predicted_demand"], name="Predicted Demand (meals)"))
style_chart(fig1, "Predicted Next-Day Demand by School")
fig1.update_layout(yaxis_title="Meals")
st.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=comparison["school_name"], y=comparison["expected_daily_cost"], name="Expected Daily Cost"))
style_chart(fig2, "Expected Daily Cost by School (KSh)")
fig2.update_layout(yaxis_title="KSh")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### Attendance Rate vs Estimated Waste")
fig3 = go.Figure()
fig3.add_trace(go.Bar(x=comparison["school_name"], y=comparison["attendance_rate"], name="Attendance Rate (%)"))
fig3.add_trace(go.Bar(x=comparison["school_name"], y=comparison["estimated_waste_pct"], name="Estimated Waste (%)"))
style_chart(fig3, "Attendance Rate vs Estimated Waste (%)")
fig3.update_layout(barmode="group", yaxis_title="%")
st.plotly_chart(fig3, use_container_width=True)

st.caption(
    "All figures are recent 30-day averages plus a next-school-day forecast per school — useful for "
    "NGOs, donors, and county governments comparing programme performance across schools."
)
