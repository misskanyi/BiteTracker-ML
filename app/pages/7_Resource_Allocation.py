"""BiteTracker — Resource Allocation: forecast demand vs planned capacity, per school."""

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

from data_access import artifacts_ready, load_final_model, load_processed_data
from src.resource_allocation import build_reallocation_note, compute_allocation, summarize_school

configure_page("Resource Allocation", icon="🧮")
page_header(
    "Resource Allocation",
    subtitle="Compare forecast demand against currently planned meal capacity, per school.",
    icon="🧮",
)

status = artifacts_ready()
if not (status["processed_data"] and status["final_model"]):
    st.error("Processed data or final model missing. Run `python run_pipeline.py` then `python run_tuning.py`.")
    st.stop()

df = load_processed_data()
model = load_final_model()

schools = df[["school_id", "school_name"]].drop_duplicates().sort_values("school_name")
school_options = dict(zip(schools["school_name"], schools["school_id"]))

st.markdown("### Enter Planned Meal Capacity")
st.caption(
    "This is a decision-support recommendation, not an automatic operational command — final "
    "allocation decisions remain with school/programme staff."
)

selected_names = st.multiselect(
    "Select schools", list(school_options.keys()), default=list(school_options.keys())
)
if not selected_names:
    st.info("Select at least one school above.")
    st.stop()

capacities = {}
cols = st.columns(min(len(selected_names), 3))
for i, name in enumerate(selected_names):
    school_id = school_options[name]
    summary = summarize_school(df, school_id, model=model)
    default_capacity = summary["predicted_demand"]
    with cols[i % len(cols)]:
        capacities[name] = st.number_input(
            f"{name} — available capacity", min_value=0, value=default_capacity, step=10, key=f"cap_{school_id}"
        )

allocations = []
for name in selected_names:
    school_id = school_options[name]
    summary = summarize_school(df, school_id, model=model)
    allocation = compute_allocation(summary["predicted_demand"], capacities[name])
    allocation["school_name"] = name
    allocation["risk_level"] = summary["risk_level"]
    allocations.append(allocation)

st.markdown("### Allocation Overview")
for a in allocations:
    with st.container():
        st.markdown(f"#### {a['school_name']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Demand", f"{a['predicted_demand']} meals")
        c2.metric("Available Capacity", f"{a['available_capacity']} meals")
        gap_label = "Shortage" if a["gap"] > 0 else "Surplus" if a["gap"] < 0 else "Balanced"
        c3.metric(gap_label, f"{abs(a['gap'])} meals")
        c4.metric("Forecast Risk", a["risk_level"])
        st.info(a["recommendation"])

note = build_reallocation_note(allocations)
if note:
    st.warning(note)

st.markdown("### Demand vs Capacity")
fig = go.Figure()
fig.add_trace(go.Bar(x=[a["school_name"] for a in allocations], y=[a["predicted_demand"] for a in allocations], name="Predicted Demand"))
fig.add_trace(go.Bar(x=[a["school_name"] for a in allocations], y=[a["available_capacity"] for a in allocations], name="Available Capacity"))
style_chart(fig, "Predicted Demand vs Available Capacity")
fig.update_layout(barmode="group", yaxis_title="Meals")
st.plotly_chart(fig, use_container_width=True)
