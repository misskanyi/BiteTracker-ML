"""BiteTracker — Dashboard page: current state at a glance."""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_APP_DIR.parent))

import plotly.express as px
import streamlit as st
from theme import configure_page, page_header, style_chart

from data_access import artifacts_ready, load_json, load_processed_data
from config import settings

configure_page("Dashboard", icon="📊")
page_header("Dashboard", subtitle="Current meal-demand state at a glance, across all schools.", icon="📊")

if not artifacts_ready()["processed_data"]:
    st.error("Processed dataset not found. Run `python run_pipeline.py` first.")
    st.stop()

df = load_processed_data()
latest_date = df[settings.DATE_COLUMN].max()
latest_rows = df[df[settings.DATE_COLUMN] == latest_date]

total_schools = df[settings.SCHOOL_ID_COLUMN].nunique()
total_enrollment = int(df.groupby(settings.SCHOOL_ID_COLUMN)["enrolled_students"].last().sum())
avg_attendance = df["attendance_rate"].mean()
latest_meals_served = int(latest_rows["meals_served"].sum())
avg_leftover = df["leftover_meals"].mean()
waste_rate = (df["leftover_meals"].sum() / df["meals_prepared"].sum() * 100) if df["meals_prepared"].sum() else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Schools", total_schools)
col2.metric("Total Enrolment", f"{total_enrollment:,}")
col3.metric("Average Attendance Rate", f"{avg_attendance:.1%}")
col4.metric("Latest Day Meals Served (all schools)", f"{latest_meals_served:,}", help=f"As of {latest_date.date()}")

col5, col6 = st.columns(2)
col5.metric("Average Leftover Meals / Day / School", f"{avg_leftover:.1f}")
col6.metric("Food Waste Rate (leftover / prepared)", f"{waste_rate:.1f}%")

model_metadata_path = settings.MODELS_DIR / "final_best_model_metadata.json"
if model_metadata_path.exists():
    meta = load_json(model_metadata_path)
    st.info(
        f"Current forecasting model: **{meta['selected_model']}** "
        f"(Test MAE={meta['final_test_mae']:.1f} meals, Test R²={meta['final_test_r2']:.3f}). "
        f"See the Forecast page to generate a prediction, or Model Performance for full comparison."
    )
else:
    st.warning("No final model found yet — run `python run_tuning.py` to select one.")

st.divider()
st.markdown("#### 📈 Historical Meal Demand")
schools = sorted(df["school_name"].unique())
selected_schools = st.multiselect("Filter by school", schools, default=schools)
filtered = df[df["school_name"].isin(selected_schools)] if selected_schools else df

fig = px.line(
    filtered, x=settings.DATE_COLUMN, y="meals_served", color="school_name",
    labels={settings.DATE_COLUMN: "Date", "meals_served": "Meals Served", "school_name": "School"},
)
style_chart(fig, "Daily Meal Demand")
st.plotly_chart(fig, use_container_width=True)

col_a, col_b = st.columns(2, gap="medium")
with col_a:
    summary = filtered.groupby("school_name", as_index=False)["meals_served"].mean()
    fig2 = px.bar(summary, x="school_name", y="meals_served", color="school_name", labels={"school_name": "School", "meals_served": "Avg Meals Served"})
    style_chart(fig2, "Average Meals Served by School")
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    fig3 = px.line(filtered, x=settings.DATE_COLUMN, y="leftover_meals", color="school_name", labels={settings.DATE_COLUMN: "Date", "leftover_meals": "Leftover Meals", "school_name": "School"})
    style_chart(fig3, "Leftover Meals Over Time (Waste Indicator)")
    st.plotly_chart(fig3, use_container_width=True)
