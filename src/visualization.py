"""Reusable Plotly chart builders for BiteTracker.

Every function takes the processed DataFrame (output of `src.pipeline`) and
returns a `plotly.graph_objects.Figure`. Kept framework-agnostic from
Streamlit so the same functions serve the EDA report (saved as HTML) and,
later, the app pages (rendered live with `st.plotly_chart`).
"""

# In simple terms: draws the charts/graphs (trends, comparisons) used in
# the EDA report and the Streamlit app pages.

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Green-and-gold palette matching the project's visual identity.
COLOR_PRIMARY = "#2E7D32"    # green
COLOR_SECONDARY = "#C9A227"  # gold
COLOR_SEQUENCE = px.colors.qualitative.Set2
TEMPLATE = "plotly_white"

NUMERIC_COLUMNS_FOR_CORRELATION = [
    "enrolled_students",
    "students_present",
    "students_absent",
    "meals_prepared",
    "meals_served",
    "leftover_meals",
    "average_cost_per_meal",
    "rainfall_mm",
    "temperature_c",
    "is_holiday",
    "is_exam_period",
    "is_school_event",
    "attendance_rate",
    "lag_1_meals",
    "lag_7_meals",
    "lag_14_meals",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "target_meals_served",
]


def _style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(template=TEMPLATE, title=title, title_x=0.02, margin=dict(t=60, l=40, r=30, b=40))
    return fig


def plot_daily_meal_demand(df: pd.DataFrame) -> go.Figure:
    """Meals served over time, one line per school."""
    fig = px.line(df, x="date", y="meals_served", color="school_name", color_discrete_sequence=COLOR_SEQUENCE)
    return _style(fig, "Daily Meal Demand Over Time")


def plot_attendance_trends(df: pd.DataFrame) -> go.Figure:
    """Attendance rate over time, one line per school."""
    fig = px.line(df, x="date", y="attendance_rate", color="school_name", color_discrete_sequence=COLOR_SEQUENCE)
    return _style(fig, "Attendance Rate Trends Over Time")


def plot_rainfall_vs_attendance(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="rainfall_mm", y="attendance_rate", color="school_name", opacity=0.6,
        trendline="ols", color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _style(fig, "Rainfall vs Attendance Rate")


def plot_rainfall_vs_meals_served(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="rainfall_mm", y="meals_served", color="school_name", opacity=0.6,
        trendline="ols", color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _style(fig, "Rainfall vs Meals Served")


def plot_temperature_vs_attendance(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="temperature_c", y="attendance_rate", color="school_name", opacity=0.6,
        trendline="ols", color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _style(fig, "Temperature vs Attendance Rate")


def plot_leftover_meals_over_time(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="date", y="leftover_meals", color="school_name", color_discrete_sequence=COLOR_SEQUENCE)
    return _style(fig, "Leftover Meals Over Time")


def plot_meals_served_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="meals_served", nbins=50, color_discrete_sequence=[COLOR_PRIMARY])
    return _style(fig, "Distribution of Meals Served")


def plot_attendance_rate_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="attendance_rate", nbins=50, color_discrete_sequence=[COLOR_SECONDARY])
    return _style(fig, "Distribution of Attendance Rate")


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in NUMERIC_COLUMNS_FOR_CORRELATION if c in df.columns]
    corr = df[cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
        aspect="auto",
    )
    return _style(fig, "Correlation Heatmap of Numerical Variables")


def plot_outlier_boxplots(df: pd.DataFrame) -> go.Figure:
    cols = ["meals_served", "students_present", "leftover_meals", "enrolled_students"]
    melted = df.melt(id_vars=["school_name"], value_vars=cols, var_name="metric", value_name="value")
    fig = px.box(melted, x="metric", y="value", color="metric", color_discrete_sequence=COLOR_SEQUENCE)
    return _style(fig, "Outlier Detection Boxplots")


def plot_school_comparison(df: pd.DataFrame) -> go.Figure:
    summary = df.groupby("school_name", as_index=False)["meals_served"].mean()
    fig = px.bar(summary, x="school_name", y="meals_served", color="school_name", color_discrete_sequence=COLOR_SEQUENCE)
    return _style(fig, "Average Meal Demand by School")


def plot_monthly_trends(df: pd.DataFrame) -> go.Figure:
    summary = df.groupby(["month", "school_name"], as_index=False)["meals_served"].mean()
    fig = px.line(summary, x="month", y="meals_served", color="school_name", markers=True, color_discrete_sequence=COLOR_SEQUENCE)
    return _style(fig, "Monthly Meal Demand Trends")


def plot_day_of_week_pattern(df: pd.DataFrame) -> go.Figure:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    summary = df.groupby("day_of_week", as_index=False)["meals_served"].mean()
    summary["day_of_week"] = pd.Categorical(summary["day_of_week"], categories=order, ordered=True)
    summary = summary.sort_values("day_of_week")
    fig = px.bar(summary, x="day_of_week", y="meals_served", color_discrete_sequence=[COLOR_PRIMARY])
    return _style(fig, "Day-of-Week Meal Demand Pattern")


def plot_school_term_comparison(df: pd.DataFrame) -> go.Figure:
    summary = df.groupby(["school_term", "school_name"], as_index=False)["meals_served"].mean()
    fig = px.bar(summary, x="school_term", y="meals_served", color="school_name", barmode="group", color_discrete_sequence=COLOR_SEQUENCE)
    return _style(fig, "School Term Comparison")


def plot_rolling_average(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="date", y="rolling_mean_7", color="school_name", color_discrete_sequence=COLOR_SEQUENCE)
    return _style(fig, "Rolling 7-Day Average Meal Demand")


# Registry used by the EDA report generator to produce every chart consistently.
CHART_REGISTRY: dict[str, callable] = {
    "daily_meal_demand": plot_daily_meal_demand,
    "attendance_trends": plot_attendance_trends,
    "rainfall_vs_attendance": plot_rainfall_vs_attendance,
    "rainfall_vs_meals_served": plot_rainfall_vs_meals_served,
    "temperature_vs_attendance": plot_temperature_vs_attendance,
    "leftover_meals_over_time": plot_leftover_meals_over_time,
    "meals_served_distribution": plot_meals_served_distribution,
    "attendance_rate_distribution": plot_attendance_rate_distribution,
    "correlation_heatmap": plot_correlation_heatmap,
    "outlier_boxplots": plot_outlier_boxplots,
    "school_comparison": plot_school_comparison,
    "monthly_trends": plot_monthly_trends,
    "day_of_week_pattern": plot_day_of_week_pattern,
    "school_term_comparison": plot_school_term_comparison,
    "rolling_average": plot_rolling_average,
}
