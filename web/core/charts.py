"""Plotly chart specifications for the Django templates.

Each function returns a JSON-serialisable dict ({"data": ..., "layout": ...})
that the `_chart.html` partial renders with plotly.js. Only presentation
lives here; the numbers come from the service layer.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go

from config import settings

# Colour roles. Each trace names its role in `meta`; static/js/charts.js resolves the role
# to the active theme's --bt-chart-<role> CSS token. The hex values here are the light-theme
# tokens, used only before that script runs. The five school hues were checked for
# colour-vision-deficiency separation and contrast against both theme surfaces.
ROLE_COLOURS = {
    "series-1": "#2563EB", "series-2": "#0D9488", "series-3": "#D97706", "series-4": "#DB2777", "series-5": "#7C3AED",
    "other": "#8A8298",
    "actual": "#7C3AED", "secondary": "#8A8298", "predicted": "#D97706", "recommended": "#0D9488",
    "mae": "#7C3AED", "rmse": "#D97706", "selected": "#7C3AED", "candidate": "#B9A6F2",
}
SCHOOL_ROLES = ["series-1", "series-2", "series-3", "series-4", "series-5"]


def _role(role: str) -> dict:
    return {"meta": role, "line": {"color": ROLE_COLOURS[role]}}


def _school_role(i: int) -> str:
    """Fixed hue per school position; schools beyond the fifth share a neutral rather than reusing a hue."""
    return SCHOOL_ROLES[i] if i < len(SCHOOL_ROLES) else "other"


def _spec(fig: go.Figure, title: str, y_title: str, x_title: str = "Date") -> dict:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        template="plotly_white",
        font={"family": "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 56, "r": 20, "t": 50, "b": 90},
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "xanchor": "left", "x": 0},
        hovermode="x unified",
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=420,
    )
    return json.loads(fig.to_json())


def school_history_chart(history: pd.DataFrame, school_name: str) -> dict:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history[settings.DATE_COLUMN], y=history["meals_served"], mode="lines+markers",
        name="Meals served", marker={"size": 4, "color": ROLE_COLOURS["actual"]}, **_role("actual"),
    ))
    if "students_present" in history:
        fig.add_trace(go.Scatter(
            x=history[settings.DATE_COLUMN], y=history["students_present"], mode="lines",
            name="Students present", meta="secondary", line={"color": ROLE_COLOURS["secondary"], "dash": "dot"},
        ))
    return _spec(fig, f"{school_name}: daily meal demand", "Count")


def all_schools_history_chart(df: pd.DataFrame, last_n_records: int = 120) -> dict:
    fig = go.Figure()
    for i, (name, group) in enumerate(df.sort_values(settings.DATE_COLUMN).groupby("school_name")):
        recent = group.tail(last_n_records)
        fig.add_trace(go.Scatter(
            x=recent[settings.DATE_COLUMN], y=recent["meals_served"], mode="lines", name=name,
            **_role(_school_role(i)),
        ))
    return _spec(fig, f"Meals served per school (last {last_n_records} recorded school days)", "Meals served")


def forecast_chart(recent_history: pd.DataFrame | None, days: list[dict], school_name: str) -> dict:
    fig = go.Figure()
    if recent_history is not None and not recent_history.empty:
        fig.add_trace(go.Scatter(
            x=recent_history[settings.DATE_COLUMN], y=recent_history["meals_served"], mode="lines+markers",
            name="Recorded meals served", marker={"size": 5, "color": ROLE_COLOURS["actual"]}, **_role("actual"),
        ))
    dates = [d["date"] for d in days]
    fig.add_trace(go.Scatter(
        x=dates, y=[d["predicted_meals"] for d in days], mode="lines+markers",
        name="Predicted demand", meta="predicted", line={"color": ROLE_COLOURS["predicted"], "dash": "dash"},
        marker={"size": 6, "color": ROLE_COLOURS["predicted"]},
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=[d["recommended_meals"] for d in days], mode="lines",
        name="Recommended preparation", meta="recommended", line={"color": ROLE_COLOURS["recommended"], "dash": "dot"},
    ))
    return _spec(fig, f"{school_name}: recent demand and forecast", "Meals")


def model_vs_baselines_chart(performance: dict) -> dict:
    names = [performance["selected_model"]] + [b["name"] for b in performance["naive_baselines"]]
    mae = [performance["test"]["mae"]] + [b["test_mae"] for b in performance["naive_baselines"]]
    rmse = [performance["test"]["rmse"]] + [b["test_rmse"] for b in performance["naive_baselines"]]
    fig = go.Figure([
        go.Bar(name="Test MAE", x=names, y=mae, meta="mae", marker_color=ROLE_COLOURS["mae"], text=[f"{v:.1f}" for v in mae], textposition="outside"),
        go.Bar(name="Test RMSE", x=names, y=rmse, meta="rmse", marker_color=ROLE_COLOURS["rmse"], text=[f"{v:.1f}" for v in rmse], textposition="outside"),
    ])
    fig.update_layout(barmode="group", hovermode="closest")
    return _spec(fig, "Held-out test error: selected model vs naive baselines (lower is better)", "Meals", x_title="")


def validation_ranking_chart(ranking: list[dict]) -> dict:
    fig = go.Figure(go.Bar(
        x=[r["Model"] for r in ranking],
        y=[r["Validation MAE"] for r in ranking],
        error_y={"type": "data", "array": [r.get("Validation MAE Std") or 0 for r in ranking], "visible": True},
        meta=["selected" if r.get("Selected") else "candidate" for r in ranking],
        marker_color=[ROLE_COLOURS["selected" if r.get("Selected") else "candidate"] for r in ranking],
        text=[f"{r['Validation MAE']:.1f}" for r in ranking], textposition="outside",
    ))
    fig.update_layout(hovermode="closest")
    return _spec(fig, "Mean validation MAE across TimeSeriesSplit folds (bars: ± 1 std)", "MAE (meals)", x_title="")
