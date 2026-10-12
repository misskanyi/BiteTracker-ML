"""Cross-school comparison and resource-allocation decision support.

Pure functions over the processed dataset + trained model — no web-framework
imports — so pages just collect inputs and display these results.
"""

# In simple terms: compares schools against each other so donors/officials
# can see which ones need more resources than others.

from __future__ import annotations

import pandas as pd

from config import settings
from src.decision_support import classify_risk, compute_safety_buffer
from src.forecasting import forecast_school_demand, get_school_history


def summarize_school(df: pd.DataFrame, school_id: str, model=None) -> dict:
    """One school's headline metrics: recent history + a next-school-day forecast.

    Used by both the School Comparison and Resource Allocation pages so the
    two stay consistent with each other.
    """
    history = get_school_history(df, school_id)
    if history.empty:
        raise ValueError(f"No history found for school_id={school_id}")

    school_name = history["school_name"].iloc[-1]
    recent = history.tail(30)

    avg_attendance = float(recent["students_present"].mean())
    avg_enrollment = float(recent["enrolled_students"].mean())
    attendance_rate = (avg_attendance / avg_enrollment) if avg_enrollment else 0.0
    avg_meals_served = float(recent["meals_served"].mean())
    avg_leftovers = float(recent["leftover_meals"].mean()) if "leftover_meals" in recent else 0.0
    avg_prepared = float(recent["meals_prepared"].mean()) if "meals_prepared" in recent else avg_meals_served
    estimated_waste_pct = (avg_leftovers / avg_prepared * 100) if avg_prepared else 0.0
    avg_cost_per_meal = float(recent["average_cost_per_meal"].mean()) if "average_cost_per_meal" in recent else 0.0

    forecast = forecast_school_demand(df, school_id, horizon_days=1, model=model)
    predicted_demand = float(forecast["predicted_meals_served"].iloc[0])
    recent_std = float(recent["meals_served"].std() or 0.0)
    buffer = compute_safety_buffer(predicted_demand, recent_std, horizon_days=1)
    risk_level = classify_risk(buffer["buffer_pct"])
    expected_daily_cost = predicted_demand * avg_cost_per_meal

    return {
        "school_id": school_id,
        "school_name": school_name,
        "predicted_demand": round(predicted_demand),
        "avg_attendance": round(avg_attendance),
        "attendance_rate": round(attendance_rate * 100, 1),
        "avg_meals_served": round(avg_meals_served),
        "avg_leftovers": round(avg_leftovers),
        "estimated_waste_pct": round(estimated_waste_pct, 1),
        "avg_cost_per_meal": round(avg_cost_per_meal, 2),
        "expected_daily_cost": round(expected_daily_cost, 2),
        "risk_level": risk_level,
    }


def compare_schools(df: pd.DataFrame, school_ids: list[str], model=None) -> pd.DataFrame:
    """Summary table across multiple schools for the School Comparison page."""
    rows = [summarize_school(df, sid, model=model) for sid in school_ids]
    return pd.DataFrame(rows)


def compute_allocation(predicted_demand: float, available_capacity: float) -> dict:
    """Gap between forecast demand and currently-planned meal capacity for one school.

    Positive `gap` = shortage (demand exceeds capacity); negative = surplus.
    This is a decision-support recommendation, not an automatic reallocation
    instruction — the caller/UI must present it as such.
    """
    gap = round(predicted_demand - available_capacity)
    if gap > 0:
        status = "Shortage"
        recommendation = f"Forecast demand exceeds planned capacity by approximately {gap} meals."
    elif gap < 0:
        status = "Surplus"
        recommendation = f"Planned capacity exceeds forecast demand by approximately {abs(gap)} meals."
    else:
        status = "Balanced"
        recommendation = "Planned capacity matches forecast demand."

    return {
        "predicted_demand": round(predicted_demand),
        "available_capacity": round(available_capacity),
        "gap": gap,
        "status": status,
        "recommendation": recommendation,
    }


def build_reallocation_note(allocations: list[dict]) -> str | None:
    """If one school has a shortage and another a surplus, suggest reviewing allocation between them.

    A suggestion only — never phrased as an automatic operational command.
    """
    shortages = [a for a in allocations if a["gap"] > 0]
    surpluses = [a for a in allocations if a["gap"] < 0]
    if shortages and surpluses:
        return (
            "Consider reviewing resource allocation between schools with a forecast shortage and those "
            "with a forecast surplus. This is a suggestion for further review, not an automatic reallocation."
        )
    return None
