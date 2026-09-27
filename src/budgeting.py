"""Budget projections from predicted meal demand and per-meal cost.

Pure arithmetic on top of already-computed predictions/history — kept
separate from `src.forecasting`/`src.decision_support` since cost is a
downstream business calculation, not a model output.
"""

# In simple terms: takes the predicted number of meals and multiplies by
# cost-per-meal to tell schools how much money they'll need to spend.

from __future__ import annotations

import pandas as pd

# School-week/month conventions used consistently across the app: 5 school
# days per week, ~22 school days per month (matches the Mon-Fri calendar
# `src.forecasting._next_school_days` already assumes).
SCHOOL_DAYS_PER_WEEK = 5
SCHOOL_DAYS_PER_MONTH = 22


def estimate_meal_cost(predicted_meals: float, cost_per_meal: float) -> dict:
    """Daily/weekly/monthly cost projection from one day's predicted demand.

    Weekly/monthly figures assume the same daily demand repeats — a simple,
    transparent extrapolation, not a re-forecast of future days.
    """
    daily = predicted_meals * cost_per_meal
    return {
        "daily_cost": round(daily, 2),
        "weekly_cost": round(daily * SCHOOL_DAYS_PER_WEEK, 2),
        "monthly_cost": round(daily * SCHOOL_DAYS_PER_MONTH, 2),
    }


def historical_expenditure(history: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Daily historical spend = meals_served * average_cost_per_meal, for charting.

    Returns an empty-cost series (cost=0) if `average_cost_per_meal` isn't
    present, so callers can still plot demand without crashing.
    """
    result = history[[date_column, "meals_served"]].copy()
    if "average_cost_per_meal" in history.columns:
        result["expenditure"] = history["meals_served"] * history["average_cost_per_meal"]
    else:
        result["expenditure"] = 0.0
    return result
