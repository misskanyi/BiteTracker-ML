"""Tests for src.budgeting: cost projection helpers.

What this file checks:
- Meal cost scales with demand, and zero demand gives zero cost.
- Historical spend is calculated both with and without a cost column.
"""

from __future__ import annotations

import pandas as pd

from src.budgeting import estimate_meal_cost, historical_expenditure


def test_estimate_meal_cost_scaling():
    result = estimate_meal_cost(predicted_meals=520, cost_per_meal=25)
    assert result["daily_cost"] == 13000
    assert result["weekly_cost"] == 13000 * 5
    assert result["monthly_cost"] == 13000 * 22


def test_estimate_meal_cost_zero_demand():
    result = estimate_meal_cost(predicted_meals=0, cost_per_meal=25)
    assert result["daily_cost"] == 0
    assert result["weekly_cost"] == 0


def test_historical_expenditure_with_cost_column():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3),
        "meals_served": [100, 200, 300],
        "average_cost_per_meal": [10, 10, 10],
    })
    result = historical_expenditure(df)
    assert list(result["expenditure"]) == [1000, 2000, 3000]


def test_historical_expenditure_without_cost_column():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3),
        "meals_served": [100, 200, 300],
    })
    result = historical_expenditure(df)
    assert (result["expenditure"] == 0.0).all()
