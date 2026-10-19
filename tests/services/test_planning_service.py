"""Meal-budget planning arithmetic (services/planning_service.py) and weather availability (data_service).

What this file checks:
- Budget = meals × cost per meal, exactly (1,000 meals × KES 27 = KES 27,000).
- Multi-day totals add each day's meals once; the horizon budget equals the sum of daily budgets.
- The default cost is KES 27 and comes from settings; bad costs are rejected.
- Weather columns count as "not recorded" for the Google Sheet source or when entirely empty.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from config import settings
from services import data_service, planning_service
from web.core.templatetags.bitetracker import kes


def test_thousand_meals_at_27_cost_27000():
    assert planning_service.meal_budget(1000, Decimal("27")) == Decimal("27000")


def test_default_cost_is_27(monkeypatch):
    assert planning_service.default_cost_per_meal() == Decimal("27")
    monkeypatch.setattr(settings, "PLANNING_COST_PER_MEAL_KES", "30.5")
    assert planning_service.default_cost_per_meal() == Decimal("30.5")


@pytest.mark.parametrize("value", ["", None, "abc", "0", "-5", "10001", "nan", "inf", True])
def test_invalid_costs_are_rejected(value):
    with pytest.raises(planning_service.InvalidCostError):
        planning_service.parse_cost(value)


@pytest.mark.parametrize("value, expected", [("27", Decimal("27")), (" 32.50 ", Decimal("32.50")), (1, Decimal("1"))])
def test_valid_costs_parse_exactly(value, expected):
    assert planning_service.parse_cost(value) == expected


def test_forecast_budget_totals_add_daily_meals_once():
    days = [
        {"day_number": 1, "predicted_meals": 100, "recommended_meals": 110},
        {"day_number": 2, "predicted_meals": 200, "recommended_meals": 220},
        {"day_number": 3, "predicted_meals": 150, "recommended_meals": 170},
    ]
    budget = planning_service.forecast_budget(days, Decimal("27"))
    assert budget["day1_predicted_cost"] == Decimal("2700")
    assert budget["day1_recommended_cost"] == Decimal("2970")
    assert budget["horizon_predicted_meals"] == 450 and budget["horizon_recommended_meals"] == 500
    assert budget["horizon_recommended_cost"] == Decimal("13500")
    assert budget["horizon_recommended_cost"] == sum(d["recommended_cost"] for d in budget["per_day"])
    assert budget["horizon_predicted_cost"] == sum(d["predicted_cost"] for d in budget["per_day"])


def test_budget_does_not_modify_forecast_days():
    days = [{"day_number": 1, "predicted_meals": 100, "recommended_meals": 110}]
    snapshot = [dict(d) for d in days]
    planning_service.forecast_budget(days, Decimal("27"))
    assert days == snapshot


@pytest.mark.parametrize("value, expected", [
    (Decimal("27000"), "KES 27,000"), (Decimal("2970.5"), "KES 2,970.50"), (27, "KES 27"), ("x", "—"),
])
def test_kes_filter(value, expected):
    assert kes(value) == expected


# --- Weather availability -----------------------------------------------------------------


def test_weather_recorded_for_synthetic_csv(stub_artifacts):
    assert data_service.unrecorded_columns() == set()


def test_google_sheet_source_has_no_weather(stub_artifacts, monkeypatch):
    monkeypatch.setattr(settings, "USE_GOOGLE_SHEET", True)
    assert data_service.unrecorded_columns() == {"rainfall_mm", "temperature_c"}


def test_empty_weather_column_is_unrecorded(stub_artifacts):
    frame = stub_artifacts.frame.copy()
    frame["temperature_c"] = np.nan
    stub_artifacts.write_processed(frame)
    assert data_service.unrecorded_columns() == {"temperature_c"}
