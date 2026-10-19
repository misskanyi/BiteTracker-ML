"""Meal-budget planning arithmetic on top of stored forecasts.

A budget is meals × an assumed cost per meal (default KES 27, see
`config.settings.PLANNING_COST_PER_MEAL_KES`). This is a planning assumption
supplied by the user, not verified Food4Education expenditure, and nothing
here feeds back into the model: predictions and recommendations are read,
never changed. Money uses `Decimal` so totals are exact.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from config import settings
from services import ServiceError

MIN_COST_PER_MEAL = Decimal("1")
MAX_COST_PER_MEAL = Decimal("10000")


class InvalidCostError(ServiceError):
    """The cost per meal is not a usable positive amount."""


def default_cost_per_meal() -> Decimal:
    return parse_cost(settings.PLANNING_COST_PER_MEAL_KES)


def parse_cost(value) -> Decimal:
    """A cost per meal in KES as a Decimal, rejecting blanks, non-numbers and out-of-range values."""
    if isinstance(value, bool) or value is None or str(value).strip() == "":
        raise InvalidCostError("Enter a cost per meal in KES.")
    try:
        cost = Decimal(str(value).strip())
    except InvalidOperation:
        raise InvalidCostError("The cost per meal must be a number.") from None
    if not cost.is_finite():
        raise InvalidCostError("The cost per meal must be a finite number.")
    if not MIN_COST_PER_MEAL <= cost <= MAX_COST_PER_MEAL:
        raise InvalidCostError(f"The cost per meal must be between KES {MIN_COST_PER_MEAL} and KES {MAX_COST_PER_MEAL:,}.")
    return cost


def meal_budget(meals, cost_per_meal: Decimal) -> Decimal:
    """Meals × cost per meal. 1,000 meals at KES 27 = KES 27,000."""
    return Decimal(str(meals)) * cost_per_meal


def forecast_budget(days: list[dict], cost_per_meal: Decimal) -> dict:
    """Budget for a stored forecast: per day and across the whole horizon.

    Each day's whole-meal predicted and recommended counts are costed once, and the
    horizon totals are sums of the daily meal counts, so cumulative figures are never
    multiplied twice.
    """
    per_day = [
        {
            "day_number": day["day_number"],
            "predicted_cost": meal_budget(day["predicted_meals"], cost_per_meal),
            "recommended_cost": meal_budget(day["recommended_meals"], cost_per_meal),
        }
        for day in days
    ]
    predicted_meals = sum(day["predicted_meals"] for day in days)
    recommended_meals = sum(day["recommended_meals"] for day in days)
    return {
        "cost_per_meal": cost_per_meal,
        "per_day": per_day,
        "day1_predicted_cost": per_day[0]["predicted_cost"] if per_day else Decimal(0),
        "day1_recommended_cost": per_day[0]["recommended_cost"] if per_day else Decimal(0),
        "horizon_days": len(days),
        "horizon_predicted_meals": predicted_meals,
        "horizon_recommended_meals": recommended_meals,
        "horizon_predicted_cost": meal_budget(predicted_meals, cost_per_meal),
        "horizon_recommended_cost": meal_budget(recommended_meals, cost_per_meal),
    }
