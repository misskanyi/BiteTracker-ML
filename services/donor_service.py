"""Donor view: how each school's meal programme is doing and where support is needed most.

Everything is derived from the processed dataset (via `data_service`); nothing is
predicted here. For each school, over its most recent `WINDOW` recorded school days:

- reach: mean meals served per day ÷ registered students (the dataset's enrolment
  figure, which for the Food4Education sheet is a proxy: unique students seen tapping);
- trend: that mean against the `WINDOW` records before it;
- unserved students: registered students not served a meal on an average recent day.
  This includes absent students, so it is an upper bound on unmet need;
- monthly funding gap: unserved students × cost per meal × `SCHOOL_DAYS_PER_MONTH`,
  using the planning cost assumption (`planning_service`), not verified expenditure.

Schools are ranked by unserved students per day: the most children a donation could
reach. Ties fall back to lower reach.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from config import settings
from services import data_service, planning_service
from src.forecasting import get_school_history as _src_get_school_history

WINDOW = 20  # recorded school days, about one month
SCHOOL_DAYS_PER_MONTH = 20
STALE_AFTER_DAYS = 14  # a school whose last record is this far behind the dataset's latest date is "not reporting"
TREND_FLAT_PCT = 3.0  # changes smaller than this (either way) count as steady
RECOMMENDED_COUNT = 3

REACH_BANDS = ((0.75, "Strong"), (0.5, "Moderate"), (0.0, "Low"))


def _reach_band(reach: float | None) -> str | None:
    if reach is None:
        return None
    return next(label for floor, label in REACH_BANDS if reach >= floor)


def _trend(recent_mean: float, previous_mean: float | None) -> tuple[float | None, str]:
    if not previous_mean:
        return None, "new"
    pct = (recent_mean - previous_mean) / previous_mean * 100
    if abs(pct) < TREND_FLAT_PCT:
        return round(pct, 1), "steady"
    return round(pct, 1), "up" if pct > 0 else "down"


def school_impact(cost_per_meal: Decimal | None = None, school_ids: set[str] | None = None) -> dict:
    """Per-school donor metrics, ranked by need, plus programme totals.

    `school_ids` limits the result (and its totals) to those schools; None means every school.
    """
    cost = planning_service.default_cost_per_meal() if cost_per_meal is None else cost_per_meal
    df = data_service.load_processed_data()
    dataset_end = df[settings.DATE_COLUMN].max().date()

    schools = []
    for school in data_service.list_schools():
        if school_ids is not None and school["school_id"] not in school_ids:
            continue
        history = _src_get_school_history(df, school["school_id"])
        meals = history["meals_served"]
        recent = meals.tail(WINDOW)
        previous = meals.iloc[-2 * WINDOW:-WINDOW] if len(meals) > WINDOW else meals.iloc[0:0]
        recent_mean = float(recent.mean())
        previous_mean = float(previous.mean()) if len(previous) else None
        trend_pct, trend = _trend(recent_mean, previous_mean)

        registered = school.get("enrolled_students") or None
        reach = min(recent_mean / registered, 1.0) if registered else None
        unserved = max(round(registered - recent_mean), 0) if registered else None
        latest_date = history[settings.DATE_COLUMN].iloc[-1].date()

        schools.append({
            **school,
            "registered_students": registered,
            "recent_records": int(len(recent)),
            "recent_mean_meals": round(recent_mean, 1),
            "previous_mean_meals": round(previous_mean, 1) if previous_mean is not None else None,
            "trend_pct": trend_pct,
            "trend": trend,
            "reach": reach,
            "reach_pct": round(reach * 100) if reach is not None else None,
            "reach_band": _reach_band(reach),
            "unserved_per_day": unserved,
            "monthly_gap_kes": planning_service.meal_budget(unserved * SCHOOL_DAYS_PER_MONTH, cost) if unserved is not None else None,
            "meals_last_window": float(recent.sum()),
            "latest_date": latest_date,
            "reporting": latest_date >= dataset_end - timedelta(days=STALE_AFTER_DAYS),
        })

    ranked = sorted(schools, key=lambda s: (-(s["unserved_per_day"] or 0), s["reach"] if s["reach"] is not None else 1.0))
    for rank, s in enumerate(ranked, start=1):
        s["priority_rank"] = rank
        s["recommended"] = rank <= RECOMMENDED_COUNT and bool(s["unserved_per_day"])

    with_registered = [s for s in ranked if s["registered_students"]]
    total_registered = sum(s["registered_students"] for s in with_registered)
    total_served = sum(s["recent_mean_meals"] for s in with_registered)
    return {
        "schools": ranked,
        "recommended": [s for s in ranked if s["recommended"]],
        "cost_per_meal": cost,
        "window": WINDOW,
        "school_days_per_month": SCHOOL_DAYS_PER_MONTH,
        "dataset_end": dataset_end,
        "totals": {
            "schools": len(ranked),
            "meals_per_day": round(sum(s["recent_mean_meals"] for s in ranked)),
            "registered_students": total_registered,
            "reach_pct": round(total_served / total_registered * 100) if total_registered else None,
            "unserved_per_day": sum(s["unserved_per_day"] or 0 for s in ranked),
            "monthly_gap_kes": sum((s["monthly_gap_kes"] or Decimal(0) for s in ranked), Decimal(0)),
        },
    }
