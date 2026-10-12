"""Persist forecast results as ForecastRun audit records and read them back for display."""

from __future__ import annotations

from datetime import date

from services.forecast_service import ForecastResult
from web.schools.models import ForecastRun, School


def record_forecast_run(result: ForecastResult, user) -> ForecastRun:
    """Save one successful forecast. Stores the result summary, never raw historical data."""
    summary = result.to_dict()
    summary.pop("forecast_frame", None)
    summary.pop("recent_history", None)
    first_day = result.days[0]
    return ForecastRun.objects.create(
        user=user if getattr(user, "is_authenticated", False) else None,
        school=School.objects.filter(school_id=result.school_id).first(),
        school_code=result.school_id,
        school_name=result.school_name,
        horizon_days=result.horizon_days,
        origin_date=result.origin_date,
        first_forecast_date=result.first_forecast_date,
        last_forecast_date=result.last_forecast_date,
        assumptions={k: v for k, v in result.inputs.items() if v is not None},
        model_label=result.model_name,
        model_version=result.model_trained_at or "",
        results_label=result.results_label,
        day1_predicted_meals=first_day["predicted_meals"],
        day1_recommended_meals=first_day["recommended_meals"],
        day1_risk_level=first_day["risk_level"],
        total_predicted_meals=result.totals["predicted_meals"],
        total_recommended_meals=result.totals["recommended_meals"],
        result_summary=summary,
    )


def stored_days(run: ForecastRun) -> list[dict]:
    """Per-day rows from a stored run, with dates parsed back to `date` objects."""
    days = []
    for day in run.result_summary.get("days", []):
        day = dict(day)
        day["date"] = date.fromisoformat(day["date"])
        days.append(day)
    return days
