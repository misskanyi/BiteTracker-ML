"""Forecast + planning orchestration over the existing src/ modules.

    request → validate → school history (data_service)
            → src.forecasting.forecast_school_demand (existing feature building)
            → saved model (model_service, verified, cached)
            → src.decision_support (buffer, recommended preparation, risk)
            → src.budgeting / src.resource_allocation (cost, capacity gap)
            → ForecastResult (plain Python)

No forecasting, feature-engineering or decision-support logic is
reimplemented here, and no model is trained. The day-1 decision-support
summary and cost estimate follow the former Streamlit Forecast page; the per-day
planning rows follow the former Streamlit Planning calendar (buffer grows with how
far ahead each day is).

Forecasts start from the last recorded date in the dataset, not from today.
`origin_date` and `origin_note` carry that so a UI can say so plainly.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd

from config import settings
from services import ServiceError
from services import calendar_service, data_service, model_service
from src.budgeting import estimate_meal_cost
from src.decision_support import build_decision_support_result
from src.feature_engineering import FEATURE_COLUMNS
from src.forecasting import forecast_school_demand
from src.resource_allocation import compute_allocation

# Horizons are counted in school days (Mon-Fri), matching
# src.forecasting._next_school_days.
HORIZONS: dict[int, str] = {
    1: "Next school day",
    7: "Next 7 school days",
    30: "Next 30 school days",
}

# The longest look-back any feature needs (lag_14_meals / rolling_mean_14).
# With less history those features would be NaN and the prediction meaningless.
# FEATURE_COLUMNS includes every Food4Education feature, so this covers both sets.
MIN_HISTORY_RECORDS = max(int(n) for column in FEATURE_COLUMNS for n in re.findall(r"_(\d+)", column))

# Same 14-day window the former Streamlit pages used for volatility and cost.
_RECENT_WINDOW = 14

TEMPERATURE_RANGE_C = (-10.0, 50.0)


class InvalidForecastRequestError(ServiceError):
    """The forecast request parameters are invalid."""


class InsufficientHistoryError(ServiceError):
    """The school does not have enough history to build the model's lag features."""


@dataclass
class ForecastResult:
    school_id: str
    school_name: str
    horizon_days: int
    horizon_label: str
    model_name: str
    results_label: str
    origin_date: date
    origin_note: str
    first_forecast_date: date
    last_forecast_date: date
    inputs: dict
    days: list[dict]
    decision_support: dict
    cost_per_meal: float
    cost_estimate: dict | None
    totals: dict
    allocation: dict | None
    warnings: list[str] = field(default_factory=list)
    # Training timestamp from the model metadata, identifying the model version.
    model_trained_at: str | None = None
    # The unmodified output of src.forecasting.forecast_school_demand.
    forecast_frame: pd.DataFrame | None = None
    # The school's last 21 records, for a "recent vs forecast" chart.
    recent_history: pd.DataFrame | None = None

    def to_dict(self) -> dict:
        """JSON-friendly view (dates as ISO strings, DataFrames as record lists)."""
        data = asdict(self)
        data["forecast_frame"] = _records(self.forecast_frame)
        data["recent_history"] = _records(self.recent_history)
        return _isoformat_dates(data)


def _records(frame: pd.DataFrame | None) -> list[dict] | None:
    return None if frame is None else frame.to_dict(orient="records")


def _isoformat_dates(value):
    if isinstance(value, dict):
        return {k: _isoformat_dates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_isoformat_dates(v) for v in value]
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    return value


def _validate_number(name: str, value, minimum: float | None = None, maximum: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidForecastRequestError(f"{name} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InvalidForecastRequestError(f"{name} must be a number.") from None
    if not math.isfinite(number):
        raise InvalidForecastRequestError(f"{name} must be a finite number.")
    if minimum is not None and number < minimum:
        raise InvalidForecastRequestError(f"{name} must be at least {minimum:g}.")
    if maximum is not None and number > maximum:
        raise InvalidForecastRequestError(f"{name} must be at most {maximum:g}.")
    return number


def validate_horizon(horizon_days) -> int:
    if isinstance(horizon_days, bool) or horizon_days not in HORIZONS:
        allowed = ", ".join(f"{days} ({label})" for days, label in HORIZONS.items())
        raise InvalidForecastRequestError(f"Forecast horizon must be one of: {allowed}.")
    return int(horizon_days)


def format_origin_note(origin: date, results_label: str) -> str:
    kind = "development data" if "SYNTHETIC" in results_label.upper() else "historical data"
    return f"Forecast generated from available {kind} ending {origin.day} {origin:%B %Y}."


def default_assumptions(school_id: str) -> dict:
    """What `generate_forecast` assumes when optional inputs are left blank, for one school.

    Uses the same 14-record window as `src.forecasting._recent_average`, so a
    form can show users the values that will be used.
    """
    history = data_service.get_school_history(school_id)
    recent = history.tail(_RECENT_WINDOW)
    return {
        "school_id": school_id,
        "origin_date": history[settings.DATE_COLUMN].max().date(),
        "last_recorded_attendance": float(history["students_present"].iloc[-1]),
        "recent_attendance": round(float(recent["students_present"].mean()), 1),
        "recent_rainfall_mm": round(float(recent["rainfall_mm"].mean()), 1),
        "recent_temperature_c": round(float(recent["temperature_c"].mean()), 1),
        "records": int(len(history)),
    }


def generate_forecast(
    school_id: str,
    horizon_days: int,
    expected_attendance: float | None = None,
    expected_rainfall_mm: float | None = None,
    expected_temperature_c: float | None = None,
    available_capacity: float | None = None,
) -> ForecastResult:
    """Forecast `horizon_days` school days for one school and build the planning recommendation.

    Optional inputs behave exactly as in `src.forecasting.forecast_school_demand`:
    omitted weather falls back to the school's recent 14-day average, and
    expected attendance (day 1) only affects day 2 onwards. `available_capacity`,
    if given, is compared with day-1 demand via `src.resource_allocation`.
    """
    horizon_days = validate_horizon(horizon_days)
    expected_attendance = _validate_number("Expected attendance", expected_attendance, minimum=0)
    expected_rainfall_mm = _validate_number("Expected rainfall (mm)", expected_rainfall_mm, minimum=0)
    expected_temperature_c = _validate_number(
        "Expected temperature (°C)", expected_temperature_c, *TEMPERATURE_RANGE_C
    )
    available_capacity = _validate_number("Available capacity", available_capacity, minimum=0)

    df = data_service.load_processed_data()
    history = data_service.get_school_history(school_id, df=df)
    if len(history) < MIN_HISTORY_RECORDS:
        raise InsufficientHistoryError(
            f"School '{school_id}' has {len(history)} recorded school day(s); at least "
            f"{MIN_HISTORY_RECORDS} are needed to build the model's lag and rolling features."
        )

    loaded = model_service.get_model()

    forecast_frame = forecast_school_demand(
        df,
        school_id,
        horizon_days,
        expected_attendance=expected_attendance,
        expected_rainfall_mm=expected_rainfall_mm,
        expected_temperature_c=expected_temperature_c,
        model=loaded.model,
        feature_columns=loaded.feature_columns,
    )

    recent = history.tail(_RECENT_WINDOW)
    recent_std = float(recent["meals_served"].std() or 0.0)
    cost_per_meal = float(recent["average_cost_per_meal"].mean()) if "average_cost_per_meal" in history else 0.0
    if math.isnan(cost_per_meal):
        cost_per_meal = 0.0

    days = []
    for i, row in forecast_frame.iterrows():
        predicted = float(row["predicted_meals_served"])
        ds = build_decision_support_result(
            predicted_meals=predicted,
            recent_std=recent_std,
            horizon_days=i + 1,
            horizon_label=f"{i + 1} school day(s) ahead",
            model_label=loaded.selected_model_name,
        )
        days.append(
            {
                "day_number": i + 1,
                "date": row["date"],
                "predicted_meals_served": predicted,
                "predicted_meals": ds.predicted_meals,
                "recommended_meals": ds.recommended_meals,
                "buffer_meals": ds.buffer_meals,
                "buffer_pct": ds.buffer_pct,
                "risk_level": ds.risk_level,
                "estimated_cost": estimate_meal_cost(predicted, cost_per_meal)["daily_cost"] if cost_per_meal else None,
                "attendance_assumption": row["attendance_assumption"],
                "rainfall_assumption": row["rainfall_assumption"],
                "temperature_assumption": row["temperature_assumption"],
            }
        )

    first = forecast_frame.iloc[0]
    day1_meals = float(first["predicted_meals_served"])
    day1 = build_decision_support_result(
        predicted_meals=day1_meals,
        recent_std=recent_std,
        horizon_days=1,
        horizon_label=HORIZONS[1],
        model_label=loaded.selected_model_name,
        assumptions=[
            f"Attendance basis: {first['attendance_assumption']}",
            f"Rainfall basis: {first['rainfall_assumption']}",
            f"Temperature basis: {first['temperature_assumption']}",
        ],
    )

    origin = history[settings.DATE_COLUMN].max().date()
    warnings = []
    calendar = calendar_service.get_active_calendar()
    holiday_days = [d["date"] for d in days if calendar.is_holiday(d["date"])]
    if holiday_days:
        warnings.append(
            f"{len(holiday_days)} of the {horizon_days} forecast day(s) fall within a school holiday in the "
            f"{calendar.name.lower()}; demand on those days is expected to be low."
        )
    if horizon_days > 1:
        warnings.append(
            "Later days reuse earlier predictions as if they were observed, so uncertainty grows across the horizon."
        )

    return ForecastResult(
        school_id=school_id,
        school_name=str(history["school_name"].iloc[-1]),
        horizon_days=horizon_days,
        horizon_label=HORIZONS[horizon_days],
        model_name=loaded.selected_model_name,
        results_label=loaded.results_label,
        origin_date=origin,
        origin_note=format_origin_note(origin, loaded.results_label),
        first_forecast_date=days[0]["date"],
        last_forecast_date=days[-1]["date"],
        inputs={
            "expected_attendance": expected_attendance,
            "expected_rainfall_mm": expected_rainfall_mm,
            "expected_temperature_c": expected_temperature_c,
            "available_capacity": available_capacity,
        },
        days=days,
        decision_support=asdict(day1),
        cost_per_meal=round(cost_per_meal, 2),
        cost_estimate=estimate_meal_cost(day1_meals, cost_per_meal) if cost_per_meal else None,
        totals={
            "predicted_meals": sum(d["predicted_meals"] for d in days),
            "recommended_meals": sum(d["recommended_meals"] for d in days),
            "estimated_cost": round(sum(d["estimated_cost"] for d in days), 2) if cost_per_meal else None,
        },
        allocation=compute_allocation(day1_meals, available_capacity) if available_capacity is not None else None,
        warnings=warnings,
        model_trained_at=loaded.metadata.get("timestamp"),
        forecast_frame=forecast_frame,
        recent_history=history.tail(21).reset_index(drop=True),
    )
