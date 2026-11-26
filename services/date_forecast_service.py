"""Meal estimates for user-selected school days (a single day or a date range).

The 1-, 7- and 30-school-day horizons (`forecast_service.generate_forecast`) are
unchanged. This module handles dates the user picks:

- The next supported school day after a school's latest usable record uses the
  saved model (`src.forecasting.forecast_school_demand`, horizon 1), exactly as
  day 1 of a horizon forecast.
- Any later date uses a DIRECT estimate (no recursion through predicted days):
  either the planning estimate or the school weekday-average baseline,
  whichever had the lower MAE on held-out observed data for that distance
  ahead (see `evaluate_methods`).

Planning estimate (multiplicative profile, per school, from recorded data only):

    estimate = level × weekday factor(day of week) × month factor(month)

    weekday / month factor = mean meals on that weekday / in that month ÷ the
        school's mean, shrunk towards 1 by SHRINK_RECORDS pseudo-records so
        thin months do not swing the estimate.
    level = mean of the last LEVEL_WINDOW records after dividing out their own
        weekday and month factors (the school's current, de-seasonalised level).

Baseline: the school's mean meals served on the same weekday.

Evaluation: both methods are fitted on records up to the saved model's training
cut-off (or the first 80% of dates if the metadata has none) and scored on every
later observed record, grouped by how many school days ahead of the school's last
training record it falls. Horizons with fewer than MIN_EVAL_POINTS held-out
records have no evaluation evidence, so their estimates are marked experimental.

Weather is not used here (the weather advisory is separate and display-only).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import settings
from services import ServiceError, calendar_service, data_service, model_service
from services.forecast_service import (
    MIN_HISTORY_RECORDS, InsufficientHistoryError, InvalidForecastRequestError, ForecastResult, _RECENT_WINDOW,
    _validate_number, format_origin_note,
)
from src.budgeting import estimate_meal_cost
from src.decision_support import build_decision_support_result
from src.forecasting import forecast_school_demand
from src.resource_allocation import compute_allocation

SINGLE_DAY, DATE_RANGE = "single", "range"
MODES = {SINGLE_DAY: "Single day", DATE_RANGE: "Date range"}
MAX_RANGE_SCHOOL_DAYS = 30
MAX_DAYS_AFTER_TODAY = 366

LEVEL_WINDOW = 20
SHRINK_RECORDS = 10
MIN_EVAL_POINTS = 20
# School days ahead of the latest record (day 1 is always the saved model).
HORIZON_BUCKETS = ((2, 5), (6, 20), (21, 60), (61, 130), (131, None))

MODEL, PLANNING, BASELINE = "model", "planning", "weekday_baseline"
METHOD_LABELS = {
    PLANNING: "Planning estimate (meal level × weekday × month)",
    BASELINE: "School weekday average (baseline)",
}


@dataclass
class SchoolProfile:
    overall_mean: float
    weekday_mean: dict[int, float]
    weekday_factor: dict[int, float]
    month_factor: dict[int, float]
    level: float

    def planning_estimate(self, d: date) -> float:
        return max(0.0, self.level * self.weekday_factor.get(d.weekday(), 1.0) * self.month_factor.get(d.month, 1.0))

    def weekday_baseline(self, d: date) -> float:
        return max(0.0, self.weekday_mean.get(d.weekday(), self.overall_mean))

    def estimate(self, method: str, d: date) -> float:
        return self.planning_estimate(d) if method == PLANNING else self.weekday_baseline(d)


def school_days_between(last: date, d: date) -> int:
    """Weekdays in (last, d]: 1 for the next weekday after `last`."""
    return int(np.busday_count(last + timedelta(days=1), d + timedelta(days=1)))


def _shrunk_factors(meals: pd.Series, keys: pd.Series, overall: float) -> dict[int, float]:
    grouped = meals.groupby(keys).agg(["sum", "count"])
    return {
        int(k): float((row["sum"] + SHRINK_RECORDS * overall) / (row["count"] + SHRINK_RECORDS) / overall)
        for k, row in grouped.iterrows()
    }


def fit_profile(history: pd.DataFrame) -> SchoolProfile:
    """A school's profile from its recorded rows (needs `date` and `meals_served`)."""
    history = history.dropna(subset=["meals_served"]).sort_values(settings.DATE_COLUMN)
    if history.empty:
        raise InsufficientHistoryError("The school has no recorded meals to estimate from.")
    meals = history["meals_served"].astype(float)
    dates = pd.to_datetime(history[settings.DATE_COLUMN])
    overall = float(meals.mean())
    if overall <= 0:
        zero = {i: 0.0 for i in range(5)}
        return SchoolProfile(0.0, zero, {i: 1.0 for i in range(5)}, {}, 0.0)
    weekday, month = dates.dt.weekday, dates.dt.month
    weekday_factor = _shrunk_factors(meals, weekday, overall)
    month_factor = _shrunk_factors(meals, month, overall)
    weekday_mean = {int(k): float(v) for k, v in meals.groupby(weekday).mean().items()}
    recent = history.tail(LEVEL_WINDOW)
    recent_dates = pd.to_datetime(recent[settings.DATE_COLUMN])
    divisors = [weekday_factor.get(d.weekday(), 1.0) * month_factor.get(d.month, 1.0) for d in recent_dates]
    level = float(np.mean([m / f if f else m for m, f in zip(recent["meals_served"].astype(float), divisors)]))
    return SchoolProfile(overall, weekday_mean, weekday_factor, month_factor, level)


def _bucket_label(low: int, high: int | None) -> str:
    return f"{low}+ school days ahead" if high is None else f"{low}–{high} school days ahead"


def _train_end(df: pd.DataFrame, metadata: dict | None) -> date:
    start = ((metadata or {}).get("test_date_range") or {}).get("start")
    if start:
        return date.fromisoformat(start) - timedelta(days=1)
    dates = np.sort(df[settings.DATE_COLUMN].dt.date.unique())
    return dates[max(int(len(dates) * 0.8) - 1, 0)]


def evaluate_methods(df: pd.DataFrame, metadata: dict | None = None) -> dict:
    """Held-out MAE of the planning estimate and weekday baseline by school days ahead.

    Fitted on records up to the training cut-off, scored on every later observed record.
    """
    train_end = _train_end(df, metadata)
    data = df.dropna(subset=["meals_served"])
    rows = []
    for school_id, school in data.groupby(settings.SCHOOL_ID_COLUMN):
        train = school[school[settings.DATE_COLUMN].dt.date <= train_end]
        test = school[school[settings.DATE_COLUMN].dt.date > train_end]
        if len(train) < MIN_HISTORY_RECORDS or test.empty:
            continue
        profile = fit_profile(train)
        last = train[settings.DATE_COLUMN].max().date()
        for d, actual in zip(test[settings.DATE_COLUMN].dt.date, test["meals_served"].astype(float)):
            rows.append({
                "ahead": school_days_between(last, d),
                PLANNING: abs(profile.planning_estimate(d) - actual),
                BASELINE: abs(profile.weekday_baseline(d) - actual),
            })
    errors = pd.DataFrame(rows, columns=["ahead", PLANNING, BASELINE])

    buckets = []
    for low, high in HORIZON_BUCKETS:
        in_bucket = errors[(errors["ahead"] >= low) & ((errors["ahead"] <= high) if high is not None else True)]
        n = int(len(in_bucket))
        evidence = n >= MIN_EVAL_POINTS
        planning_mae = round(float(in_bucket[PLANNING].mean()), 1) if n else None
        baseline_mae = round(float(in_bucket[BASELINE].mean()), 1) if n else None
        chosen = None
        if evidence:
            chosen = BASELINE if baseline_mae < planning_mae else PLANNING
        buckets.append({
            "low": low, "high": high, "label": _bucket_label(low, high), "n": n, "evidence": evidence,
            "planning_mae": planning_mae, "baseline_mae": baseline_mae, "chosen": chosen,
        })
    evidenced = [b for b in buckets if b["evidence"]]
    return {
        "train_end": train_end.isoformat(),
        "held_out_records": int(len(errors)),
        "max_evidenced_ahead": max((int(errors["ahead"].max()) if b["high"] is None else b["high"]) for b in evidenced)
        if evidenced else None,
        "buckets": buckets,
        "fallback_method": evidenced[-1]["chosen"] if evidenced else PLANNING,
    }


def method_for(ahead: int, evaluation: dict) -> tuple[str, bool, dict | None]:
    """(method, experimental, bucket) for a date `ahead` school days after the latest record (ahead >= 2)."""
    for bucket in evaluation["buckets"]:
        if ahead >= bucket["low"] and (bucket["high"] is None or ahead <= bucket["high"]):
            if bucket["evidence"]:
                return bucket["chosen"], False, bucket
            return evaluation["fallback_method"], True, bucket
    return evaluation["fallback_method"], True, None


def _calendar_excluded(calendar, d: date) -> bool:
    return calendar.is_holiday(d)


def school_days_in_range(start: date, end: date, calendar=None) -> tuple[list[date], list[date]]:
    """(school days, excluded holiday weekdays) from `start` to `end` inclusive; weekends are skipped."""
    calendar = calendar or calendar_service.get_active_calendar()
    days, holidays, current = [], [], start
    while current <= end:
        if current.weekday() < 5:
            (holidays if _calendar_excluded(calendar, current) else days).append(current)
        current += timedelta(days=1)
    return days, holidays


def _decision_support(result, first_day: dict) -> dict:
    """The shared decision-support bundle, with uncertainty worded for a direct (non-recursive) estimate."""
    data = asdict(result)
    if first_day["method"] != MODEL:
        ahead = first_day["school_days_ahead"]
        data["uncertainty"] = (
            f"Direct estimate {ahead} school days after the latest record"
            + (" with no held-out evidence at this distance; experimental." if first_day["experimental"]
               else "; treat as directional.")
        )
    return data


def forecast_selected_dates(
    school_id: str,
    mode: str,
    start: date,
    end: date | None = None,
    expected_attendance: float | None = None,
    available_capacity: float | None = None,
    today: date | None = None,
) -> ForecastResult:
    """Meal estimates and planning recommendations for the selected school day(s)."""
    if mode not in MODES:
        raise InvalidForecastRequestError("Choose a single day or a date range.")
    end = start if mode == SINGLE_DAY else end
    if end is None or end < start:
        raise InvalidForecastRequestError("The end date must be on or after the start date.")
    today = today or date.today()
    if end > today + timedelta(days=MAX_DAYS_AFTER_TODAY):
        raise InvalidForecastRequestError(f"Choose dates up to {MAX_DAYS_AFTER_TODAY} days from today.")
    expected_attendance = _validate_number("Expected attendance", expected_attendance, minimum=0)
    available_capacity = _validate_number("Available capacity", available_capacity, minimum=0)

    df = data_service.load_processed_data()
    history = data_service.get_school_history(school_id, df=df).dropna(subset=["meals_served"]).reset_index(drop=True)
    if len(history) < MIN_HISTORY_RECORDS:
        raise InsufficientHistoryError(
            f"School '{school_id}' has {len(history)} recorded school day(s); at least "
            f"{MIN_HISTORY_RECORDS} are needed."
        )
    latest = history[settings.DATE_COLUMN].max().date()
    if start <= latest:
        raise InvalidForecastRequestError(
            f"Choose dates after this school's latest record ({latest.day} {latest:%B %Y})."
        )

    calendar = calendar_service.get_active_calendar()
    if mode == SINGLE_DAY:
        if start.weekday() >= 5:
            raise InvalidForecastRequestError(f"{start:%A} {start.day} {start:%B %Y} is not a school day (Mon–Fri).")
        if _calendar_excluded(calendar, start):
            raise InvalidForecastRequestError(
                f"{start:%A} {start.day} {start:%B %Y} is a school holiday in the {calendar.name.lower()}."
            )
        dates, holidays = [start], []
    else:
        dates, holidays = school_days_in_range(start, end, calendar)
        if not dates:
            raise InvalidForecastRequestError("The selected range has no school days (Mon–Fri, outside holidays).")
        if len(dates) > MAX_RANGE_SCHOOL_DAYS:
            raise InvalidForecastRequestError(
                f"The selected range has {len(dates)} school days; choose at most {MAX_RANGE_SCHOOL_DAYS}."
            )

    loaded_metadata = model_service.load_model_metadata()
    evaluation = evaluate_methods(df, loaded_metadata)
    profile = fit_profile(history)

    recent = history.tail(_RECENT_WINDOW)
    recent_std = float(recent["meals_served"].std() or 0.0)
    cost_per_meal = float(recent["average_cost_per_meal"].mean()) if "average_cost_per_meal" in history else 0.0
    if math.isnan(cost_per_meal):
        cost_per_meal = 0.0

    model_label, model_trained_at = None, loaded_metadata.get("timestamp")
    days = []
    for i, d in enumerate(dates):
        ahead = school_days_between(latest, d)
        if ahead == 1:
            loaded = model_service.get_model()
            model_label = loaded.selected_model_name
            frame = forecast_school_demand(
                df, school_id, 1, expected_attendance=expected_attendance,
                model=loaded.model, feature_columns=loaded.feature_columns,
            )
            predicted = float(frame.iloc[0]["predicted_meals_served"])
            method, experimental, label = MODEL, False, f"{model_label} (saved model)"
            evidence = "Held-out test period of the saved model"
        else:
            method, experimental, bucket = method_for(ahead, evaluation)
            predicted = profile.estimate(method, d)
            label = METHOD_LABELS[method]
            if method == BASELINE:
                label += " · beat the planning estimate on held-out data"
            evidence = (f"{bucket['label']}: planning MAE {bucket['planning_mae']}, baseline MAE {bucket['baseline_mae']} "
                        f"({bucket['n']} held-out records)") if bucket and bucket["evidence"] else "No held-out evidence"
        ds = build_decision_support_result(
            predicted_meals=predicted, recent_std=recent_std, horizon_days=ahead,
            horizon_label=f"{ahead} school day(s) ahead", model_label=label,
        )
        days.append({
            "day_number": i + 1,
            "date": d,
            "school_days_ahead": ahead,
            "predicted_meals_served": predicted,
            "predicted_meals": ds.predicted_meals,
            "recommended_meals": ds.recommended_meals,
            "buffer_meals": ds.buffer_meals,
            "buffer_pct": ds.buffer_pct,
            "risk_level": ds.risk_level,
            "estimated_cost": estimate_meal_cost(predicted, cost_per_meal)["daily_cost"] if cost_per_meal else None,
            "method": method,
            "method_label": label,
            "experimental": experimental,
            "evidence": evidence,
            "attendance_assumption": "last recorded day (observed)" if method == MODEL else "not used",
            "rainfall_assumption": "not used",
            "temperature_assumption": "not used",
        })

    first = days[0]
    day1 = build_decision_support_result(
        predicted_meals=first["predicted_meals_served"], recent_std=recent_std,
        horizon_days=first["school_days_ahead"], horizon_label=f"{first['school_days_ahead']} school day(s) ahead",
        model_label=first["method_label"],
        assumptions=[f"Method: {first['method_label']}", f"Latest usable record: {latest.isoformat()}"],
    )

    warnings = []
    experimental = [d for d in days if d["experimental"]]
    if experimental:
        limit = evaluation["max_evidenced_ahead"]
        warnings.append(
            f"{len(experimental)} of the {len(days)} selected day(s) are further ahead than any held-out evaluation"
            + (f" (evidence covers up to {limit} school days after the latest record)" if limit else "")
            + ", so their estimates are experimental."
        )
    if any(d["method"] == BASELINE for d in days):
        warnings.append("For some distances ahead the school weekday average beat the planning estimate on "
                        "held-out data, so it is used and labelled as the baseline.")
    if holidays:
        warnings.append(f"{len(holidays)} weekday(s) in the range are school holidays in the "
                        f"{calendar.name.lower()} and were left out.")
    unknown = [d for d in dates if hasattr(calendar, "classify") and calendar.classify(d).kind == calendar_service.UNKNOWN]
    if unknown:
        warnings.append("The school calendar for some selected dates is not configured, so holidays could not be "
                        "excluded. Check they are school days.")
    if expected_attendance is not None and any(d["method"] != MODEL for d in days):
        warnings.append("Expected attendance is used only by the saved model's day; direct estimates do not use it.")

    methods_used = sorted({d["method_label"].split(" · ")[0] for d in days})
    selection = {"mode": mode, "mode_label": MODES[mode], "start": dates[0], "end": dates[-1],
                 "requested_start": start, "requested_end": end, "latest_record": latest,
                 "excluded_holidays": holidays}
    return ForecastResult(
        school_id=school_id,
        school_name=str(history["school_name"].iloc[-1]),
        horizon_days=len(days),
        horizon_label=MODES[mode],
        model_name="; ".join(methods_used)[:100],
        results_label=loaded_metadata.get("results_label", ""),
        origin_date=latest,
        origin_note=format_origin_note(latest, loaded_metadata.get("results_label", "")),
        first_forecast_date=dates[0],
        last_forecast_date=dates[-1],
        inputs={"expected_attendance": expected_attendance, "available_capacity": available_capacity},
        days=days,
        decision_support=_decision_support(day1, first),
        cost_per_meal=round(cost_per_meal, 2),
        cost_estimate=estimate_meal_cost(first["predicted_meals_served"], cost_per_meal) if cost_per_meal else None,
        totals={
            "predicted_meals": sum(d["predicted_meals"] for d in days),
            "recommended_meals": sum(d["recommended_meals"] for d in days),
            "estimated_cost": round(sum(d["estimated_cost"] for d in days), 2) if cost_per_meal else None,
        },
        allocation=compute_allocation(first["predicted_meals_served"], available_capacity)
        if available_capacity is not None else None,
        warnings=warnings,
        model_trained_at=model_trained_at,
        forecast_frame=None,
        recent_history=history.tail(21).reset_index(drop=True),
        selection=selection,
        method_evaluation=evaluation,
    )


__all__ = [
    "SINGLE_DAY", "DATE_RANGE", "MODES", "MAX_RANGE_SCHOOL_DAYS", "evaluate_methods", "fit_profile",
    "forecast_selected_dates", "method_for", "school_days_between", "school_days_in_range", "ServiceError",
]
