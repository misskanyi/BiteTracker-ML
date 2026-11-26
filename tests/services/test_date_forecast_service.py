"""Meal estimates for selected school days (services.date_forecast_service).

What this file checks:
- The next school day after the latest record uses the saved model, matching day 1 of a horizon forecast.
- Later dates use a direct estimate chosen by held-out MAE; the weekday baseline is used and
  labelled when it wins; distances without held-out evidence are marked experimental.
- Date validation: on/before the latest record, weekends, empty and over-long ranges, too far ahead.
- Ranges skip weekends; weather is never an input.
"""

from __future__ import annotations

import inspect
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from services import date_forecast_service as dfs
from services import forecast_service
from services.forecast_service import InvalidForecastRequestError

TODAY = date(2023, 6, 1)


@pytest.fixture
def latest(stub_artifacts):
    frame = stub_artifacts.frame
    return pd.to_datetime(frame.loc[frame["school_id"] == "SCH001", "date"]).max().date()


def next_weekday(d: date, n: int = 1) -> date:
    while n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def test_school_days_between_counts_weekdays():
    friday = date(2026, 11, 27)
    assert dfs.school_days_between(friday, date(2026, 11, 30)) == 1  # Monday
    assert dfs.school_days_between(friday, date(2026, 12, 4)) == 5


def test_next_school_day_uses_saved_model(latest):
    day = next_weekday(latest)
    result = dfs.forecast_selected_dates("SCH001", dfs.SINGLE_DAY, day, today=TODAY)
    horizon = forecast_service.generate_forecast("SCH001", 1)
    (row,) = result.days
    assert row["method"] == dfs.MODEL and row["school_days_ahead"] == 1 and not row["experimental"]
    assert row["predicted_meals"] == horizon.days[0]["predicted_meals"]
    assert result.origin_date == latest and result.selection["latest_record"] == latest


def test_later_dates_use_direct_estimate_with_evidence_label(latest):
    day = next_weekday(latest, 8)
    result = dfs.forecast_selected_dates("SCH001", dfs.SINGLE_DAY, day, today=TODAY)
    (row,) = result.days
    assert row["method"] in (dfs.PLANNING, dfs.BASELINE)
    assert row["school_days_ahead"] == 8
    assert result.method_evaluation["buckets"] and row["predicted_meals"] > 0


def test_far_dates_without_evidence_are_experimental(latest):
    day = next_weekday(latest, 200)
    result = dfs.forecast_selected_dates("SCH001", dfs.SINGLE_DAY, day, today=day)
    (row,) = result.days
    assert row["experimental"] and "experimental" in " ".join(result.warnings)
    assert result.decision_support["uncertainty"].startswith("Direct estimate 200 school days")
    assert "recursive" not in result.decision_support["uncertainty"]


def _school_frame(meals: list[float], start: date) -> pd.DataFrame:
    days, d = [], start
    while len(days) < len(meals):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return pd.DataFrame({"date": pd.to_datetime(days), "school_id": "S", "meals_served": meals})


def test_baseline_is_chosen_when_it_beats_planning_on_held_out_data():
    """A stable weekday pattern with a one-off spike just before the cut-off misleads the recent level."""
    pattern = [100.0, 200.0, 300.0, 400.0, 500.0]
    meals = [pattern[i % 5] for i in range(400)]
    frame = _school_frame(meals, date(2023, 1, 2))
    cut = frame["date"].iloc[299].date()
    frame.loc[280:299, "meals_served"] *= 3  # last 20 training records: a temporary spike
    evaluation = dfs.evaluate_methods(frame, {"test_date_range": {"start": (cut + timedelta(days=1)).isoformat()}})
    evidenced = [b for b in evaluation["buckets"] if b["evidence"]]
    assert evidenced and all(b["chosen"] == dfs.BASELINE for b in evidenced)
    assert all(b["baseline_mae"] < b["planning_mae"] for b in evidenced)


def test_planning_is_chosen_when_level_shift_makes_it_better():
    pattern = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    meals = [pattern[i % 5] * (1 if i < 250 else 2) for i in range(400)]  # level doubles and stays
    frame = _school_frame(meals, date(2023, 1, 2))
    cut = frame["date"].iloc[299].date()
    evaluation = dfs.evaluate_methods(frame, {"test_date_range": {"start": (cut + timedelta(days=1)).isoformat()}})
    evidenced = [b for b in evaluation["buckets"] if b["evidence"]]
    assert evidenced and all(b["chosen"] == dfs.PLANNING for b in evidenced)


def test_method_for_marks_unevidenced_buckets_experimental():
    evaluation = {"fallback_method": dfs.BASELINE, "buckets": [
        {"low": 2, "high": 5, "evidence": False, "chosen": None},
        {"low": 6, "high": 20, "evidence": True, "chosen": dfs.BASELINE},
        {"low": 21, "high": None, "evidence": False, "chosen": None},
    ]}
    assert dfs.method_for(3, evaluation)[:2] == (dfs.BASELINE, True)
    assert dfs.method_for(10, evaluation)[:2] == (dfs.BASELINE, False)
    assert dfs.method_for(500, evaluation)[:2] == (dfs.BASELINE, True)


def test_baseline_days_are_labelled(latest, monkeypatch):
    original = dfs.evaluate_methods

    def baseline_wins(df, metadata=None):
        evaluation = original(df, metadata)
        for b in evaluation["buckets"]:
            b.update(evidence=True, chosen=dfs.BASELINE, n=50, planning_mae=20.0, baseline_mae=10.0)
        return evaluation

    monkeypatch.setattr(dfs, "evaluate_methods", baseline_wins)
    result = dfs.forecast_selected_dates("SCH001", dfs.SINGLE_DAY, next_weekday(latest, 10), today=TODAY)
    (row,) = result.days
    assert row["method"] == dfs.BASELINE and "baseline" in row["method_label"].lower()
    assert "beat the planning estimate" in row["method_label"]


def test_range_skips_weekends_and_mixes_methods(latest):
    start, end = next_weekday(latest), next_weekday(latest, 6)
    result = dfs.forecast_selected_dates("SCH001", dfs.DATE_RANGE, start, end, today=TODAY)
    assert all(d["date"].weekday() < 5 for d in result.days)
    assert result.days[0]["method"] == dfs.MODEL
    assert {d["method"] for d in result.days[1:]} <= {dfs.PLANNING, dfs.BASELINE}
    assert result.horizon_days == len(result.days) == 6


@pytest.mark.parametrize("case", ["on_latest", "weekend", "reversed", "too_long", "too_far"])
def test_invalid_selections(latest, case):
    if case == "on_latest":
        args = (dfs.SINGLE_DAY, latest, None)
    elif case == "weekend":
        saturday = latest + timedelta(days=(5 - latest.weekday()) % 7 or 7)
        args = (dfs.SINGLE_DAY, saturday, None)
    elif case == "reversed":
        args = (dfs.DATE_RANGE, next_weekday(latest, 3), next_weekday(latest))
    elif case == "too_long":
        args = (dfs.DATE_RANGE, next_weekday(latest), next_weekday(latest, dfs.MAX_RANGE_SCHOOL_DAYS + 1))
    else:
        args = (dfs.SINGLE_DAY, TODAY + timedelta(days=dfs.MAX_DAYS_AFTER_TODAY + 7), None)
    with pytest.raises(InvalidForecastRequestError):
        dfs.forecast_selected_dates("SCH001", *args, today=TODAY)


def test_weather_is_not_an_input():
    params = inspect.signature(dfs.forecast_selected_dates).parameters
    assert not any("rain" in p or "temperature" in p or "weather" in p for p in params)


class _Calendar:
    name = "Test calendar"

    def __init__(self, holidays):
        self.holidays = set(holidays)

    def is_holiday(self, d):
        return d in self.holidays


def test_range_excludes_calendar_holidays_and_single_holiday_is_rejected(latest, monkeypatch):
    start = next_weekday(latest, 2)
    holiday = next_weekday(latest, 3)
    end = next_weekday(latest, 5)
    monkeypatch.setattr(dfs.calendar_service, "get_active_calendar", lambda: _Calendar({holiday}))
    result = dfs.forecast_selected_dates("SCH001", dfs.DATE_RANGE, start, end, today=TODAY)
    assert holiday not in [d["date"] for d in result.days] and len(result.days) == 3
    assert any("school holidays" in w for w in result.warnings)
    with pytest.raises(InvalidForecastRequestError, match="school holiday"):
        dfs.forecast_selected_dates("SCH001", dfs.SINGLE_DAY, holiday, today=TODAY)
