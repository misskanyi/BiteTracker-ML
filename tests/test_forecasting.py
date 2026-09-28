"""Tests for src/forecasting.py — multi-horizon, recursive, leakage-safe forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from config import settings
from src.feature_engineering import FEATURE_COLUMNS
from src.forecasting import (
    _build_feature_row,
    _next_school_days,
    forecast_school_demand,
    get_school_history,
    predict_specific_date,
)


def _fake_processed_df(n_days: int = 60) -> pd.DataFrame:
    """Minimal processed-shape DataFrame with real column names for two schools."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=n_days)  # business days only
    rows = []
    for school, base in [("SCH001", 300), ("SCH002", 500)]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "school_id": school,
                    "school_name": f"{school} Primary",
                    "enrolled_students": 400 if school == "SCH001" else 600,
                    "students_present": base * 0.9 + rng.normal(0, 5),
                    "meals_served": base + rng.normal(0, 10),
                    "rainfall_mm": max(0.0, rng.normal(3, 2)),
                    "temperature_c": rng.normal(22, 1),
                    "is_holiday": 0,
                    "is_exam_period": 0,
                    "is_school_event": 0,
                    "month": d.month,
                }
            )
    return pd.DataFrame(rows)


class _StubModel:
    """Deterministic stand-in for a trained regressor: predicts lag_1_meals + 1."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["lag_1_meals"].fillna(300) + 1).to_numpy()


def test_get_school_history_sorted_and_filtered():
    df = _fake_processed_df()
    history = get_school_history(df, "SCH001")
    assert (history["school_id"] == "SCH001").all()
    assert history[settings.DATE_COLUMN].is_monotonic_increasing


def test_get_school_history_invalid_school_id_raises():
    df = _fake_processed_df()
    with pytest.raises(ValueError):
        forecast_school_demand(df, "NOT_A_REAL_SCHOOL", horizon_days=1, model=_StubModel())


def test_one_day_forecast_shape_and_columns():
    df = _fake_processed_df()
    result = forecast_school_demand(df, "SCH001", horizon_days=1, model=_StubModel())
    assert len(result) == 1
    expected_cols = {
        "date", "school_id", "predicted_meals_served",
        "attendance_assumption", "rainfall_assumption", "temperature_assumption",
    }
    assert expected_cols.issubset(result.columns)


def test_seven_day_forecast_returns_seven_school_days():
    df = _fake_processed_df()
    result = forecast_school_demand(df, "SCH001", horizon_days=7, model=_StubModel())
    assert len(result) == 7
    # All forecast dates must be strictly after the last historical date.
    last_hist_date = df[df["school_id"] == "SCH001"][settings.DATE_COLUMN].max()
    assert (pd.to_datetime(result["date"]) > last_hist_date).all()
    # Forecast dates must be weekdays only (school days).
    assert all(pd.Timestamp(d).weekday() < 5 for d in result["date"])


def test_thirty_day_forecast_returns_thirty_school_days():
    df = _fake_processed_df()
    result = forecast_school_demand(df, "SCH002", horizon_days=30, model=_StubModel())
    assert len(result) == 30
    assert result["date"].is_monotonic_increasing


def test_recursive_feature_updates_use_predicted_values_as_history():
    """With the stub model (predicts lag_1_meals + 1), each day's forecast must be
    exactly 1 higher than the previous day's forecast — proving predictions are fed
    back into the lag history rather than the model always seeing the same lag_1."""
    df = _fake_processed_df()
    result = forecast_school_demand(df, "SCH001", horizon_days=5, model=_StubModel())
    diffs = result["predicted_meals_served"].diff().dropna()
    assert np.allclose(diffs, 1.0)


def test_missing_optional_inputs_falls_back_to_recent_average():
    df = _fake_processed_df()
    result = forecast_school_demand(df, "SCH001", horizon_days=3, model=_StubModel())
    # Day 1's attendance features come from the last recorded (observed) day.
    assert result.iloc[0]["attendance_assumption"] == "last recorded day (observed)"
    assert (result.iloc[1:]["attendance_assumption"] == "recent 14-day average").all()
    assert (result["rainfall_assumption"] == "recent 14-day average").all()
    assert (result["temperature_assumption"] == "recent 14-day average").all()


def test_user_provided_inputs_are_labelled_as_such():
    df = _fake_processed_df()
    result = forecast_school_demand(
        df, "SCH001", horizon_days=3,
        expected_attendance=350, expected_rainfall_mm=10.0, expected_temperature_c=25.0,
        model=_StubModel(),
    )
    # Expected day-1 attendance can only feed day 2 (as its previous-day attendance).
    assert list(result["attendance_assumption"]) == [
        "last recorded day (observed)", "user-provided (day 1)", "recent 14-day average",
    ]
    assert (result["rainfall_assumption"] == "user-provided").all()
    assert (result["temperature_assumption"] == "user-provided").all()


def test_day1_forecast_ignores_expected_attendance():
    """A day's own attendance is not a model input, so day 1 can't depend on it."""

    class _AttendanceEcho:
        def predict(self, X):
            return X["previous_attendance"].to_numpy()

    df = _fake_processed_df()
    last_present = get_school_history(df, "SCH001")["students_present"].iloc[-1]
    low = forecast_school_demand(df, "SCH001", horizon_days=2, expected_attendance=10, model=_AttendanceEcho())
    high = forecast_school_demand(df, "SCH001", horizon_days=2, expected_attendance=999, model=_AttendanceEcho())
    assert low.iloc[0]["predicted_meals_served"] == high.iloc[0]["predicted_meals_served"] == last_present
    assert low.iloc[1]["predicted_meals_served"] == 10
    assert high.iloc[1]["predicted_meals_served"] == 999


def test_predictions_never_negative():
    df = _fake_processed_df()

    class _NegativeStub:
        def predict(self, X):
            return np.full(len(X), -50.0)

    result = forecast_school_demand(df, "SCH001", horizon_days=3, model=_NegativeStub())
    assert (result["predicted_meals_served"] >= 0).all()


def test_forecast_reproducible_with_same_inputs():
    df = _fake_processed_df()
    result_a = forecast_school_demand(df, "SCH001", horizon_days=7, model=_StubModel())
    result_b = forecast_school_demand(df, "SCH001", horizon_days=7, model=_StubModel())
    pd.testing.assert_frame_equal(result_a, result_b)


def test_forecast_with_real_model_type_runs_end_to_end():
    """Sanity check against a real (if untuned) sklearn regressor, not just the stub."""
    df = _fake_processed_df(n_days=40)
    train_rows = []
    for school in df["school_id"].unique():
        hist = get_school_history(df, school)
        for i in range(14, len(hist)):
            feature_row = _build_feature_row(
                forecast_date=hist.iloc[i][settings.DATE_COLUMN].date(),
                meals_history=hist["meals_served"].iloc[:i].tolist(),
                students_present_history=hist["students_present"].iloc[:i].tolist(),
                enrolled_students=float(hist["enrolled_students"].iloc[i]),
                rainfall_mm=float(hist["rainfall_mm"].iloc[i]),
                temperature_c=float(hist["temperature_c"].iloc[i]),
            )
            feature_row[settings.TARGET_COLUMN] = hist["meals_served"].iloc[i]
            train_rows.append(feature_row)
    train_df = pd.DataFrame(train_rows)
    model = DecisionTreeRegressor(random_state=42).fit(train_df[FEATURE_COLUMNS], train_df[settings.TARGET_COLUMN])

    result = forecast_school_demand(df, "SCH001", horizon_days=7, model=model)
    assert len(result) == 7
    assert result["predicted_meals_served"].notna().all()


def test_next_school_days_skips_weekends():
    from datetime import date

    days = _next_school_days(date(2024, 1, 5), 3)  # Friday
    assert all(d.weekday() < 5 for d in days)
    assert days[0] == date(2024, 1, 8)  # next Monday


def test_predict_specific_date_returns_expected_keys():
    from datetime import timedelta

    df = _fake_processed_df()
    last_date = get_school_history(df, "SCH001")[settings.DATE_COLUMN].max().date()
    target = last_date + timedelta(days=10)
    while target.weekday() >= 5:
        target += timedelta(days=1)

    result = predict_specific_date(
        df, "SCH001", target,
        expected_rainfall_mm=2.0, expected_temperature_c=23.0,
        school_term="Term 2", is_holiday=False, is_exam_period=False,
        model=_StubModel(),
    )
    assert result["predicted_meals_served"] >= 0
    assert result["target_date"] == target
    assert result["days_ahead"] > 0
    assert set(FEATURE_COLUMNS).issubset(result["feature_row"].keys())


def test_predict_specific_date_rejects_past_or_current_date():
    df = _fake_processed_df()
    last_date = get_school_history(df, "SCH001")[settings.DATE_COLUMN].max().date()
    with pytest.raises(ValueError):
        predict_specific_date(
            df, "SCH001", last_date,
            expected_rainfall_mm=2.0, expected_temperature_c=23.0,
            school_term="Term 2", is_holiday=False, is_exam_period=False,
            model=_StubModel(),
        )


def test_predict_specific_date_invalid_school_raises():
    df = _fake_processed_df()
    from datetime import date

    with pytest.raises(ValueError):
        predict_specific_date(
            df, "NOT_A_REAL_SCHOOL", date(2030, 1, 1),
            expected_rainfall_mm=2.0, expected_temperature_c=23.0,
            school_term="Term 2", is_holiday=False, is_exam_period=False,
            model=_StubModel(),
        )
