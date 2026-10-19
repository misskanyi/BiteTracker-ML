"""Regression tests: training and inference must build identical features for the same target date.

Training (src/feature_engineering.py) makes one row per target date T. Inference
(src/forecasting.py) builds the row for T from history strictly before T. For
the same school and T, every feature in FEATURE_COLUMNS must be equal; if the
two paths drift apart again (e.g. one of them shifts calendar features by a
day), these tests fail.

What this file checks:
- The feature row built at forecast time matches the training row for every date,
  including Mondays and holidays.
- The day-1 forecast uses exactly the training row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.feature_engineering import FEATURE_COLUMNS, engineer_features
from src.forecasting import _build_feature_row, forecast_school_demand
from src.preprocessing import run_preprocessing
from src.synthetic_data import generate_synthetic_dataset

# Named features the alignment must hold for, at minimum.
KEY_FEATURES = [
    "lag_1_meals",
    "lag_7_meals",
    "rolling_mean_7",
    "previous_attendance",
    "previous_attendance_rate",
    "month",
    "is_holiday",
    "school_term_Term 2",
    "day_of_week_Tuesday",
    "day_of_week_Wednesday",
    "day_of_week_Thursday",
    "day_of_week_Friday",
]


@pytest.fixture(scope="module")
def clean_and_features():
    # Jan–May 2023 includes Mondays after weekends and the April holiday window.
    clean = run_preprocessing(generate_synthetic_dataset(start_date="2023-01-01", end_date="2023-05-31"))
    features = engineer_features(clean)
    return clean, features


def _training_row(features: pd.DataFrame, school_id: str, target_date: pd.Timestamp) -> pd.Series:
    match = features[(features[settings.SCHOOL_ID_COLUMN] == school_id) & (features[settings.DATE_COLUMN] == target_date)]
    assert len(match) == 1
    return match.iloc[0]


def _inference_row(clean: pd.DataFrame, school_id: str, target_date: pd.Timestamp) -> dict:
    history = clean[clean[settings.SCHOOL_ID_COLUMN] == school_id].sort_values(settings.DATE_COLUMN)
    before = history[history[settings.DATE_COLUMN] < target_date]
    on_target = history[history[settings.DATE_COLUMN] == target_date].iloc[0]
    return _build_feature_row(
        forecast_date=target_date.date(),
        meals_history=before["meals_served"].tolist(),
        students_present_history=before["students_present"].tolist(),
        enrolled_students=float(before["enrolled_students"].iloc[-1]),
        rainfall_mm=float(on_target["rainfall_mm"]),
        temperature_c=float(on_target["temperature_c"]),
        is_school_event=bool(on_target["is_school_event"]),
    )


def _assert_rows_equal(training: pd.Series, inference: dict, target_date) -> None:
    for feature in FEATURE_COLUMNS:
        assert np.isclose(float(training[feature]), float(inference[feature])), (
            f"{feature} differs for target {target_date:%a %Y-%m-%d}: "
            f"training={training[feature]!r} inference={inference[feature]!r}"
        )


def test_lag_and_attendance_features_refer_to_days_before_target(clean_and_features):
    """Worked example: for target T, lag_1 and previous attendance come from T-1, never T."""
    clean, features = clean_and_features
    school = clean[clean[settings.SCHOOL_ID_COLUMN] == "SCH001"].sort_values(settings.DATE_COLUMN).reset_index(drop=True)
    t = school.index[(school[settings.DATE_COLUMN] >= "2023-03-01") & (school[settings.DATE_COLUMN].dt.dayofweek == 3)][0]
    target_date = school.loc[t, settings.DATE_COLUMN]  # a Thursday
    row = _training_row(features, "SCH001", target_date)

    assert row[settings.TARGET_COLUMN] == school.loc[t, "meals_served"]           # Thursday's meals
    assert row["lag_1_meals"] == school.loc[t - 1, "meals_served"]               # Wednesday's meals
    assert row["lag_7_meals"] == school.loc[t - 7, "meals_served"]
    assert np.isclose(row["rolling_mean_7"], school.loc[t - 7:t - 1, "meals_served"].mean())
    assert row["previous_attendance"] == school.loc[t - 1, "students_present"]   # Wednesday's attendance
    assert row["day_of_week_Thursday"] == 1 and row["day_of_week_Wednesday"] == 0
    assert row["month"] == target_date.month


def test_training_and_build_feature_row_match_for_every_target_date(clean_and_features):
    clean, features = clean_and_features
    checked = 0
    for _, training in features.iterrows():
        target_date = training[settings.DATE_COLUMN]
        inference = _inference_row(clean, training[settings.SCHOOL_ID_COLUMN], target_date)
        _assert_rows_equal(training, inference, target_date)
        checked += 1
    assert checked > 250


def test_alignment_covers_mondays_and_holidays(clean_and_features):
    """Make sure the checked rows include the cases a one-day shift would break."""
    _, features = clean_and_features
    assert (features[settings.DATE_COLUMN].dt.dayofweek == 0).any()
    assert (features["is_holiday"] == 1).any()
    assert (features["is_holiday"].diff() != 0).any()  # a term/holiday boundary


@pytest.mark.parametrize("weekday", [0, 3])  # a Monday (after a weekend) and a Thursday
def test_forecast_school_demand_day1_matches_training_row(clean_and_features, weekday):
    """End-to-end: forecasting the next school day from history truncated at T-1
    must feed the model exactly the training row for T."""
    clean, features = clean_and_features
    school_rows = features[features[settings.SCHOOL_ID_COLUMN] == "SCH002"]
    candidates = school_rows[school_rows[settings.DATE_COLUMN].dt.dayofweek == weekday]
    training = candidates.iloc[len(candidates) // 2]
    target_date = training[settings.DATE_COLUMN]

    history = clean[(clean[settings.SCHOOL_ID_COLUMN] == "SCH002") & (clean[settings.DATE_COLUMN] < target_date)]
    on_target = clean[(clean[settings.SCHOOL_ID_COLUMN] == "SCH002") & (clean[settings.DATE_COLUMN] == target_date)].iloc[0]

    captured = {}

    class _CapturingModel:
        def predict(self, X):
            captured["X"] = X.iloc[0]
            return np.array([0.0])

    result = forecast_school_demand(
        history, "SCH002", horizon_days=1,
        expected_rainfall_mm=float(on_target["rainfall_mm"]),
        expected_temperature_c=float(on_target["temperature_c"]),
        school_events={target_date.date()} if on_target["is_school_event"] else set(),
        model=_CapturingModel(),
    )
    assert result.iloc[0]["date"] == target_date.date()
    for feature in KEY_FEATURES:
        assert feature in FEATURE_COLUMNS
    _assert_rows_equal(training, captured["X"].to_dict(), target_date)
