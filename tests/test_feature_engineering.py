"""Tests for src/feature_engineering.py — correctness and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from src.feature_engineering import (
    FEATURE_COLUMNS,
    add_attendance_features,
    add_lag_features,
    add_rolling_features,
    add_target,
    drop_incomplete_rows,
    engineer_features,
    get_feature_target_split,
)


def _multi_school_df(n_days: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for school in ["A", "B"]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "school_id": school,
                    "enrolled_students": 100,
                    "students_present": 80 + i,
                    "meals_served": 50 + i * 2 + (0 if school == "A" else 100),
                    "rainfall_mm": 1.0,
                    "temperature_c": 20.0,
                    "is_holiday": 0,
                    "is_exam_period": 0,
                    "is_school_event": 0,
                    "month": d.month,
                    "school_term": "Term 1",
                    "day_of_week": d.day_name(),
                }
            )
    df = pd.DataFrame(rows)
    category_orders = {
        "school_term": ["Term 1", "Term 2", "Term 3"],
        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    }
    for col, categories in category_orders.items():
        ordered = pd.Categorical(df[col], categories=categories)
        dummies = pd.get_dummies(ordered, prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    return df


def test_lag_features_do_not_cross_schools():
    df = _multi_school_df()
    result = add_lag_features(df)
    row_b_day0 = result[(result.school_id == "B") & (result.date == "2024-01-01")]
    # First day of school B must not pick up school A's meals_served as a lag.
    assert row_b_day0["lag_1_meals"].isna().all()


def test_lag_1_matches_previous_day_value():
    df = _multi_school_df()
    result = add_lag_features(df).sort_values(["school_id", "date"]).reset_index(drop=True)
    school_a = result[result.school_id == "A"].reset_index(drop=True)
    assert school_a.loc[5, "lag_1_meals"] == school_a.loc[4, "meals_served"]


def test_rolling_mean_excludes_current_day():
    df = _multi_school_df()
    result = add_rolling_features(df).sort_values(["school_id", "date"]).reset_index(drop=True)
    school_a = result[result.school_id == "A"].reset_index(drop=True)
    # rolling_mean_7 at row 7 should be the mean of rows 0..6 (meals_served), not include row 7 itself.
    expected = school_a.loc[0:6, "meals_served"].mean()
    assert np.isclose(school_a.loc[7, "rolling_mean_7"], expected)


def test_target_is_meals_served_on_the_rows_own_date():
    df = _multi_school_df()
    result = add_target(df)
    assert (result[settings.TARGET_COLUMN] == result["meals_served"]).all()


def test_target_defined_for_each_schools_last_row():
    df = _multi_school_df()
    result = add_target(df)
    last_rows = result.groupby("school_id").tail(1)
    assert last_rows[settings.TARGET_COLUMN].notna().all()


def test_attendance_rate_bounded_between_0_and_1():
    df = _multi_school_df()
    result = add_attendance_features(df)
    assert result["attendance_rate"].between(0, 1).all()
    assert result["previous_attendance_rate"].dropna().between(0, 1).all()


def test_attendance_features_use_previous_day_per_school():
    df = _multi_school_df()
    result = add_attendance_features(df).sort_values(["school_id", "date"]).reset_index(drop=True)
    school_a = result[result.school_id == "A"].reset_index(drop=True)
    assert school_a.loc[5, "previous_attendance"] == school_a.loc[4, "students_present"]
    assert school_a.loc[5, "previous_attendance_rate"] == school_a.loc[4, "attendance_rate"]
    # First row of each school has no previous day.
    assert result.groupby("school_id")["previous_attendance_rate"].head(1).isna().all()


def test_same_day_attendance_is_not_a_model_feature():
    assert "attendance_rate" not in FEATURE_COLUMNS
    assert "students_present" not in FEATURE_COLUMNS


def test_drop_incomplete_rows_removes_insufficient_history_and_missing_target():
    df = _multi_school_df()
    df = add_lag_features(df)
    df = add_target(df)
    result = drop_incomplete_rows(df)
    assert result["lag_14_meals"].isna().sum() == 0
    assert result[settings.TARGET_COLUMN].isna().sum() == 0


def test_engineer_features_end_to_end_no_leakage_columns_missing():
    df = _multi_school_df(n_days=30)
    result = engineer_features(df)
    X, y = get_feature_target_split(result)
    assert len(X) == len(y)
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0
