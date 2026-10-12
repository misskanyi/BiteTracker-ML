"""Tests for src/preprocessing.py — date parsing, chronological sorting, duplicate removal,
missing-value imputation, outlier handling and non-negative enforcement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from src.preprocessing import (
    drop_duplicates,
    enforce_non_negative,
    handle_missing_values,
    handle_outliers,
    parse_dates,
    sort_chronologically,
)


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "school_id": ["A", "A", "A"],
            "enrolled_students": [100, 100, 100],
            "students_present": [90.0, 80.0, np.nan],
            "students_absent": [10, 20, 10],
            "meals_prepared": [95.0, 85.0, 90.0],
            "meals_served": [90, 80, 85],
            "leftover_meals": [5, 5, 5],
            "average_cost_per_meal": [15.0, 15.0, 15.0],
            "rainfall_mm": [0.0, 5.0, 3.0],
            "temperature_c": [22.0, 21.0, 23.0],
        }
    )


def test_parse_dates_converts_to_datetime():
    df = parse_dates(_base_df())
    assert pd.api.types.is_datetime64_any_dtype(df[settings.DATE_COLUMN])


def test_parse_dates_drops_unparsable_rows():
    df = _base_df()
    df.loc[0, "date"] = "not-a-date"
    result = parse_dates(df)
    assert len(result) == 2


def test_sort_chronologically_orders_by_school_then_date():
    df = parse_dates(_base_df())
    result = sort_chronologically(df)
    dates = result[settings.DATE_COLUMN].tolist()
    assert dates == sorted(dates)


def test_drop_duplicates_removes_same_school_same_date():
    df = _base_df()
    dup = df.iloc[[0]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    result = drop_duplicates(parse_dates(df))
    assert len(result) == 3


def test_enforce_non_negative_clips_negatives():
    df = _base_df()
    df.loc[0, "meals_served"] = -10
    result = enforce_non_negative(df)
    assert (result["meals_served"] >= 0).all()


def test_handle_missing_values_fills_nan():
    df = _base_df()
    result = handle_missing_values(df)
    assert result["students_present"].isna().sum() == 0


def test_handle_outliers_caps_extreme_values():
    rows = []
    for i in range(20):
        rows.append(
            {
                "date": f"2024-01-{i + 1:02d}",
                "school_id": "A",
                "meals_served": 100,
                "leftover_meals": 5,
                "enrolled_students": 200,
            }
        )
    df = pd.DataFrame(rows)
    df.loc[0, "meals_served"] = 100000  # extreme outlier
    result = handle_outliers(df)
    assert result["meals_served"].max() < 100000
