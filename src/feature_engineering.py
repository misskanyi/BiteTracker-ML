"""Feature engineering for next-school-day meal-demand forecasting.

Target: `meals_served_next_day` = tomorrow's meals_served for that school.
All predictive features are built from information available *as of* the
current row's date (today), so a model trained on them never sees the
future. Concretely:

- lag / rolling features use `.shift(1)` before rolling, so "today" is
  excluded from its own rolling window — only strictly-past days feed in.
- `attendance_rate` / `previous_attendance` use today's (or earlier) values,
  which are known before tomorrow's meal count is decided.
- The target is `meals_served.shift(-1)`, computed per-school so a school's
  last recorded day (no known "tomorrow") gets a NaN target and is dropped.

All shift/rolling operations are grouped by `school_id` so one school's
history never leaks into another's lag/rolling values.
"""

from __future__ import annotations

import pandas as pd

from config import settings
from src.utils import get_logger

logger = get_logger(__name__)

# Final feature list used for model training (see model_training.py).
FEATURE_COLUMNS = [
    "lag_1_meals",
    "lag_7_meals",
    "lag_14_meals",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "attendance_rate",
    "previous_attendance",
    "month",
    "rainfall_mm",
    "temperature_c",
    "is_holiday",
    "is_exam_period",
    "is_school_event",
    # one-hot encoded in preprocessing.encode_categoricals() (drop_first=True,
    # so "Term 1" and "Monday" are the implicit reference categories)
    "school_term_Term 2",
    "school_term_Term 3",
    "day_of_week_Tuesday",
    "day_of_week_Wednesday",
    "day_of_week_Thursday",
    "day_of_week_Friday",
]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag_1/7/14_meals: meals_served N school-days before today, per school."""
    df = df.copy()
    grouped = df.groupby(settings.SCHOOL_ID_COLUMN)["meals_served"]
    df["lag_1_meals"] = grouped.shift(1)
    df["lag_7_meals"] = grouped.shift(7)
    df["lag_14_meals"] = grouped.shift(14)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean/std of meals_served over the trailing 7/14 days.

    `.shift(1)` before `.rolling()` excludes today from its own window, so
    each rolling value summarizes only strictly-past school days.
    """
    df = df.copy()
    shifted = df.groupby(settings.SCHOOL_ID_COLUMN)["meals_served"].shift(1)
    df["rolling_mean_7"] = shifted.groupby(df[settings.SCHOOL_ID_COLUMN]).transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df["rolling_mean_14"] = shifted.groupby(df[settings.SCHOOL_ID_COLUMN]).transform(
        lambda s: s.rolling(window=14, min_periods=1).mean()
    )
    df["rolling_std_7"] = shifted.groupby(df[settings.SCHOOL_ID_COLUMN]).transform(
        lambda s: s.rolling(window=7, min_periods=2).std()
    )
    return df


def add_attendance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add attendance_rate (today) and previous_attendance (yesterday's headcount)."""
    df = df.copy()
    df["attendance_rate"] = (df["students_present"] / df["enrolled_students"]).clip(0, 1)
    df["previous_attendance"] = df.groupby(settings.SCHOOL_ID_COLUMN)["students_present"].shift(1)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the forecasting target: next school day's meals_served, per school."""
    df = df.copy()
    df[settings.TARGET_COLUMN] = df.groupby(settings.SCHOOL_ID_COLUMN)["meals_served"].shift(-1)
    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that can't have a complete feature set or a known target.

    This removes each school's first 14 rows (insufficient lag history) and
    each school's last row (no known "next day" to form the target).
    """
    df = df.copy()
    before = len(df)
    required = ["lag_14_meals", settings.TARGET_COLUMN]
    df = df.dropna(subset=required)
    logger.info("Dropped %d rows lacking full lag history or a target", before - len(df))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline in leakage-safe order."""
    logger.info("Starting feature engineering on %d rows", len(df))
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_attendance_features(df)
    df = add_target(df)
    df = drop_incomplete_rows(df)
    logger.info("Feature engineering complete: %d rows, %d features", len(df), len(FEATURE_COLUMNS))
    return df


def get_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature-engineered DataFrame into (X, y), keeping only model features."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered feature columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df[settings.TARGET_COLUMN].copy()
    return X, y
