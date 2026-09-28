"""Feature engineering for school-day meal-demand forecasting.

Each row represents the TARGET DATE T for one school, and
`target_meals_served` is the meals_served recorded on T. Features describe T
using only information that is known before T's meals are served, which is
exactly what `src.forecasting._build_feature_row` builds at prediction time:

- lag / rolling features: `.shift(1)` first, so they cover school days
  strictly before T (lag_1 = the previous school day).
- attendance: `previous_attendance` / `previous_attendance_rate` come from
  the previous school day. T's own attendance is NOT a feature, because it
  is not known until T happens and moves almost one-for-one with meals.
- calendar (weekday, month, term, holiday, exam period, school event):
  T's own values, which the school calendar publishes in advance.
- weather (rainfall_mm, temperature_c): T's values, standing in for a
  weather forecast. Training uses observed weather, so it assumes a
  perfect forecast; at prediction time the user supplies expected values.

`attendance_rate` (T's own attendance / enrolment) is still computed for
dashboards and EDA, but it is deliberately excluded from FEATURE_COLUMNS.

All shift/rolling operations are grouped by `school_id` so one school's
history never leaks into another's lag/rolling values.
"""

# In simple terms: builds the "clues" (past meals, past attendance, the
# day's calendar) the model uses to guess a school day's meal count.

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
    "previous_attendance_rate",
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
    """Add previous-school-day attendance features, per school.

    - previous_attendance: students present on the previous school day.
    - previous_attendance_rate: previous school day's present / enrolled.
    - attendance_rate: the row day's own rate, for dashboards/EDA only
      (not a model feature — it is unknown before that day's meals).
    """
    df = df.copy()
    df["attendance_rate"] = (df["students_present"] / df["enrolled_students"]).clip(0, 1)
    grouped = df.groupby(settings.SCHOOL_ID_COLUMN)
    df["previous_attendance"] = grouped["students_present"].shift(1)
    df["previous_attendance_rate"] = grouped["attendance_rate"].shift(1)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the forecasting target: meals_served on the row's own date (the target date)."""
    df = df.copy()
    df[settings.TARGET_COLUMN] = df["meals_served"]
    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that can't have a complete feature set or a known target.

    This removes each school's first 14 rows (insufficient lag history) and
    any row whose meals_served (the target) is missing.
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
