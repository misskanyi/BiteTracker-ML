"""Data cleaning pipeline for the school-meals dataset.

Each function does one job and returns a new DataFrame so steps can be
composed, tested, and reordered independently. `run_preprocessing()` chains
them in the order that matters for correctness (e.g. dates must be parsed
and sorted before outlier/duplicate logic that depends on order).
"""

from __future__ import annotations

import pandas as pd

from config import settings
from src.data_loader import validate_schema
from src.utils import get_logger

logger = get_logger(__name__)

# Columns where physically-impossible negative values indicate bad data.
_NON_NEGATIVE_COLUMNS = [
    "enrolled_students",
    "students_present",
    "students_absent",
    "meals_prepared",
    "meals_served",
    "leftover_meals",
    "average_cost_per_meal",
    "rainfall_mm",
]

# Numeric columns that may contain missing values in raw data.
_NUMERIC_FILLABLE_COLUMNS = [
    "rainfall_mm",
    "temperature_c",
    "meals_prepared",
    "students_present",
    "average_cost_per_meal",
]

# Columns checked for statistical outliers (IQR method), capped rather than dropped
# so we don't lose school-days of history.
_OUTLIER_COLUMNS = ["meals_served", "leftover_meals", "enrolled_students"]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the date column to datetime64, coercing unparsable values to NaT."""
    df = df.copy()
    df[settings.DATE_COLUMN] = pd.to_datetime(df[settings.DATE_COLUMN], errors="coerce")
    n_bad = df[settings.DATE_COLUMN].isna().sum()
    if n_bad:
        logger.warning("Dropping %d rows with unparsable dates", n_bad)
        df = df.dropna(subset=[settings.DATE_COLUMN])
    return df


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by school then date, ascending. Required before any lag/rolling logic.

    Sorting per-school-then-date (rather than date-only) guarantees that when
    feature_engineering.py later groups by school_id, each group's row order
    is already chronological — lag/rolling operations assume this.
    """
    df = df.copy()
    df = df.sort_values([settings.SCHOOL_ID_COLUMN, settings.DATE_COLUMN]).reset_index(drop=True)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and duplicate (school, date) records.

    A duplicate (school_id, date) pair would mean two conflicting meal-count
    records for the same school-day — we keep the first occurrence.
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=[settings.SCHOOL_ID_COLUMN, settings.DATE_COLUMN], keep="first")
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows", removed)
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to numeric dtype, non-numeric junk becomes NaN."""
    df = df.copy()
    numeric_cols = [
        "enrolled_students",
        "students_present",
        "students_absent",
        "meals_prepared",
        "meals_served",
        "leftover_meals",
        "average_cost_per_meal",
        "rainfall_mm",
        "temperature_c",
        "is_holiday",
        "is_exam_period",
        "is_school_event",
        "month",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def enforce_non_negative(df: pd.DataFrame) -> pd.DataFrame:
    """Clip physically-impossible negative values to zero (data-entry errors)."""
    df = df.copy()
    for col in _NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            n_negative = (df[col] < 0).sum()
            if n_negative:
                logger.warning("Clipping %d negative values in %s to 0", n_negative, col)
                df[col] = df[col].clip(lower=0)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values per-school using forward/backward fill,
    falling back to the per-school median for any values still missing at
    the start/end of a school's history.

    Per-school (not global) imputation avoids blending one school's typical
    attendance/weather pattern into another's gaps.
    """
    df = df.copy()
    for col in _NUMERIC_FILLABLE_COLUMNS:
        if col not in df.columns:
            continue
        n_missing_before = df[col].isna().sum()
        if n_missing_before == 0:
            continue
        df[col] = df.groupby(settings.SCHOOL_ID_COLUMN)[col].transform(lambda s: s.ffill().bfill())
        df[col] = df.groupby(settings.SCHOOL_ID_COLUMN)[col].transform(lambda s: s.fillna(s.median()))
        logger.info("Imputed %d missing values in %s", n_missing_before, col)
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers per-school using the IQR method (1.5x fences).

    Values are capped rather than removed, preserving each school's
    chronological record for lag/rolling features while still preventing a
    single bad sensor reading from distorting trained models.
    """
    df = df.copy()
    for col in _OUTLIER_COLUMNS:
        if col not in df.columns:
            continue

        def _cap(series: pd.Series) -> pd.Series:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return series.clip(lower=lower, upper=upper)

        before = df[col].copy()
        df[col] = df.groupby(settings.SCHOOL_ID_COLUMN)[col].transform(_cap)
        n_capped = (before != df[col]).sum()
        if n_capped:
            logger.info("Capped %d outliers in %s", n_capped, col)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode low-cardinality categorical columns used as model features.

    `school_term` and `day_of_week` are nominal (no ordinal relationship), so
    one-hot encoding avoids implying a false ordering. Identifier columns
    (school_id, school_name, county, sub_county) are left as-is — they're
    used for filtering/display, not as raw model inputs.
    """
    df = df.copy()
    # Explicit category order so drop_first always drops the intended
    # reference category (Term 1 / Monday), not whichever sorts first
    # alphabetically ("Friday" would otherwise be dropped instead of "Monday").
    category_orders = {
        "school_term": ["Term 1", "Term 2", "Term 3"],
        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    }
    for col, categories in category_orders.items():
        if col in df.columns:
            ordered = pd.Categorical(df[col], categories=categories)
            # drop_first avoids the dummy-variable trap (perfectly collinear
            # one-hot columns), which destabilizes linear models.
            dummies = pd.get_dummies(ordered, prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
    return df


def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in the order correctness requires."""
    logger.info("Starting preprocessing on %d raw rows", len(df))
    validate_schema(df)
    df = parse_dates(df)
    df = fix_dtypes(df)
    df = sort_chronologically(df)
    df = drop_duplicates(df)
    df = enforce_non_negative(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = encode_categoricals(df)
    logger.info("Preprocessing complete: %d rows remain", len(df))
    return df
