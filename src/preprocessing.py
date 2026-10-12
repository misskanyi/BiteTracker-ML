"""Data cleaning pipeline for the school-meals dataset.

Each function does one job and returns a new DataFrame so steps can be
composed, tested, and reordered independently. `run_preprocessing()` chains
them in the order that matters for correctness (e.g. dates must be parsed
and sorted before outlier/duplicate logic that depends on order).

Cleaning has two kinds of step:

- Stateless steps (`clean_raw_data`): parse dates, fix dtypes, sort, drop
  duplicates, clip negatives. Each row is cleaned on its own, so these are
  safe to run on the whole dataset.
- Learned steps: the per-school imputation medians and IQR outlier fences.
  `fit_preprocessing` learns them from the TRAINING PERIOD ONLY and
  `apply_preprocessing` applies the same numbers to training, validation,
  test and inference rows, so no statistic ever sees the future.

Missing values are forward-filled within each school (yesterday's value
carried forward) and then filled with the fitted median. They are never
back-filled, because that would copy a later day's value into an earlier one.
"""

# In simple terms: cleans up messy raw data (fixes dates, removes
# duplicates/outliers) so it's ready for feature-building and modelling.
# Anything "learned" from the data (typical values, outlier limits) is
# learned from the training period only.

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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

# Key for the all-schools fallback in fitted statistics, used for a school
# with no usable training rows (e.g. one that first appears in the test period).
ALL_SCHOOLS = "__all_schools__"


@dataclass
class PreprocessingParams:
    """Statistics learned by `fit_preprocessing` from the training period only.

    Plain JSON-friendly values so they can be saved next to a model and
    reapplied unchanged at evaluation and inference time.
    """

    fit_end_date: str | None
    fit_rows: int
    # column -> {school_id or ALL_SCHOOLS: median}
    fill_values: dict[str, dict[str, float | None]] = field(default_factory=dict)
    # column -> {school_id or ALL_SCHOOLS: [lower fence, upper fence]}
    outlier_bounds: dict[str, dict[str, list[float] | None]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PreprocessingParams:
        return cls(**data)


def _median_or_none(series: pd.Series) -> float | None:
    """Median of a series, or None if it has no values (e.g. a column the source lacks)."""
    series = series.dropna()
    return None if series.empty else float(series.median())


def _iqr_fences(series: pd.Series) -> list[float] | None:
    """1.5x IQR fences of a series, or None if it has no values."""
    series = series.dropna()
    if series.empty:
        return None
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return [float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)]


def _per_school(df: pd.DataFrame, col: str, stat) -> dict:
    """`stat` of `col` for each school, plus an all-schools fallback."""
    values = {str(school): stat(group) for school, group in df.groupby(settings.SCHOOL_ID_COLUMN)[col]}
    values[ALL_SCHOOLS] = stat(df[col])
    return values


def _school_lookup(df: pd.DataFrame, values: dict) -> pd.Series:
    """Map each row's school to its fitted value, falling back to the all-schools value."""
    fallback = values.get(ALL_SCHOOLS)
    return df[settings.SCHOOL_ID_COLUMN].astype(str).map(lambda s: values.get(s, fallback))


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


def fit_preprocessing(train_df: pd.DataFrame, fit_end_date=None) -> PreprocessingParams:
    """Learn imputation medians and IQR outlier fences, per school, from `train_df`.

    Pass training-period rows only (already through `clean_raw_data`); the
    returned numbers are then reused unchanged for every later row.
    """
    fill_values = {
        col: _per_school(train_df, col, _median_or_none)
        for col in _NUMERIC_FILLABLE_COLUMNS
        if col in train_df.columns
    }
    outlier_bounds = {
        col: _per_school(train_df, col, _iqr_fences)
        for col in _OUTLIER_COLUMNS
        if col in train_df.columns
    }
    return PreprocessingParams(
        fit_end_date=None if fit_end_date is None else str(pd.Timestamp(fit_end_date).date()),
        fit_rows=int(len(train_df)),
        fill_values=fill_values,
        outlier_bounds=outlier_bounds,
    )


def handle_missing_values(df: pd.DataFrame, params: PreprocessingParams | None = None) -> pd.DataFrame:
    """Impute missing numeric values per school: forward fill, then the fitted median.

    Forward fill only carries a school's earlier value into a later gap, so
    nothing moves backwards in time. Gaps still left (e.g. a school's first
    rows) get that school's median from `params`, which `fit_preprocessing`
    learned from the training period. Without `params`, the medians are
    fitted on `df` itself (fine for descriptive work, not for model evaluation).

    Per-school (not global) imputation avoids blending one school's typical
    attendance/weather pattern into another's gaps.
    """
    params = params or fit_preprocessing(df)
    df = df.copy()
    for col in _NUMERIC_FILLABLE_COLUMNS:
        if col not in df.columns:
            continue
        n_missing_before = df[col].isna().sum()
        if n_missing_before == 0:
            continue
        df[col] = df.groupby(settings.SCHOOL_ID_COLUMN)[col].ffill()
        if col in params.fill_values:
            df[col] = df[col].fillna(_school_lookup(df, params.fill_values[col]))
        n_filled = n_missing_before - df[col].isna().sum()
        if n_filled:
            logger.info("Imputed %d missing values in %s", n_filled, col)
    return df


def handle_outliers(df: pd.DataFrame, params: PreprocessingParams | None = None) -> pd.DataFrame:
    """Cap outliers per school at the fitted IQR fences (1.5x).

    Values are capped rather than removed, preserving each school's
    chronological record for lag/rolling features while still preventing a
    single bad sensor reading from distorting trained models. The fences come
    from `params` (training period only); without `params` they are fitted
    on `df` itself.
    """
    params = params or fit_preprocessing(df)
    df = df.copy()
    for col in _OUTLIER_COLUMNS:
        if col not in df.columns or col not in params.outlier_bounds:
            continue
        fences = _school_lookup(df, params.outlier_bounds[col])
        known = fences.notna()
        if not known.any():
            continue
        lower = fences[known].map(lambda f: f[0]).astype(float)
        upper = fences[known].map(lambda f: f[1]).astype(float)
        df[col] = df[col].astype(float)
        before = df.loc[known, col].copy()
        df.loc[known, col] = before.clip(lower=lower, upper=upper)
        n_capped = (before.ne(df.loc[known, col]) & before.notna()).sum()
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


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless cleaning steps: nothing here is learned from the data."""
    logger.info("Starting preprocessing on %d raw rows", len(df))
    validate_schema(df)
    df = parse_dates(df)
    df = fix_dtypes(df)
    df = sort_chronologically(df)
    df = drop_duplicates(df)
    df = enforce_non_negative(df)
    return df


def apply_preprocessing(clean_df: pd.DataFrame, params: PreprocessingParams) -> pd.DataFrame:
    """Apply fitted imputation, outlier capping and encoding to `clean_raw_data` output."""
    df = handle_missing_values(clean_df, params)
    df = handle_outliers(df, params)
    df = encode_categoricals(df)
    logger.info("Preprocessing complete: %d rows remain", len(df))
    return df


def run_preprocessing(
    df: pd.DataFrame,
    params: PreprocessingParams | None = None,
    fit_end_date=None,
) -> pd.DataFrame:
    """Run the full cleaning pipeline in the order correctness requires.

    Learned statistics come from `params` if given, otherwise they are fitted
    on rows dated up to `fit_end_date`. With neither, they are fitted on
    every row: acceptable for EDA and dashboards, but model training must
    use `src.pipeline.prepare_training_data`, which fits on the training
    period only.
    """
    df = clean_raw_data(df)
    if params is None:
        fit_rows = df if fit_end_date is None else df[df[settings.DATE_COLUMN] <= pd.Timestamp(fit_end_date)]
        params = fit_preprocessing(fit_rows, fit_end_date)
    return apply_preprocessing(df, params)
