"""Validation and quality scoring for user-uploaded school-meal datasets.

Kept separate from `src.data_loader`/`src.preprocessing` (which assume a
trusted dataset) because uploaded files need a *report*, not just a
pass/fail: schools and NGOs uploading their own data need to see exactly
what is wrong before any modelling happens.

Required vs optional columns mirror `src.synthetic_data.SCHEMA_COLUMNS`,
minus any personally-identifying fields (none are part of the schema —
records are school-level aggregates, never per-student).
"""

# In simple terms: checks a file someone uploads (from a school/NGO) and
# gives back a report card on how usable/clean it is.

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
import pandas as pd

from config import settings

REQUIRED_COLUMNS = [
    "date",
    "school_id",
    "school_name",
    "meals_served",
]

OPTIONAL_COLUMNS = [
    "county",
    "sub_county",
    "enrolled_students",
    "students_present",
    "students_absent",
    "meals_prepared",
    "leftover_meals",
    "average_cost_per_meal",
    "rainfall_mm",
    "temperature_c",
    "school_term",
    "is_holiday",
    "is_exam_period",
    "is_school_event",
]

ALL_KNOWN_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Columns that must be non-negative when present.
_NUMERIC_RANGE_COLUMNS = [
    "enrolled_students",
    "students_present",
    "students_absent",
    "meals_prepared",
    "meals_served",
    "leftover_meals",
    "average_cost_per_meal",
    "rainfall_mm",
]

# Minimum per-school daily records for each readiness tier. Matches the
# 14-day lag window `src.feature_engineering.drop_incomplete_rows` requires
# before a row has a complete feature set.
MIN_RECORDS_READY = 60  # roughly 3 months of school days
MIN_RECORDS_LIMITED = 14


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file is neither CSV nor Excel."""


def load_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV/XLSX file into a DataFrame.

    Takes raw bytes + filename rather than a Streamlit UploadedFile so this
    stays testable without importing streamlit.
    """
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    buffer = BytesIO(file_bytes)
    if suffix == "csv":
        return pd.read_csv(buffer)
    if suffix in ("xlsx", "xls"):
        return pd.read_excel(buffer)
    raise UnsupportedFileTypeError(f"Unsupported file type '.{suffix}'. Please upload a CSV or Excel file.")


@dataclass
class ValidationReport:
    """Human-readable + machine-readable summary of an uploaded dataset's quality."""

    n_records: int
    missing_required_columns: list[str]
    optional_columns_present: list[str]
    optional_columns_missing: list[str]
    unparsable_dates: int
    missing_values: int
    duplicate_rows: int
    invalid_numeric_values: int
    date_range: tuple[str, str] | None
    coverage_months: float
    quality_score: int
    score_breakdown: dict[str, int]
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def forecast_readiness(self) -> str:
        if not self.is_valid:
            return "NOT READY"
        return "READY" if self.n_records >= MIN_RECORDS_READY else "LIMITED"


def validate_upload(df: pd.DataFrame) -> ValidationReport:
    """Run the full validation + quality-scoring pass over an uploaded dataset.

    Never raises on data-quality problems — bad data produces a report with
    `is_valid=False` and populated `errors`, so the UI can show *why* rather
    than a stack trace. Only genuinely malformed input (not a DataFrame,
    empty file) escalates, and even that is caught by the caller page.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        errors.append(f"Missing required columns: {', '.join(missing_required)}")

    optional_present = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    optional_missing = [c for c in OPTIONAL_COLUMNS if c not in df.columns]

    n_records = len(df)
    if n_records == 0:
        errors.append("File contains no data rows.")

    # If required columns are missing outright, later checks would crash —
    # report what we can and stop there.
    if missing_required or n_records == 0:
        return ValidationReport(
            n_records=n_records,
            missing_required_columns=missing_required,
            optional_columns_present=optional_present,
            optional_columns_missing=optional_missing,
            unparsable_dates=0,
            missing_values=0,
            duplicate_rows=0,
            invalid_numeric_values=0,
            date_range=None,
            coverage_months=0.0,
            quality_score=0,
            score_breakdown={},
            is_valid=False,
            errors=errors,
            warnings=warnings,
        )

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    unparsable_dates = int(parsed_dates.isna().sum())
    if unparsable_dates:
        warnings.append(f"{unparsable_dates} row(s) have an unparsable date and will be dropped.")

    valid_dates = parsed_dates.dropna()
    if valid_dates.empty:
        date_range = None
        coverage_months = 0.0
    else:
        date_range = (valid_dates.min().strftime("%b %Y"), valid_dates.max().strftime("%b %Y"))
        coverage_months = round((valid_dates.max() - valid_dates.min()).days / 30.44, 1)

    missing_values = int(df[REQUIRED_COLUMNS + optional_present].isna().sum().sum())

    duplicate_rows = int(df.duplicated().sum())
    if "school_id" in df.columns:
        duplicate_rows = max(duplicate_rows, int(df.duplicated(subset=["school_id", "date"]).sum()))

    invalid_numeric_values = 0
    for col in _NUMERIC_RANGE_COLUMNS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            invalid_numeric_values += int((numeric < 0).sum())
            invalid_numeric_values += int(numeric.isna().sum() - df[col].isna().sum())

    score_breakdown = _score_breakdown(
        n_records=n_records,
        optional_present=optional_present,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        unparsable_dates=unparsable_dates,
        invalid_numeric_values=invalid_numeric_values,
        coverage_months=coverage_months,
    )
    quality_score = sum(score_breakdown.values())

    if n_records < MIN_RECORDS_LIMITED:
        warnings.append(
            f"Only {n_records} record(s) found — at least {MIN_RECORDS_LIMITED} are needed for any "
            "forecasting features, and 60+ for a reliable forecast."
        )

    is_valid = not errors

    return ValidationReport(
        n_records=n_records,
        missing_required_columns=missing_required,
        optional_columns_present=optional_present,
        optional_columns_missing=optional_missing,
        unparsable_dates=unparsable_dates,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        invalid_numeric_values=invalid_numeric_values,
        date_range=date_range,
        coverage_months=coverage_months,
        quality_score=quality_score,
        score_breakdown=score_breakdown,
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
    )


def _score_breakdown(
    *,
    n_records: int,
    optional_present: list[str],
    missing_values: int,
    duplicate_rows: int,
    unparsable_dates: int,
    invalid_numeric_values: int,
    coverage_months: float,
) -> dict[str, int]:
    """Transparent 0-100 data quality score, broken into named factors.

    Each factor is capped so no single dimension can dominate. Weights sum
    to 100: completeness 30, duplicates 15, date validity 15, coverage 20,
    optional-column richness 10, numeric validity 10.
    """
    cell_count = max(n_records * (len(REQUIRED_COLUMNS) + len(optional_present)), 1)
    completeness = round(30 * (1 - min(missing_values / cell_count, 1)))
    duplicates_score = round(15 * (1 - min(duplicate_rows / max(n_records, 1), 1)))
    date_validity = round(15 * (1 - min(unparsable_dates / max(n_records, 1), 1)))
    coverage_score = round(20 * min(coverage_months / 12, 1))  # full marks at 12+ months
    richness = round(10 * min(len(optional_present) / len(OPTIONAL_COLUMNS), 1))
    numeric_validity = round(10 * (1 - min(invalid_numeric_values / cell_count, 1)))

    return {
        "completeness": completeness,
        "no_duplicates": duplicates_score,
        "date_validity": date_validity,
        "historical_coverage": coverage_score,
        "column_richness": richness,
        "numeric_validity": numeric_validity,
    }


def forecast_readiness_message(n_records: int, coverage_months: float) -> str:
    """Human-readable explanation of forecast readiness for one school's data."""
    if n_records >= MIN_RECORDS_READY:
        return (
            f"This school contains {n_records} valid daily records covering "
            f"{coverage_months:.0f} months and is suitable for model evaluation."
        )
    if n_records >= MIN_RECORDS_LIMITED:
        return (
            f"Only {n_records} historical records are available (covering {coverage_months:.0f} months). "
            "More data is recommended before generating reliable forecasts."
        )
    return (
        f"Only {n_records} historical record(s) are available — this is below the minimum "
        f"({MIN_RECORDS_LIMITED}) needed to compute the lag/rolling features the model relies on. "
        "Forecasting is not available for this school yet."
    )


def check_feature_compatibility(df: pd.DataFrame) -> list[str]:
    """Return a list of reasons the existing trained model may not validly apply to `df`.

    The model was trained on `FEATURE_COLUMNS` from `src.feature_engineering`,
    which assumes specific one-hot categories (school_term Term 1-3,
    Monday-Friday). Data using different category labels won't align —
    surfacing this explicitly beats a silently-wrong prediction.
    """
    reasons = []
    if "school_term" in df.columns:
        known_terms = {"Term 1", "Term 2", "Term 3"}
        unknown = set(df["school_term"].dropna().unique()) - known_terms
        if unknown:
            reasons.append(f"Unrecognised school_term value(s): {sorted(unknown)}. Expected Term 1/2/3.")
    if "enrolled_students" not in df.columns or "students_present" not in df.columns:
        reasons.append("Missing enrolled_students/students_present — attendance-rate features cannot be computed.")
    return reasons
