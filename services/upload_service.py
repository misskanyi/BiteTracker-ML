"""Validate an uploaded dataset file and summarise the result.

Wraps the existing `src.data_validation` checks. Returns schema-level and
aggregate information only (counts, column names, date range, scores), never
data rows, so the summary can be stored and displayed safely.

Validating a file never changes the active dataset and never retrains the model.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from src.data_validation import (
    UnsupportedFileTypeError,
    check_feature_compatibility,
    load_uploaded_file,
    validate_upload,
)

logger = logging.getLogger(__name__)

# .xls is not offered: reading it needs the xlrd package, which is not installed.
ALLOWED_EXTENSIONS = (".csv", ".xlsx")

VALID, INVALID, UNREADABLE = "valid", "invalid", "unreadable"
MAX_SCHOOL_IDS = 500


def _unreadable(message: str) -> dict:
    return {"status": UNREADABLE, "errors": [message], "warnings": []}


def validate_dataset(file_bytes: bytes, filename: str) -> dict:
    """Read and validate one uploaded file. Never raises for bad file content."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return _unreadable(f"Unsupported file type '{suffix or 'none'}'. Upload a CSV (.csv) or Excel (.xlsx) file.")
    if not file_bytes:
        return _unreadable("The file is empty.")
    try:
        df = load_uploaded_file(file_bytes, filename)
    except UnsupportedFileTypeError as exc:
        return _unreadable(str(exc))
    except Exception:  # malformed CSV/Excel content: parser errors vary by library
        logger.warning("Uploaded file %r could not be parsed", filename, exc_info=True)
        return _unreadable("The file could not be read. Check that it is a valid CSV or Excel (.xlsx) file.")

    report = validate_upload(df)
    n_schools = int(df[settings.SCHOOL_ID_COLUMN].nunique()) if settings.SCHOOL_ID_COLUMN in df.columns else None
    # School codes only (no data rows), so access checks can confirm the uploader may submit for them.
    school_ids = (sorted({str(v).strip() for v in df[settings.SCHOOL_ID_COLUMN].dropna()} - {""})[:MAX_SCHOOL_IDS]
                  if settings.SCHOOL_ID_COLUMN in df.columns else [])
    return {
        "status": VALID if report.is_valid else INVALID,
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "n_records": report.n_records,
        "n_columns": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "n_schools": n_schools,
        "school_ids": school_ids,
        "date_range": list(report.date_range) if report.date_range else None,
        "coverage_months": report.coverage_months,
        "missing_values": report.missing_values,
        "duplicate_rows": report.duplicate_rows,
        "unparsable_dates": report.unparsable_dates,
        "invalid_numeric_values": report.invalid_numeric_values,
        "missing_required_columns": list(report.missing_required_columns),
        "optional_columns_missing": list(report.optional_columns_missing),
        "quality_score": report.quality_score,
        "score_breakdown": dict(report.score_breakdown),
        "forecast_readiness": report.forecast_readiness,
        "compatibility_issues": check_feature_compatibility(df) if report.is_valid else [],
    }
