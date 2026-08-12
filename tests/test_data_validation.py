"""Tests for src.data_validation: upload parsing, validation, and quality scoring."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from src.data_validation import (
    MIN_RECORDS_LIMITED,
    MIN_RECORDS_READY,
    UnsupportedFileTypeError,
    check_feature_compatibility,
    forecast_readiness_message,
    load_uploaded_file,
    validate_upload,
)


def _good_df(n=70):
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "school_id": ["SCH100"] * n,
            "school_name": ["Test Primary"] * n,
            "meals_served": [300 + i for i in range(n)],
            "enrolled_students": [400] * n,
            "students_present": [350] * n,
            "rainfall_mm": [5.0] * n,
        }
    )


def test_load_uploaded_file_csv():
    df = _good_df(5)
    csv_bytes = df.to_csv(index=False).encode()
    loaded = load_uploaded_file(csv_bytes, "data.csv")
    assert len(loaded) == 5


def test_load_uploaded_file_excel():
    df = _good_df(5)
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    loaded = load_uploaded_file(buffer.getvalue(), "data.xlsx")
    assert len(loaded) == 5


def test_load_uploaded_file_unsupported_type():
    with pytest.raises(UnsupportedFileTypeError):
        load_uploaded_file(b"junk", "data.txt")


def test_validate_upload_missing_required_columns():
    df = pd.DataFrame({"date": ["2025-01-01"], "meals_served": [100]})
    report = validate_upload(df)
    assert not report.is_valid
    assert "school_id" in " ".join(report.missing_required_columns)


def test_validate_upload_good_dataset_is_valid():
    report = validate_upload(_good_df(70))
    assert report.is_valid
    assert report.n_records == 70
    assert report.forecast_readiness == "READY"
    assert 0 <= report.quality_score <= 100


def test_validate_upload_detects_duplicates():
    df = _good_df(20)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    report = validate_upload(df)
    assert report.duplicate_rows >= 1


def test_validate_upload_detects_missing_values():
    df = _good_df(20)
    df.loc[0, "rainfall_mm"] = None
    report = validate_upload(df)
    assert report.missing_values >= 1


def test_validate_upload_detects_invalid_numeric_values():
    df = _good_df(20)
    df.loc[0, "meals_served"] = -50
    report = validate_upload(df)
    assert report.invalid_numeric_values >= 1


def test_validate_upload_empty_file():
    report = validate_upload(pd.DataFrame(columns=["date", "school_id", "school_name", "meals_served"]))
    assert not report.is_valid


def test_forecast_readiness_thresholds():
    assert "suitable for model evaluation" in forecast_readiness_message(MIN_RECORDS_READY, 12)
    assert "More data is recommended" in forecast_readiness_message(MIN_RECORDS_LIMITED, 1)
    assert "not available" in forecast_readiness_message(5, 0.2)


def test_check_feature_compatibility_flags_unknown_term():
    df = _good_df(5)
    df["school_term"] = "Term X"
    issues = check_feature_compatibility(df)
    assert any("school_term" in i for i in issues)


def test_check_feature_compatibility_flags_missing_attendance():
    df = _good_df(5).drop(columns=["enrolled_students", "students_present"])
    issues = check_feature_compatibility(df)
    assert any("attendance" in i for i in issues)
