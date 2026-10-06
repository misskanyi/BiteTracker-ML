"""upload_service: validation summaries for uploaded files (no Django involved)."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from services import upload_service


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


@pytest.fixture
def valid_frame(stub_processed_frame):
    return stub_processed_frame[["date", "school_id", "school_name", "meals_served"]].head(70)


def test_valid_csv_summary_contains_aggregates_not_rows(valid_frame):
    summary = upload_service.validate_dataset(_csv(valid_frame), "data.csv")
    assert summary["status"] == "valid"
    assert summary["n_records"] == 70 and summary["n_columns"] == 4 and summary["n_schools"] >= 1
    assert summary["forecast_readiness"] == "READY"
    assert summary["columns"] == ["date", "school_id", "school_name", "meals_served"]
    # Only aggregate/schema information — no data values from the file.
    text = repr(summary)
    assert str(valid_frame["meals_served"].iloc[3]) not in summary["columns"]
    assert "Greenfield Primary" not in text


def test_missing_required_column_is_invalid(valid_frame):
    summary = upload_service.validate_dataset(_csv(valid_frame.drop(columns=["school_id"])), "data.csv")
    assert summary["status"] == "invalid"
    assert summary["missing_required_columns"] == ["school_id"]


def test_xlsx_is_supported(valid_frame):
    buffer = io.BytesIO()
    valid_frame.to_excel(buffer, index=False)
    assert upload_service.validate_dataset(buffer.getvalue(), "data.XLSX")["status"] == "valid"


@pytest.mark.parametrize(
    "content, name, message",
    [
        (b"a,b\n1,2\n", "data.xls", "Unsupported file type '.xls'"),
        (b"a,b\n1,2\n", "data", "Unsupported file type 'none'"),
        (b"", "data.csv", "The file is empty"),
        (b"\x00\x01garbage", "data.xlsx", "could not be read"),
    ],
)
def test_unreadable_files_reported_not_raised(content, name, message):
    summary = upload_service.validate_dataset(content, name)
    assert summary["status"] == "unreadable"
    assert message in summary["errors"][0]
