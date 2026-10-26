"""Tests for src.google_sheet_loader: share-link parsing and workbook-to-schema mapping.

Covers export URL building, mapping the Food4Education workbook onto the
BiteTracker schema (target column, reporting-only columns, columns left empty
rather than invented, calendar facts, enrolment), and errors for missing or
malformed sheets. Uses an in-memory workbook only; no network access.

What this file checks:
- A share link is converted to an export URL; other links are rejected without repeating
  the link back.
- Sheet columns are mapped to the full schema; the target is daily unique students
  tapped.
- Columns the sheet doesn't have are left empty, not made up; calendar facts come from
  the date.
- Enrolment is never lower than the number of students present.
- A missing meals sheet, missing target column or non-numeric target raises an error.
"""

import pandas as pd
import pytest

from config import settings
from src.feature_engineering import FOOD4EDUCATION_FEATURE_COLUMNS
from src.google_sheet_loader import (
    MEALS_SHEET,
    TARGET_SOURCE_COLUMN,
    GoogleSheetError,
    export_url,
    to_bitetracker_schema,
)
from src.synthetic_data import SCHEMA_COLUMNS


def _workbook() -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(["2025-04-25", "2025-04-28", "2025-08-25"])
    meals = pd.DataFrame({
        "deidentified_school": ["School_1", "School_1", "School_2"],
        "activity_date": dates,
        "daily_unique_students_tapped": [8, 10, 5],
        "projected_students": [12, 9, 6],
        "actual_students": [9, 11, 7],
    })
    taps = pd.DataFrame({
        "school": ["School_1"] * 3,
        "activity_date": dates[:1].repeat(3),
        "student_id": ["Student_1", "Student_2", "Student_2"],
    })
    return {MEALS_SHEET: meals, "school_1": taps}


def test_export_url_from_share_link():
    url = "https://docs.google.com/spreadsheets/d/abc_DEF-123/edit?usp=sharing&rtpof=true"
    assert export_url(url) == "https://docs.google.com/spreadsheets/d/abc_DEF-123/export?format=xlsx"


def test_export_url_rejects_other_links_without_echoing_them():
    with pytest.raises(GoogleSheetError) as excinfo:
        export_url("https://example.com/secret-file.xlsx")
    assert "secret-file" not in str(excinfo.value)


def test_mapping_produces_full_schema():
    df = to_bitetracker_schema(_workbook())
    assert list(df.columns) == SCHEMA_COLUMNS
    assert df["school_id"].tolist() == ["SCHOOL_1", "SCHOOL_1", "SCHOOL_2"]


def test_target_is_daily_unique_students_tapped():
    df = to_bitetracker_schema(_workbook())
    assert TARGET_SOURCE_COLUMN == "daily_unique_students_tapped"
    assert df["meals_served"].tolist() == [8, 10, 5]
    # actual_students ([9, 11, 7]) is not carried into any column.
    for col in df.columns:
        assert df[col].tolist() != [9, 11, 7], f"actual_students leaked into {col}"


def test_projected_students_is_reporting_only():
    df = to_bitetracker_schema(_workbook())
    assert df["meals_prepared"].tolist() == [12, 9, 6]
    assert df["leftover_meals"].tolist() == [4, 0, 1]
    assert "meals_prepared" not in FOOD4EDUCATION_FEATURE_COLUMNS
    assert "leftover_meals" not in FOOD4EDUCATION_FEATURE_COLUMNS


def test_unavailable_columns_are_left_empty_not_invented(monkeypatch):
    monkeypatch.setattr(settings, "COST_PER_MEAL_KSH", None)
    df = to_bitetracker_schema(_workbook())
    for col in ["rainfall_mm", "temperature_c", "is_holiday", "is_exam_period", "is_school_event", "school_term"]:
        assert df[col].isna().all(), col
        assert col not in FOOD4EDUCATION_FEATURE_COLUMNS
    assert df["average_cost_per_meal"].isna().all()


def test_calendar_facts_come_from_the_date():
    df = to_bitetracker_schema(_workbook())
    assert df["day_of_week"].tolist() == ["Friday", "Monday", "Monday"]
    assert df["month"].tolist() == [4, 4, 8]


def test_enrolment_never_below_present_students():
    df = to_bitetracker_schema(_workbook())
    # School_1 has 2 unique tapping students but 10 present on its busiest day.
    assert (df["enrolled_students"] >= df["students_present"]).all()
    # School_2 has no tap sheet, so its busiest day is the fallback.
    assert df.loc[df["school_id"] == "SCHOOL_2", "enrolled_students"].iloc[0] == 5


def test_school_day_data_sheet_name_is_accepted():
    # The 2023-2025 export calls the school-day sheet "School Day Data" and has no tap sheets.
    workbook = _workbook()
    df = to_bitetracker_schema({"School Day Data": workbook[MEALS_SHEET]})
    assert df["meals_served"].tolist() == [8, 10, 5]
    assert (df["enrolled_students"] >= df["students_present"]).all()


def test_missing_meals_sheet_raises():
    with pytest.raises(GoogleSheetError):
        to_bitetracker_schema({"school_1": pd.DataFrame()})


def test_missing_target_column_raises():
    workbook = _workbook()
    workbook[MEALS_SHEET] = workbook[MEALS_SHEET].drop(columns=["daily_unique_students_tapped"])
    with pytest.raises(GoogleSheetError, match="daily_unique_students_tapped"):
        to_bitetracker_schema(workbook)


def test_non_numeric_target_raises():
    workbook = _workbook()
    workbook[MEALS_SHEET]["daily_unique_students_tapped"] = [8, "ten", 5]
    with pytest.raises(GoogleSheetError, match="non-numeric"):
        to_bitetracker_schema(workbook)
