"""Load the Food4Education dataset straight from its Google Sheets share link.

The workbook is fetched into memory on every call and never written to disk,
so the shared sheet stays the single source of truth. Access is read-only:
the sheet's .xlsx export link is downloaded, nothing is ever sent back. It is
then mapped onto the canonical BiteTracker schema
(`src.synthetic_data.SCHEMA_COLUMNS`), so preprocessing, feature engineering
and training need no source-specific code paths.

Workbook layout expected (checked by `to_bitetracker_schema`):
- "meals produced and served" (or "School Day Data" in the 2023-2025
  export): one row per school-day with
  deidentified_school, activity_date, daily_unique_students_tapped,
  projected_students, actual_students.
- "school_1" ... "school_N": one row per student tap with
  school, activity_date, student_id.

Field mapping (source -> BiteTracker schema):
- daily_unique_students_tapped -> meals_served, the forecasting target.
  The same count is also stored as students_present for dashboards, so the
  previous-attendance features are not used for this source (they would
  repeat lag_1_meals; see feature_engineering.FOOD4EDUCATION_FEATURE_COLUMNS).
- projected_students -> meals_prepared, for reporting only (planned vs
  served). It is never a model feature or a naive baseline.
- actual_students is not carried over: it is not used as a feature, a
  target or a baseline.
- unique student_ids per school (tap sheets) -> enrolled_students, a proxy
  for enrolment used for dashboards only (not a model feature).
- day_of_week and month are derived from the date.

The sheet has no weather, cost, location, term, holiday, exam or event data.
Those columns are left empty (NaN, or "Not provided" for the location
labels) rather than filled with invented values, and are not model
features for this source. average_cost_per_meal is
BITETRACKER_COST_PER_MEAL_KSH when that is configured, otherwise empty.
"""

# In simple terms: downloads the real school-meals spreadsheet into memory
# from its share link and reshapes it to look like the dataset the rest of
# BiteTracker already understands, without making up any missing values.

from __future__ import annotations

import io
import re
import urllib.request

import numpy as np
import pandas as pd

from config import settings
from src.synthetic_data import SCHEMA_COLUMNS
from src.utils import get_logger

logger = get_logger(__name__)

MEALS_SHEET = "meals produced and served"
# The 2023-2025 export names the school-day sheet differently; same columns.
MEALS_SHEET_NAMES = [MEALS_SHEET, "School Day Data"]
TAP_SHEET_PREFIX = "school_"
_REQUEST_TIMEOUT_SECONDS = 120

# Source column used as the forecasting target (recorded in model metadata).
TARGET_SOURCE_COLUMN = "daily_unique_students_tapped"
REQUIRED_MEALS_COLUMNS = [
    "deidentified_school",
    "activity_date",
    "daily_unique_students_tapped",
    "projected_students",
    "actual_students",
]
_COUNT_COLUMNS = ["daily_unique_students_tapped", "projected_students"]


class GoogleSheetError(RuntimeError):
    """Raised when the shared sheet cannot be fetched or has an unexpected layout."""


def export_url(share_url: str) -> str:
    """Turn any Google Sheets share/edit link into its .xlsx export link."""
    match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", share_url)
    if not match:
        # The link itself is not echoed: it may grant access to the data.
        raise GoogleSheetError("Not a Google Sheets link (expected .../spreadsheets/d/<id>/...).")
    return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=xlsx"


def fetch_workbook(share_url: str) -> dict[str, pd.DataFrame]:
    """Fetch every sheet of the shared workbook into memory as DataFrames (read-only)."""
    url = export_url(share_url)
    logger.info("Fetching the Food4Education Google Sheet (read-only export)")
    try:
        with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            content = response.read()
    except OSError as exc:
        raise GoogleSheetError(
            f"Could not fetch the Google Sheet ({type(exc).__name__}). Check the link is readable "
            "with your access settings and that you are online."
        ) from exc
    try:
        return pd.read_excel(io.BytesIO(content), sheet_name=None)
    except ValueError as exc:
        raise GoogleSheetError(
            "The link did not return a spreadsheet. It may require sign-in; check the sheet's "
            "sharing settings with its owner."
        ) from exc


def _school_id(name: str) -> str:
    """'School_1' -> 'SCHOOL_1', so ids are stable whatever the sheet's casing."""
    return str(name).strip().upper()


def _enrolment_by_school(sheets: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Unique students seen tapping at each school: the closest available proxy for enrolment."""
    enrolment: dict[str, int] = {}
    for name, taps in sheets.items():
        if not name.lower().startswith(TAP_SHEET_PREFIX):
            continue
        if not {"school", "student_id"}.issubset(taps.columns):
            logger.warning("Skipping sheet %r: missing school/student_id columns", name)
            continue
        for school, group in taps.groupby("school"):
            enrolment[_school_id(school)] = int(group["student_id"].nunique())
    return enrolment


def _mark_unavailable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the schema columns the sheet does not provide, left empty rather than invented."""
    df["county"] = "Not provided"
    df["sub_county"] = "Not provided"
    df["average_cost_per_meal"] = np.nan if settings.COST_PER_MEAL_KSH is None else settings.COST_PER_MEAL_KSH
    for col in ["rainfall_mm", "temperature_c", "is_holiday", "is_exam_period", "is_school_event"]:
        df[col] = np.nan
    df["school_term"] = None
    return df


def _validate_meals_sheet(meals: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Check required columns exist and counts/dates parse; return a typed copy."""
    missing = [c for c in REQUIRED_MEALS_COLUMNS if c not in meals.columns]
    if missing:
        raise GoogleSheetError(f"Sheet {sheet_name!r} is missing columns: {missing}")
    meals = meals.copy()
    meals["activity_date"] = pd.to_datetime(meals["activity_date"], errors="coerce")
    if meals["activity_date"].isna().any():
        raise GoogleSheetError(
            f"Sheet {sheet_name!r} has {int(meals['activity_date'].isna().sum())} unparsable activity_date values"
        )
    for col in _COUNT_COLUMNS:
        numeric = pd.to_numeric(meals[col], errors="coerce")
        bad = int((numeric.isna() & meals[col].notna()).sum())
        if bad:
            raise GoogleSheetError(f"Sheet {sheet_name!r} column {col!r} has {bad} non-numeric values")
        meals[col] = numeric
    return meals


def to_bitetracker_schema(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Map the Food4Education workbook onto `SCHEMA_COLUMNS`."""
    sheet_name = next((name for name in MEALS_SHEET_NAMES if name in sheets), None)
    if sheet_name is None:
        raise GoogleSheetError(f"None of the sheets {MEALS_SHEET_NAMES} found; got {list(sheets)}")
    meals = _validate_meals_sheet(sheets[sheet_name], sheet_name)

    df = pd.DataFrame({
        "date": meals["activity_date"],
        "school_id": meals["deidentified_school"].map(_school_id),
        "school_name": meals["deidentified_school"].astype(str).str.replace("_", " "),
        "meals_served": meals[TARGET_SOURCE_COLUMN],
        "students_present": meals[TARGET_SOURCE_COLUMN],
        "meals_prepared": meals["projected_students"],
    })

    enrolment = _enrolment_by_school(sheets)
    # Fall back to the busiest recorded day for schools without a tap sheet.
    busiest_day = df.groupby("school_id")["students_present"].transform("max")
    df["enrolled_students"] = df["school_id"].map(enrolment).fillna(busiest_day)
    df["enrolled_students"] = df[["enrolled_students", "students_present"]].max(axis=1)
    df["students_absent"] = df["enrolled_students"] - df["students_present"]
    df["leftover_meals"] = (df["meals_prepared"] - df["meals_served"]).clip(lower=0)

    df["day_of_week"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df = _mark_unavailable_columns(df)

    logger.info(
        "Mapped Google Sheet to %d school-day rows across %d schools (target: %s)",
        len(df), df["school_id"].nunique(), TARGET_SOURCE_COLUMN,
    )
    return df[SCHEMA_COLUMNS]


def load_google_sheet_dataset(share_url: str | None = None) -> pd.DataFrame:
    """Fetch the shared sheet and return it in the BiteTracker schema."""
    share_url = share_url or settings.GOOGLE_SHEET_URL
    if not share_url:
        raise GoogleSheetError("No Google Sheet link configured (set BITETRACKER_GOOGLE_SHEET_URL).")
    return to_bitetracker_schema(fetch_workbook(share_url))
