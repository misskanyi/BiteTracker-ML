"""Read access to the historical meal-demand data used for forecasting.

The processed CSV written by the ML pipeline
(`data/processed/school_meals_processed.csv`) remains the single source of
truth; nothing here copies demand records into a database. The file is read
once and cached per process, keyed on its modification time, and callers
always receive a copy so they cannot alter the cached frame.
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path

import pandas as pd

from config import settings
from services import ServiceError
from src.forecasting import get_school_history as _src_get_school_history

PROCESSED_FILENAME = "school_meals_processed.csv"

SCHOOL_INFO_COLUMNS = ["school_id", "school_name", "county", "sub_county", "enrolled_students"]


class DataNotFoundError(ServiceError):
    """The processed dataset has not been generated."""


class DataLoadError(ServiceError):
    """The processed dataset exists but is unreadable or lacks required columns."""


class UnknownSchoolError(ServiceError):
    """The requested school is not in the dataset."""


_cache: dict[Path, tuple[int, pd.DataFrame]] = {}
_cache_lock = threading.Lock()


def processed_data_path() -> Path:
    return settings.PROCESSED_DATA_DIR / PROCESSED_FILENAME


def load_processed_data() -> pd.DataFrame:
    """The full processed dataset (all schools), with `date` parsed."""
    path = processed_data_path()
    if not path.exists():
        raise DataNotFoundError(
            f"Processed dataset not found at {path}. Run `python scripts/train_models.py` to generate it."
        )
    mtime = path.stat().st_mtime_ns
    with _cache_lock:
        cached = _cache.get(path)
        if cached is None or cached[0] != mtime:
            try:
                df = pd.read_csv(path)
            except (OSError, ValueError) as exc:
                raise DataLoadError(f"Processed dataset at {path} could not be read: {exc}") from exc
            required = [settings.DATE_COLUMN, settings.SCHOOL_ID_COLUMN, "school_name", "meals_served"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise DataLoadError(f"Processed dataset is missing required columns: {', '.join(missing)}.")
            df[settings.DATE_COLUMN] = pd.to_datetime(df[settings.DATE_COLUMN])
            _cache[path] = (mtime, df)
            cached = _cache[path]
    return cached[1].copy()


def list_schools() -> list[dict]:
    """One dict per school (id, name, location, latest enrolment), sorted by name."""
    df = load_processed_data().sort_values(settings.DATE_COLUMN)
    columns = [c for c in SCHOOL_INFO_COLUMNS if c in df.columns]
    latest = df.groupby(settings.SCHOOL_ID_COLUMN, as_index=False).last()[columns]
    records = latest.sort_values("school_name").to_dict(orient="records")
    for record in records:
        if "enrolled_students" in record:
            record["enrolled_students"] = int(record["enrolled_students"])
    return records


def school_ids() -> list[str]:
    return [s["school_id"] for s in list_schools()]


def get_school_history(
    school_id: str, start: date | None = None, end: date | None = None, df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """A school's records in date order (via `src.forecasting.get_school_history`), optionally date-filtered.

    Raises `UnknownSchoolError` if the school has no records at all.
    """
    df = load_processed_data() if df is None else df
    history = _src_get_school_history(df, school_id)
    if history.empty:
        raise UnknownSchoolError(f"School '{school_id}' was not found in the dataset.")
    if start is not None:
        history = history[history[settings.DATE_COLUMN].dt.date >= start]
    if end is not None:
        history = history[history[settings.DATE_COLUMN].dt.date <= end]
    return history.reset_index(drop=True)


def get_date_range(school_id: str | None = None) -> dict:
    """First and last recorded dates, for one school or the whole dataset."""
    df = get_school_history(school_id) if school_id else load_processed_data()
    return {
        "start": df[settings.DATE_COLUMN].min().date(),
        "end": df[settings.DATE_COLUMN].max().date(),
    }


def get_latest_date(school_id: str | None = None) -> date:
    """The most recent date with recorded data: the point forecasts start from."""
    return get_date_range(school_id)["end"]


def recent_demand_summary(window: int = 14) -> list[dict]:
    """Per school: latest recorded day and the mean/min/max meals served over its last `window` records.

    Counts are returned as stored (floats), not truncated: preprocessing caps outliers at
    IQR fences, so a few processed counts are non-integer and the display must show them as such.
    """
    df = load_processed_data()
    original = original_count_column()
    summary = []
    for school in list_schools():
        history = _src_get_school_history(df, school["school_id"])
        recent = history.tail(window)["meals_served"]
        latest = history.iloc[-1]
        latest_original = float(latest[original]) if original and original in history else None
        summary.append(
            {
                **school,
                "latest_date": latest[settings.DATE_COLUMN].date(),
                "latest_meals_served": float(latest["meals_served"]),
                "latest_original_meals": latest_original,
                "latest_is_processed": is_processed_value(latest["meals_served"], latest_original),
                "recent_records": int(len(recent)),
                "recent_mean_meals": round(float(recent.mean()), 1),
                "recent_min_meals": float(recent.min()),
                "recent_max_meals": float(recent.max()),
                "total_records": int(len(history)),
            }
        )
    return summary


def original_count_column() -> str | None:
    """Column that still holds the unprocessed source count behind `meals_served`, if the source has one.

    The Google Sheet loader fills `meals_served` and `students_present` from the same source
    column (daily unique students tapped), and only `meals_served` is outlier-capped during
    preprocessing, so `students_present` keeps the original observation. Other sources record
    them separately, so no original is available there.
    """
    return "students_present" if settings.USE_GOOGLE_SHEET else None


def is_processed_value(meals, original) -> bool:
    """True when a processed meal count differs from its original observation (or, without one, is non-integer)."""
    if original is not None and not pd.isna(original):
        return float(meals) != float(original)
    return not float(meals).is_integer()


WEATHER_COLUMNS = ("rainfall_mm", "temperature_c")


def unrecorded_columns(df: pd.DataFrame | None = None) -> set[str]:
    """Weather columns the active data does not genuinely record.

    The Food4Education Google Sheet has no weather data (the loader leaves these
    columns empty), so placeholder values in the processed file must not be shown
    as measurements. A column that is absent or entirely empty is also unrecorded.
    """
    df = load_processed_data() if df is None else df
    missing = {c for c in WEATHER_COLUMNS if c not in df.columns or df[c].isna().all()}
    if settings.USE_GOOGLE_SHEET:
        missing.update(WEATHER_COLUMNS)
    return missing


def clear_data_cache() -> None:
    with _cache_lock:
        _cache.clear()
