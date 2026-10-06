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
    """Per school: latest recorded day and the mean/min/max meals served over its last `window` records."""
    df = load_processed_data()
    summary = []
    for school in list_schools():
        history = _src_get_school_history(df, school["school_id"])
        recent = history.tail(window)["meals_served"]
        latest = history.iloc[-1]
        summary.append(
            {
                **school,
                "latest_date": latest[settings.DATE_COLUMN].date(),
                "latest_meals_served": int(latest["meals_served"]),
                "recent_records": int(len(recent)),
                "recent_mean_meals": round(float(recent.mean()), 1),
                "recent_min_meals": int(recent.min()),
                "recent_max_meals": int(recent.max()),
                "total_records": int(len(history)),
            }
        )
    return summary


def clear_data_cache() -> None:
    with _cache_lock:
        _cache.clear()
