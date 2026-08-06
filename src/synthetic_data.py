"""Synthetic school-meal dataset generator.

IMPORTANT: This module produces SYNTHETIC data for development and testing
only. It exists because the real-world dataset for this project is not yet
available. The generator encodes believable domain relationships (rainfall
depressing attendance, holidays halting meal service, etc.) so that the rest
of the pipeline (preprocessing, feature engineering, modelling) can be built
and validated before real data arrives.

To swap in a real dataset later: point `config.settings.ACTIVE_DATA_PATH`
(or the `BITETRACKER_DATA_PATH` env var) at the new CSV. As long as it has
the same column schema as `SCHEMA_COLUMNS` below, nothing else needs to
change.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from src.utils import get_logger, set_global_seed

logger = get_logger(__name__)

# Canonical raw-data schema. `data_loader.py` validates incoming CSVs against
# this list so a real dataset can be swapped in without touching downstream code.
SCHEMA_COLUMNS = [
    "date",
    "school_id",
    "school_name",
    "county",
    "sub_county",
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
    "school_term",
    "day_of_week",
    "month",
]

# Approximate Kenyan school-holiday windows (month-day tuples, inclusive) that
# recur every calendar year. Long rains / short rains inform rainfall seasonality.
_HOLIDAY_WINDOWS = [
    ((4, 1), (4, 25)),    # April holiday (between Term 1 and Term 2)
    ((8, 1), (8, 25)),    # August holiday (between Term 2 and Term 3)
    ((11, 20), (12, 31)), # Long December holiday
    ((1, 1), (1, 3)),     # New year tail
]


def _in_window(d: date, window: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    (sm, sd), (em, ed) = window
    start = date(d.year, sm, sd)
    end = date(d.year, em, ed)
    return start <= d <= end


def _is_holiday(d: date) -> bool:
    return any(_in_window(d, w) for w in _HOLIDAY_WINDOWS)


def _school_term(d: date) -> str:
    if d.month <= 3 or (d.month == 4 and d.day < 1):
        return "Term 1"
    if 5 <= d.month <= 7 or (d.month == 4) or (d.month == 8 and d.day < 1):
        return "Term 2"
    return "Term 3"


def _is_exam_period(d: date, term: str) -> bool:
    # Exam weeks fall near the end of each term.
    exam_windows = {
        "Term 1": ((3, 18), (3, 31)),
        "Term 2": ((7, 15), (7, 31)),
        "Term 3": ((11, 1), (11, 15)),
    }
    window = exam_windows[term]
    return _in_window(d, window)


def _seasonal_rainfall(d: date, rng: np.random.Generator) -> float:
    # Long rains: Mar-May, Short rains: Oct-Dec. Otherwise drier.
    if d.month in (3, 4, 5):
        base = 18.0
    elif d.month in (10, 11, 12):
        base = 12.0
    else:
        base = 3.0
    noise = rng.gamma(shape=1.2, scale=base / 1.5 + 1e-6)
    return round(float(max(0.0, noise)), 1)


def _seasonal_temperature(d: date, rng: np.random.Generator) -> float:
    # Mild tropical-highland pattern: warmest Jan-Mar, coolest Jun-Jul.
    seasonal = 22.0 + 3.0 * np.cos((d.month - 2) / 12 * 2 * np.pi)
    noise = rng.normal(0, 1.2)
    return round(float(seasonal + noise), 1)


def _school_day_dates(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday-Friday
            days.append(current)
        current += timedelta(days=1)
    return days


def generate_synthetic_dataset(
    schools: list[dict] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic school-meal-demand dataset.

    One row per school per school day (Mon-Fri), across the given date
    range, for every school in `schools`. Encodes believable relationships
    between attendance, weather, holidays, exams, and meal counts, plus a
    small proportion of missing values and outliers so the preprocessing
    pipeline has real data quality issues to handle.
    """
    schools = schools or settings.SYNTHETIC_SCHOOLS
    start = date.fromisoformat(start_date or settings.SYNTHETIC_START_DATE)
    end = date.fromisoformat(end_date or settings.SYNTHETIC_END_DATE)
    seed = seed if seed is not None else settings.RANDOM_SEED

    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    all_days = _school_day_dates(start, end)
    logger.info("Generating synthetic data for %d schools across %d school days", len(schools), len(all_days))

    rows: list[dict] = []

    for school in schools:
        enrollment = float(school["base_enrollment"])
        cost_per_meal = rng.uniform(14.0, 20.0)  # KES, drifts slowly (inflation)
        prev_present = enrollment * 0.9

        for d in all_days:
            term = _school_term(d)
            holiday = _is_holiday(d)
            exam_period = (not holiday) and _is_exam_period(d, term)
            school_event = (not holiday) and rng.random() < 0.02

            # Enrolment drifts slightly term to term, stays broadly stable.
            enrollment += rng.normal(0, 0.15)
            enrollment = max(50.0, enrollment)

            rainfall = _seasonal_rainfall(d, rng)
            temperature = _seasonal_temperature(d, rng)

            if holiday:
                attendance_rate = float(np.clip(rng.normal(0.03, 0.02), 0.0, 0.15))
            else:
                base_rate = 0.90
                if rainfall > 25:
                    base_rate -= 0.12
                elif rainfall > 15:
                    base_rate -= 0.05
                if exam_period:
                    base_rate += 0.03
                if school_event:
                    base_rate += 0.06
                # Attendance is sticky day-to-day (autocorrelated) plus noise.
                momentum = 0.3 * (prev_present / enrollment - base_rate)
                attendance_rate = float(np.clip(base_rate - momentum + rng.normal(0, 0.04), 0.05, 0.99))

            students_present = round(enrollment * attendance_rate)
            students_present = max(0, min(students_present, round(enrollment)))
            students_absent = round(enrollment) - students_present
            prev_present = students_present if students_present > 0 else prev_present

            # Caterers estimate demand from recent attendance; imperfect -> leftovers.
            prep_bias = rng.normal(1.05, 0.06) if not holiday else rng.normal(0.2, 0.1)
            meals_prepared = max(0, round(students_present * max(prep_bias, 0)))
            serving_efficiency = float(np.clip(rng.normal(0.96, 0.03), 0.7, 1.0))
            meals_served = min(meals_prepared, round(students_present * serving_efficiency))
            leftover_meals = max(0, meals_prepared - meals_served)

            cost_per_meal += rng.normal(0, 0.03)  # slow inflation drift
            cost_per_meal = float(np.clip(cost_per_meal, 10.0, 35.0))

            rows.append(
                {
                    "date": d.isoformat(),
                    "school_id": school["school_id"],
                    "school_name": school["school_name"],
                    "county": school["county"],
                    "sub_county": school["sub_county"],
                    "enrolled_students": round(enrollment),
                    "students_present": students_present,
                    "students_absent": students_absent,
                    "meals_prepared": meals_prepared,
                    "meals_served": meals_served,
                    "leftover_meals": leftover_meals,
                    "average_cost_per_meal": round(cost_per_meal, 2),
                    "rainfall_mm": rainfall,
                    "temperature_c": temperature,
                    "is_holiday": int(holiday),
                    "is_exam_period": int(exam_period),
                    "is_school_event": int(school_event),
                    "school_term": term,
                    "day_of_week": d.strftime("%A"),
                    "month": d.month,
                }
            )

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    df = _inject_data_quality_issues(df, rng)
    logger.info("Generated %d rows, %d columns", *df.shape)
    return df


def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Randomly null out a small fraction of values and inject a few outliers.

    This mirrors real-world messiness (sensor gaps, data-entry errors) so the
    preprocessing pipeline has something genuine to clean.
    """
    df = df.copy()
    n = len(df)

    missing_cols = ["rainfall_mm", "temperature_c", "meals_prepared", "students_present", "average_cost_per_meal"]
    for col in missing_cols:
        n_missing = max(1, int(n * rng.uniform(0.005, 0.02)))
        idx = rng.choice(n, size=n_missing, replace=False)
        df.loc[idx, col] = np.nan

    outlier_cols = ["meals_served", "leftover_meals", "enrolled_students"]
    for col in outlier_cols:
        n_outliers = max(1, int(n * rng.uniform(0.002, 0.006)))
        idx = rng.choice(n, size=n_outliers, replace=False)
        multiplier = rng.choice([0.1, 3.5, 4.0], size=n_outliers)
        df.loc[idx, col] = (df.loc[idx, col].fillna(df[col].median()) * multiplier).round()

    return df


def save_synthetic_dataset(df: pd.DataFrame | None = None, path: Path | None = None) -> Path:
    """Generate (if needed) and persist the synthetic dataset to CSV."""
    if df is None:
        df = generate_synthetic_dataset()
    out_path = path or settings.SYNTHETIC_DATA_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Synthetic dataset saved to %s (%d rows)", out_path, len(df))
    return out_path


if __name__ == "__main__":
    dataset = generate_synthetic_dataset()
    save_synthetic_dataset(dataset)
