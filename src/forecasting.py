"""Multi-horizon meal-demand forecasting for a single school.

Honesty note (per project scope): real future attendance and weather are
*not* known. Every forecast either uses a user-supplied assumption (entered
on the Forecast page) or falls back to that school's recent
historical average, and every output is labelled as an estimate built on
that assumption — never presented as a guaranteed figure.

Multi-day forecasts (7/30 days) are iterative: each day's predicted
meals_served is fed back in as if it were observed, to compute the next
day's lag/rolling features. This is standard for autoregressive forecasting
but means errors can compound over the horizon — documented in the
web UI, not hidden.

A forecast day's own attendance is never a model input (it isn't known in
advance). Attendance features use the previous school day: the last recorded
day for day 1, and an assumed attendance for each later day.

`_build_feature_row` always builds the full synthetic feature set
(`FEATURE_COLUMNS`); each model is given only the columns it was trained on.
A Food4Education model uses a subset (no weather, term, holiday or
attendance features), so those inputs are reported as not used.
"""

# In simple terms: uses the trained model to predict meal demand for the
# next day, next week, or next month for a chosen school.

from __future__ import annotations

from datetime import date, timedelta

import joblib
import pandas as pd

from config import settings
from src.feature_engineering import FEATURE_COLUMNS
from src.synthetic_data import _is_exam_period, _is_holiday, _school_term
from src.utils import get_logger

logger = get_logger(__name__)


def load_forecasting_model():
    """Load the Phase 5 final selected model."""
    path = settings.MODELS_DIR / "final_best_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No final best model found at {path}. Run run_tuning.py first.")
    return joblib.load(path)


_NOT_USED = "not used by this model"
_ATTENDANCE_FEATURES = {"previous_attendance", "previous_attendance_rate"}


def model_feature_columns(model) -> list[str]:
    """The feature columns `model` was fitted on, in order; `FEATURE_COLUMNS` if it doesn't record them."""
    names = getattr(model, "feature_names_in_", None)
    return list(FEATURE_COLUMNS) if names is None else [str(name) for name in names]


def get_school_history(df: pd.DataFrame, school_id: str) -> pd.DataFrame:
    """Return a school's rows sorted chronologically."""
    history = df[df[settings.SCHOOL_ID_COLUMN] == school_id].copy()
    history = history.sort_values(settings.DATE_COLUMN).reset_index(drop=True)
    return history


def _recent_average(history: pd.DataFrame, column: str, window: int = 14) -> float:
    """Fallback assumption: mean of the last `window` school-day records."""
    return float(history[column].tail(window).mean())


def _build_feature_row(
    forecast_date: date,
    meals_history: list[float],
    students_present_history: list[float],
    enrolled_students: float,
    rainfall_mm: float,
    temperature_c: float,
    is_school_event: bool = False,
    term_override: str | None = None,
    holiday_override: bool | None = None,
    exam_override: bool | None = None,
) -> dict:
    """Build one row of model features for `forecast_date` from running history lists.

    `meals_history`/`students_present_history` must contain school days
    strictly before `forecast_date`, most recent last. Every feature then has
    the same meaning as in `src.feature_engineering` for a row dated
    `forecast_date`: lags/rolling stats and attendance come from earlier
    days, calendar values and (expected) weather describe `forecast_date`.
    `enrolled_students` should be the enrolment of the most recent history
    day, matching `previous_attendance_rate` in training.

    `term_override`/`holiday_override`/`exam_override` let a caller (e.g. the
    Specific Date Prediction page, where the user states these explicitly)
    override the calendar-derived defaults from `src.synthetic_data`.
    """
    term = term_override if term_override is not None else _school_term(forecast_date)
    holiday = holiday_override if holiday_override is not None else _is_holiday(forecast_date)
    exam_period = (
        exam_override if exam_override is not None else ((not holiday) and _is_exam_period(forecast_date, term))
    )

    def lag(n: int) -> float:
        return meals_history[-n] if len(meals_history) >= n else float("nan")

    def rolling_mean(n: int) -> float:
        window = meals_history[-n:] if meals_history else []
        return sum(window) / len(window) if window else float("nan")

    def rolling_std(n: int) -> float:
        window = meals_history[-n:] if meals_history else []
        if len(window) < 2:
            return 0.0
        m = sum(window) / len(window)
        return (sum((x - m) ** 2 for x in window) / (len(window) - 1)) ** 0.5

    previous_attendance = students_present_history[-1] if students_present_history else enrolled_students * 0.9
    previous_attendance_rate = (
        min(max(previous_attendance / enrolled_students, 0.0), 1.0) if enrolled_students else 0.0
    )

    row = {
        "lag_1_meals": lag(1),
        "lag_7_meals": lag(7),
        "lag_14_meals": lag(14),
        "rolling_mean_7": rolling_mean(7),
        "rolling_mean_14": rolling_mean(14),
        "rolling_std_7": rolling_std(7),
        "previous_attendance_rate": previous_attendance_rate,
        "previous_attendance": previous_attendance,
        "month": forecast_date.month,
        "rainfall_mm": rainfall_mm,
        "temperature_c": temperature_c,
        "is_holiday": int(holiday),
        "is_exam_period": int(exam_period),
        "is_school_event": int(is_school_event),
        "school_term_Term 2": int(term == "Term 2"),
        "school_term_Term 3": int(term == "Term 3"),
        "day_of_week_Tuesday": int(forecast_date.weekday() == 1),
        "day_of_week_Wednesday": int(forecast_date.weekday() == 2),
        "day_of_week_Thursday": int(forecast_date.weekday() == 3),
        "day_of_week_Friday": int(forecast_date.weekday() == 4),
    }
    return {col: row[col] for col in FEATURE_COLUMNS}


def _next_school_days(start: date, n: int) -> list[date]:
    """Return the next `n` weekdays (Mon-Fri) starting the day after `start`."""
    days = []
    current = start
    while len(days) < n:
        current = current + timedelta(days=1)
        if current.weekday() < 5:
            days.append(current)
    return days


def forecast_school_demand(
    df: pd.DataFrame,
    school_id: str,
    horizon_days: int,
    expected_attendance: float | None = None,
    expected_rainfall_mm: float | None = None,
    expected_temperature_c: float | None = None,
    school_events: set[date] | None = None,
    model=None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Forecast the next `horizon_days` school days of meal demand for one school.

    `expected_rainfall_mm`/`expected_temperature_c`: expected weather on the
    forecast days; if None, the school's trailing 14-day average is used.

    `expected_attendance`: expected students present on day 1 of the horizon.
    A day's own attendance is not a model input, so this only affects day 2
    onwards (through their previous-day attendance features). Day 1 uses the
    last recorded day's actual attendance; days after day 2 use the 14-day
    average.

    `feature_columns`: the columns the model was trained on; defaults to
    `model_feature_columns(model)`.

    The `*_assumption` columns say which basis each row used, so the UI can
    label it honestly.
    """
    model = model or load_forecasting_model()
    features = list(feature_columns) if feature_columns is not None else model_feature_columns(model)
    uses_attendance = bool(_ATTENDANCE_FEATURES & set(features))
    uses_rainfall = "rainfall_mm" in features
    uses_temperature = "temperature_c" in features
    history = get_school_history(df, school_id)
    if history.empty:
        raise ValueError(f"No history found for school_id={school_id}")

    school_events = school_events or set()
    last_date = history[settings.DATE_COLUMN].max()
    if hasattr(last_date, "date"):
        last_date = last_date.date()

    enrolled = float(history["enrolled_students"].iloc[-1])
    meals_hist = history["meals_served"].tolist()
    present_hist = history["students_present"].tolist()

    attendance_assumed = expected_attendance is None
    rainfall_assumed = expected_rainfall_mm is None
    temperature_assumed = expected_temperature_c is None

    rainfall = expected_rainfall_mm if not rainfall_assumed else _recent_average(history, "rainfall_mm")
    temperature = expected_temperature_c if not temperature_assumed else _recent_average(history, "temperature_c")
    recent_attendance = _recent_average(history, "students_present")
    day1_attendance = expected_attendance if not attendance_assumed else recent_attendance

    forecast_dates = _next_school_days(last_date, horizon_days)
    results = []

    for i, forecast_date in enumerate(forecast_dates):
        # Attendance features describe the previous school day: the last
        # recorded day for day 1, then the attendance assumed for the
        # previous forecast day.
        if not uses_attendance:
            attendance_basis = _NOT_USED
        elif i == 0:
            attendance_basis = "last recorded day (observed)"
        elif i == 1 and not attendance_assumed:
            attendance_basis = "user-provided (day 1)"
        else:
            attendance_basis = "recent 14-day average"

        feature_row = _build_feature_row(
            forecast_date=forecast_date,
            meals_history=meals_hist,
            students_present_history=present_hist,
            enrolled_students=enrolled,
            rainfall_mm=rainfall,
            temperature_c=temperature,
            is_school_event=forecast_date in school_events,
        )
        X = pd.DataFrame([feature_row])[features]
        predicted_meals = float(model.predict(X)[0])
        predicted_meals = max(0.0, predicted_meals)

        results.append(
            {
                "date": forecast_date,
                "school_id": school_id,
                "predicted_meals_served": predicted_meals,
                "attendance_assumption": attendance_basis,
                "rainfall_assumption": _NOT_USED if not uses_rainfall
                else "user-provided" if not rainfall_assumed else "recent 14-day average",
                "temperature_assumption": _NOT_USED if not uses_temperature
                else "user-provided" if not temperature_assumed else "recent 14-day average",
            }
        )

        # Feed the prediction back in as if observed, so the next day's
        # lag/rolling features reflect it (compounding uncertainty — see
        # module docstring). This day's attendance is an assumption too.
        meals_hist = meals_hist + [predicted_meals]
        present_hist = present_hist + [day1_attendance if i == 0 else recent_attendance]

    return pd.DataFrame(results)


def predict_specific_date(
    df: pd.DataFrame,
    school_id: str,
    target_date: date,
    expected_rainfall_mm: float,
    expected_temperature_c: float,
    school_term: str,
    is_holiday: bool,
    is_exam_period: bool,
    is_school_event: bool = False,
    model=None,
    feature_columns: list[str] | None = None,
) -> dict:
    """Predict meal demand for one specific, user-chosen future date.

    Unlike `forecast_school_demand` (which derives term/holiday/exam from the
    calendar and lets weather default to a recent average), the weather and
    calendar inputs here are supplied by the user on the Planning page. The
    function auto-derives day-of-week/month (from `target_date` itself) and
    lag/rolling/attendance history from the school's actual past records.

    Attendance on `target_date` is not an input: it isn't known in advance.
    The attendance features use the last recorded day, and lag features use
    the most recent recorded days even when `target_date` is further ahead
    (an approximation that grows with `days_ahead`).

    Raises `ValueError` if `target_date` is not strictly after the school's
    last recorded date — the model has no lag history to predict backward
    or into an already-observed day.
    """
    model = model or load_forecasting_model()
    history = get_school_history(df, school_id)
    if history.empty:
        raise ValueError(f"No history found for school_id={school_id}")

    last_date = history[settings.DATE_COLUMN].max()
    if hasattr(last_date, "date"):
        last_date = last_date.date()
    if target_date <= last_date:
        raise ValueError(
            f"Target date {target_date} must be after this school's last recorded date ({last_date})."
        )

    enrolled = float(history["enrolled_students"].iloc[-1])
    meals_hist = history["meals_served"].tolist()
    present_hist = history["students_present"].tolist()

    feature_row = _build_feature_row(
        forecast_date=target_date,
        meals_history=meals_hist,
        students_present_history=present_hist,
        enrolled_students=enrolled,
        rainfall_mm=expected_rainfall_mm,
        temperature_c=expected_temperature_c,
        is_school_event=is_school_event,
        term_override=school_term,
        holiday_override=is_holiday,
        exam_override=is_exam_period,
    )
    features = list(feature_columns) if feature_columns is not None else model_feature_columns(model)
    X = pd.DataFrame([feature_row])[features]
    predicted_meals = max(0.0, float(model.predict(X)[0]))

    days_ahead = (target_date - last_date).days

    return {
        "school_id": school_id,
        "target_date": target_date,
        "days_ahead": days_ahead,
        "predicted_meals_served": predicted_meals,
        "recent_std_7": feature_row["rolling_std_7"],
        "recent_history": history.tail(21),
        "feature_row": feature_row,
    }
