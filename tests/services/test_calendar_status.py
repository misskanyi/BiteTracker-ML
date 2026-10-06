"""Stage 3: calendar_service (synthetic + 2025 Kenyan calendar) and status_service (data labels)."""

from __future__ import annotations

from datetime import date

import pytest

from config import settings
from services import calendar_service, status_service
from services.calendar_service import (
    HALF_TERM,
    HOLIDAY,
    SCHOOL_DAY,
    UNKNOWN,
    WEEKEND,
    KenyaPrimaryCalendar,
    SyntheticDevelopmentCalendar,
)
from src.synthetic_data import _is_holiday, _school_term

# --- 2025 Kenyan primary-school calendar ---------------------------------------------

KENYA = KenyaPrimaryCalendar()


@pytest.mark.parametrize(
    "day, kind, term",
    [
        (date(2025, 1, 6), SCHOOL_DAY, "Term 1"),   # first day of Term 1 (Monday)
        (date(2025, 1, 11), WEEKEND, "Term 1"),
        (date(2025, 2, 26), HALF_TERM, "Term 1"),   # half-term start (Wednesday)
        (date(2025, 2, 28), HALF_TERM, "Term 1"),
        (date(2025, 3, 3), SCHOOL_DAY, "Term 1"),   # back after half-term
        (date(2025, 4, 4), SCHOOL_DAY, "Term 1"),   # last day of Term 1
        (date(2025, 4, 7), HOLIDAY, None),          # April holiday
        (date(2025, 4, 25), HOLIDAY, None),
        (date(2025, 4, 28), SCHOOL_DAY, "Term 2"),
        (date(2025, 6, 25), HALF_TERM, "Term 2"),
        (date(2025, 6, 30), SCHOOL_DAY, "Term 2"),
        (date(2025, 8, 1), SCHOOL_DAY, "Term 2"),
        (date(2025, 8, 4), HOLIDAY, None),          # August holiday
        (date(2025, 8, 25), SCHOOL_DAY, "Term 3"),
        (date(2025, 10, 24), SCHOOL_DAY, "Term 3"), # last day of Term 3
        (date(2025, 10, 27), UNKNOWN, None),        # beyond the supplied calendar
        (date(2024, 12, 23), UNKNOWN, None),
    ],
)
def test_kenya_2025_calendar_classification(day, kind, term):
    info = KENYA.classify(day)
    assert info.kind == kind
    assert info.term == term


def test_kenya_calendar_holidays_and_half_terms_are_not_school_days():
    assert KENYA.is_holiday(date(2025, 4, 10))
    assert KENYA.is_holiday(date(2025, 6, 26))
    assert not KENYA.is_holiday(date(2025, 5, 5))


def test_kenya_next_school_days_skips_weekends_and_half_term():
    days = KENYA.next_school_days(date(2025, 2, 24), 3)  # Mon 24 Feb → Tue 25, then half-term until Sun 2 Mar
    assert days == [date(2025, 2, 25), date(2025, 3, 3), date(2025, 3, 4)]


def test_kenya_next_school_days_skips_holiday_between_terms():
    days = KENYA.next_school_days(date(2025, 4, 3), 3)
    assert days == [date(2025, 4, 4), date(2025, 4, 28), date(2025, 4, 29)]


def test_kenya_next_school_days_stops_at_end_of_known_calendar():
    assert KENYA.next_school_days(date(2025, 10, 22), 10) == [date(2025, 10, 23), date(2025, 10, 24)]


# --- Synthetic calendar must match what the synthetic model was trained with ---------------


@pytest.mark.parametrize("day", [date(2024, 12, 23), date(2024, 4, 15), date(2024, 6, 3), date(2024, 8, 14)])
def test_synthetic_calendar_matches_src_synthetic_data(day):
    calendar = SyntheticDevelopmentCalendar()
    assert calendar.is_holiday(day) == _is_holiday(day)
    assert calendar.classify(day).term == _school_term(day)


def test_active_calendar_is_synthetic_while_synthetic_model_active(stub_artifacts):
    assert isinstance(calendar_service.get_active_calendar(), SyntheticDevelopmentCalendar)


def test_active_calendar_is_kenya_calendar_for_real_data(stub_artifacts, make_metadata, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ACTIVE_DATA_PATH", tmp_path / "food4education.csv")
    stub_artifacts.write_metadata(make_metadata(results_label="FOOD4EDUCATION RESEARCH RESULTS"))
    assert isinstance(calendar_service.get_active_calendar(), KenyaPrimaryCalendar)


# --- Data status labels ---------------------------------------------------------------------------


def test_status_synthetic_from_metadata(stub_artifacts):
    status = status_service.get_data_status()
    assert status["kind"] == "synthetic"
    assert status["label"] == "SYNTHETIC DEVELOPMENT DATA"
    assert status["model_name"] == "Linear Regression"


def test_status_synthetic_when_dataset_path_is_synthetic_even_if_label_is_not(stub_artifacts, make_metadata):
    """A real-looking label must never hide that the active dataset is the synthetic one."""
    stub_artifacts.write_metadata(make_metadata(results_label="REAL RESULTS"))
    assert status_service.get_data_status()["kind"] == "synthetic"


def test_status_real_only_when_both_dataset_and_metadata_are_real(stub_artifacts, make_metadata, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ACTIVE_DATA_PATH", tmp_path / "food4education.csv")
    stub_artifacts.write_metadata(make_metadata(results_label="FOOD4EDUCATION RESEARCH RESULTS"))
    status = status_service.get_data_status()
    assert status["kind"] == "real"
    assert status["label"] == "FOOD4EDUCATION RESEARCH RESULTS"

    stub_artifacts.write_metadata(make_metadata(results_label="SYNTHETIC DEVELOPMENT RESULTS"))
    assert status_service.get_data_status()["kind"] == "synthetic"


def test_status_unknown_when_metadata_missing_and_dataset_not_synthetic(stub_artifacts, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ACTIVE_DATA_PATH", tmp_path / "food4education.csv")
    stub_artifacts.metadata_path.unlink()
    status = status_service.get_data_status()
    assert status["kind"] == "unknown"
    assert status["label"] == "DATA STATUS UNKNOWN"


def test_status_synthetic_when_metadata_missing_but_dataset_synthetic(stub_artifacts):
    stub_artifacts.metadata_path.unlink()
    assert status_service.get_data_status()["kind"] == "synthetic"


def test_real_status_is_synthetic(real_artifacts):
    assert status_service.get_data_status()["label"] == "SYNTHETIC DEVELOPMENT DATA"
