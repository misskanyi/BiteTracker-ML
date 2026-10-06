"""School calendars: which dates are school days, holidays or half-terms, and in which term.

Two calendars are provided:

- `SyntheticDevelopmentCalendar` wraps the approximate calendar in
  `src.synthetic_data`. It is the calendar the current synthetic model was
  trained with, so it is the one used while synthetic artifacts are active;
  using any other calendar would feed the model features it never saw.
- `KenyaPrimaryCalendar` holds the supplied 2025 Kenyan primary-school
  calendar. It is ready for the real-data pipeline but is not used by the
  current synthetic model. Dates outside the configured year are reported as
  "unknown" rather than guessed.

`get_active_calendar()` picks the calendar that matches the active data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from services import status_service
from src.synthetic_data import _is_exam_period, _is_holiday, _school_term

SCHOOL_DAY, HALF_TERM, HOLIDAY, WEEKEND, UNKNOWN = "school_day", "half_term", "holiday", "weekend", "unknown"


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


@dataclass(frozen=True)
class DayInfo:
    date: date
    kind: str  # one of SCHOOL_DAY, HALF_TERM, HOLIDAY, WEEKEND, UNKNOWN
    term: str | None

    @property
    def is_school_day(self) -> bool:
        return self.kind == SCHOOL_DAY


# Supplied 2025 Kenyan primary-school calendar. No Term 3 half-term was supplied.
KENYA_PRIMARY_2025 = {
    "terms": [
        Period("Term 1", date(2025, 1, 6), date(2025, 4, 4)),
        Period("Term 2", date(2025, 4, 28), date(2025, 8, 1)),
        Period("Term 3", date(2025, 8, 25), date(2025, 10, 24)),
    ],
    "half_terms": [
        Period("Term 1 half-term", date(2025, 2, 26), date(2025, 3, 2)),
        Period("Term 2 half-term", date(2025, 6, 25), date(2025, 6, 29)),
    ],
    "holidays": [
        Period("April holiday", date(2025, 4, 7), date(2025, 4, 25)),
        Period("August holiday", date(2025, 8, 4), date(2025, 8, 22)),
    ],
}


class KenyaPrimaryCalendar:
    """Calendar built from explicit term, half-term and holiday periods."""

    name = "Kenya primary-school calendar"

    def __init__(self, definition: dict | None = None):
        definition = definition or KENYA_PRIMARY_2025
        self.terms: list[Period] = list(definition["terms"])
        self.half_terms: list[Period] = list(definition.get("half_terms", []))
        self.holidays: list[Period] = list(definition.get("holidays", []))
        self.first_day = min(p.start for p in self.terms)
        self.last_day = max(p.end for p in self.terms)

    def classify(self, d: date) -> DayInfo:
        term = next((p.name for p in self.terms if p.contains(d)), None)
        if not (self.first_day <= d <= self.last_day):
            return DayInfo(d, UNKNOWN, None)
        if d.weekday() >= 5:
            return DayInfo(d, WEEKEND, term)
        if any(p.contains(d) for p in self.half_terms):
            return DayInfo(d, HALF_TERM, term)
        if term is None or any(p.contains(d) for p in self.holidays):
            return DayInfo(d, HOLIDAY, None)
        return DayInfo(d, SCHOOL_DAY, term)

    def is_holiday(self, d: date) -> bool:
        return self.classify(d).kind in (HOLIDAY, HALF_TERM)

    def next_school_days(self, after: date, n: int) -> list[date]:
        """The next `n` school days after `after`; stops early at the end of the known calendar."""
        days, current = [], after
        while len(days) < n and current < self.last_day:
            current += timedelta(days=1)
            if self.classify(current).is_school_day:
                days.append(current)
        return days


class SyntheticDevelopmentCalendar:
    """The approximate calendar the synthetic dataset (and current model) were generated with."""

    name = "Synthetic development calendar"

    def classify(self, d: date) -> DayInfo:
        if d.weekday() >= 5:
            return DayInfo(d, WEEKEND, _school_term(d))
        if _is_holiday(d):
            return DayInfo(d, HOLIDAY, _school_term(d))
        return DayInfo(d, SCHOOL_DAY, _school_term(d))

    def is_holiday(self, d: date) -> bool:
        return _is_holiday(d)

    def is_exam_period(self, d: date) -> bool:
        return (not _is_holiday(d)) and _is_exam_period(d, _school_term(d))


def get_active_calendar():
    """The calendar matching the active model's data (synthetic unless real data is active)."""
    if status_service.get_data_status()["kind"] == "real":
        return KenyaPrimaryCalendar()
    return SyntheticDevelopmentCalendar()
