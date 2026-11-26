"""School calendars: which dates are school days, holidays or half-terms, and in which term.

Two calendars are provided:

- `SyntheticDevelopmentCalendar` wraps the approximate calendar in
  `src.synthetic_data`. It is the calendar the current synthetic model was
  trained with, so it is the one used while synthetic artifacts are active;
  using any other calendar would feed the model features it never saw.
- `KenyaPrimaryCalendar` holds the Kenyan national school calendar, used with
  the real data. By default it is loaded from
  data/external/calendar/kenya_basic_education_calendar_2023_2026.csv (terms,
  half-terms, holidays, closures; compiled from news reports of Ministry of
  Education circulars, each row with its source and verification status).
  If that file is missing it falls back to the supplied 2025 calendar. Dates
  outside the calendar are reported as "unknown" rather than guessed.

`get_active_calendar()` picks the calendar that matches the active data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from services import status_service
from src.synthetic_data import _is_exam_period, _is_holiday, _school_term

SCHOOL_DAY, HALF_TERM, HOLIDAY, WEEKEND, UNKNOWN = "school_day", "half_term", "holiday", "weekend", "unknown"


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date
    verified: bool = True
    source_url: str = ""

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


NATIONAL_CALENDAR_CSV = (
    Path(__file__).resolve().parents[1] / "data" / "external" / "calendar" / "kenya_basic_education_calendar_2023_2026.csv"
)
# CSV event types that close schools, and how each is classified.
_CLOSED_EVENT_TYPES = {"half_term_break": "half_terms", "term_break": "holidays",
                       "school_closure": "holidays", "public_holiday": "holidays"}


@lru_cache(maxsize=4)
def load_national_calendar(path: Path = NATIONAL_CALENDAR_CSV) -> dict:
    """Calendar definition (terms, half_terms, holidays) read from the national calendar CSV."""
    definition: dict[str, list[Period]] = {"terms": [], "half_terms": [], "holidays": []}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kind = row["event_type"].strip()
            key = "terms" if kind == "term" else _CLOSED_EVENT_TYPES.get(kind)
            if key is None:
                continue
            definition[key].append(Period(
                name=row["title"].strip(),
                start=date.fromisoformat(row["start_date"]),
                end=date.fromisoformat(row["end_date"]),
                verified=row.get("verification_status", "").strip() == "verified",
                source_url=row.get("source_url", "").strip(),
            ))
    if not definition["terms"]:
        raise ValueError(f"{path} lists no school terms.")
    return definition


def default_kenya_definition() -> dict:
    """The national 2023–2026 calendar when its file is present, else the supplied 2025 calendar."""
    if NATIONAL_CALENDAR_CSV.exists():
        return load_national_calendar()
    return KENYA_PRIMARY_2025


class KenyaPrimaryCalendar:
    """Calendar built from explicit term, half-term and holiday periods."""

    name = "Kenyan national school calendar"

    def __init__(self, definition: dict | None = None):
        definition = definition or default_kenya_definition()
        self.terms: list[Period] = list(definition["terms"])
        self.half_terms: list[Period] = list(definition.get("half_terms", []))
        self.holidays: list[Period] = list(definition.get("holidays", []))
        self.first_day = min(p.start for p in self.terms)
        # A closing holiday (e.g. December) can run past the last term, so it counts as known.
        self.last_day = max(p.end for p in [*self.terms, *self.half_terms, *self.holidays])

    def closure(self, d: date) -> Period | None:
        """The half-term, holiday or closure that likely shuts schools on weekday `d`, if any."""
        if d.weekday() >= 5 or not (self.first_day <= d <= self.last_day):
            return None
        match = next((p for p in [*self.half_terms, *self.holidays] if p.contains(d)), None)
        if match is None and not any(p.contains(d) for p in self.terms):
            match = Period("Between terms", d, d, verified=False)
        return match

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

    def closure(self, d: date) -> Period | None:
        if d.weekday() < 5 and _is_holiday(d):
            return Period("School holiday", d, d, verified=False)
        return None

    def is_exam_period(self, d: date) -> bool:
        return (not _is_holiday(d)) and _is_exam_period(d, _school_term(d))


def get_active_calendar():
    """The calendar matching the active model's data (synthetic unless real data is active)."""
    if status_service.get_data_status()["kind"] == "real":
        return KenyaPrimaryCalendar()
    return SyntheticDevelopmentCalendar()
