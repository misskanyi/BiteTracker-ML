"""School calendar month view shows the Kenyan national calendar (closures greyed, terms marked, sources listed)."""

from __future__ import annotations

import pytest
from django.urls import reverse

from tests.support.web_fixtures import operator


@pytest.fixture
def page(client, db):
    client.force_login(operator())

    def get(month: str) -> str:
        return client.get(reverse("schoolcalendar:calendar"), {"month": month}).content.decode()
    return get


def test_december_holiday_weekdays_are_greyed_and_labelled(page):
    html = page("2026-12")
    # 23 weekdays in December 2026 plus the grid's Mon 30 Nov and Fri 1 Jan, all in the December holiday.
    assert html.count('data-closed="December holiday"') == 23 + 2
    assert 'data-testid="closure-label"' in html and "December holiday" in html
    assert 'data-testid="national-calendar"' in html and "Likely closed" in html
    assert "the-star.co.ke" in html  # source link


def test_term_open_and_close_marked_and_half_term_listed(page):
    html = page("2026-02")
    assert "Term 1 half-term break" in html and 'data-closed="Term 1 half-term break"' in html
    html = page("2026-10")
    assert "Term 3 closes" in html and 'data-testid="term-mark"' in html


def test_single_source_period_is_flagged(page):
    assert "single source or inferred" in page("2025-10")  # Mashujaa Day: one article only


def test_months_outside_the_calendar_are_not_shaded(page):
    html = page("2027-03")
    assert "data-closed=" not in html and "covers January 2023 to January 2027 only" in html
