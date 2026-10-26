"""Donor insights page: per-school reach, need ranking and funding gap.

What this file checks:
- Reach, unserved students and the monthly funding gap follow the documented arithmetic.
- Schools are ranked by unserved students per day; the top three are recommended.
- Trend compares the latest window with the one before it.
- The page renders the recommendations, table, chart and method note, and falls back
  to a friendly message when there is no processed data.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from django.urls import reverse

from config import settings
from services import donor_service
from tests.support.web_fixtures import operator

pytestmark = pytest.mark.django_db


@pytest.fixture
def logged_in(client):
    client.force_login(operator(codes=("A", "B", "C", "D")))
    return client


def _school_rows(school_id: str, name: str, enrolled: int, meals: list[float], start: str = "2024-01-01"):
    dates = pd.bdate_range(start, periods=len(meals))
    return [
        {settings.DATE_COLUMN: d.date().isoformat(), settings.SCHOOL_ID_COLUMN: school_id, "school_name": name,
         "county": "Nairobi", "sub_county": "Kibra", "enrolled_students": enrolled, "meals_served": m}
        for d, m in zip(dates, meals)
    ]


@pytest.fixture
def donor_data(stub_artifacts):
    w = donor_service.WINDOW
    rows = (
        _school_rows("A", "Alpha", 1000, [400] * w + [500] * w)  # 50% reach, 500 unserved, up 25%
        + _school_rows("B", "Bravo", 200, [190] * (2 * w))  # 95% reach, 10 unserved, steady
        + _school_rows("C", "Charlie", 3000, [1200] * w + [1000] * w)  # 33% reach, 2000 unserved, down
        + _school_rows("D", "Delta", 600, [300] * (2 * w))  # 50% reach, 300 unserved
    )
    stub_artifacts.write_processed(pd.DataFrame(rows))
    return stub_artifacts


def test_metrics_follow_documented_arithmetic(donor_data):
    impact = donor_service.school_impact(cost_per_meal=Decimal("27"))
    alpha = next(s for s in impact["schools"] if s["school_id"] == "A")
    assert alpha["reach_pct"] == 50 and alpha["reach_band"] == "Moderate"
    assert alpha["unserved_per_day"] == 500
    assert alpha["monthly_gap_kes"] == Decimal(500 * donor_service.SCHOOL_DAYS_PER_MONTH * 27)
    assert alpha["trend"] == "up" and alpha["trend_pct"] == 25.0
    bravo = next(s for s in impact["schools"] if s["school_id"] == "B")
    assert bravo["reach_band"] == "Strong" and bravo["trend"] == "steady"
    assert impact["totals"]["unserved_per_day"] == 500 + 10 + 2000 + 300


def test_schools_ranked_by_unserved_students_and_top_three_recommended(donor_data):
    impact = donor_service.school_impact()
    assert [s["school_id"] for s in impact["schools"]] == ["C", "A", "D", "B"]
    assert [s["school_id"] for s in impact["recommended"]] == ["C", "A", "D"]
    charlie = impact["schools"][0]
    assert charlie["trend"] == "down" and charlie["reach_band"] == "Low"


def test_donor_page_shows_recommendations_table_and_method(logged_in, donor_data):
    response = logged_in.get(reverse("core:donors"))
    assert response.status_code == 200
    html = response.content.decode()
    assert html.count('data-testid="recommended-school"') == 3
    assert 'data-testid="donor-table"' in html and 'data-testid="donor-method-note"' in html
    assert 'data-chart="donor-reach-chart"' in html
    assert "Charlie" in html and "includes students who were absent" in html


def test_donor_page_without_data_shows_message(logged_in, stub_artifacts):
    stub_artifacts.processed_path.unlink()
    response = logged_in.get(reverse("core:donors"))
    assert response.status_code == 200
    assert 'data-testid="data-error"' in response.content.decode()


def test_sidebar_links_to_donor_page(logged_in, donor_data):
    html = logged_in.get(reverse("core:dashboard")).content.decode()
    assert reverse("core:donors") in html and "Donor insights" in html
