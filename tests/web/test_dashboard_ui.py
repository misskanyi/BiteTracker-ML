"""Dashboard redesign: honest display of processed values, locations, baselines and run provenance.

What this file checks:
- Meal counts are shown as recorded: capped (fractional) values are not truncated and are marked.
- Placeholder locations ("Not provided") become a single "Location unavailable" label.
- The model card names a naive baseline when it beat the selected model on held-out test MAE.
- Recent forecast runs label synthetic-data runs separately from Food4Education runs.
- Quick forecast only opens the forecast form (GET); it never creates a forecast.
"""

from __future__ import annotations

from datetime import date

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

from tests.support.service_fixtures import stub_metadata
from web.core.templatetags.bitetracker import is_fractional, location, meals
from web.schools.models import ForecastRun
from tests.support.web_fixtures import operator



@pytest.fixture
def logged_in(client, db):
    client.force_login(operator())
    return client


def _dashboard(client) -> str:
    response = client.get(reverse("core:dashboard"))
    assert response.status_code == 200
    return response.content.decode()


def _run(label: str, school: str = "School 1") -> ForecastRun:
    return ForecastRun.objects.create(
        school_code="SCH001", school_name=school, horizon_days=1,
        origin_date=date(2025, 10, 24), first_forecast_date=date(2025, 10, 27), last_forecast_date=date(2025, 10, 27),
        model_label="Random Forest", results_label=label,
        day1_predicted_meals=1696, day1_recommended_meals=1913, day1_risk_level="High",
        total_predicted_meals=1696, total_recommended_meals=1913,
    )


# --- Template filters ----------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    (57.5, "57.5"), (1546.0, "1,546"), (2723, "2,723"), (1234.25, "1,234.25"), (1234.75, "1,234.75"), (1913.375, "1,913.375"), (None, "—"), (float("nan"), "—"),
])
def test_meals_filter_never_truncates(value, expected):
    assert meals(value) == expected


def test_is_fractional_only_flags_non_integer_counts():
    assert is_fractional(57.5) and not is_fractional(57.0) and not is_fractional(None)


@pytest.mark.parametrize("record, expected", [
    ({"county": "Not provided", "sub_county": "Not provided"}, ""),
    ({"county": "Nairobi", "sub_county": "Not provided"}, "Nairobi"),
    ({"county": "Nairobi", "sub_county": "Kibra"}, "Kibra, Nairobi"),
    ({}, ""),
])
def test_location_filter_skips_placeholders(record, expected):
    assert location(record) == expected


# --- Dashboard ----------------------------------------------------------------------------------


def test_capped_values_are_shown_unrounded_and_marked(logged_in, stub_artifacts):
    frame = stub_artifacts.frame.copy()
    last = frame[frame["school_id"] == "SCH001"].index.max()
    frame.loc[last, "meals_served"] = 57.5
    stub_artifacts.write_processed(frame)
    html = _dashboard(logged_in)
    assert '57.5<sup class="bt-capped"' in html
    assert 'data-testid="processed-values-note"' in html


def test_placeholder_locations_show_location_unavailable(logged_in, stub_artifacts):
    frame = stub_artifacts.frame.copy()
    frame["county"] = "Not provided"
    frame["sub_county"] = "Not provided"
    stub_artifacts.write_processed(frame)
    html = _dashboard(logged_in)
    assert "Location unavailable" in html
    assert "Not provided" not in html


def test_no_baseline_note_when_model_beats_every_baseline(logged_in, stub_artifacts):
    assert 'data-testid="baseline-note"' not in _dashboard(logged_in)


@pytest.fixture
def admin_logged_in(client, db):
    """Model evaluation (the dashboard model card) is for System Administrators only."""
    client.force_login(get_user_model().objects.create_superuser("evaluator_admin", password="Str0ng-test-passw0rd!"))
    return client


def test_baseline_note_when_a_baseline_beats_the_model_on_test_mae(admin_logged_in, stub_artifacts):
    metadata = stub_metadata()
    metadata["naive_baseline_comparison"][0]["Baseline Test MAE"] = 8.5  # Persistence beats model's 10.0
    stub_artifacts.write_metadata(metadata)
    html = _dashboard(admin_logged_in)
    assert 'data-testid="baseline-note"' in html
    assert "<strong>Persistence / Last Observation</strong> had a lower MAE (8.5 vs 10.0 meals)" in html


def test_model_card_separates_validation_and_test_metrics(admin_logged_in, stub_artifacts):
    html = _dashboard(admin_logged_in)
    assert "Validation · training period" in html and "Held-out test period" in html
    assert "accuracy" not in html.lower()


def test_recent_runs_label_synthetic_and_real_data(logged_in, stub_artifacts):
    _run("SYNTHETIC DEVELOPMENT RESULTS", school="Greenfield Primary")
    _run("FOOD4EDUCATION RESEARCH RESULTS")
    html = _dashboard(logged_in)
    assert html.count('data-testid="run-data-synthetic"') == 1
    assert html.count('data-testid="run-data-real"') == 1


def test_quick_forecast_only_opens_the_form(logged_in, stub_artifacts):
    html = _dashboard(logged_in)
    assert f'<form method="get" action="{reverse("core:forecast")}"' in html
    assert ForecastRun.objects.count() == 0
