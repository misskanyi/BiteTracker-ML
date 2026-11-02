"""Stage 4 pages: historical demand and forecast & planning.

What this file checks:
- Forecasts store the cost per meal they were priced with (default KES 27) without changing predictions.
- The result page shows day-1 and whole-horizon budgets, and ?cost= re-prices without creating a run.
- Invalid costs are rejected on the form and on the result page.
- Unrecorded weather shows "Not recorded" (history table, forecast form, result assumptions), never 0.
- History statistics show processed values exactly (no truncation) and offer quick ranges and a reset.
"""

from __future__ import annotations

import re
from decimal import Decimal

import pytest
from django.urls import reverse

from config import settings
from web.schools.models import ForecastRun
from tests.support.web_fixtures import operator



@pytest.fixture
def logged_in(client, db):
    client.force_login(operator())
    return client


def _forecast(client, **data) -> ForecastRun:
    response = client.post(reverse("core:forecast"), {"school": "SCH001", "horizon": 1, **data})
    assert response.status_code == 302, response.content.decode()[:500]
    return ForecastRun.objects.latest("created_at")


def _result(client, run, **params) -> str:
    response = client.get(reverse("core:forecast_run", args=[run.pk]), params)
    assert response.status_code == 200
    return response.content.decode()


# --- Meal budget ------------------------------------------------------------------------------


def test_forecast_stores_default_cost_of_27(logged_in, stub_artifacts):
    run = _forecast(logged_in)
    assert Decimal(run.result_summary["planning"]["cost_per_meal_kes"]) == Decimal("27")
    assert "cost_per_meal_kes" not in run.assumptions  # model inputs stay separate from planning assumptions


def test_cost_never_changes_predictions(logged_in, stub_artifacts):
    cheap = _forecast(logged_in, horizon=7, cost_per_meal_kes="27")
    dear = _forecast(logged_in, horizon=7, cost_per_meal_kes="80")
    assert [d["predicted_meals_served"] for d in cheap.result_summary["days"]] == [d["predicted_meals_served"] for d in dear.result_summary["days"]]
    assert cheap.total_recommended_meals == dear.total_recommended_meals


def test_result_page_shows_day1_and_horizon_budget(logged_in, stub_artifacts):
    run = _forecast(logged_in, horizon=7)
    html = _result(logged_in, run)
    assert 'data-testid="meal-budget"' in html and "KES 27 per meal" in html
    day1 = run.day1_recommended_meals * 27
    horizon = run.total_recommended_meals * 27
    assert f'data-testid="day1-budget">KES {day1:,}<' in html
    assert f'data-testid="horizon-budget">KES {horizon:,}<' in html
    assert "not verified Food4Education expenditure" in html


def test_custom_cost_is_used(logged_in, stub_artifacts):
    run = _forecast(logged_in, cost_per_meal_kes="30")
    html = _result(logged_in, run)
    assert f'data-testid="day1-budget">KES {run.day1_recommended_meals * 30:,}<' in html


def test_cost_override_reprices_without_new_run(logged_in, stub_artifacts):
    run = _forecast(logged_in)
    html = _result(logged_in, run, cost="40")
    assert f'data-testid="day1-budget">KES {run.day1_recommended_meals * 40:,}<' in html
    assert "recalculated" in html
    assert ForecastRun.objects.count() == 1
    assert Decimal(ForecastRun.objects.get().result_summary["planning"]["cost_per_meal_kes"]) == Decimal("27")


def test_invalid_override_falls_back_with_message(logged_in, stub_artifacts):
    run = _forecast(logged_in)
    html = _result(logged_in, run, cost="abc")
    assert 'data-testid="cost-error"' in html
    assert f'data-testid="day1-budget">KES {run.day1_recommended_meals * 27:,}<' in html


@pytest.mark.parametrize("cost", ["0", "-3", "20000"])
def test_invalid_cost_on_form_records_nothing(logged_in, stub_artifacts, cost):
    response = logged_in.post(reverse("core:forecast"), {"school": "SCH001", "horizon": 1, "cost_per_meal_kes": cost})
    assert response.status_code == 200
    assert ForecastRun.objects.count() == 0


def test_runs_saved_before_budgeting_use_default(logged_in, stub_artifacts):
    run = _forecast(logged_in)
    run.result_summary.pop("planning")
    run.save()
    assert "default assumption" in _result(logged_in, run)


# --- Weather "not recorded" ------------------------------------------------------------------------


def test_history_shows_not_recorded_instead_of_zero(logged_in, stub_artifacts, monkeypatch):
    monkeypatch.setattr(settings, "USE_GOOGLE_SHEET", True)
    html = logged_in.get(reverse("core:history")).content.decode()
    rows = html.count('<td class="text-nowrap">')  # one date cell per row
    assert html.count('data-testid="not-recorded"') == 2 * rows
    assert "Rainfall (mm) and Temp (°C) are not measured" in html


def test_history_shows_weather_values_when_recorded(logged_in, stub_artifacts):
    assert 'data-testid="not-recorded"' not in logged_in.get(reverse("core:history")).content.decode()


def test_forecast_form_disables_unrecorded_weather(logged_in, stub_artifacts, monkeypatch):
    monkeypatch.setattr(settings, "USE_GOOGLE_SHEET", True)
    html = logged_in.get(reverse("core:forecast")).content.decode()
    assert 'data-testid="weather-not-recorded"' in html
    assert re.search(r'<input[^>]*name="expected_rainfall_mm"[^>]*disabled', html)
    assert re.search(r'<input[^>]*name="expected_temperature_c"[^>]*disabled', html)


def test_result_replaces_weather_assumptions_for_real_data_runs(logged_in, stub_artifacts, monkeypatch):
    run = _forecast(logged_in)
    run.results_label = "FOOD4EDUCATION RESEARCH RESULTS"
    run.save()
    monkeypatch.setattr(settings, "USE_GOOGLE_SHEET", True)
    html = _result(logged_in, run)
    assert 'data-testid="weather-not-recorded"' in html
    assert "Rainfall basis" not in html and "Temperature basis" not in html


def test_synthetic_runs_keep_their_weather_assumptions(logged_in, stub_artifacts, monkeypatch):
    run = _forecast(logged_in)
    monkeypatch.setattr(settings, "USE_GOOGLE_SHEET", True)
    assert "Rainfall basis" in _result(logged_in, run)


# --- History precision and controls -------------------------------------------------------------------


def test_history_stats_are_not_truncated(logged_in, stub_artifacts):
    frame = stub_artifacts.frame.copy()
    school = frame["school_id"] == "SCH001"
    frame.loc[frame[school].index.max(), "meals_served"] = 1.375  # far below every other day: the minimum
    stub_artifacts.write_processed(frame)
    response = logged_in.get(reverse("core:history"), {"school": "SCH001"})
    assert response.context["stats"]["min"] == 1.375
    html = response.content.decode()
    assert "1.375 –" in html
    assert '1.375<sup class="bt-capped"' in html


def test_history_offers_quick_ranges_and_reset(logged_in, stub_artifacts):
    html = logged_in.get(reverse("core:history"), {"school": "SCH002"}).content.decode()
    for label in ("Last 30 days", "Last 90 days", "Last 6 months", "All records"):
        assert label in html
    assert f'href="{reverse("core:history")}?school=SCH002">Reset' in html


def test_google_sheet_marks_whole_number_capped_values_with_original(logged_in, stub_artifacts, monkeypatch):
    """The sheet keeps the original count in students_present, so whole-number capped values are marked too."""
    frame = stub_artifacts.frame.copy()
    last = frame[frame["school_id"] == "SCH001"].index.max()
    frame.loc[last, ["meals_served", "students_present"]] = [1546.0, 572]
    stub_artifacts.write_processed(frame)
    monkeypatch.setattr(settings, "USE_GOOGLE_SHEET", True)
    html = logged_in.get(reverse("core:history"), {"school": "SCH001"}).content.decode()
    assert re.search(r'1,546<sup class="bt-capped" data-testid="processed-mark" title="[^"]*original recorded count: 572"', html)
    dashboard = logged_in.get(reverse("core:dashboard")).content.decode()
    assert "original recorded count: 572" in dashboard


def test_unchanged_values_are_not_marked(logged_in, stub_artifacts, monkeypatch):
    frame = stub_artifacts.frame.copy()
    frame["students_present"] = frame["meals_served"]
    stub_artifacts.write_processed(frame)
    monkeypatch.setattr(settings, "USE_GOOGLE_SHEET", True)
    html = logged_in.get(reverse("core:history"), {"school": "SCH001"}).content.decode()
    assert 'data-testid="processed-mark"' not in html
