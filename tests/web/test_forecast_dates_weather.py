"""Forecast page: single-day / date-range selection and the morning weather advisory.

What this file checks (provider responses are controlled; no network):
- The existing horizon flow still works and shows a weather status without breaking.
- Single day: the next school day uses the saved model; a later upcoming date uses the planning
  method; the page shows latest record date, method and daily estimate.
- Heavy rain 6-9 a.m. on an upcoming selected day shows an advisory with source, forecast
  date and last update; a normal day does not.
- In a date range only the advisory days are flagged.
- A date beyond the weather window, an API failure, and a school with no location each show a
  short status while the meal forecast loads normally.
- The API key never reaches the page; weather never changes the stored meal estimates.
- Users can only pick authorised schools; administrators can set a school's location.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone

from services import calendar_service, weather_service
from tests.support.web_fixtures import PASSWORD, operator
from web.schools.models import ForecastRun, School

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def _fresh_weather_cache():
    weather_service.clear_weather_cache()
    yield
    weather_service.clear_weather_cache()


class _TermTimeCalendar:
    """No holidays: the synthetic development calendar treats late Nov–Dec as holiday, which would
    reject every upcoming date these tests pick. Holiday handling is covered in the service tests."""

    name = "Test calendar"

    def is_holiday(self, d):
        return False


@pytest.fixture(autouse=True)
def _term_time(monkeypatch):
    monkeypatch.setattr(calendar_service, "get_active_calendar", lambda: _TermTimeCalendar())


@pytest.fixture
def logged_in(client):
    client.force_login(operator())
    return client


@pytest.fixture
def located(db):
    """SCH001 has a configured location (the Nairobi point used in the project's weather study)."""
    School.objects.update_or_create(school_id="SCH001", defaults={"latitude": Decimal("-1.28640"),
                                                                  "longitude": Decimal("36.81720")})


class Provider:
    """Stands in for the HTTP client; records calls."""

    def __init__(self, rain: dict[date, float] | None = None, error: Exception | None = None):
        self.rain, self.error, self.calls = rain or {}, error, []

    def __call__(self, url, timeout):
        self.calls.append(url)
        if self.error:
            raise self.error
        times, mm = [], []
        start = date.fromisoformat(url.split("start_date=")[1][:10])
        end = date.fromisoformat(url.split("end_date=")[1][:10])
        d = start
        while d <= end:
            for h in range(24):
                times.append(f"{d.isoformat()}T{h:02d}:00")
                mm.append(self.rain.get(d, 0.0) if h == 8 else 0.0)
            d += timedelta(days=1)
        return {"hourly": {"time": times, "precipitation": mm, "precipitation_probability": [70] * len(times)}}


@pytest.fixture
def provider(monkeypatch):
    def install(**kwargs):
        p = Provider(**kwargs)
        monkeypatch.setattr(weather_service, "_http_get_json", p)
        return p
    return install


def upcoming_weekdays(n: int, skip: int = 1) -> list[date]:
    """The next `n` weekdays starting `skip` days from today (inside the 16-day weather window)."""
    days, d = [], timezone.localdate() + timedelta(days=skip)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def latest_record(stub_artifacts, school="SCH001") -> date:
    frame = stub_artifacts.frame
    return pd.to_datetime(frame.loc[frame["school_id"] == school, "date"]).max().date()


def post(client, **data):
    return client.post(reverse("core:forecast"), {"school": "SCH001", **data})


# --- Existing horizon flow ------------------------------------------------------------------


def test_horizon_flow_unchanged_and_shows_weather_status(logged_in, stub_artifacts, located, provider):
    p = provider()
    response = post(logged_in, horizon=7)
    run = ForecastRun.objects.get()
    assert response.status_code == 302 and run.selection_mode == "horizon" and run.horizon_days == 7
    html = logged_in.get(response.url).content.decode()
    assert "Next 7 school days".lower() in html and 'data-testid="forecast-table"' in html
    # Stub data ends in 2023, so every forecast day is outside the weather window: no request made.
    assert 'data-weather-status="out_of_window"' in html and p.calls == []


def test_form_offers_three_modes_and_keeps_horizons(logged_in, stub_artifacts):
    html = logged_in.get(reverse("core:forecast")).content.decode()
    for value in ("horizon", "single", "range"):
        assert f'value="{value}"' in html
    for label in ("Next school day", "Next 7 school days", "Next 30 school days", "Single day", "Date range"):
        assert label in html


def test_only_authorised_schools_can_be_selected(client, stub_artifacts):
    client.force_login(operator("narrow", codes=("SCH002",)))
    response = post(client, mode="single", target_date=upcoming_weekdays(1)[0].isoformat())
    assert response.status_code == 200 and ForecastRun.objects.count() == 0
    assert "Select a valid choice" in response.content.decode()


# --- Date selection -------------------------------------------------------------------------


def test_next_school_day_uses_saved_model(logged_in, stub_artifacts, provider):
    provider()
    latest = latest_record(stub_artifacts)
    day = latest + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    response = post(logged_in, mode="single", target_date=day.isoformat())
    run = ForecastRun.objects.get()
    assert response.status_code == 302 and run.selection_mode == "single"
    (row,) = run.result_summary["days"]
    assert row["method"] == "model" and row["school_days_ahead"] == 1
    html = logged_in.get(response.url).content.decode()
    assert "Linear Regression (saved model)" in html
    assert f"Latest record: {latest.day} {latest:%b %Y}" in html


def test_upcoming_date_uses_planning_method_with_weather_advisory(logged_in, stub_artifacts, located, provider):
    wet = upcoming_weekdays(1)[0]
    p = provider(rain={wet: 9.5})
    response = post(logged_in, mode="single", target_date=wet.isoformat())
    run = ForecastRun.objects.get()
    (row,) = run.result_summary["days"]
    assert row["method"] in ("planning", "weekday_baseline") and row["experimental"]  # years past the stub data
    html = logged_in.get(response.url).content.decode()
    assert 'data-testid="selection-method"' in html and "Experimental" in html
    assert 'data-testid="weather-advisory"' in html and "Heavy-rain advisory" in html
    assert "Open-Meteo" in html and f"{wet.day} {wet:%b %Y}" in html and "Last updated" in html
    assert "9.5 mm in one hour" in html
    assert len(p.calls) == 1
    # Advisory only: the stored meal estimate is untouched by weather.
    assert run.result_summary["days"][0]["predicted_meals"] == row["predicted_meals"]
    assert "rainfall" not in str(run.assumptions)


def test_normal_day_shows_no_advisory(logged_in, stub_artifacts, located, provider):
    day = upcoming_weekdays(1)[0]
    provider(rain={day: 0.5})
    response = post(logged_in, mode="single", target_date=day.isoformat())
    html = logged_in.get(response.url).content.decode()
    assert 'data-weather-status="clear"' in html and 'data-testid="weather-advisory"' not in html
    assert "No heavy rain forecast" in html and 'data-testid="rain-flag"' not in html
    assert "Last updated" in html and "Open-Meteo" in html


def test_range_flags_only_advisory_days(logged_in, stub_artifacts, located, provider):
    days = upcoming_weekdays(4)
    provider(rain={days[1]: 12.0, days[3]: 5.0, days[0]: 1.0})
    response = post(logged_in, mode="range", range_start=days[0].isoformat(), range_end=days[3].isoformat())
    run = ForecastRun.objects.get()
    assert run.selection_mode == "range" and run.horizon_days == 4
    html = logged_in.get(response.url).content.decode()
    assert html.count('data-testid="rain-flag"') == 2
    assert html.count('data-testid="weather-advisory-day"') == 2
    assert f"{days[1]:%a} {days[1].day} {days[1]:%b %Y}" in html


def test_date_beyond_weather_window_still_shows_estimate(logged_in, stub_artifacts, located, provider):
    p = provider()
    far = timezone.localdate() + timedelta(days=60)
    while far.weekday() >= 5:
        far += timedelta(days=1)
    response = post(logged_in, mode="single", target_date=far.isoformat())
    html = logged_in.get(response.url).content.decode()
    assert 'data-weather-status="out_of_window"' in html and "outside the weather forecast window" in html
    assert 'data-testid="predicted-meals"' in html and p.calls == []


def test_api_failure_shows_status_and_forecast_loads(logged_in, stub_artifacts, located, provider):
    p = provider(error=TimeoutError("timed out"))
    day = upcoming_weekdays(1)[0]
    response = post(logged_in, mode="single", target_date=day.isoformat())
    page = logged_in.get(response.url)
    html = page.content.decode()
    assert page.status_code == 200
    assert 'data-weather-status="unavailable"' in html and "meal forecast is not affected" in html
    assert 'data-testid="predicted-meals"' in html and 'data-testid="forecast-table"' in html
    logged_in.get(response.url)
    assert len(p.calls) == 1  # failure cached: the second view does not wait on the provider again


def test_no_location_shows_status(logged_in, stub_artifacts, provider):
    p = provider()
    response = post(logged_in, mode="single", target_date=upcoming_weekdays(1)[0].isoformat())
    html = logged_in.get(response.url).content.decode()
    assert 'data-weather-status="no_location"' in html and "no location configured" in html
    assert p.calls == []


def test_api_key_stays_server_side(logged_in, stub_artifacts, located, provider, settings):
    settings.WEATHER_API_KEY = "TOP-SECRET-WEATHER-KEY"
    p = provider(rain={})
    response = post(logged_in, mode="single", target_date=upcoming_weekdays(1)[0].isoformat())
    html = logged_in.get(response.url).content.decode()
    assert "apikey=TOP-SECRET-WEATHER-KEY" in p.calls[0]
    assert "TOP-SECRET-WEATHER-KEY" not in html


def test_threshold_setting_controls_advisory(logged_in, stub_artifacts, located, provider, settings):
    settings.WEATHER_HEAVY_RAIN_MM_PER_HOUR = 20.0
    day = upcoming_weekdays(1)[0]
    provider(rain={day: 9.5})
    response = post(logged_in, mode="single", target_date=day.isoformat())
    html = logged_in.get(response.url).content.decode()
    assert 'data-weather-status="clear"' in html and "Heavy rain = 20 mm" in html


@pytest.mark.parametrize("data, error", [
    ({"mode": "single"}, "Choose a school day."),
    ({"mode": "range", "range_start": "2026-12-10", "range_end": "2026-12-01"}, "on or after the first date"),
])
def test_form_validation(logged_in, stub_artifacts, data, error):
    response = post(logged_in, **data)
    assert response.status_code == 200 and error in response.content.decode()
    assert ForecastRun.objects.count() == 0


def test_date_on_or_before_latest_record_rejected(logged_in, stub_artifacts):
    latest = latest_record(stub_artifacts)
    response = post(logged_in, mode="single", target_date=latest.isoformat())
    assert response.status_code == 200 and "latest record" in response.content.decode()
    assert ForecastRun.objects.count() == 0


def test_history_lists_selected_dates(logged_in, stub_artifacts, provider):
    provider()
    day = upcoming_weekdays(1, skip=30)[0]
    post(logged_in, mode="single", target_date=day.isoformat())
    html = logged_in.get(reverse("core:forecast_history")).content.decode()
    assert f"{day:%a} {day.day} {day:%b %Y}" in html


# --- School location --------------------------------------------------------------------------


@pytest.fixture
def sysadmin_client(client, db):
    client.force_login(get_user_model().objects.create_superuser("root", "root@example.com", PASSWORD))
    return client


def test_admin_sets_school_location(sysadmin_client):
    school = School.objects.create(school_id="SCH001", name="One")
    url = reverse("administration:school_edit", args=[school.pk])
    response = sysadmin_client.post(url, {"name": "One", "latitude": "-1.28640", "longitude": "36.81720", "is_active": "on"})
    assert response.status_code == 302
    school.refresh_from_db()
    assert school.has_location and school.latitude == Decimal("-1.28640")
    detail = sysadmin_client.get(reverse("administration:school_detail", args=[school.pk])).content.decode()
    assert "-1.28640, 36.81720" in detail


@pytest.mark.parametrize("lat, lon", [("-1.2", ""), ("95", "36.8"), ("-1.2", "200")])
def test_admin_location_validation(sysadmin_client, lat, lon):
    school = School.objects.create(school_id="SCH001", name="One")
    response = sysadmin_client.post(reverse("administration:school_edit", args=[school.pk]),
                                    {"name": "One", "latitude": lat, "longitude": lon, "is_active": "on"})
    assert response.status_code == 200
    school.refresh_from_db()
    assert not school.has_location
