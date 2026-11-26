"""Morning heavy-rain advisory (services.weather_service), with a controlled provider response.

What this file checks:
- A heavy-rain morning raises an advisory; a normal morning does not.
- Only the 06:00-09:00 window counts (Open-Meteo stamps 07, 08 and 09 hold the preceding hour).
- The threshold is configurable.
- Dates beyond the provider's forecast window are not requested and give a status message.
- Provider failures (exception, HTTP error, bad JSON, missing fields) give "unavailable", never raise.
- Responses and failures are cached; the API key never appears in logs or messages.
- A school without a location gets a status message and no request.
"""

from __future__ import annotations

import io
import logging
import urllib.error
from datetime import date, datetime, timedelta

import pytest

from services import weather_service as ws

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=ws.NAIROBI)
TODAY = NOW.date()
CONFIG = ws.WeatherConfig(api_key="SECRET-KEY-123", cache_seconds=600, failure_cache_seconds=60)
LAT, LON = -1.2864, 36.8172


@pytest.fixture(autouse=True)
def _fresh_cache():
    ws.clear_weather_cache()
    yield
    ws.clear_weather_cache()


def hourly_payload(rain_by_day: dict[date, dict[int, float]], probability: float = 60.0) -> dict:
    """An Open-Meteo-shaped response: 24 hours per day, rain given per local hour stamp."""
    times, rain, prob = [], [], []
    for d, hours in sorted(rain_by_day.items()):
        for h in range(24):
            times.append(f"{d.isoformat()}T{h:02d}:00")
            rain.append(hours.get(h, 0.0))
            prob.append(probability)
    return {"latitude": LAT, "longitude": LON, "timezone": "Africa/Nairobi",
            "hourly": {"time": times, "precipitation": rain, "precipitation_probability": prob}}


class FakeProvider:
    def __init__(self, payload=None, error: Exception | None = None):
        self.payload, self.error, self.urls = payload, error, []

    def __call__(self, url, timeout):
        self.urls.append(url)
        if self.error:
            raise self.error
        return self.payload


def advise(dates, fetch, config=CONFIG, clock=lambda: 1000.0, lat=LAT, lon=LON):
    return ws.morning_advisory(lat, lon, dates, config, fetch=fetch, now=NOW, clock=clock)


def test_heavy_rain_morning_gives_advisory():
    day = TODAY + timedelta(days=2)
    provider = FakeProvider(hourly_payload({day: {7: 1.0, 8: 6.2, 9: 2.0}}))
    advisory = advise([day], provider)
    assert advisory.status == ws.ADVISORY
    weather = advisory.for_day(day)
    assert weather.heavy_rain and weather.max_hour_mm == 6.2 and weather.total_mm == 9.2
    assert weather.max_probability_pct == 60.0
    assert [d.date for d in advisory.advisory_days] == [day]
    assert advisory.source == "Open-Meteo" and advisory.retrieved_at is not None


def test_normal_morning_gives_no_advisory():
    day = TODAY + timedelta(days=1)
    advisory = advise([day], FakeProvider(hourly_payload({day: {7: 0.4, 8: 1.1, 9: 0.0}})))
    assert advisory.status == ws.CLEAR
    assert not advisory.for_day(day).heavy_rain
    assert advisory.advisory_days == []


def test_only_six_to_nine_counts():
    """Rain stamped 06:00 fell 05-06 and 10:00 fell 09-10: both outside the window."""
    day = TODAY + timedelta(days=1)
    advisory = advise([day], FakeProvider(hourly_payload({day: {6: 25.0, 10: 25.0, 8: 1.0}})))
    assert advisory.status == ws.CLEAR
    assert advisory.for_day(day).hourly_mm == [0.0, 1.0, 0.0]


def test_threshold_is_configurable():
    day = TODAY + timedelta(days=1)
    payload = hourly_payload({day: {8: 6.0}})
    assert advise([day], FakeProvider(payload)).status == ws.ADVISORY
    ws.clear_weather_cache()
    strict = ws.WeatherConfig(heavy_rain_mm_per_hour=10.0)
    assert advise([day], FakeProvider(payload), config=strict).status == ws.CLEAR


def test_date_beyond_forecast_window_is_not_requested():
    provider = FakeProvider(hourly_payload({}))
    far = TODAY + timedelta(days=CONFIG.forecast_days)  # first day after the window
    advisory = advise([far], provider)
    assert advisory.status == ws.OUT_OF_WINDOW
    assert provider.urls == []
    assert "outside the weather forecast window" in advisory.message
    assert advisory.out_of_window == [far]


def test_past_date_is_outside_window():
    advisory = advise([TODAY - timedelta(days=1)], FakeProvider(hourly_payload({})))
    assert advisory.status == ws.OUT_OF_WINDOW


def test_range_flags_only_advisory_days_and_lists_out_of_window():
    wet, dry = TODAY + timedelta(days=1), TODAY + timedelta(days=2)
    far = TODAY + timedelta(days=40)
    provider = FakeProvider(hourly_payload({wet: {7: 8.0}, dry: {7: 0.2}}))
    advisory = advise([wet, dry, far], provider)
    assert advisory.status == ws.ADVISORY
    assert [d.date for d in advisory.advisory_days] == [wet]
    assert advisory.for_day(dry).heavy_rain is False
    assert advisory.for_day(far) is None and advisory.out_of_window == [far]
    # One request covering only the in-window dates.
    assert len(provider.urls) == 1
    assert f"start_date={wet.isoformat()}" in provider.urls[0] and f"end_date={dry.isoformat()}" in provider.urls[0]


@pytest.mark.parametrize("error", [ws.WeatherProviderError("provider returned HTTP 503"), TimeoutError("timed out"),
                                   ValueError("boom")])
def test_api_failure_is_a_status_not_an_error(error, caplog):
    day = TODAY + timedelta(days=1)
    with caplog.at_level(logging.WARNING):
        advisory = advise([day], FakeProvider(error=error))
    assert advisory.status == ws.UNAVAILABLE
    assert "meal forecast is not affected" in advisory.message
    assert "SECRET-KEY-123" not in caplog.text and "SECRET-KEY-123" not in advisory.message


@pytest.mark.parametrize("payload", [{"error": True, "reason": "bad"}, {"hourly": {}}, ["not", "a", "dict"],
                                     {"hourly": {"time": ["x"], "precipitation": []}}])
def test_unusable_response_is_unavailable(payload):
    assert advise([TODAY + timedelta(days=1)], FakeProvider(payload)).status == ws.UNAVAILABLE


def test_http_error_from_real_client_hides_key(monkeypatch):
    def fail(request, timeout):
        assert timeout == CONFIG.timeout_seconds
        raise urllib.error.HTTPError(request.full_url, 500, "err", {}, io.BytesIO(b""))

    monkeypatch.setattr(ws.urllib.request, "urlopen", fail)
    url = ws.build_url(CONFIG, LAT, LON, TODAY, TODAY)
    with pytest.raises(ws.WeatherProviderError) as exc:
        ws._http_get_json(url, CONFIG.timeout_seconds)
    assert "SECRET-KEY-123" not in str(exc.value)


def test_responses_are_cached_until_expiry():
    day = TODAY + timedelta(days=1)
    provider = FakeProvider(hourly_payload({day: {7: 0.0}}))
    clock = [1000.0]
    advise([day], provider, clock=lambda: clock[0])
    advise([day], provider, clock=lambda: clock[0])
    assert len(provider.urls) == 1
    clock[0] += CONFIG.cache_seconds + 1
    advise([day], provider, clock=lambda: clock[0])
    assert len(provider.urls) == 2


def test_failures_are_cached_briefly():
    day = TODAY + timedelta(days=1)
    provider = FakeProvider(error=TimeoutError())
    clock = [1000.0]
    assert advise([day], provider, clock=lambda: clock[0]).status == ws.UNAVAILABLE
    assert advise([day], provider, clock=lambda: clock[0]).status == ws.UNAVAILABLE
    assert len(provider.urls) == 1  # outage costs one timeout, not one per page view
    clock[0] += CONFIG.failure_cache_seconds + 1
    provider.error, provider.payload = None, hourly_payload({day: {7: 0.0}})
    assert advise([day], provider, clock=lambda: clock[0]).status == ws.CLEAR


def test_no_location_makes_no_request():
    provider = FakeProvider(hourly_payload({}))
    advisory = advise([TODAY + timedelta(days=1)], provider, lat=None, lon=None)
    assert advisory.status == ws.NO_LOCATION and provider.urls == []


def test_disabled():
    advisory = advise([TODAY + timedelta(days=1)], FakeProvider(), config=ws.WeatherConfig(enabled=False))
    assert advisory.status == ws.DISABLED


def test_request_uses_nairobi_time_and_key_only_in_url():
    url = ws.build_url(CONFIG, LAT, LON, TODAY, TODAY)
    assert "timezone=Africa%2FNairobi" in url and "hourly=precipitation%2Cprecipitation_probability" in url
    assert "apikey=SECRET-KEY-123" in url
    assert "apikey" not in ws.build_url(ws.WeatherConfig(), LAT, LON, TODAY, TODAY)
