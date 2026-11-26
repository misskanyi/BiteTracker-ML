"""Morning heavy-rain advisory for forecast results (advisory only).

The advisory answers one question for each selected school day: is heavy rain
forecast at the school between 06:00 and 09:00 Nairobi time, when pupils
travel and the kitchen starts serving? It is shown beside the meal forecast
and is NEVER a model input: it does not change predicted demand,
recommended preparation or budget.

Provider: Open-Meteo hourly forecast API (https://open-meteo.com, CC BY 4.0).
No key is needed for the free API; a key (commercial plan) is only sent from
the server and never rendered or logged.

Window: Open-Meteo's `precipitation` value at hour H is the total of the
PRECEDING hour, so 06:00–09:00 is covered by the values stamped 07:00, 08:00
and 09:00 (Africa/Nairobi).

Heavy rain (default, configurable via `WEATHER_HEAVY_RAIN_MM_PER_HOUR`):
any one hour in that window with at least 4.0 mm of rain. 4 mm/h is the
lower bound of "heavy rain" in the UK Met Office rainfall-rate scale
(slight < 0.5, moderate 0.5–4, heavy > 4 mm/h).

Dates outside the provider window (today to today + `forecast_days` - 1, in
Nairobi) are not requested. Successful responses are cached for
`cache_seconds`; failures for `failure_cache_seconds`, so an outage costs at
most one short timeout per school and date range in that time.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

NAIROBI = ZoneInfo("Africa/Nairobi")
SOURCE_NAME = "Open-Meteo"
SOURCE_URL = "https://open-meteo.com"
SOURCE_LICENCE = "CC BY 4.0"
DEFAULT_BASE_URL = "https://api.open-meteo.com/v1/forecast"
# Hour stamps (local time) whose preceding-hour totals cover 06:00–09:00.
WINDOW_HOURS = (7, 8, 9)
WINDOW_LABEL = "6–9 a.m. (Nairobi time)"

# Status of the advisory for a whole result page.
ADVISORY = "advisory"           # at least one day meets the threshold
CLEAR = "clear"                 # forecast available, no day meets the threshold
NO_LOCATION = "no_location"     # school has no configured coordinates
OUT_OF_WINDOW = "out_of_window"  # no selected date is inside the provider window
UNAVAILABLE = "unavailable"     # provider error, timeout or unusable response
DISABLED = "disabled"           # turned off in settings


@dataclass(frozen=True)
class WeatherConfig:
    enabled: bool = True
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    timeout_seconds: float = 3.0
    cache_seconds: int = 1800
    failure_cache_seconds: int = 300
    forecast_days: int = 16
    heavy_rain_mm_per_hour: float = 4.0


@dataclass
class DayWeather:
    date: date
    hourly_mm: list[float]
    max_hour_mm: float
    total_mm: float
    max_probability_pct: float | None
    heavy_rain: bool


@dataclass
class WeatherAdvisory:
    status: str
    message: str
    threshold_mm_per_hour: float
    days: dict[date, DayWeather] = field(default_factory=dict)
    # Dates that were requested but are outside the provider's forecast window.
    out_of_window: list[date] = field(default_factory=list)
    retrieved_at: datetime | None = None
    source: str = SOURCE_NAME
    source_url: str = SOURCE_URL
    source_licence: str = SOURCE_LICENCE
    window_label: str = WINDOW_LABEL

    @property
    def advisory_days(self) -> list[DayWeather]:
        return [d for d in sorted(self.days.values(), key=lambda d: d.date) if d.heavy_rain]

    def for_day(self, d: date) -> DayWeather | None:
        return self.days.get(d)


class WeatherProviderError(Exception):
    """The provider could not be reached or returned something unusable (message never includes the key)."""


Fetch = Callable[[str, float], dict]

_cache: dict[tuple, tuple[float, object]] = {}
_cache_lock = threading.Lock()


def clear_weather_cache() -> None:
    with _cache_lock:
        _cache.clear()


def today_in_nairobi(now: datetime | None = None) -> date:
    return (now or datetime.now(tz=NAIROBI)).astimezone(NAIROBI).date()


def forecast_window(config: WeatherConfig, today: date) -> tuple[date, date]:
    """First and last date the provider can forecast (inclusive)."""
    return today, today + timedelta(days=max(config.forecast_days, 1) - 1)


def _http_get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "BiteTracker weather advisory", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https base URL
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise WeatherProviderError(f"provider returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise WeatherProviderError(f"provider unreachable ({type(reason).__name__})") from None
    except (ValueError, UnicodeDecodeError):
        raise WeatherProviderError("provider returned invalid JSON") from None


def build_url(config: WeatherConfig, latitude: float, longitude: float, start: date, end: date) -> str:
    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "hourly": "precipitation,precipitation_probability",
        "timezone": "Africa/Nairobi",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    if config.api_key:
        params["apikey"] = config.api_key
    return f"{config.base_url}?{urllib.parse.urlencode(params)}"


def parse_hourly(payload: dict, dates: Iterable[date], threshold: float) -> dict[date, DayWeather]:
    """Morning-window summary per date from an Open-Meteo hourly response."""
    try:
        hourly = payload["hourly"]
        times, rain = hourly["time"], hourly["precipitation"]
        probability = hourly.get("precipitation_probability") or [None] * len(times)
    except (KeyError, TypeError):
        raise WeatherProviderError("provider response has no hourly precipitation") from None
    if not (len(times) == len(rain) == len(probability)):
        raise WeatherProviderError("provider response has mismatched hourly arrays")

    by_stamp = {}
    for stamp, mm, pct in zip(times, rain, probability):
        try:
            by_stamp[datetime.fromisoformat(stamp)] = (mm, pct)
        except (TypeError, ValueError):
            continue

    result = {}
    for d in dates:
        values = [by_stamp.get(datetime(d.year, d.month, d.day, hour)) for hour in WINDOW_HOURS]
        if any(v is None or v[0] is None for v in values):
            continue  # incomplete morning: no claim either way for this day
        hourly_mm = [float(v[0]) for v in values]
        pcts = [float(v[1]) for v in values if v[1] is not None]
        result[d] = DayWeather(
            date=d,
            hourly_mm=hourly_mm,
            max_hour_mm=max(hourly_mm),
            total_mm=round(sum(hourly_mm), 1),
            max_probability_pct=max(pcts) if pcts else None,
            heavy_rain=max(hourly_mm) >= threshold,
        )
    return result


def _cached_fetch(url: str, config: WeatherConfig, fetch: Fetch, clock: Callable[[], float]) -> tuple[dict, float]:
    """(payload, retrieved_at epoch); raises WeatherProviderError, also from a cached failure."""
    key = (url,)
    now = clock()
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None and cached[0] > now:
        value = cached[1]
        if isinstance(value, WeatherProviderError):
            raise value
        return value
    try:
        payload = fetch(url, config.timeout_seconds)
        if not isinstance(payload, dict):
            raise WeatherProviderError("provider returned an unexpected response")
        if payload.get("error"):
            raise WeatherProviderError("provider rejected the request")
    except WeatherProviderError as exc:
        with _cache_lock:
            _cache[key] = (now + config.failure_cache_seconds, exc)
        raise
    except Exception as exc:  # never let weather break the forecast page
        error = WeatherProviderError(f"provider call failed ({type(exc).__name__})")
        with _cache_lock:
            _cache[key] = (now + config.failure_cache_seconds, error)
        raise error from None
    value = (payload, now)
    with _cache_lock:
        _cache[key] = (now + config.cache_seconds, value)
    return value


def morning_advisory(
    latitude: float | None,
    longitude: float | None,
    dates: Iterable[date],
    config: WeatherConfig,
    *,
    fetch: Fetch | None = None,
    now: datetime | None = None,
    clock: Callable[[], float] = time.time,
) -> WeatherAdvisory:
    """Heavy-rain advisory for 06:00–09:00 Nairobi time on each of `dates` (never raises)."""
    dates = sorted(set(dates))
    threshold = config.heavy_rain_mm_per_hour

    def status(kind: str, message: str, **extra) -> WeatherAdvisory:
        return WeatherAdvisory(status=kind, message=message, threshold_mm_per_hour=threshold, **extra)

    if not config.enabled:
        return status(DISABLED, "Weather advisory is turned off.")
    if latitude is None or longitude is None:
        return status(NO_LOCATION, "No weather advisory: this school has no location configured.")

    first, last = forecast_window(config, today_in_nairobi(now))
    in_window = [d for d in dates if first <= d <= last]
    outside = [d for d in dates if d not in in_window]
    if not in_window:
        return status(
            OUT_OF_WINDOW,
            f"No weather advisory: the selected date{'s are' if len(dates) != 1 else ' is'} outside the weather "
            f"forecast window ({first:%d %b} – {last:%d %b %Y}).",
            out_of_window=outside,
        )

    url = build_url(config, float(latitude), float(longitude), in_window[0], in_window[-1])
    try:
        payload, retrieved = _cached_fetch(url, config, fetch or _http_get_json, clock)
        days = parse_hourly(payload, in_window, threshold)
    except WeatherProviderError as exc:
        logger.warning("Weather advisory unavailable: %s", exc)  # exc never contains the URL or key
        return status(UNAVAILABLE, "Weather advisory unavailable right now. The meal forecast is not affected.",
                      out_of_window=outside)
    if not days:
        return status(UNAVAILABLE, "Weather advisory unavailable: the provider returned no morning forecast.",
                      out_of_window=outside)

    retrieved_at = datetime.fromtimestamp(retrieved, tz=NAIROBI)
    heavy = [d for d in days.values() if d.heavy_rain]
    if heavy:
        message = (f"Heavy rain forecast {WINDOW_LABEL} on {len(heavy)} selected day{'s' if len(heavy) != 1 else ''}. "
                   "Advisory only: the meal estimates below do not include weather.")
        kind = ADVISORY
    else:
        message = f"No heavy rain forecast {WINDOW_LABEL} on the selected day{'s' if len(days) != 1 else ''} in the forecast window."
        kind = CLEAR
    return status(kind, message, days=days, out_of_window=outside, retrieved_at=retrieved_at)
