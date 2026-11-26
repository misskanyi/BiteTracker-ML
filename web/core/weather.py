"""Django glue for the morning weather advisory: settings in, school location lookup, advisory out.

The advisory is fetched when a forecast result is displayed (cached in services.weather_service),
never stored with the run and never passed to any forecasting code.
"""

from __future__ import annotations

from datetime import date

from django.conf import settings

from services import weather_service
from web.schools.models import School


def config() -> weather_service.WeatherConfig:
    return weather_service.WeatherConfig(
        enabled=settings.WEATHER_ADVISORY_ENABLED,
        base_url=settings.WEATHER_API_BASE_URL,
        api_key=settings.WEATHER_API_KEY,
        timeout_seconds=settings.WEATHER_TIMEOUT_SECONDS,
        cache_seconds=settings.WEATHER_CACHE_SECONDS,
        failure_cache_seconds=settings.WEATHER_FAILURE_CACHE_SECONDS,
        forecast_days=settings.WEATHER_FORECAST_DAYS,
        heavy_rain_mm_per_hour=settings.WEATHER_HEAVY_RAIN_MM_PER_HOUR,
    )


def advisory_for(school: School | None, school_code: str, dates: list[date]) -> weather_service.WeatherAdvisory:
    """Morning heavy-rain advisory for a school's forecast dates. Never raises."""
    school = school or School.objects.filter(school_id=school_code).first()
    latitude = school.latitude if school else None
    longitude = school.longitude if school else None
    advisory = weather_service.morning_advisory(latitude, longitude, dates, config())
    if school is not None and school.has_location:
        advisory.location_label = school.location_label or f"{float(latitude):.4f}, {float(longitude):.4f}"
        advisory.location_verified = school.location_verified
    return advisory
