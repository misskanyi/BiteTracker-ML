"""Facts about this BiteTracker installation for the About sections: read live, never hardcoded."""

from __future__ import annotations

import tomllib
from datetime import datetime
from functools import lru_cache

from django.conf import settings

from services import ServiceError, model_service

DESCRIPTION = (
    "BiteTracker is a machine learning-based decision support system for school meal programmes. "
    "It forecasts daily school meal demand from historical meal-service records and turns forecasts "
    "into planning guidance: recommended meal quantities, shortage and waste risk, and meal budgets."
)


@lru_cache(maxsize=1)
def app_version() -> str | None:
    try:
        with open(settings.BASE_DIR / "pyproject.toml", "rb") as handle:
            return tomllib.load(handle)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None


def model_facts() -> dict:
    """Selected model name and training timestamp from the saved metadata, or blanks if unavailable."""
    try:
        metadata = model_service.load_model_metadata()
    except ServiceError:
        return {"model_name": None, "trained_at": None, "results_label": None}
    try:
        trained_at = datetime.fromisoformat(metadata["timestamp"])
    except (KeyError, TypeError, ValueError):
        trained_at = None
    return {
        "model_name": metadata.get("selected_model"),
        "trained_at": trained_at,
        "results_label": metadata.get("results_label"),
    }


def about_context() -> dict:
    return {"description": DESCRIPTION, "version": app_version(), **model_facts()}
