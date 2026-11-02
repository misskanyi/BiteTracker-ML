"""Unauthenticated health check for load balancers, monitoring and the deploy script.

GET /health/ answers 200 when Django, the database and the production
forecasting artifacts are all usable, and 503 otherwise. The body only says
which check passed or failed ("ok" / "fail"): no file paths, exception text,
versions, school names or metrics, so it is safe to expose publicly.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.contrib.auth.decorators import login_not_required
from django.db import connection
from django.http import JsonResponse
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_safe

from services import data_service, model_service
from src.feature_engineering import SYNTHETIC_SOURCE

logger = logging.getLogger(__name__)


def _check_database() -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        return cursor.fetchone() == (1,)


def _check_model() -> bool:
    # Cached after the first load (keyed on file modification times), so repeat probes are cheap.
    return model_service.get_model() is not None


def _check_processed_data() -> bool:
    return data_service.processed_data_path().is_file()


def _check_real_model() -> bool:
    """A production server must never serve a model trained on the synthetic sample."""
    metadata = model_service.load_model_metadata()
    synthetic = metadata.get("data_source", SYNTHETIC_SOURCE) == SYNTHETIC_SOURCE or (
        "SYNTHETIC" in str(metadata.get("results_label", "")).upper()
    )
    return not synthetic


def _run(name: str, check) -> bool:
    try:
        return bool(check())
    except Exception:  # any failure is a failed check; details go to the server log only
        logger.warning("Health check %r failed", name, exc_info=True)
        return False


@login_not_required
@never_cache
@require_safe
def health(request):
    checks = {
        "database": _check_database,
        "model": _check_model,
        "processed_data": _check_processed_data,
    }
    # Synthetic development models are fine locally; in production (DEBUG off) they fail the check.
    if not settings.DEBUG:
        checks["real_data_model"] = _check_real_model
    results = {name: _run(name, check) for name, check in checks.items()}
    healthy = all(results.values())
    body = {
        "status": "ok" if healthy else "fail",
        "checks": {name: "ok" if passed else "fail" for name, passed in results.items()},
    }
    return JsonResponse(body, status=200 if healthy else 503)
