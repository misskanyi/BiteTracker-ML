"""Template context shared by every page."""

from __future__ import annotations

import logging

from services import status_service

logger = logging.getLogger(__name__)


def data_provenance(request) -> dict:
    """Data-status label for the banner on every page, derived from the active model metadata."""
    try:
        status = status_service.get_data_status()
    except Exception:  # the banner must never break a page
        logger.exception("Could not determine data status")
        status = {
            "kind": "unknown",
            "label": status_service.UNKNOWN_LABEL,
            "detail": "The data behind results could not be confirmed. Treat results as unverified.",
        }
    return {"DATA_STATUS": status, "DATA_LABEL": status["label"]}
