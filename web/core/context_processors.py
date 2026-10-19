"""Template context shared by every page."""

from __future__ import annotations

import logging

from django.urls import reverse

from services import status_service

logger = logging.getLogger(__name__)

# Sidebar sections: (heading, [(label, icon, url name, url names that mark it active, visibility check)]).
# Visibility mirrors the permissions the views themselves enforce; hiding a link is not access control.
NAV_SECTIONS = [
    ("Main", [
        ("Dashboard", "bi-grid-1x2", "core:dashboard", {"dashboard"}, None),
        ("Historical demand", "bi-graph-up", "core:history", {"history"}, None),
        ("Forecast & planning", "bi-calendar2-week", "core:forecast", {"forecast", "forecast_run"}, None),
        ("Forecast history", "bi-clock-history", "core:forecast_history", {"forecast_history"}, None),
        ("Model performance", "bi-bar-chart-line", "core:performance", {"performance"}, None),
        ("Donor insights", "bi-heart", "core:donors", {"donors"}, None),
    ]),
    ("Data management", [
        ("Data upload", "bi-cloud-arrow-up", "schools:upload", {"upload", "upload_detail"},
         lambda user: user.has_perm("schools.add_datasetupload")),
        ("Schools", "bi-buildings", "admin:schools_school_changelist", set(),
         lambda user: user.is_staff and user.has_perm("schools.view_school")),
    ]),
    ("Administration", [
        ("Django admin", "bi-shield-lock", "admin:index", set(), lambda user: user.is_staff),
    ]),
]


def navigation(request) -> dict:
    """Sidebar items the signed-in user may see, with the active page marked."""
    user = getattr(request, "user", None)
    if user is None or not user.is_authenticated:
        return {}
    match = getattr(request, "resolver_match", None)
    current = match.url_name if match else None
    sections, current_label = [], None
    for heading, entries in NAV_SECTIONS:
        items = []
        for label, icon, url_name, active_on, visible in entries:
            if visible is not None and not visible(user):
                continue
            active = current in active_on
            if active:
                current_label = label
            items.append({"label": label, "icon": icon, "url": reverse(url_name), "active": active})
        if items:
            sections.append({"heading": heading, "items": items})
    return {"NAV_SECTIONS": sections, "NAV_CURRENT": current_label}


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
