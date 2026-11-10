"""Template context shared by every page."""

from __future__ import annotations

import logging

from django.conf import settings
from django.urls import reverse

from services import status_service
from web.accounts import access
from web.notifications.models import Notification

logger = logging.getLogger(__name__)

# Sidebar sections: (heading, [(label, icon, url name, view names that mark it active, visibility check)]).
# Active sets hold bare url names ("history") or, where names repeat across apps, full view names
# ("administration:users"). Visibility mirrors the permissions the views themselves enforce; hiding a
# link is not access control.
NAV_SECTIONS = [
    ("Main", [
        ("Dashboard", "bi-grid-1x2", "core:dashboard", {"dashboard"}, None),
        ("Historical demand", "bi-graph-up", "core:history", {"history"}, None),
        ("Forecast & planning", "bi-calendar2-week", "core:forecast", {"forecast", "forecast_run"}, None),
        ("Forecast history", "bi-clock-history", "core:forecast_history", {"forecast_history"}, None),
        ("School calendar", "bi-calendar3", "schoolcalendar:calendar", {
            "schoolcalendar:calendar", "schoolcalendar:event_detail", "schoolcalendar:event_create",
            "schoolcalendar:event_update", "schoolcalendar:event_delete",
        }, None),
        ("Model performance", "bi-bar-chart-line", "core:performance", {"performance"}, access.can_view_model_evaluation),
        ("Donor insights", "bi-heart", "core:donors", {"donors"}, None),
    ]),
    ("Data management", [
        ("Data upload", "bi-cloud-arrow-up", "schools:upload", {"upload", "upload_detail"}, access.can_upload),
        ("Schools", "bi-buildings", "admin:schools_school_changelist", set(),
         lambda user: user.is_staff and user.has_perm("schools.view_school")),
    ]),
    ("Administration", [
        ("Administration", "bi-people", "administration:users", {
            "administration:users", "administration:user_create", "administration:user_detail",
            "administration:user_edit", "administration:user_password", "administration:schools",
            "administration:school_detail", "administration:school_edit", "administration:logs",
            "administration:issues", "administration:issue_detail", "administration:announce",
            "administration:uploads", "administration:upload_review",
            "administration:access_requests", "administration:access_request_detail",
        }, access.is_admin),
        ("Django admin", "bi-shield-lock", "admin:index", set(), lambda user: user.is_staff),
    ]),
]

# Personal pages, reached from the topbar menus rather than the sidebar; listed so the topbar
# still names the current page.
ACCOUNT_NAV = [
    ("My profile", {"accounts:profile", "accounts:profile_edit"}),
    ("Settings", {"accounts:settings"}),
    ("Help & support", {"support:help", "support:issue_detail"}),
    ("Notifications", {"notifications:list"}),
]

# Pages the server takes noticeably longer to prepare; their links show the branded page loader
# (layout.js) with this message if the next page has not arrived after a short delay.
NAV_LOADING_MESSAGES = {
    "core:dashboard": "Loading dashboard…",
    "core:history": "Loading school records…",
    "core:forecast_history": "Loading forecast results…",
    "core:performance": "Analysing model performance…",
    "core:donors": "Loading donor insights…",
}


def navigation(request) -> dict:
    """Sidebar items the signed-in user may see, with the active page marked."""
    user = getattr(request, "user", None)
    if user is None or not user.is_authenticated:
        return {}
    match = getattr(request, "resolver_match", None)
    current = {match.url_name, match.view_name} if match else set()
    sections, current_label = [], None
    for heading, entries in NAV_SECTIONS:
        items = []
        for label, icon, url_name, active_on, visible in entries:
            if visible is not None and not visible(user):
                continue
            active = bool(current & active_on)
            if active:
                current_label = label
            items.append({"label": label, "icon": icon, "url": reverse(url_name), "active": active,
                          "loading": NAV_LOADING_MESSAGES.get(url_name)})
        if items:
            sections.append({"heading": heading, "items": items})
    for label, active_on in ACCOUNT_NAV:
        if current & active_on:
            current_label = label
    return {"NAV_SECTIONS": sections, "NAV_CURRENT": current_label}


def initials(user) -> str:
    first, last = (user.first_name or "").strip(), (user.last_name or "").strip()
    if first or last:
        return ((first[:1] + last[:1]) or first[:2]).upper()
    return user.get_username()[:2].upper()


def account(request) -> dict:
    """The signed-in user's role, preferences, avatar and unread notifications, for the shell."""
    user = getattr(request, "user", None)
    if user is None or not user.is_authenticated:
        return {}
    profile = access.get_profile(user)
    return {
        "USER_PROFILE": profile,
        "USER_ROLE": access.role_label(user),
        "USER_INITIALS": initials(user),
        "USER_AVATAR_URL": reverse("accounts:avatar", args=[user.pk]) + f"?v={int(profile.updated_at.timestamp())}" if profile.avatar else "",
        "IS_ADMIN": access.is_admin(user),
        "IS_SYSTEM_ADMIN": access.is_system_admin(user),
        "NOTIFICATIONS_UNREAD": Notification.objects.filter(recipient=user, is_read=False).count(),
        "NOTIFICATIONS_RECENT": list(Notification.objects.filter(recipient=user)[:6]),
        "NOTIFICATIONS_LATEST_ID": Notification.objects.filter(recipient=user).order_by("-pk").values_list(
            "pk", flat=True).first() or 0,
    }


def debug(request) -> dict:
    """Whether browser-console debug logging (static/js/debug.js) is on by default."""
    return {"BT_DEBUG": settings.DEBUG}


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
