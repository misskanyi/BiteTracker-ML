"""The signed-in user's own notifications. Every query is filtered by recipient=request.user."""

from __future__ import annotations

from django.contrib import messages
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.urls import reverse
from django.utils import dateformat
from django.utils.timesince import timesince
from django.template.defaultfilters import pluralize
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET, require_POST

from web.notifications.models import Notification


PANEL_SIZE = 6
FEED_MAX_NEW = 10


def _wants_json(request) -> bool:
    return request.headers.get("Accept", "").startswith("application/json")


def _unread_count(user) -> int:
    return Notification.objects.filter(recipient=user, is_read=False).count()


def _as_json(n: Notification) -> dict:
    """Only what the bell and toasts display. Destinations are internal paths (see open_notification)."""
    return {
        "id": n.pk,
        "type": n.type,
        "type_label": n.get_type_display(),
        "icon": n.icon,
        "title": n.title,
        "message": n.message,
        "is_read": n.is_read,
        "created_at": timezone.localtime(n.created_at).isoformat(),
        "created_display": dateformat.format(timezone.localtime(n.created_at), "j M Y, H:i"),
        "ago": f"{timesince(n.created_at)} ago",
        "has_destination": bool(n.url),
        "open_url": reverse("notifications:open", args=[n.pk]),
        "read_url": reverse("notifications:mark_read", args=[n.pk]),
        "delete_url": reverse("notifications:delete", args=[n.pk]),
    }


@never_cache
@require_GET
def feed(request):
    """Polled by static/js/notifications.js: unread count, the bell panel's items, and
    unread notifications newer than `since` (shown as toasts). Always the requester's own."""
    mine = Notification.objects.filter(recipient=request.user)
    since = request.GET.get("since", "")
    # No (or a malformed) `since`: nothing is "new", so a first poll never floods the screen with toasts.
    # since=0 is valid: a user who had no notifications yet must still see their first one.
    new = (list(mine.filter(pk__gt=int(since), is_read=False).order_by("-pk")[:FEED_MAX_NEW])
           if since.isdigit() else [])
    latest = mine.order_by("-pk").values_list("pk", flat=True).first() or 0
    return JsonResponse({
        "unread_count": _unread_count(request.user),
        "latest_id": latest,
        "items": [_as_json(n) for n in mine[:PANEL_SIZE]],
        "new": [_as_json(n) for n in reversed(new)],
    })


def _back(request, fallback="notifications:list"):
    target = request.POST.get("next", "")
    if target and url_has_allowed_host_and_scheme(target, allowed_hosts={request.get_host()}, require_https=request.is_secure()):
        return redirect(target)
    return redirect(fallback)


def notification_list(request):
    notifications = Notification.objects.filter(recipient=request.user)
    has_any = notifications.exists()
    show = request.GET.get("show")
    if show == "unread":
        notifications = notifications.filter(is_read=False)
    page = Paginator(notifications, 20).get_page(request.GET.get("page"))
    return render(request, "notifications/list.html", {
        "page": page, "show": show, "has_any": has_any, "querystring": "show=unread" if show == "unread" else "",
    })


@require_POST
def mark_read(request, pk: int):
    notification = get_object_or_404(Notification, pk=pk, recipient=request.user)
    if not notification.is_read:
        notification.is_read, notification.read_at = True, timezone.now()
        notification.save(update_fields=["is_read", "read_at"])
    if _wants_json(request):
        return JsonResponse({"ok": True, "unread_count": _unread_count(request.user)})
    return _back(request)


@require_POST
def mark_all_read(request):
    Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True, read_at=timezone.now())
    if _wants_json(request):
        return JsonResponse({"ok": True, "unread_count": 0})
    return _back(request)


@require_POST
def delete_notification(request, pk: int):
    get_object_or_404(Notification, pk=pk, recipient=request.user).delete()
    if _wants_json(request):
        return JsonResponse({"ok": True, "unread_count": _unread_count(request.user)})
    return _back(request)


@require_POST
def clear_all(request):
    """Delete every notification the user has (read or not). Asked to confirm in the page first."""
    deleted, _ = Notification.objects.filter(recipient=request.user).delete()
    if deleted:
        messages.success(request, f"Cleared {deleted} notification{pluralize(deleted)}.")
    return _back(request)


@require_POST
def open_notification(request, pk: int):
    """Mark as read, then go to the page it is about (internal links only)."""
    notification = get_object_or_404(Notification, pk=pk, recipient=request.user)
    if not notification.is_read:
        notification.is_read, notification.read_at = True, timezone.now()
        notification.save(update_fields=["is_read", "read_at"])
    url = notification.url
    if url and url.startswith("/") and url_has_allowed_host_and_scheme(url, allowed_hosts={request.get_host()}):
        return redirect(url)
    return redirect("notifications:list")
