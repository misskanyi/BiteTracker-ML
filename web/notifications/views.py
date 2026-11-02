"""The signed-in user's own notifications. Every query is filtered by recipient=request.user."""

from __future__ import annotations

from django.contrib import messages
from django.core.paginator import Paginator
from django.template.defaultfilters import pluralize
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from web.notifications.models import Notification


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
    return _back(request)


@require_POST
def mark_all_read(request):
    Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True, read_at=timezone.now())
    return _back(request)


@require_POST
def delete_notification(request, pk: int):
    get_object_or_404(Notification, pk=pk, recipient=request.user).delete()
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
