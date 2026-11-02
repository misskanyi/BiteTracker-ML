"""Create notifications for real application events, respecting each user's preferences.

`notify` is the single entry point. It drops recipients who:
- switched that kind of notification off in Settings (account, security and
  support notices are always delivered),
- may not see the school the notification is about,
- are inactive, or performed the action themselves (unless `include_actor`),
- already have the same unread notification from the last minute (double submits).
"""

from __future__ import annotations

from datetime import timedelta

from django.contrib.auth import get_user_model
from django.db.models import Q
from django.utils import timezone

from web.accounts import access
from web.accounts.models import Role
from web.notifications.models import Notification

Type = Notification.Type

PREFERENCE_FIELDS = {
    Type.FORECAST_COMPLETED: "notify_forecast",
    Type.FORECAST_FAILED: "notify_forecast",
    Type.DATASET_UPLOADED: "notify_dataset_upload",
    Type.DATASET_INVALID: "notify_dataset_validation",
    Type.DATASET_REVIEW: "notify_dataset_upload",
    Type.CALENDAR: "notify_calendar",
    Type.ANNOUNCEMENT: "notify_announcements",
}

DUPLICATE_WINDOW = timedelta(minutes=1)


def wants(user, notification_type: str) -> bool:
    field = PREFERENCE_FIELDS.get(notification_type)
    return True if field is None else getattr(access.get_profile(user), field)


def notify(recipients, notification_type: str, title: str, message: str = "", *, url: str = "", school=None,
           actor=None, include_actor: bool = False) -> list[Notification]:
    seen, created = set(), []
    recent = timezone.now() - DUPLICATE_WINDOW
    for user in recipients:
        if user.pk in seen or not user.is_active:
            continue
        seen.add(user.pk)
        if actor is not None and user.pk == actor.pk and not include_actor:
            continue
        if not wants(user, notification_type):
            continue
        if school is not None and not access.can_access_school_code(user, school.school_id):
            continue
        if Notification.objects.filter(recipient=user, type=notification_type, title=title, url=url,
                                       is_read=False, created_at__gte=recent).exists():
            continue
        created.append(Notification(recipient=user, type=notification_type, title=title[:200], message=message,
                                    url=url, school=school))
    return Notification.objects.bulk_create(created)


def users_for_school(school):
    """Active users who can see a school: those assigned to it and every System Admin."""
    users = get_user_model().objects.filter(is_active=True)
    if school is not None:
        users = users.filter(
            Q(school_assignments__school=school) | Q(is_superuser=True) | Q(profile__role=Role.SYSTEM_ADMIN)
        ).distinct()
    return users.select_related("profile")


def system_admins():
    return (
        get_user_model().objects.filter(is_active=True)
        .filter(Q(is_superuser=True) | Q(profile__role=Role.SYSTEM_ADMIN)).distinct().select_related("profile")
    )
