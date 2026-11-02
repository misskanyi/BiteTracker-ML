"""Write audit entries and decide who may read them."""

from __future__ import annotations

from django.db.models import Q, QuerySet

from web.accounts import access
from web.audit.models import AuditLog

# Never stored, whatever a caller passes in.
_SECRET_MARKERS = ("password", "token", "secret", "session", "csrf")


def _clean(details: dict | None) -> dict:
    return {k: v for k, v in (details or {}).items() if not any(m in k.lower() for m in _SECRET_MARKERS)}


def log_action(actor, action: str, *, target=None, target_label: str = "", school=None, target_user=None,
               details: dict | None = None) -> AuditLog:
    actor = actor if getattr(actor, "is_authenticated", False) else None
    return AuditLog.objects.create(
        actor=actor,
        actor_name=actor.get_username() if actor else "",
        action=action,
        target_type=target._meta.model_name if target is not None else "",
        target_id=str(target.pk) if target is not None else "",
        target_label=(target_label or (str(target) if target is not None else ""))[:255],
        school=school,
        target_user=target_user,
        details=_clean(details),
    )


def visible_logs(user) -> QuerySet[AuditLog]:
    """System Admins read everything; Programme Admins read entries about their own
    schools, the operators they manage, and their own actions."""
    logs = AuditLog.objects.select_related("actor", "school", "target_user")
    if access.is_system_admin(user):
        return logs
    if access.is_programme_admin(user):
        return logs.filter(
            Q(school__in=access.accessible_schools(user))
            | Q(target_user__in=access.manageable_users(user))
            | Q(actor=user)
        ).distinct()
    return logs.none()
