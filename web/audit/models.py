"""Read-only audit trail of important actions.

Rows are written only by `web.audit.services.log_action`; nothing in the
application edits them. Details never contain passwords, tokens or other secrets.
"""

from __future__ import annotations

from django.conf import settings
from django.db import models

from web.schools.models import School


class AuditLog(models.Model):
    class Action(models.TextChoices):
        USER_CREATED = "user_created", "User created"
        USER_UPDATED = "user_updated", "User details updated"
        ROLE_CHANGED = "role_changed", "Role changed"
        SCHOOLS_ASSIGNED = "schools_assigned", "School assignments changed"
        USER_ACTIVATED = "user_activated", "Account reactivated"
        USER_DEACTIVATED = "user_deactivated", "Account suspended"
        PASSWORD_RESET = "password_reset", "Password reset by administrator"
        PASSWORD_CHANGED = "password_changed", "Password changed"
        PROFILE_UPDATED = "profile_updated", "Profile updated"
        SCHOOL_UPDATED = "school_updated", "School record updated"
        CALENDAR_CREATED = "calendar_created", "Calendar event created"
        CALENDAR_UPDATED = "calendar_updated", "Calendar event updated"
        CALENDAR_DELETED = "calendar_deleted", "Calendar event deleted"
        DATASET_UPLOADED = "dataset_uploaded", "Dataset uploaded"
        DATASET_REVIEWED = "dataset_reviewed", "Dataset upload reviewed"
        ANNOUNCEMENT_SENT = "announcement_sent", "Announcement sent"
        ISSUE_REPORTED = "issue_reported", "Problem reported"
        ISSUE_STATUS_CHANGED = "issue_status_changed", "Problem status changed"
        ACCESS_REQUESTED = "access_requested", "Registration requested"
        ACCESS_APPROVED = "access_approved", "Registration approved"
        ACCESS_REJECTED = "access_rejected", "Registration rejected"
        ACCESS_DENIED = "access_denied", "Blocked sign-in attempt"
        GOOGLE_LINKED = "google_linked", "Google account linked"

    actor = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    # Kept as text so the log still reads correctly after an account is deleted.
    actor_name = models.CharField(max_length=150, blank=True)
    action = models.CharField(max_length=40, choices=Action.choices, db_index=True)
    target_type = models.CharField(max_length=50, blank=True)
    target_id = models.CharField(max_length=64, blank=True)
    target_label = models.CharField(max_length=255, blank=True)
    # Links used to decide which administrators may read the entry.
    school = models.ForeignKey(School, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    target_user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at", "-pk"]

    def __str__(self) -> str:
        return f"{self.created_at:%Y-%m-%d %H:%M} · {self.actor_name or 'system'} · {self.get_action_display()}"
