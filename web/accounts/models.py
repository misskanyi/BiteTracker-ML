"""Per-user profile, preferences and school assignments.

Django's built-in User model is kept unchanged. Everything BiteTracker adds about
a person lives in `UserProfile` (one row per user, created automatically) and
`SchoolAssignment` (which schools a user may work with).

Roles (as defined in the project documentation):
- School Manager: their assigned schools only. Views history, generates forecasts,
  uploads school records for verification.
- Resource Planning Stakeholder: read-only access to the schools assigned to them,
  for comparing schools and planning resources.
- System Administrator: every school and every account. Approves registrations,
  assigns roles and schools, verifies uploads. Kept in sync with Django's
  `is_superuser` flag, so existing superusers are System Administrators.

`AccessRequest` records a self-registration (Google sign-in) and the administrator's
decision. Accounts created by an administrator have no request and are approved.
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import models
from django.utils import timezone

from web.schools.models import School


class Role(models.TextChoices):
    SCHOOL_MANAGER = "school_manager", "School Manager"
    PLANNING_STAKEHOLDER = "planning_stakeholder", "Resource Planning Stakeholder"
    SYSTEM_ADMIN = "system_admin", "System Administrator"


def avatar_upload_path(instance: UserProfile, filename: str) -> str:
    # The user id, not the original filename, names the file: nothing user-supplied reaches the path.
    return f"avatars/user_{instance.user_id}_{timezone.now():%Y%m%d%H%M%S}{Path(filename).suffix.lower()}"


class UserProfile(models.Model):
    class Theme(models.TextChoices):
        SYSTEM = "system", "System default"
        LIGHT = "light", "Light"
        DARK = "dark", "Dark"

    class LandingPage(models.TextChoices):
        DASHBOARD = "dashboard", "Dashboard"
        FORECAST = "forecast", "Forecasts"

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.SCHOOL_MANAGER)
    phone = models.CharField(max_length=20, blank=True)
    avatar = models.FileField(upload_to=avatar_upload_path, blank=True, help_text="Stored under data/uploads/avatars/.")

    # Appearance
    theme = models.CharField(max_length=10, choices=Theme.choices, default=Theme.SYSTEM)
    compact_tables = models.BooleanField(default=False)
    reduce_motion = models.BooleanField(default=False)
    landing_page = models.CharField(max_length=20, choices=LandingPage.choices, default=LandingPage.DASHBOARD)

    # Optional in-app notifications (account and security notices cannot be switched off).
    notify_forecast = models.BooleanField("Forecast completion", default=True)
    notify_dataset_upload = models.BooleanField("Dataset upload completion", default=True)
    notify_dataset_validation = models.BooleanField("Dataset validation errors", default=True)
    notify_calendar = models.BooleanField("School calendar updates", default=True)
    notify_announcements = models.BooleanField("Important system announcements", default=True)

    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"Profile of {self.user.get_username()}"


class SchoolAssignment(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="school_assignments")
    school = models.ForeignKey(School, on_delete=models.CASCADE, related_name="assignments")
    assigned_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    assigned_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["user", "school"], name="unique_user_school_assignment")]
        ordering = ["school__name", "school__school_id"]

    def __str__(self) -> str:
        return f"{self.user.get_username()} → {self.school}"


class AccessRequest(models.Model):
    """A self-registration waiting for, or decided by, a System Administrator.

    One per user: signing in again never creates a second request. The row is kept
    after a decision as the audit record of who decided what, and when.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending approval"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="access_request")
    provider = models.CharField(max_length=30, default="google")
    # Copied at registration from the provider's verified identity, so the record stays readable.
    email = models.EmailField()
    full_name = models.CharField(max_length=300, blank=True)
    requested_role = models.CharField(max_length=30, choices=Role.choices, blank=True)
    applicant_note = models.CharField(
        "note from applicant", max_length=500, blank=True,
        help_text="Optional: the school(s) or organisation the applicant works with.",
    )
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    decided_at = models.DateTimeField(null=True, blank=True)
    decided_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    granted_role = models.CharField(max_length=30, choices=Role.choices, blank=True)
    decision_note = models.CharField(max_length=1000, blank=True)

    class Meta:
        ordering = ["-created_at", "-pk"]

    def __str__(self) -> str:
        return f"{self.email} ({self.get_status_display()})"
