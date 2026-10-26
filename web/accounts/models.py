"""Per-user profile, preferences and school assignments.

Django's built-in User model is kept unchanged. Everything BiteTracker adds about
a person lives in `UserProfile` (one row per user, created automatically) and
`SchoolAssignment` (which schools a user may work with).

Roles:
- School Operator: assigned schools only.
- Programme Admin: the schools assigned to them form their programme; they manage
  School Operators within it.
- System Admin: every school and every account. Kept in sync with Django's
  `is_superuser` flag, so existing superusers are System Admins.
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import models
from django.utils import timezone

from web.schools.models import School


class Role(models.TextChoices):
    SCHOOL_OPERATOR = "school_operator", "School Operator"
    PROGRAMME_ADMIN = "programme_admin", "Programme Admin"
    SYSTEM_ADMIN = "system_admin", "System Admin"


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
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.SCHOOL_OPERATOR)
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
