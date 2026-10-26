"""Problems reported from Help & Support. A simple Open → In progress → Resolved workflow."""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import models
from django.utils import timezone


def screenshot_upload_path(instance: SupportIssue, filename: str) -> str:
    return f"support/{timezone.now():%Y%m%d_%H%M%S}_user{instance.reporter_id}{Path(filename).suffix.lower()}"


class SupportIssue(models.Model):
    class Category(models.TextChoices):
        ACCOUNT = "account", "Account access"
        FORECAST = "forecast", "Forecasts"
        DATA = "data", "Data uploads"
        CALENDAR = "calendar", "School calendar"
        REPORTS = "reports", "Reports and charts"
        OTHER = "other", "Something else"

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        IN_PROGRESS = "in_progress", "In progress"
        RESOLVED = "resolved", "Resolved"

    reporter = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name="support_issues")
    category = models.CharField(max_length=20, choices=Category.choices)
    subject = models.CharField(max_length=200)
    description = models.TextField()
    screenshot = models.FileField(upload_to=screenshot_upload_path, blank=True,
                                  help_text="Stored under data/uploads/support/ and never served publicly.")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.OPEN, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"#{self.pk} {self.subject}"
