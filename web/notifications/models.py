"""In-app notifications (no email or SMS)."""

from __future__ import annotations

from django.conf import settings
from django.db import models

from web.schools.models import School


class Notification(models.Model):
    class Type(models.TextChoices):
        FORECAST_COMPLETED = "forecast_completed", "Forecast completed"
        FORECAST_FAILED = "forecast_failed", "Forecast failed"
        DATASET_UPLOADED = "dataset_uploaded", "Dataset upload completed"
        DATASET_INVALID = "dataset_invalid", "Dataset validation errors"
        DATASET_REVIEW = "dataset_review", "Dataset review"
        CALENDAR = "calendar", "School calendar update"
        ANNOUNCEMENT = "announcement", "System announcement"
        ACCOUNT = "account", "Account update"
        SECURITY = "security", "Security"
        SUPPORT = "support", "Help & support"

    recipient = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="notifications")
    type = models.CharField(max_length=30, choices=Type.choices)
    title = models.CharField(max_length=200)
    message = models.TextField(blank=True)
    url = models.CharField(max_length=500, blank=True, help_text="Page inside BiteTracker to open.")
    school = models.ForeignKey(School, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    is_read = models.BooleanField(default=False)
    read_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-pk"]
        indexes = [models.Index(fields=["recipient", "is_read", "-created_at"], name="notif_recipient_unread")]

    def __str__(self) -> str:
        return f"{self.recipient.get_username()}: {self.title}"

    @property
    def icon(self) -> str:
        return {
            self.Type.FORECAST_COMPLETED: "bi-graph-up-arrow",
            self.Type.FORECAST_FAILED: "bi-exclamation-octagon",
            self.Type.DATASET_UPLOADED: "bi-cloud-check",
            self.Type.DATASET_INVALID: "bi-file-earmark-x",
            self.Type.DATASET_REVIEW: "bi-clipboard-check",
            self.Type.CALENDAR: "bi-calendar-event",
            self.Type.ANNOUNCEMENT: "bi-megaphone",
            self.Type.ACCOUNT: "bi-person-gear",
            self.Type.SECURITY: "bi-shield-lock",
            self.Type.SUPPORT: "bi-life-preserver",
        }.get(self.type, "bi-bell")
