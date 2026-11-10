"""In-app notifications, and the outbox for the few account events that are also emailed."""

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
        ACCESS_REQUEST = "access_request", "Registration request"

    recipient = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="notifications")
    type = models.CharField(max_length=30, choices=Type.choices)
    title = models.CharField(max_length=200)
    message = models.TextField(blank=True)
    url = models.CharField(max_length=500, blank=True, help_text="Page inside BiteTracker to open.")
    school = models.ForeignKey(School, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    # Identifies the event a notification is about (e.g. "access_request:12:submitted"), so the
    # same event never notifies the same person twice. Blank for events that may repeat.
    event_key = models.CharField(max_length=120, blank=True, default="")
    is_read = models.BooleanField(default=False)
    read_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-pk"]
        indexes = [models.Index(fields=["recipient", "is_read", "-created_at"], name="notif_recipient_unread")]
        constraints = [
            models.UniqueConstraint(fields=["recipient", "event_key"], condition=~models.Q(event_key=""),
                                    name="unique_notification_event_per_recipient"),
        ]

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
            self.Type.ACCESS_REQUEST: "bi-person-plus",
        }.get(self.type, "bi-bell")


class EmailDelivery(models.Model):
    """One account email to one person, recorded in the same transaction as the event it reports.

    The row is the outbox and the duplicate guard: (recipient, event_key) is unique, so the same
    event never emails the same person twice, however often a page is refreshed or a form is
    resubmitted. Rows exist only for events that happened after this table was created, so
    nothing is ever sent retrospectively. Only account emails are recorded here; routine
    activity (forecasts, uploads, calendar) stays in-app.
    """

    class Kind(models.TextChoices):
        REGISTRATION_RECEIVED = "registration_received", "Registration received"
        ACCESS_REQUEST_ADMIN = "access_request_admin", "New registration request (administrators)"
        REGISTRATION_APPROVED = "registration_approved", "Registration approved"
        REGISTRATION_REJECTED = "registration_rejected", "Registration not approved"
        ACCOUNT_SUSPENDED = "account_suspended", "Account suspended"
        ACCOUNT_REACTIVATED = "account_reactivated", "Account reactivated"
        ACCESS_CHANGED = "access_changed", "Role or school assignments changed"

    class Status(models.TextChoices):
        PENDING = "pending", "Waiting to send"
        SENDING = "sending", "Sending"
        SENT = "sent", "Sent"
        FAILED = "failed", "Failed (will be retried)"
        DISABLED = "disabled", "Not sent: email delivery is switched off"
        SUPPRESSED = "suppressed", "Not sent: recipient not in EMAIL_ALLOWED_RECIPIENTS"

    recipient = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="email_deliveries")
    # The address at the time of the event: later edits to the account don't redirect an old email.
    to_email = models.EmailField()
    kind = models.CharField(max_length=40, choices=Kind.choices)
    event_key = models.CharField(max_length=120)
    subject = models.CharField(max_length=200)
    body_text = models.TextField()
    body_html = models.TextField()
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING, db_index=True)
    # Delivery mode when queued (console / smtp / disabled). A row is only ever sent under the same
    # mode, so emails recorded in console mode are never sent later through SMTP.
    channel = models.CharField(max_length=20, blank=True, default="")
    attempts = models.PositiveSmallIntegerField(default=0)
    # Exception class name only: SMTP error text can echo addresses or server details.
    last_error = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_attempt_at = models.DateTimeField(null=True, blank=True)
    sent_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at", "-pk"]
        constraints = [
            models.UniqueConstraint(fields=["recipient", "event_key"], name="unique_email_event_per_recipient"),
        ]

    def __str__(self) -> str:
        return f"{self.get_kind_display()} → user {self.recipient_id} ({self.get_status_display()})"
