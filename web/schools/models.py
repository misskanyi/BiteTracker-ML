"""Application state stored in the Django database.

Historical meal-demand records are NOT stored here: the processed CSV used
by the ML pipeline stays the single source of truth for them. The database
holds school metadata, a record of every dataset upload, and an audit trail
of forecasts generated through the web application.
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.text import get_valid_filename


class School(models.Model):
    school_id = models.CharField(
        "school ID", max_length=50, unique=True, help_text="Stable identifier used in the dataset (e.g. SCH001)."
    )
    name = models.CharField(max_length=200, blank=True)
    county = models.CharField(max_length=100, blank=True)
    sub_county = models.CharField(max_length=100, blank=True)
    verified_enrolment = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text=(
            "Verified total school enrolment, entered only when confirmed by the school or programme. "
            "Used for planning comparisons after prediction, not as a model feature. Leave blank if unknown."
        ),
    )
    data_source = models.CharField(
        max_length=100, blank=True, help_text="Data status of the dataset this record was imported from."
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name", "school_id"]

    def __str__(self) -> str:
        return f"{self.name} ({self.school_id})" if self.name else self.school_id


def dataset_upload_path(instance: DatasetUpload, filename: str) -> str:
    path = Path(filename)
    stem = get_valid_filename(path.stem)[:50] or "dataset"
    return f"datasets/{timezone.now():%Y%m%d_%H%M%S}_{stem}{path.suffix.lower()}"


def supporting_letter_path(instance: DatasetUpload, filename: str) -> str:
    # Only the extension is kept from the original name: nothing user-supplied reaches the path.
    return f"letters/{timezone.now():%Y%m%d_%H%M%S}_user{instance.uploaded_by_id}{Path(filename).suffix.lower()}"


class DatasetUpload(models.Model):
    class ValidationStatus(models.TextChoices):
        VALID = "valid", "Valid"
        INVALID = "invalid", "Invalid"
        UNREADABLE = "unreadable", "Unreadable"

    class ProcessingStatus(models.TextChoices):
        NOT_PROCESSED = "not_processed", "Not processed (stored for review only)"

    class ReviewStatus(models.TextChoices):
        # Blank for uploads that failed validation: nothing was stored, so there is nothing to review.
        PENDING = "pending", "Awaiting review"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"
        CLARIFICATION = "clarification", "Clarification requested"

    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="dataset_uploads"
    )
    original_filename = models.CharField(max_length=255)
    file = models.FileField(
        upload_to=dataset_upload_path,
        blank=True,
        help_text="Stored only for valid uploads, under data/uploads/ (never committed to Git).",
    )
    file_size = models.PositiveIntegerField(help_text="Bytes.")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    validation_status = models.CharField(max_length=20, choices=ValidationStatus.choices)
    forecast_readiness = models.CharField(max_length=20, blank=True)
    n_records = models.PositiveIntegerField(null=True, blank=True)
    quality_score = models.PositiveSmallIntegerField(null=True, blank=True)
    validation_summary = models.JSONField(default=dict, blank=True)
    processing_status = models.CharField(
        max_length=20, choices=ProcessingStatus.choices, default=ProcessingStatus.NOT_PROCESSED
    )
    supporting_letter = models.FileField(
        upload_to=supporting_letter_path,
        blank=True,
        help_text="Signed cover letter (PDF, DOCX, PNG or JPEG) stored with valid uploads, never served publicly.",
    )
    letter_filename = models.CharField(max_length=255, blank=True)
    review_status = models.CharField(max_length=20, choices=ReviewStatus.choices, blank=True, db_index=True)
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="reviewed_uploads"
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    review_comment = models.TextField(blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return f"{self.original_filename} ({self.get_validation_status_display()})"


class ForecastRun(models.Model):
    class Horizon(models.IntegerChoices):
        NEXT_DAY = 1, "Next school day"
        NEXT_7 = 7, "Next 7 school days"
        NEXT_30 = 30, "Next 30 school days"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="forecast_runs"
    )
    school = models.ForeignKey(School, null=True, blank=True, on_delete=models.SET_NULL, related_name="forecast_runs")
    # Kept even if no School row exists (e.g. before import_schools has run).
    school_code = models.CharField("school ID", max_length=50)
    school_name = models.CharField(max_length=200, blank=True)
    horizon_days = models.PositiveSmallIntegerField(choices=Horizon.choices)
    origin_date = models.DateField(help_text="Last recorded date the forecast starts from.")
    first_forecast_date = models.DateField()
    last_forecast_date = models.DateField()
    assumptions = models.JSONField(default=dict, blank=True, help_text="User-provided inputs (blank = recent average).")
    model_label = models.CharField(max_length=100)
    model_version = models.CharField(max_length=64, blank=True, help_text="Training timestamp from the model metadata.")
    results_label = models.CharField(max_length=100)
    day1_predicted_meals = models.PositiveIntegerField()
    day1_recommended_meals = models.PositiveIntegerField()
    day1_risk_level = models.CharField(max_length=20)
    total_predicted_meals = models.PositiveIntegerField()
    total_recommended_meals = models.PositiveIntegerField()
    result_summary = models.JSONField(
        default=dict, help_text="Per-day predictions, recommendations, warnings and decision-support summary."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.school_code} · {self.get_horizon_days_display()} · {self.created_at:%Y-%m-%d %H:%M}"
