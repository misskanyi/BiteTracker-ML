"""Application state stored in the Django database.

The database holds school metadata, a record of every dataset upload, the
approved meal-demand history (`DemandRecord`: the processed baseline plus rows
from approved uploads, see `web.schools.demand_history`), and an audit trail
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
    # Weather advisories only: never a model input. Both blank = no advisory for this school.
    latitude = models.DecimalField(
        max_digits=8, decimal_places=5, null=True, blank=True,
        help_text="Decimal degrees, e.g. -1.28640. Used only to look up the morning weather advisory.",
    )
    longitude = models.DecimalField(
        max_digits=8, decimal_places=5, null=True, blank=True,
        help_text="Decimal degrees, e.g. 36.81720. Used only to look up the morning weather advisory.",
    )
    location_label = models.CharField(
        "location name", max_length=120, blank=True,
        help_text="Place the weather is looked up for, e.g. “Westlands, Nairobi”. Shown with the weather advisory.",
    )
    location_verified = models.BooleanField(
        "location confirmed by the school or programme", default=False,
        help_text="Leave unticked when the coordinates are an approximate area, not the school's confirmed location.",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name", "school_id"]

    def __str__(self) -> str:
        return f"{self.name} ({self.school_id})" if self.name else self.school_id

    @property
    def has_location(self) -> bool:
        return self.latitude is not None and self.longitude is not None


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
        PROCESSED = "processed", "Added to the forecasting data"

    class ReviewStatus(models.TextChoices):
        # Blank for uploads that failed validation: nothing was stored, so there is nothing to review.
        # "approved" means verified: its rows only feed forecasting once the upload is also published.
        PENDING = "pending", "Pending review"
        APPROVED = "approved", "Verified"
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
    # Set when a System Administrator publishes the verified rows into forecasting history.
    published_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="published_uploads"
    )
    published_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return f"{self.original_filename} ({self.get_validation_status_display()})"

    @property
    def stage(self) -> str:
        """Pipeline stage shown in the interface: pending, verified, published, rejected or clarification."""
        if self.review_status == self.ReviewStatus.APPROVED:
            return "published" if self.published_at else "verified"
        return self.review_status

    @property
    def stage_label(self) -> str:
        return "Published" if self.stage == "published" else self.get_review_status_display()


class DemandRecord(models.Model):
    """One school-day of meal-demand history. Only `approved` (published) rows feed forecasting and training."""

    class Status(models.TextChoices):
        # Same values as DatasetUpload.ReviewStatus, so an upload's decision can be copied onto its rows.
        # The stored value "approved" is kept for existing rows; it means published.
        APPROVED = "approved", "Published"
        VERIFIED = "verified", "Verified (not yet published)"
        PENDING = "pending", "Pending review"
        REJECTED = "rejected", "Rejected"
        CLARIFICATION = "clarification", "Clarification requested"

    class Source(models.TextChoices):
        BASELINE = "baseline", "Historical dataset"
        UPLOAD = "upload", "Verified upload"
        ACTUAL = "actual", "Recorded actual"

    upload = models.ForeignKey(
        DatasetUpload, null=True, blank=True, on_delete=models.CASCADE, related_name="demand_records",
        help_text="Empty for rows seeded from the processed baseline dataset and for recorded actuals.",
    )
    actual = models.OneToOneField(
        "ActualMealRecord", null=True, blank=True, on_delete=models.CASCADE, related_name="demand_record",
        help_text="The recorded actual this row was published from.",
    )
    published_at = models.DateTimeField(
        null=True, blank=True, db_index=True,
        help_text="When the row entered forecasting history (later publications win for the same school-day).",
    )
    status = models.CharField(max_length=20, choices=Status.choices, db_index=True)
    school_id = models.CharField("school ID", max_length=50, db_index=True)
    date = models.DateField()
    school_name = models.CharField(max_length=200, blank=True)
    county = models.CharField(max_length=100, blank=True)
    sub_county = models.CharField(max_length=100, blank=True)
    enrolled_students = models.FloatField(null=True, blank=True)
    students_present = models.FloatField(null=True, blank=True)
    students_absent = models.FloatField(null=True, blank=True)
    meals_prepared = models.FloatField(null=True, blank=True)
    meals_served = models.FloatField(null=True, blank=True)
    leftover_meals = models.FloatField(null=True, blank=True)
    average_cost_per_meal = models.FloatField(null=True, blank=True)
    rainfall_mm = models.FloatField(null=True, blank=True)
    temperature_c = models.FloatField(null=True, blank=True)
    is_holiday = models.FloatField(null=True, blank=True)
    is_exam_period = models.FloatField(null=True, blank=True)
    is_school_event = models.FloatField(null=True, blank=True)
    school_term = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["school_id", "date"]
        constraints = [
            models.UniqueConstraint(
                fields=["school_id", "date"], condition=models.Q(upload__isnull=True, actual__isnull=True),
                name="unique_baseline_school_day",
            ),
            models.UniqueConstraint(fields=["upload", "school_id", "date"], name="unique_upload_school_day"),
        ]

    def __str__(self) -> str:
        return f"{self.school_id} · {self.date} ({self.get_status_display()})"

    @property
    def source(self) -> str:
        if self.upload_id:
            return self.Source.UPLOAD
        if self.actual_id:
            return self.Source.ACTUAL
        return self.Source.BASELINE

    @property
    def source_label(self) -> str:
        return self.Source(self.source).label


class ActualMealRecord(models.Model):
    """Meals actually served at one school on one day, as recorded by a School Manager.

    One row per school-day, shared by every saved forecast that covers that day (forecasts
    are never changed). Pending review → Verified → Published; only published values enter
    forecasting history (through a linked `DemandRecord`), and only verified or published
    values count towards accuracy figures.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending review"
        CORRECTION_REQUESTED = "correction_requested", "Correction requested"
        VERIFIED = "verified", "Verified"
        PUBLISHED = "published", "Published"
        REJECTED = "rejected", "Rejected"

    VERIFIED_STATUSES = (Status.VERIFIED, Status.PUBLISHED)
    # The submitter may change the values directly while the record is in one of these states.
    EDITABLE_STATUSES = (Status.PENDING, Status.CORRECTION_REQUESTED, Status.REJECTED)

    school = models.ForeignKey(School, on_delete=models.PROTECT, related_name="actual_records")
    date = models.DateField()
    meals_served = models.PositiveIntegerField(help_text="Meals actually served. Zero is a valid count.")
    meals_prepared = models.PositiveIntegerField(null=True, blank=True, help_text="Optional: meals prepared that day.")
    status = models.CharField(max_length=24, choices=Status.choices, default=Status.PENDING, db_index=True)
    submitted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="actual_records"
    )
    submitted_at = models.DateTimeField(default=timezone.now)
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="reviewed_actuals"
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    review_comment = models.TextField(blank=True)
    published_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="published_actuals"
    )
    published_at = models.DateTimeField(null=True, blank=True)
    # A correction proposed for a verified or published record; the current values stay in use until approved.
    proposed_meals_served = models.PositiveIntegerField(null=True, blank=True)
    proposed_meals_prepared = models.PositiveIntegerField(null=True, blank=True)
    proposed_reason = models.TextField(blank=True)
    proposed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    proposed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-date", "school__school_id"]
        constraints = [models.UniqueConstraint(fields=["school", "date"], name="unique_actual_school_day")]

    def __str__(self) -> str:
        return f"{self.school.school_id} · {self.date} · {self.meals_served} served ({self.get_status_display()})"

    @property
    def is_verified(self) -> bool:
        return self.status in self.VERIFIED_STATUSES

    @property
    def is_editable(self) -> bool:
        return self.status in self.EDITABLE_STATUSES

    @property
    def has_pending_correction(self) -> bool:
        return self.proposed_at is not None

    @property
    def prepared_minus_served(self) -> int | None:
        """Planning figure: positive = more prepared than served, negative = fewer. None if prepared is unknown."""
        return None if self.meals_prepared is None else self.meals_prepared - self.meals_served


class ActualMealRevision(models.Model):
    """Every change to an actual record: submission, edit, review decision, correction and publication."""

    class Kind(models.TextChoices):
        SUBMITTED = "submitted", "Submitted"
        EDITED = "edited", "Edited before review"
        VERIFIED = "verified", "Verified"
        REJECTED = "rejected", "Rejected"
        CORRECTION_REQUESTED = "correction_requested", "Correction requested"
        CORRECTION_PROPOSED = "correction_proposed", "Correction submitted for review"
        CORRECTION_APPROVED = "correction_approved", "Correction approved"
        CORRECTION_DECLINED = "correction_declined", "Correction declined"
        PUBLISHED = "published", "Published to forecasting history"

    record = models.ForeignKey(ActualMealRecord, on_delete=models.CASCADE, related_name="revisions")
    kind = models.CharField(max_length=24, choices=Kind.choices)
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    actor_name = models.CharField(max_length=150, blank=True)
    meals_served_before = models.PositiveIntegerField(null=True, blank=True)
    meals_served_after = models.PositiveIntegerField(null=True, blank=True)
    meals_prepared_before = models.PositiveIntegerField(null=True, blank=True)
    meals_prepared_after = models.PositiveIntegerField(null=True, blank=True)
    status_after = models.CharField(max_length=24, blank=True)
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-pk"]

    def __str__(self) -> str:
        return f"{self.record_id} · {self.get_kind_display()}"


class ForecastRun(models.Model):
    class Horizon(models.IntegerChoices):
        NEXT_DAY = 1, "Next school day"
        NEXT_7 = 7, "Next 7 school days"
        NEXT_30 = 30, "Next 30 school days"

    class SelectionMode(models.TextChoices):
        HORIZON = "horizon", "Next school days"
        SINGLE_DAY = "single", "Single day"
        DATE_RANGE = "range", "Date range"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="forecast_runs"
    )
    school = models.ForeignKey(School, null=True, blank=True, on_delete=models.SET_NULL, related_name="forecast_runs")
    # Kept even if no School row exists (e.g. before import_schools has run).
    school_code = models.CharField("school ID", max_length=50)
    school_name = models.CharField(max_length=200, blank=True)
    # For a horizon run, one of Horizon; for a single day or date range, the number of school days estimated.
    horizon_days = models.PositiveSmallIntegerField(choices=Horizon.choices)
    selection_mode = models.CharField(max_length=10, choices=SelectionMode.choices, default=SelectionMode.HORIZON)
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
        return f"{self.school_code} · {self.period_label} · {self.created_at:%Y-%m-%d %H:%M}"

    @property
    def period_label(self) -> str:
        """What was forecast: a horizon ("Next 7 school days") or the selected date(s)."""
        if self.selection_mode == self.SelectionMode.SINGLE_DAY:
            return f"{self.first_forecast_date:%a} {self.first_forecast_date.day} {self.first_forecast_date:%b %Y}"
        if self.selection_mode == self.SelectionMode.DATE_RANGE:
            first, last = self.first_forecast_date, self.last_forecast_date
            return f"{first.day} {first:%b %Y} – {last.day} {last:%b %Y} ({self.horizon_days} school days)"
        return self.get_horizon_days_display()

    @property
    def is_date_selection(self) -> bool:
        return self.selection_mode != self.SelectionMode.HORIZON

    @property
    def is_synthetic(self) -> bool:
        """Produced by a model trained on synthetic development data (not a research result)."""
        return "SYNTHETIC" in (self.results_label or "").upper()
