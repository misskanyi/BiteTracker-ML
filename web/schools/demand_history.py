"""The demand-history table (`DemandRecord`) and how it feeds forecasting.

Records move Pending review → Verified → Published. Only published rows (stored status
"approved") feed forecasting and training; verifying never changes forecasting data.

- `seed_baseline` copies the processed dataset into the table as published baseline rows.
- `apply_review_decision` runs whenever an administrator decides on an upload: verifying
  writes the upload's rows as verified (not yet used); any other decision gives existing
  rows that status and withdraws a publication, so they stop being used (FR-30, FR-31).
- `publish_upload` / `publish_actual` move verified rows into forecasting history. Both are
  atomic and claim the publication with a conditional update, so a retry or double click
  publishes once. Overlaps with existing history are reported by `upload_conflicts` /
  `actual_conflicts`; differing values must be acknowledged before publishing.
- `DatabaseHistoryProvider` hands only published rows to `services.data_service`, which the
  forecasting service reads (registered in `SchoolsConfig.ready`). The saved model is
  unchanged by publishing: retraining is a separate process.
- `approved_training_frame` gives the training pipeline the same published rows (FR-33).
"""

from __future__ import annotations

import math
from pathlib import Path

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from django.db import transaction
from django.db.models import Count, Max, Q
from django.utils import timezone

from config import settings as ml_settings
from services import ServiceError, data_service
from src.data_validation import load_uploaded_file
from web.schools.models import ActualMealRecord, ActualMealRevision, DatasetUpload, DemandRecord

# Rows that are neither from an upload nor from a recorded actual: the retained historical dataset.
BASELINE = Q(upload__isnull=True, actual__isnull=True)

TEXT_FIELDS = ("school_id", "school_name", "county", "sub_county", "school_term")
NUMBER_FIELDS = (
    "enrolled_students", "students_present", "students_absent", "meals_prepared", "meals_served", "leftover_meals",
    "average_cost_per_meal", "rainfall_mm", "temperature_c", "is_holiday", "is_exam_period", "is_school_event",
)
FIELDS = ("date",) + TEXT_FIELDS + NUMBER_FIELDS
BATCH_SIZE = 1000


class DemandHistoryError(Exception):
    """An approved upload's file could not be turned into demand rows."""


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return ""
    return str(value).strip()


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _records(frame: pd.DataFrame, *, status: str, upload: DatasetUpload | None = None) -> list[DemandRecord]:
    """DemandRecord objects for every row with a parsable date and a school ID (unsaved)."""
    frame = frame.copy()
    frame[ml_settings.DATE_COLUMN] = pd.to_datetime(frame[ml_settings.DATE_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[ml_settings.DATE_COLUMN])
    records = []
    for row in frame.to_dict(orient="records"):
        school = _text(row.get("school_id"))
        if not school:
            continue
        values = {f: _text(row.get(f)) for f in TEXT_FIELDS} | {f: _number(row.get(f)) for f in NUMBER_FIELDS}
        records.append(DemandRecord(upload=upload, status=status, date=row["date"].date(), **values))
    return records


def seed_baseline(path: Path | None = None, *, replace: bool = False) -> int:
    """Copy the processed dataset into the table as approved baseline rows; return rows written.

    Does nothing if a baseline already exists, unless `replace` (e.g. after a model release ships a
    new processed dataset), which swaps the baseline in one transaction. Upload rows are untouched.
    """
    path = path or data_service.processed_data_path()
    if not Path(path).exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}.")
    with transaction.atomic():
        baseline = DemandRecord.objects.filter(BASELINE)
        if baseline.exists():
            if not replace:
                return 0
            baseline.delete()
        # Duplicate school-days cannot exist in the processed file; keep the first if they ever do.
        frame = pd.read_csv(path).drop_duplicates([ml_settings.SCHOOL_ID_COLUMN, ml_settings.DATE_COLUMN])
        records = _records(frame, status=DemandRecord.Status.APPROVED)
        DemandRecord.objects.bulk_create(records, batch_size=BATCH_SIZE)
    data_service.clear_data_cache()
    return len(records)


def apply_review_decision(upload: DatasetUpload) -> int:
    """Bring the upload's demand rows in line with its review status; return rows written or updated.

    Verified ("approved"): the upload's rows are stored once as verified, not yet used for
    forecasting (re-verifying reuses them). Publishing is a separate step (`publish_upload`).
    Any other status: rows take that status and any publication is withdrawn, so they stop being used.
    """
    rows = DemandRecord.objects.filter(upload=upload)
    if upload.review_status == DatasetUpload.ReviewStatus.APPROVED:
        if rows.exists():
            target = DemandRecord.Status.APPROVED if upload.published_at else DemandRecord.Status.VERIFIED
            changed = rows.update(status=target)
        else:
            changed = _write_upload_rows(upload)
    else:
        changed = rows.update(status=upload.review_status or DemandRecord.Status.PENDING, published_at=None)
        if upload.published_at or upload.published_by_id:
            upload.published_at = upload.published_by = None
            upload.save(update_fields=["published_at", "published_by"])
    processing = (DatasetUpload.ProcessingStatus.PROCESSED
                  if upload.review_status == DatasetUpload.ReviewStatus.APPROVED and upload.published_at
                  else DatasetUpload.ProcessingStatus.NOT_PROCESSED)
    if upload.processing_status != processing:
        upload.processing_status = processing
        upload.save(update_fields=["processing_status"])
    data_service.clear_data_cache()
    return changed


def _write_upload_rows(upload: DatasetUpload) -> int:
    if not upload.file:
        raise DemandHistoryError("The approved upload has no stored data file.")
    with upload.file.open("rb") as handle:
        content = handle.read()
    try:
        frame = load_uploaded_file(content, upload.original_filename or upload.file.name)
    except Exception as exc:  # parser errors vary by library
        raise DemandHistoryError(f"The approved upload's data file could not be read: {exc}") from exc
    frame.columns = [str(c).strip() for c in frame.columns]
    # Within one file the first record of a school-day counts, as in training-data cleaning.
    frame = frame.drop_duplicates([ml_settings.SCHOOL_ID_COLUMN, ml_settings.DATE_COLUMN])
    records = _records(frame, status=DemandRecord.Status.VERIFIED, upload=upload)
    DemandRecord.objects.bulk_create(records, batch_size=BATCH_SIZE)
    return len(records)


# --- Publishing into forecasting history ------------------------------------------------------


class PublishError(ServiceError):
    """The record cannot be published in its current state."""


class ConflictsNotAcknowledged(PublishError):
    """Publishing would replace existing history values that differ; the reviewer must confirm."""

    def __init__(self, conflicts: list[dict]):
        self.conflicts = conflicts
        super().__init__(
            f"{len(conflicts)} school-day(s) already have a different value in forecasting history. "
            "Review them and confirm to publish; the newly published values will replace them."
        )


@dataclass
class PublishResult:
    published: bool  # False: it was already published (a retry or a second click), nothing changed
    rows: int = 0
    conflicts: list[dict] = field(default_factory=list)


def _history_values(pairs: set[tuple[str, date]]) -> dict[tuple[str, date], float]:
    """Current forecasting-history meals_served for the given (school_id, date) pairs."""
    if not pairs:
        return {}
    try:
        frame = data_service.load_processed_data()
    except ServiceError:
        return {}
    schools = {p[0] for p in pairs}
    frame = frame[frame[ml_settings.SCHOOL_ID_COLUMN].isin(schools)]
    values = {}
    for school_id, day, meals in zip(frame[ml_settings.SCHOOL_ID_COLUMN], frame[ml_settings.DATE_COLUMN].dt.date,
                                     frame["meals_served"]):
        if (school_id, day) in pairs and not pd.isna(meals):
            values[(school_id, day)] = float(meals)
    return values


def _differs(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) > 0.5


def upload_conflicts(upload: DatasetUpload) -> list[dict]:
    """School-days in a verified upload that overlap existing data, flagged when the values differ.

    Compared with the current forecasting history (baseline, published uploads and published
    actuals) and with recorded actuals still in review. `conflict` is True when meals served differ.
    """
    rows = list(DemandRecord.objects.filter(upload=upload).values_list("school_id", "date", "meals_served"))
    if not rows:
        return []
    pairs = {(s, d) for s, d, _ in rows}
    history = _history_values(pairs)
    actuals = {
        (a.school.school_id, a.date): a
        for a in ActualMealRecord.objects.select_related("school")
        .filter(school__school_id__in={s for s, _ in pairs}, date__in={d for _, d in pairs})
        .exclude(status=ActualMealRecord.Status.REJECTED)
    }
    found = []
    for school_id, day, meals in sorted(rows, key=lambda r: (r[0], r[1])):
        existing = history.get((school_id, day))
        actual = actuals.get((school_id, day))
        if existing is None and actual is None:
            continue
        found.append({
            "school_id": school_id, "date": day, "new_value": meals,
            "history_value": existing,
            "actual_value": actual.meals_served if actual else None,
            "actual_status": actual.get_status_display() if actual else "",
            "conflict": _differs(meals, existing) or _differs(meals, actual.meals_served if actual else None),
        })
    return found


def actual_conflicts(record: ActualMealRecord) -> list[dict]:
    """The record's school-day in current forecasting history, flagged when the value differs.

    The record's own earlier publication is not a conflict with itself.
    """
    key = (record.school.school_id, record.date)
    existing = _history_values({key}).get(key)
    if existing is None:
        return []
    own = DemandRecord.objects.filter(actual=record, status=DemandRecord.Status.APPROVED).first()
    if own is not None and not _differs(own.meals_served, existing):
        return []  # only this record's own earlier value is in history
    return [{"school_id": key[0], "date": key[1], "new_value": record.meals_served, "history_value": existing,
             "actual_value": None, "actual_status": "", "conflict": _differs(record.meals_served, existing)}]


def publish_upload(upload: DatasetUpload, user, *, acknowledge_conflicts: bool = False) -> PublishResult:
    """Publish a verified upload's rows into forecasting history, once."""
    if upload.review_status != DatasetUpload.ReviewStatus.APPROVED:
        raise PublishError("Only a verified upload can be published.")
    if upload.published_at:
        return PublishResult(published=False)
    conflicts = upload_conflicts(upload)
    if any(c["conflict"] for c in conflicts) and not acknowledge_conflicts:
        raise ConflictsNotAcknowledged([c for c in conflicts if c["conflict"]])
    now = timezone.now()
    with transaction.atomic():
        claimed = DatasetUpload.objects.filter(
            pk=upload.pk, review_status=DatasetUpload.ReviewStatus.APPROVED, published_at__isnull=True,
        ).update(published_at=now, published_by=user, processing_status=DatasetUpload.ProcessingStatus.PROCESSED)
        if not claimed:
            upload.refresh_from_db()
            return PublishResult(published=False)
        rows = DemandRecord.objects.filter(upload=upload, status=DemandRecord.Status.VERIFIED).update(
            status=DemandRecord.Status.APPROVED, published_at=now)
    upload.refresh_from_db()
    data_service.clear_data_cache()
    return PublishResult(published=True, rows=rows, conflicts=conflicts)


def publish_actual(record: ActualMealRecord, user, *, acknowledge_conflicts: bool = False) -> PublishResult:
    """Publish a verified actual into forecasting history, once (re-publishing after an approved correction updates it)."""
    if record.status == ActualMealRecord.Status.PUBLISHED:
        return PublishResult(published=False)
    if record.status != ActualMealRecord.Status.VERIFIED:
        raise PublishError("Only a verified record can be published.")
    if record.has_pending_correction:
        raise PublishError("Decide on the pending correction before publishing.")
    conflicts = actual_conflicts(record)
    if any(c["conflict"] for c in conflicts) and not acknowledge_conflicts:
        raise ConflictsNotAcknowledged(conflicts)
    now = timezone.now()
    school = record.school
    with transaction.atomic():
        claimed = ActualMealRecord.objects.filter(
            pk=record.pk, status=ActualMealRecord.Status.VERIFIED, proposed_at__isnull=True,
        ).update(status=ActualMealRecord.Status.PUBLISHED, published_at=now, published_by=user, updated_at=now)
        if not claimed:
            record.refresh_from_db()
            return PublishResult(published=False)
        DemandRecord.objects.update_or_create(actual=record, defaults=dict(
            status=DemandRecord.Status.APPROVED, published_at=now, upload=None,
            school_id=school.school_id, date=record.date, school_name=school.name,
            county=school.county, sub_county=school.sub_county,
            meals_served=float(record.meals_served),
            meals_prepared=None if record.meals_prepared is None else float(record.meals_prepared),
            leftover_meals=None if record.meals_prepared is None else float(record.meals_prepared - record.meals_served),
        ))
        record.refresh_from_db()
        add_revision(record, ActualMealRevision.Kind.PUBLISHED, user,
                     comment=f"{len([c for c in conflicts if c['conflict']])} conflicting value(s) acknowledged"
                     if any(c["conflict"] for c in conflicts) else "")
    data_service.clear_data_cache()
    return PublishResult(published=True, rows=1, conflicts=conflicts)


def withdraw_actual(record: ActualMealRecord) -> None:
    """Take a rejected record's published row out of forecasting history."""
    if DemandRecord.objects.filter(actual=record).update(status=DemandRecord.Status.REJECTED, published_at=None):
        data_service.clear_data_cache()


def add_revision(record: ActualMealRecord, kind: str, actor, *, before: tuple | None = None, comment: str = "") -> ActualMealRevision:
    """Append to the record's history. `before` is (meals_served, meals_prepared) prior to the change."""
    actor = actor if getattr(actor, "is_authenticated", False) else None
    served_before, prepared_before = before if before else (None, None)
    return ActualMealRevision.objects.create(
        record=record, kind=kind, actor=actor, actor_name=actor.get_username() if actor else "",
        meals_served_before=served_before, meals_prepared_before=prepared_before,
        meals_served_after=record.meals_served, meals_prepared_after=record.meals_prepared,
        status_after=record.status, comment=comment,
    )


def latest_usable_dates(school_codes=None) -> dict[str, date]:
    """Latest date in forecasting history per school: the newest record a forecast can start from."""
    try:
        frame = data_service.load_processed_data()
    except ServiceError:
        return {}
    if school_codes is not None:
        frame = frame[frame[ml_settings.SCHOOL_ID_COLUMN].isin(set(school_codes))]
    latest = frame.groupby(ml_settings.SCHOOL_ID_COLUMN)[ml_settings.DATE_COLUMN].max()
    return {code: ts.date() for code, ts in latest.items()}


def _frame(queryset) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(queryset.values(*FIELDS), columns=list(FIELDS))
    frame[ml_settings.DATE_COLUMN] = pd.to_datetime(frame[ml_settings.DATE_COLUMN])
    for col in NUMBER_FIELDS:
        frame[col] = pd.to_numeric(frame[col])
    for col in TEXT_FIELDS:
        frame[col] = frame[col].replace("", pd.NA)
    return frame


def approved_baseline() -> pd.DataFrame | None:
    rows = DemandRecord.objects.filter(BASELINE, status=DemandRecord.Status.APPROVED)
    return _frame(rows.order_by("school_id", "date")) if rows.exists() else None


def approved_uploads() -> pd.DataFrame:
    """Published rows beyond the baseline (verified uploads and recorded actuals), oldest publication
    first, so a later publication wins for the same school-day when combined."""
    rows = DemandRecord.objects.filter(status=DemandRecord.Status.APPROVED).filter(
        Q(upload__isnull=False, upload__review_status=DatasetUpload.ReviewStatus.APPROVED,
          upload__published_at__isnull=False)
        # An actual's row stays published while a correction is reviewed; rejection withdraws it.
        | Q(actual__isnull=False)
    )
    return _frame(rows.order_by("published_at", "pk"))


class DatabaseHistoryProvider:
    """`services.data_service.HistoryProvider` backed by `DemandRecord`."""

    def version(self) -> tuple:
        stats = DemandRecord.objects.aggregate(
            baseline=Count("pk", filter=BASELINE & Q(status=DemandRecord.Status.APPROVED)),
            total=Count("pk"), latest=Max("updated_at"), last_id=Max("pk"),
            approved=Count("pk", filter=Q(status=DemandRecord.Status.APPROVED)),
        )
        return (stats["baseline"], stats["approved"], stats["total"], stats["latest"], stats["last_id"])

    def approved_frames(self) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        return approved_baseline(), approved_uploads()


def approved_training_frame() -> pd.DataFrame:
    """Every approved row (baseline + approved uploads) in the raw schema the training pipeline expects.

    Upload rows are only cleaned statelessly here (no saved imputation or outlier settings), because
    training fits those afresh on its own training period. Without a seeded baseline the processed
    CSV stands in for it.
    """
    from src.synthetic_data import SCHEMA_COLUMNS

    baseline = approved_baseline()
    if baseline is None:
        baseline = pd.read_csv(data_service.processed_data_path(), parse_dates=[ml_settings.DATE_COLUMN])
    frame = data_service.combine_history(baseline, approved_uploads())
    frame[ml_settings.DATE_COLUMN] = pd.to_datetime(frame[ml_settings.DATE_COLUMN])
    frame["day_of_week"] = frame[ml_settings.DATE_COLUMN].dt.day_name()
    frame["month"] = frame[ml_settings.DATE_COLUMN].dt.month
    for col in SCHEMA_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame[SCHEMA_COLUMNS].sort_values([ml_settings.SCHOOL_ID_COLUMN, ml_settings.DATE_COLUMN]).reset_index(drop=True)
