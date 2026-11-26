"""Recorded actual meals served: submission, review, corrections and forecast comparison.

One `ActualMealRecord` per school-day, shared by every saved forecast covering that day.
Saved forecasts (`ForecastRun`) are never changed: the comparison is computed when shown.

State machine (who may move it):
    Pending review --(admin verify)--> Verified --(admin publish)--> Published
    Pending review / Verified --(admin reject)--> Rejected
    Pending review / Verified / Published --(admin request correction)--> Correction requested
    Correction requested / Rejected --(manager edits)--> Pending review
    Verified / Published --(manager proposes correction)--> same state, correction pending
    correction pending --(admin approves)--> Verified (new values; re-publish to update history)
    correction pending --(admin declines)--> unchanged

Every transition is a conditional update on the current state inside a transaction, so a
double click or a concurrent reviewer changes the record once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from django.db import IntegrityError, transaction
from django.utils import timezone

from services import ServiceError
from web.schools import demand_history
from web.schools.models import ActualMealRecord, ActualMealRevision, ForecastRun, School

Status = ActualMealRecord.Status
Kind = ActualMealRevision.Kind


class ActualError(ServiceError):
    """The requested change is not allowed for the record in its current state."""


class StaleState(ActualError):
    """The record changed since the page was loaded (another reviewer, or a repeated submit)."""


ALREADY_PUBLISHED = "This record was already published; nothing was changed."


@dataclass
class SubmitOutcome:
    record: ActualMealRecord
    created: bool  # False: an existing record for the school-day was updated (or left as it was)
    changed: bool


# --- Manager actions ---------------------------------------------------------------------------


def submit_actual(user, school: School, day: date, meals_served: int, meals_prepared: int | None) -> SubmitOutcome:
    """Record (or, while still editable, update) the school-day's actual. Safe to repeat.

    A verified or published record is never overwritten here: corrections go through review.
    """
    if day > timezone.localdate():
        raise ActualError("Actual meals served cannot be recorded for a future date.")
    with transaction.atomic():
        record = ActualMealRecord.objects.select_for_update().filter(school=school, date=day).first()
        if record is None:
            try:
                with transaction.atomic():
                    record = ActualMealRecord.objects.create(
                        school=school, date=day, meals_served=meals_served, meals_prepared=meals_prepared,
                        submitted_by=user, status=Status.PENDING,
                    )
            except IntegrityError:
                # A concurrent submission created it first: fall through to the update rules.
                record = ActualMealRecord.objects.select_for_update().get(school=school, date=day)
            else:
                demand_history.add_revision(record, Kind.SUBMITTED, user)
                return SubmitOutcome(record, created=True, changed=True)
        return SubmitOutcome(record, created=False, changed=_edit(record, user, meals_served, meals_prepared))


def edit_actual(user, record: ActualMealRecord, meals_served: int, meals_prepared: int | None) -> bool:
    with transaction.atomic():
        record = ActualMealRecord.objects.select_for_update().get(pk=record.pk)
        return _edit(record, user, meals_served, meals_prepared)


def _edit(record: ActualMealRecord, user, meals_served: int, meals_prepared: int | None) -> bool:
    if not record.is_editable:
        raise ActualError(
            f"This day's record is already {record.get_status_display().lower()}. "
            "Submit a correction for review instead of changing it directly."
        )
    same_values = record.meals_served == meals_served and record.meals_prepared == meals_prepared
    if same_values and record.status == Status.PENDING:
        return False  # a repeated submission: nothing to change
    before = (record.meals_served, record.meals_prepared)
    record.meals_served, record.meals_prepared = meals_served, meals_prepared
    resubmitted = record.status != Status.PENDING
    record.status = Status.PENDING
    record.submitted_by, record.submitted_at = user, timezone.now()
    record.save()
    demand_history.add_revision(record, Kind.SUBMITTED if resubmitted else Kind.EDITED, user, before=before)
    return True


def propose_correction(user, record: ActualMealRecord, meals_served: int, meals_prepared: int | None,
                       reason: str) -> bool:
    """Ask for a verified or published value to be changed. The current value stays in use meanwhile."""
    with transaction.atomic():
        record = ActualMealRecord.objects.select_for_update().get(pk=record.pk)
        if record.status not in ActualMealRecord.VERIFIED_STATUSES:
            raise ActualError("Only a verified or published record needs a correction request; edit it directly instead.")
        if (record.has_pending_correction and record.proposed_meals_served == meals_served
                and record.proposed_meals_prepared == meals_prepared):
            return False  # the same correction submitted twice
        if record.meals_served == meals_served and record.meals_prepared == meals_prepared:
            raise ActualError("The correction has the same values as the current record.")
        record.proposed_meals_served, record.proposed_meals_prepared = meals_served, meals_prepared
        record.proposed_reason, record.proposed_by, record.proposed_at = reason, user, timezone.now()
        record.save()
        demand_history.add_revision(
            record, Kind.CORRECTION_PROPOSED, user, before=(record.meals_served, record.meals_prepared),
            comment=f"Proposed {meals_served} served"
                    + (f", {meals_prepared} prepared" if meals_prepared is not None else "") + f". {reason}".rstrip(". ") + ".",
        )
        return True


# --- Administrator actions ---------------------------------------------------------------------

VERIFY, REJECT, REQUEST_CORRECTION, PUBLISH = "verify", "reject", "request_correction", "publish"
APPROVE_CORRECTION, DECLINE_CORRECTION = "approve_correction", "decline_correction"

# Action -> states it may start from.
ALLOWED_FROM = {
    VERIFY: (Status.PENDING,),
    REJECT: (Status.PENDING, Status.VERIFIED),
    REQUEST_CORRECTION: (Status.PENDING, Status.VERIFIED, Status.PUBLISHED),
    PUBLISH: (Status.VERIFIED,),
}
COMMENT_REQUIRED = {REJECT, REQUEST_CORRECTION, DECLINE_CORRECTION}


def available_actions(record: ActualMealRecord) -> list[str]:
    if record.has_pending_correction:
        return [APPROVE_CORRECTION, DECLINE_CORRECTION, REQUEST_CORRECTION]
    return [a for a, states in ALLOWED_FROM.items() if record.status in states]


def review(user, record: ActualMealRecord, action: str, comment: str = "", *, expected_status: str | None = None,
           acknowledge_conflicts: bool = False) -> str:
    """Apply an administrator decision; returns a message. Raises ActualError / StaleState / ConflictsNotAcknowledged.

    StaleState means nothing changed (e.g. a publish retry), so callers must not audit or notify.

    `expected_status` is the status the reviewer saw: if the record moved on since (a second
    click or another reviewer), nothing is changed.
    """
    comment = comment.strip()
    if action in COMMENT_REQUIRED and not comment:
        raise ActualError("Add a comment explaining the decision.")
    if expected_status and record.status != expected_status:
        if action == PUBLISH and record.status == Status.PUBLISHED:
            raise StaleState(ALREADY_PUBLISHED)
        raise StaleState(f"This record is now {record.get_status_display().lower()}; nothing was changed.")
    if action == PUBLISH:
        result = demand_history.publish_actual(record, user, acknowledge_conflicts=acknowledge_conflicts)
        record.refresh_from_db()
        if not result.published:
            raise StaleState(ALREADY_PUBLISHED)
        return "Published to forecasting history. The saved model is unchanged; new forecasts can start from this day."
    if action in (APPROVE_CORRECTION, DECLINE_CORRECTION):
        return _decide_correction(user, record, approve=action == APPROVE_CORRECTION, comment=comment)
    if action not in ALLOWED_FROM:
        raise ActualError("Unknown review action.")

    target = {VERIFY: Status.VERIFIED, REJECT: Status.REJECTED, REQUEST_CORRECTION: Status.CORRECTION_REQUESTED}[action]
    now = timezone.now()
    with transaction.atomic():
        claimed = ActualMealRecord.objects.filter(pk=record.pk, status__in=ALLOWED_FROM[action]).update(
            status=target, reviewed_by=user, reviewed_at=now, review_comment=comment, updated_at=now,
            # A pending correction is superseded by asking the manager to correct the record directly.
            **({"proposed_meals_served": None, "proposed_meals_prepared": None, "proposed_reason": "",
                "proposed_by": None, "proposed_at": None} if action == REQUEST_CORRECTION else {}),
        )
        if not claimed:
            record.refresh_from_db()
            raise StaleState(f"This record is {record.get_status_display().lower()}; nothing was changed.")
        record.refresh_from_db()
        if action == REJECT:
            record.published_at = record.published_by = None
            record.save(update_fields=["published_at", "published_by"])
            demand_history.withdraw_actual(record)
        demand_history.add_revision(record, {VERIFY: Kind.VERIFIED, REJECT: Kind.REJECTED,
                                             REQUEST_CORRECTION: Kind.CORRECTION_REQUESTED}[action], user, comment=comment)
    return {
        VERIFY: "Verified. Publish it to add it to forecasting history.",
        REJECT: "Rejected. It is not used for accuracy figures or forecasting.",
        REQUEST_CORRECTION: "Correction requested. The School Manager can now edit and resubmit it.",
    }[action]


def _decide_correction(user, record: ActualMealRecord, *, approve: bool, comment: str) -> str:
    with transaction.atomic():
        record = ActualMealRecord.objects.select_for_update().get(pk=record.pk)
        if not record.has_pending_correction:
            raise StaleState("There is no pending correction for this record; nothing was changed.")
        before = (record.meals_served, record.meals_prepared)
        if approve:
            record.meals_served, record.meals_prepared = record.proposed_meals_served, record.proposed_meals_prepared
            # Verified again: the corrected value enters forecasting history when it is (re)published.
            record.status = Status.VERIFIED
        record.proposed_meals_served = record.proposed_meals_prepared = record.proposed_by = record.proposed_at = None
        record.proposed_reason = ""
        record.reviewed_by, record.reviewed_at, record.review_comment = user, timezone.now(), comment
        record.save()
        demand_history.add_revision(record, Kind.CORRECTION_APPROVED if approve else Kind.CORRECTION_DECLINED,
                                    user, before=before, comment=comment)
    if approve:
        return ("Correction approved and the record is verified with the new values. "
                "Publish it to update forecasting history" + (" (the earlier published value stays in use until then)."
                                                              if record.published_at else "."))
    return "Correction declined. The record keeps its current values."


# --- Forecast comparison ----------------------------------------------------------------------


def run_day(run: ForecastRun, day: date) -> dict | None:
    """The stored per-day forecast for `day` in a saved run (None if the run does not cover it)."""
    for row in run.result_summary.get("days", []):
        if row.get("date") == day.isoformat():
            return row
    return None


def run_forecast_days(run: ForecastRun) -> list[date]:
    return [date.fromisoformat(row["date"]) for row in run.result_summary.get("days", []) if row.get("date")]


def forecast_error(forecast: int | float | None, actual: int | None) -> dict | None:
    """Recorded minus forecast (positive = more were served than forecast), with the percentage of actual."""
    if forecast is None or actual is None:
        return None
    error = actual - forecast
    return {"error": error, "abs_error": abs(error),
            "pct": round(error / actual * 100, 1) if actual else None,
            "direction": "under" if error > 0 else "over" if error < 0 else "exact"}


def comparison_rows(run: ForecastRun, records: dict[date, ActualMealRecord]) -> list[dict]:
    """One row per forecast day: original forecast, recorded value (if any), error and status."""
    rows = []
    for row in run.result_summary.get("days", []):
        day = date.fromisoformat(row["date"])
        record = records.get(day)
        rows.append({
            "date": day,
            "predicted": row.get("predicted_meals"),
            "recommended": row.get("recommended_meals"),
            "method": row.get("method_label") or run.model_label,
            "record": record,
            "error": forecast_error(row.get("predicted_meals"), record.meals_served if record else None),
        })
    return rows


def accuracy(rows: list[dict]) -> dict | None:
    """Aggregate accuracy from verified or published actuals only (pending values never count)."""
    errors = [r["error"]["abs_error"] for r in rows if r["error"] and r["record"] and r["record"].is_verified]
    if not errors:
        return None
    signed = [r["error"]["error"] for r in rows if r["error"] and r["record"] and r["record"].is_verified]
    return {"n": len(errors), "mae": round(sum(errors) / len(errors), 1), "bias": round(sum(signed) / len(signed), 1)}


def latest_forecasts(school_codes, start: date, end: date, runs) -> dict[tuple[str, date], tuple[ForecastRun, dict]]:
    """For each (school, day) in [start, end]: the most recently saved run covering it, with its day row.

    `runs` is the caller's already-scoped ForecastRun queryset.
    """
    found: dict[tuple[str, date], tuple[ForecastRun, dict]] = {}
    qs = runs.filter(school_code__in=list(school_codes), first_forecast_date__lte=end,
                     last_forecast_date__gte=start).order_by("created_at", "pk")
    for run in qs:  # oldest first, so the newest run wins
        for row in run.result_summary.get("days", []):
            try:
                day = date.fromisoformat(row["date"])
            except (KeyError, TypeError, ValueError):
                continue
            if start <= day <= end:
                found[(run.school_code, day)] = (run, row)
    return found
