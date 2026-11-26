"""Forecast vs actual: recording meals served, reviewing and publishing them, and the follow-up CSV.

Access (enforced here on every request; see web.accounts.access):
- Viewing: any approved user, for schools in their scope. Out-of-scope runs, records and
  schools return 404 so their existence is not revealed.
- Recording, editing and proposing corrections: School Managers, assigned schools only (403 otherwise).
- Verifying, rejecting, requesting/approving corrections and publishing: System Administrators only (403).
- Export: schools in scope only; asking for any other school returns 404.
Pending, rejected and suspended accounts are stopped earlier by AccountStatusMiddleware.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from urllib.parse import urlencode

from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from web.accounts import access
from web.audit.models import AuditLog
from web.audit.services import log_action
from web.core.forms import ActualCorrectionForm, ActualEditForm, ActualMealsForm, ActualReviewForm, FollowUpFilterForm
from web.notifications.models import Notification
from web.notifications.services import notify, system_admins
from web.schools import actuals, demand_history
from web.schools.models import ActualMealRecord, ForecastRun, School

DEFAULT_PERIOD_DAYS = 60
PAGE_SIZE = 25


def _visible_runs(user):
    from web.core.views import _visible_runs as visible

    return visible(user)


# --- Saved forecast page section -----------------------------------------------------------------


def forecast_vs_actual_context(request, run: ForecastRun, actual_form=None) -> dict:
    """Context for the "Forecast vs actual" section of a saved forecast."""
    today = timezone.localdate()
    days = actuals.run_forecast_days(run)
    school = School.objects.filter(school_id=run.school_code).first()
    records = {}
    if school is not None and days:
        records = {r.date: r for r in ActualMealRecord.objects.filter(school=school, date__in=days)}
    rows = actuals.comparison_rows(run, records)
    can_record = school is not None and access.can_record_actuals(request.user, run.school_code)
    recordable = [d for d in days if d <= today]
    if can_record and actual_form is None and recordable:
        default = next((d for d in recordable if d not in records), recordable[0])
        actual_form = ActualMealsForm(days=days, today=today, initial={"date": default.isoformat()})
    return {
        "fva_rows": rows,
        "fva_accuracy": actuals.accuracy(rows),
        "fva_pending": sum(1 for r in rows if r["record"] and not r["record"].is_verified),
        "fva_can_record": can_record,
        "fva_recordable": bool(recordable),
        "fva_no_school": school is None,
        "actual_form": actual_form if can_record else None,
    }


@require_POST
def record_actual(request, pk: int):
    run = get_object_or_404(_visible_runs(request.user), pk=pk)  # 404 outside scope
    if not access.can_record_actuals(request.user, run.school_code):
        raise PermissionDenied
    school = School.objects.filter(school_id=run.school_code).first()
    if school is None:
        raise Http404("This school has no school record to attach actuals to.")
    form = ActualMealsForm(request.POST, days=actuals.run_forecast_days(run), today=timezone.localdate())
    if not form.is_valid():
        from web.core.views import render_forecast_run

        return render_forecast_run(request, run, actual_form=form, status=400)
    data = form.cleaned_data
    try:
        outcome = actuals.submit_actual(request.user, school, data["date"], data["meals_served"], data["meals_prepared"])
    except actuals.ActualError as exc:
        form.add_error(None, str(exc))
        from web.core.views import render_forecast_run

        return render_forecast_run(request, run, actual_form=form, status=400)
    record = outcome.record
    if outcome.changed:
        log_action(request.user, AuditLog.Action.ACTUAL_SUBMITTED if outcome.created else AuditLog.Action.ACTUAL_UPDATED,
                   target=record, school=school, details=_values(record) | {"forecast_run": run.pk})
        _notify_reviewers(request.user, record, "submitted" if outcome.created else "updated")
        messages.success(request, f"Recorded {record.meals_served} meals served on {record.date:%d %b %Y}. "
                                  "It is pending review and does not count towards accuracy until verified.")
    elif outcome.created is False and record.is_editable:
        messages.info(request, "That value was already recorded; nothing was changed.")
    return redirect(reverse("core:forecast_run", args=[run.pk]) + "#forecast-vs-actual")


def _values(record: ActualMealRecord) -> dict:
    return {"date": record.date.isoformat(), "meals_served": record.meals_served,
            "meals_prepared": record.meals_prepared, "status": record.status}


def _notify_reviewers(user, record: ActualMealRecord, what: str):
    notify(system_admins(), Notification.Type.DATASET_REVIEW,
           f"Actual meals served {what}: {record.school.school_id} {record.date:%d %b %Y}",
           f"{record.meals_served} meals served recorded by {user.get_username()}; awaiting review.",
           url=reverse("core:actual_detail", args=[record.pk]), actor=user, school=record.school)


# --- Record detail: edit, correction, review ---------------------------------------------------


def _visible_record(request, pk: int) -> ActualMealRecord:
    return get_object_or_404(access.visible_actuals(request.user), pk=pk)


def _forecasts_for(request, record: ActualMealRecord) -> list[dict]:
    """Every saved forecast the user may see that covers this school-day, with its original value."""
    rows = []
    runs = _visible_runs(request.user).filter(
        school_code=record.school.school_id, first_forecast_date__lte=record.date, last_forecast_date__gte=record.date)
    for run in runs.order_by("-created_at"):
        day = actuals.run_day(run, record.date)
        if day is not None:
            rows.append({"run": run, "predicted": day.get("predicted_meals"), "recommended": day.get("recommended_meals"),
                         "method": day.get("method_label") or run.model_label,
                         "error": actuals.forecast_error(day.get("predicted_meals"), record.meals_served)})
    return rows


def _detail_context(request, record, **forms) -> dict:
    is_manager = access.can_record_actuals(request.user, record.school.school_id)
    is_reviewer = access.can_review_actuals(request.user)
    context = {
        "record": record,
        "forecasts": _forecasts_for(request, record),
        "revisions": record.revisions.all()[:50],
        "can_edit": is_manager and record.is_editable,
        "can_correct": is_manager and record.status in ActualMealRecord.VERIFIED_STATUSES,
        "is_reviewer": is_reviewer,
        "review_actions": actuals.available_actions(record) if is_reviewer else [],
        "published_row": record.demand_record if hasattr(record, "demand_record") else None,
        "latest_usable": demand_history.latest_usable_dates([record.school.school_id]).get(record.school.school_id),
    }
    if is_reviewer and actuals.PUBLISH in context["review_actions"]:
        context["conflicts"] = demand_history.actual_conflicts(record)
    if context["can_edit"]:
        context["edit_form"] = forms.get("edit_form") or ActualEditForm(
            initial={"meals_served": record.meals_served, "meals_prepared": record.meals_prepared})
    if context["can_correct"]:
        context["correction_form"] = forms.get("correction_form") or ActualCorrectionForm(
            initial={"meals_served": record.proposed_meals_served if record.has_pending_correction else record.meals_served,
                     "meals_prepared": record.meals_prepared})
    context["review_form"] = forms.get("review_form")
    return context


@require_http_methods(["GET", "POST"])
def actual_detail(request, pk: int):
    record = _visible_record(request, pk)
    if request.method == "GET":
        return render(request, "core/actual_detail.html", _detail_context(request, record))

    # Manager edits and corrections.
    if not access.can_record_actuals(request.user, record.school.school_id):
        raise PermissionDenied
    action = request.POST.get("action")
    if action == "edit":
        form = ActualEditForm(request.POST)
        if form.is_valid():
            try:
                changed = actuals.edit_actual(request.user, record, form.cleaned_data["meals_served"],
                                              form.cleaned_data["meals_prepared"])
            except actuals.ActualError as exc:
                form.add_error(None, str(exc))
            else:
                record.refresh_from_db()
                if changed:
                    log_action(request.user, AuditLog.Action.ACTUAL_UPDATED, target=record, school=record.school,
                               details=_values(record))
                    _notify_reviewers(request.user, record, "resubmitted")
                    messages.success(request, "Saved. The record is pending review.")
                else:
                    messages.info(request, "No changes; nothing was saved.")
                return redirect("core:actual_detail", pk=pk)
        record.refresh_from_db()
        return render(request, "core/actual_detail.html", _detail_context(request, record, edit_form=form), status=400)
    if action == "correction":
        form = ActualCorrectionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            try:
                changed = actuals.propose_correction(request.user, record, data["meals_served"], data["meals_prepared"],
                                                     data["reason"].strip())
            except actuals.ActualError as exc:
                form.add_error(None, str(exc))
            else:
                record.refresh_from_db()
                if changed:
                    log_action(request.user, AuditLog.Action.ACTUAL_UPDATED, target=record, school=record.school,
                               details={"correction_proposed": data["meals_served"], "date": record.date.isoformat()})
                    _notify_reviewers(request.user, record, "correction submitted")
                    messages.success(request, "Correction submitted for review. The current value stays in use until it is approved.")
                else:
                    messages.info(request, "That correction is already awaiting review; nothing was changed.")
                return redirect("core:actual_detail", pk=pk)
        record.refresh_from_db()
        return render(request, "core/actual_detail.html", _detail_context(request, record, correction_form=form),
                      status=400)
    return redirect("core:actual_detail", pk=pk)


@access.actual_review_required
@require_POST
def actual_review(request, pk: int):
    record = _visible_record(request, pk)
    form = ActualReviewForm(request.POST)
    if not form.is_valid():
        return redirect("core:actual_detail", pk=pk)
    data = form.cleaned_data
    try:
        message = actuals.review(request.user, record, data["action"], data["comment"],
                                 expected_status=data["expected_status"] or None,
                                 acknowledge_conflicts=data["acknowledge_conflicts"])
    except demand_history.ConflictsNotAcknowledged as exc:
        messages.error(request, str(exc))
        return redirect("core:actual_detail", pk=pk)
    except actuals.StaleState as exc:
        messages.info(request, str(exc))
        return redirect("core:actual_detail", pk=pk)
    except (actuals.ActualError, demand_history.PublishError) as exc:
        messages.error(request, str(exc))
        return redirect("core:actual_detail", pk=pk)
    record.refresh_from_db()
    log_action(request.user, AuditLog.Action.ACTUAL_PUBLISHED if data["action"] == actuals.PUBLISH
               else AuditLog.Action.ACTUAL_REVIEWED, target=record, school=record.school,
               target_user=record.submitted_by,
               details=_values(record) | {"action": data["action"], "comment": data["comment"].strip()})
    if record.submitted_by:
        notify([record.submitted_by], Notification.Type.DATASET_REVIEW,
               f"Actual meals served {record.get_status_display().lower()}: {record.school.school_id} {record.date:%d %b %Y}",
               data["comment"].strip() or message, url=reverse("core:actual_detail", args=[record.pk]),
               actor=request.user, school=record.school)
    messages.success(request, message)
    return redirect("core:actual_detail", pk=pk)


# --- Overview and export -----------------------------------------------------------------------


def _scope_choices(user) -> list[tuple[str, str]]:
    return [(s.school_id, f"{s.name or s.school_id} ({s.school_id})")
            for s in access.accessible_schools(user).order_by("name", "school_id")]


def _selection(request):
    """(form, school codes, start, end, status) for the request; 404 if any school is outside scope."""
    choices = _scope_choices(request.user)
    allowed = {code for code, _ in choices}
    requested = request.GET.getlist("school")
    if any(code not in allowed for code in requested):
        raise Http404  # a school outside the user's scope: not revealed, not exported
    form = FollowUpFilterForm(request.GET or None, schools=choices, statuses=ActualMealRecord.Status.choices)
    today = timezone.localdate()
    codes, start, end, status = sorted(allowed), today - timedelta(days=DEFAULT_PERIOD_DAYS), today, ""
    if request.GET and form.is_valid():
        data = form.cleaned_data
        codes = data["school"] or codes
        start, end = data["start"] or start, data["end"] or end
        if data["start"] and not data["end"]:
            end = max(start, today)
        if data["end"] and not data["start"]:
            start = end - timedelta(days=DEFAULT_PERIOD_DAYS)
        status = data["status"]
    return form, codes, start, end, status


def _freshness(user, codes) -> list[dict]:
    latest = demand_history.latest_usable_dates(codes)
    counts = {}
    for code, status in ActualMealRecord.objects.filter(school__school_id__in=codes).values_list("school__school_id", "status"):
        counts.setdefault(code, {}).setdefault(status, 0)
        counts[code][status] += 1
    schools = {s.school_id: s for s in School.objects.filter(school_id__in=codes)}
    return [{"code": c, "school": schools.get(c), "latest": latest.get(c), "counts": counts.get(c, {})} for c in codes]


def actuals_overview(request):
    form, codes, start, end, status = _selection(request)
    if not codes:
        return render(request, "core/actuals.html", {"form": form, "no_schools": True})
    records = access.visible_actuals(request.user).filter(school__school_id__in=codes, date__range=(start, end))
    if status:
        records = records.filter(status=status)
    forecasts = actuals.latest_forecasts(codes, start, end, _visible_runs(request.user))
    rows = []
    for record in records.order_by("-date", "school__school_id"):
        run, day = forecasts.get((record.school.school_id, record.date), (None, None))
        predicted = day.get("predicted_meals") if day else None
        rows.append({"record": record, "run": run, "predicted": predicted,
                     "error": actuals.forecast_error(predicted, record.meals_served)})
    accuracy = actuals.accuracy([{"record": r["record"], "error": r["error"]} for r in rows])
    page = Paginator(rows, PAGE_SIZE).get_page(request.GET.get("page"))
    query = request.GET.copy()
    query.pop("page", None)
    export_query = urlencode([("school", c) for c in codes] + [("start", start.isoformat()), ("end", end.isoformat())])
    return render(request, "core/actuals.html", {
        "form": form, "page": page, "start": start, "end": end, "accuracy": accuracy,
        "freshness": _freshness(request.user, codes), "querystring": query.urlencode(),
        "export_url": reverse("core:followup_export") + "?" + export_query,
        "is_reviewer": access.can_review_actuals(request.user),
        "pending_count": access.visible_actuals(request.user).filter(
            school__school_id__in=codes, status=ActualMealRecord.Status.PENDING).count(),
    })


FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

EXPORT_COLUMNS = [
    "school_code", "date", "recorded_meals_served", "recorded_value_status", "recorded_meals_prepared",
    "forecast_demand_estimate", "planned_quantity", "planned_quantity_source",
    "surplus_positive_or_shortfall_negative", "surplus_or_shortfall_basis",
    "forecast_run_id", "forecast_method", "latest_usable_record_date",
]


def spreadsheet_safe(value):
    """Text cells that a spreadsheet would treat as a formula are prefixed with an apostrophe."""
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def _blank(value):
    return "" if value is None else value


def followup_rows(user, codes, start: date, end: date) -> list[dict]:
    """One row per school-day in the selection that has a recorded actual or a saved forecast.

    Recorded counts and forecast estimates are separate columns. Rejected actuals are left out.
    Contains no learner identifiers, supporting letters or review comments.
    """
    records = {(r.school.school_id, r.date): r for r in access.visible_actuals(user).filter(
        school__school_id__in=codes, date__range=(start, end)).exclude(status=ActualMealRecord.Status.REJECTED)}
    forecasts = actuals.latest_forecasts(codes, start, end, _visible_runs(user))
    latest = demand_history.latest_usable_dates(codes)
    rows = []
    for code, day in sorted(set(records) | set(forecasts)):
        record = records.get((code, day))
        run, forecast = forecasts.get((code, day), (None, None))
        predicted = forecast.get("predicted_meals") if forecast else None
        recommended = forecast.get("recommended_meals") if forecast else None
        if record is not None and record.meals_prepared is not None:
            planned, planned_source = record.meals_prepared, "recorded meals prepared"
        elif recommended is not None:
            planned, planned_source = recommended, "forecast recommended preparation (estimate)"
        else:
            planned, planned_source = None, ""
        if planned is not None and record is not None:
            gap, basis = planned - record.meals_served, "recorded meals served"
        elif planned is not None and predicted is not None:
            gap, basis = planned - predicted, "forecast demand (estimate)"
        else:
            gap, basis = None, ""
        rows.append({
            "school_code": code,
            "date": day.isoformat(),
            "recorded_meals_served": _blank(record.meals_served if record else None),
            "recorded_value_status": record.get_status_display() if record else "",
            "recorded_meals_prepared": _blank(record.meals_prepared if record else None),
            "forecast_demand_estimate": _blank(predicted),
            "planned_quantity": _blank(planned),
            "planned_quantity_source": planned_source,
            "surplus_positive_or_shortfall_negative": _blank(gap),
            "surplus_or_shortfall_basis": basis,
            "forecast_run_id": _blank(run.pk if run else None),
            "forecast_method": (forecast.get("method_label") or run.model_label) if run else "",
            "latest_usable_record_date": latest[code].isoformat() if code in latest else "",
        })
    return rows


def followup_export(request):
    _, codes, start, end, _ = _selection(request)
    rows = followup_rows(request.user, codes, start, end)
    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="bitetracker_followup_{start:%Y%m%d}_{end:%Y%m%d}.csv"'
    response.write("﻿")  # so spreadsheet programs read the file as UTF-8
    writer = csv.DictWriter(response, fieldnames=EXPORT_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: spreadsheet_safe(v) for k, v in row.items()})
    log_action(request.user, AuditLog.Action.FOLLOWUP_EXPORTED, target_label="Follow-up report",
               details={"schools": list(codes), "start": start.isoformat(), "end": end.isoformat(), "rows": len(rows)})
    return response
