"""BiteTracker page views.

Views stay thin: they validate input with forms, call the framework-independent
services, and render templates. No forecasting, feature-engineering or
decision-support logic lives here, and no model is ever trained in a request.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from urllib.parse import urlencode

from django.contrib.auth.decorators import login_not_required
from django.core.paginator import Paginator
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from config import settings as ml_settings
from services import (ServiceError, data_service, donor_service, forecast_service, model_service, performance_service,
                      planning_service, status_service)
from web.accounts import access
from web.core import charts
from web.core.errors import user_message
from web.core.forecast_records import record_forecast_run, stored_cost_per_meal, stored_days
from web.core.forms import ForecastForm, HistoryFilterForm
from web.notifications.models import Notification
from web.notifications.services import notify
from web.schools.models import ForecastRun, School

logger = logging.getLogger(__name__)

NO_SCHOOLS_MESSAGE = (
    "No schools are assigned to your account yet, so there is no school data to show. "
    "Ask a System Administrator to assign your schools."
)

DEFAULT_HISTORY_DAYS = 90
HISTORY_PAGE_SIZE = 25
HISTORY_TABLE_COLUMNS = [
    ("meals_served", "Meals served"),
    ("students_present", "Students present"),
    ("enrolled_students", "Enrolled (dataset)"),
    ("rainfall_mm", "Rainfall (mm)"),
    ("temperature_c", "Temp (°C)"),
]


def _baselines_beating_model(metadata: dict) -> list[dict]:
    """Saved naive baselines whose held-out test MAE is lower than the selected model's, best first."""
    rows = []
    for row in metadata.get("naive_baseline_comparison") or []:
        try:
            baseline_mae, model_mae = float(row["Baseline Test MAE"]), float(row["Model Test MAE"])
        except (KeyError, TypeError, ValueError):
            continue
        if baseline_mae < model_mae:
            rows.append({"name": row.get("Naive Baseline", "A naive baseline"), "test_mae": baseline_mae, "model_test_mae": model_mae})
    return sorted(rows, key=lambda r: r["test_mae"])


def _visible_runs(user):
    """Forecast runs the user may open.

    System Administrators: every run (synthetic runs are labelled in the UI).
    Everyone else: runs for their assigned schools only and, unless the active model is itself
    synthetic, never synthetic development runs, so old test forecasts are not mistaken for
    research results.
    """
    runs = ForecastRun.objects.select_related("user")
    codes = access.accessible_school_codes(user)
    if codes is None:
        return runs
    runs = runs.filter(school_code__in=codes)
    if status_service.get_data_status()["kind"] != "synthetic":
        runs = runs.exclude(results_label__icontains="SYNTHETIC")
    return runs


def _schools_for(user) -> list[dict]:
    """Dataset schools within the user's scope. Raises ServiceError if the data is unavailable."""
    return access.filter_school_records(user, data_service.list_schools())


def _no_school_data_message(user) -> str:
    """Why a non-administrator sees no school data: no assignments, or assignments with no records
    in the active dataset (e.g. a legacy synthetic school)."""
    assigned = list(access.accessible_schools(user).order_by("name", "school_id"))
    if not assigned:
        return NO_SCHOOLS_MESSAGE
    names = ", ".join(f"{s.name or s.school_id} ({s.school_id})" for s in assigned)
    return (
        f"Your assigned school{'s' if len(assigned) > 1 else ''} ({names}) "
        f"{'have' if len(assigned) > 1 else 'has'} no records in the active dataset, so there is no school "
        "data to show. Ask a System Administrator to assign a school from the active dataset."
    )


def _assigned_schools_not_in_dataset(user, shown: list[dict]) -> list[School]:
    """Assigned school records (non-administrators) whose exact ID isn't among the dataset schools shown."""
    if access.is_system_admin(user):
        return []
    shown_codes = {s["school_id"] for s in shown}
    return list(access.accessible_schools(user).exclude(school_id__in=shown_codes).order_by("name", "school_id"))


def dashboard(request):
    context = {"recent_runs": _visible_runs(request.user)[:5]}
    try:
        context.update(_dashboard_data(request.user))
    except ServiceError as exc:
        context["data_error"] = user_message(exc)
    if "schools" in context:  # some data shown: still say which assigned schools are not in the dataset
        context["unavailable_schools"] = _assigned_schools_not_in_dataset(request.user, context["schools"])
    # Model evaluation (metrics, baselines, performance link) is for System Administrators only.
    context["show_model_card"] = access.can_view_model_evaluation(request.user)
    if context["show_model_card"]:
        try:
            metadata = model_service.load_model_metadata()
            context["model_metadata"] = metadata
            context["baselines_beating_model"] = _baselines_beating_model(metadata)
        except ServiceError as exc:
            context["model_error"] = user_message(exc)
    return render(request, "core/dashboard.html", context)


def _dashboard_data(user) -> dict:
    """Demand figures for the user's schools only."""
    summary = access.filter_school_records(user, data_service.recent_demand_summary())
    if not summary:
        return {"data_error": _no_school_data_message(user)}
    date_range = data_service.get_date_range()
    latest = date_range["end"]
    reporting = [s for s in summary if s["latest_date"] == latest]
    data = data_service.load_processed_data()
    codes = access.accessible_school_codes(user)
    if codes is not None:
        data = data[data[ml_settings.SCHOOL_ID_COLUMN].isin(codes)]
    return dict(
        schools=summary,
        date_range=date_range,
        latest_day_total=sum(s["latest_meals_served"] for s in reporting),
        latest_day_schools=len(reporting),
        recent_daily_total=round(sum(s["recent_mean_meals"] for s in summary)),
        history_chart=charts.all_schools_history_chart(data),
        quick_form=ForecastForm(schools=summary),
    )


def _history_presets(school_id: str, available: dict, start, end) -> list[dict]:
    """Quick date ranges as plain filter links (same start/end parameters the form submits)."""
    presets = []
    for label, days in (("Last 30 days", 30), ("Last 90 days", DEFAULT_HISTORY_DAYS), ("Last 6 months", 182), ("All records", None)):
        preset_start = available["start"] if days is None else max(available["start"], available["end"] - timedelta(days=days))
        query = urlencode({"school": school_id, "start": preset_start.isoformat(), "end": available["end"].isoformat()})
        presets.append({"label": label, "href": f"?{query}", "active": start == preset_start and end == available["end"]})
    return presets


def history(request):
    try:
        schools = _schools_for(request.user)
    except ServiceError as exc:
        return render(request, "core/history.html", {"data_error": user_message(exc)})
    if not schools:
        return render(request, "core/history.html", {"data_error": _no_school_data_message(request.user)})

    form = HistoryFilterForm(request.GET or None, schools=schools)
    context = {"form": form}
    if request.GET and not form.is_valid():
        return render(request, "core/history.html", context)

    school_id = form.cleaned_data["school"] if request.GET else schools[0]["school_id"]
    start = form.cleaned_data.get("start") if request.GET else None
    end = form.cleaned_data.get("end") if request.GET else None
    try:
        available = data_service.get_date_range(school_id)
        if start is None and end is None:
            end = available["end"]
            start = end - timedelta(days=DEFAULT_HISTORY_DAYS)
        history_df = data_service.get_school_history(school_id, start=start, end=end)
    except ServiceError as exc:
        context["data_error"] = user_message(exc)
        return render(request, "core/history.html", context)

    if not request.GET:
        form = HistoryFilterForm(initial={"school": school_id, "start": start, "end": end}, schools=schools)
    school = next(s for s in schools if s["school_id"] == school_id)
    unrecorded = data_service.unrecorded_columns()
    context.update(
        form=form, school=school, available=available, start=start, end=end, records=len(history_df),
        range_presets=_history_presets(school_id, available, start, end),
        unrecorded_labels=[label for col, label in HISTORY_TABLE_COLUMNS if col in unrecorded],
    )

    if not history_df.empty:
        meals = history_df["meals_served"]
        original = data_service.original_count_column()
        original = original if original in history_df else None
        # Processed counts can be non-integer (outlier capping), so they are shown as stored, never truncated.
        context["stats"] = {
            "mean": round(float(meals.mean()), 1), "min": float(meals.min()),
            "max": float(meals.max()), "total": float(meals.sum()),
        }
        context["original_known"] = original is not None
        context["chart"] = charts.school_history_chart(history_df, school["school_name"])
        rows = []
        for _, row in history_df.iloc[::-1].iterrows():
            original_meals = row[original] if original else None
            rows.append({
                "date": row[ml_settings.DATE_COLUMN].date(),
                "original_meals": original_meals,
                "processed": data_service.is_processed_value(row["meals_served"], original_meals),
                "cells": [
                    {"key": col, "value": row[col] if col in history_df else None, "unrecorded": col in unrecorded}
                    for col, _ in HISTORY_TABLE_COLUMNS
                ],
                "flags": [name for col, name in (("is_holiday", "Holiday"), ("is_exam_period", "Exams"), ("is_school_event", "Event")) if col in history_df and row[col]],
            })
        page = Paginator(rows, HISTORY_PAGE_SIZE).get_page(request.GET.get("page"))
        query = request.GET.copy()
        query.pop("page", None)
        context.update(page=page, columns=[label for _, label in HISTORY_TABLE_COLUMNS], querystring=query.urlencode())
    return render(request, "core/history.html", context)


@require_http_methods(["GET", "POST"])
def forecast(request):
    try:
        schools = _schools_for(request.user)
        defaults = {s["school_id"]: forecast_service.default_assumptions(s["school_id"]) for s in schools}
    except ServiceError as exc:
        return render(request, "core/forecast.html", {"data_error": user_message(exc)})
    if not schools:
        return render(request, "core/forecast.html", {"data_error": _no_school_data_message(request.user)})

    if request.method == "POST":
        form = ForecastForm(request.POST, schools=schools)
        if form.is_valid():
            data = form.cleaned_data
            try:
                result = forecast_service.generate_forecast(
                    data["school"], data["horizon"],
                    expected_attendance=data["expected_attendance"],
                    expected_rainfall_mm=data["expected_rainfall_mm"],
                    expected_temperature_c=data["expected_temperature_c"],
                    available_capacity=data["available_capacity"],
                )
            except ServiceError as exc:
                message = user_message(exc)
                form.add_error(None, message)
                school = School.objects.filter(school_id=data["school"]).first()
                notify([request.user], Notification.Type.FORECAST_FAILED, f"Forecast failed: {school or data['school']}",
                       message, url=reverse("core:forecast"), school=school, include_actor=True)
            else:
                run = record_forecast_run(result, request.user, cost_per_meal=data["cost_per_meal_kes"])
                notify([request.user], Notification.Type.FORECAST_COMPLETED,
                       f"Forecast ready: {run.school_name or run.school_code}",
                       f"{run.get_horizon_days_display()} from {run.first_forecast_date:%d %b %Y}. "
                       f"Day 1: {run.day1_predicted_meals} meals predicted, {run.day1_recommended_meals} recommended.",
                       url=reverse("core:forecast_run", args=[run.pk]), school=run.school, include_actor=True)
                return redirect("core:forecast_run", pk=run.pk)
    else:
        initial = {k: request.GET[k] for k in ("school", "horizon") if k in request.GET}
        form = ForecastForm(initial=initial, schools=schools)

    defaults_json = {
        school_id: {**d, "origin_date": d["origin_date"].isoformat()} for school_id, d in defaults.items()
    }
    weather_unrecorded = bool(data_service.unrecorded_columns())
    if weather_unrecorded:
        # Browser-side only: the model was trained without weather variation, so these inputs cannot
        # inform the forecast. Server-side validation is unchanged.
        for name in ("expected_rainfall_mm", "expected_temperature_c"):
            form.fields[name].widget.attrs.update(disabled=True, **{"aria-describedby": "weather-note"})
    attendance_unused = _model_ignores_attendance()
    if attendance_unused:
        form.fields["expected_attendance"].widget.attrs.update(disabled=True)
    return render(request, "core/forecast.html", {
        "form": form, "defaults": defaults_json, "weather_unrecorded": weather_unrecorded,
        "attendance_unused": attendance_unused,
    })


def _model_ignores_attendance() -> bool:
    """True when the saved model has no attendance features (e.g. a Food4Education model)."""
    try:
        features = set(model_service.load_model_metadata()["features_used"])
    except ServiceError:
        return False  # the forecast itself reports the model problem
    return not features & {"previous_attendance", "previous_attendance_rate"}


def forecast_run(request, pk: int):
    # Out-of-scope runs 404 rather than 403, so their existence is not revealed.
    run = get_object_or_404(_visible_runs(request.user), pk=pk)
    days = stored_days(run)
    try:
        recent = data_service.get_school_history(run.school_code, end=run.origin_date).tail(21)
    except ServiceError:
        recent = None  # chart still shows the stored forecast
    summary = run.result_summary

    # Budget: a ?cost= override re-prices the stored forecast for display only (no new run).
    cost_error, cost_source = None, "stored"
    stored_cost = stored_cost_per_meal(run)
    try:
        if request.GET.get("cost"):
            cost, cost_source = planning_service.parse_cost(request.GET["cost"]), "override"
        elif stored_cost is not None:
            cost = planning_service.parse_cost(stored_cost)
        else:
            cost, cost_source = planning_service.default_cost_per_meal(), "default"
    except planning_service.InvalidCostError as exc:
        cost_error = str(exc)
        cost = planning_service.parse_cost(stored_cost) if stored_cost else planning_service.default_cost_per_meal()
        cost_source = "stored" if stored_cost else "default"
    budget = planning_service.forecast_budget(days, cost)
    for day, day_budget in zip(days, budget["per_day"]):
        day["budget"] = day_budget

    # Weather is "not recorded" for real-data runs when the active source has no weather.
    weather_unrecorded = "SYNTHETIC" not in run.results_label.upper() and bool(_safe_unrecorded())
    ds = summary.get("decision_support", {})
    assumptions = [a for a in ds.get("assumptions", [])
                   if not (weather_unrecorded and a.startswith(("Rainfall basis", "Temperature basis")))]
    return render(request, "core/forecast_result.html", {
        "run": run,
        "days": days,
        "summary": summary,
        "ds": ds,
        "assumptions": assumptions,
        "weather_unrecorded": weather_unrecorded,
        "budget": budget,
        "cost_source": cost_source,
        "cost_error": cost_error,
        "default_cost": planning_service.default_cost_per_meal(),
        "chart": charts.forecast_chart(recent, days, run.school_name or run.school_code),
    })


def _safe_unrecorded() -> set[str]:
    try:
        return data_service.unrecorded_columns()
    except ServiceError:
        return set()


def forecast_history(request):
    visible = _visible_runs(request.user)
    runs = visible
    school = request.GET.get("school")
    if school:
        runs = runs.filter(school_code=school)
    page = Paginator(runs, 20).get_page(request.GET.get("page"))
    schools = visible.order_by().values_list("school_code", "school_name").distinct()
    return render(request, "core/forecast_history.html", {
        "page": page,
        "selected_school": school,
        "schools": sorted(set(schools)),
        "querystring": urlencode({"school": school}) if school else "",
    })


@access.model_evaluation_required
def performance(request):
    try:
        perf = performance_service.get_model_performance()
    except ServiceError as exc:
        return render(request, "core/performance.html", {"error": user_message(exc)})
    context = {"perf": perf, "baselines_chart": charts.model_vs_baselines_chart(perf)}
    if perf["validation_ranking"]:
        context["ranking_chart"] = charts.validation_ranking_chart(perf["validation_ranking"])
    return render(request, "core/performance.html", context)


def donors(request):
    """How each school's programme is doing and where a donation would reach the most children."""
    try:
        impact = donor_service.school_impact(school_ids=access.accessible_school_codes(request.user))
    except ServiceError as exc:
        return render(request, "core/donors.html", {"data_error": user_message(exc)})
    if not impact["schools"]:
        return render(request, "core/donors.html", {"data_error": _no_school_data_message(request.user)})
    return render(request, "core/donors.html", {"impact": impact, "chart": charts.donor_reach_chart(impact["schools"])})


@login_not_required
def privacy(request):
    """Public privacy notice. Not served (404) until every owner-confirmed and server-verified fact
    in web/core/privacy.py is filled in, so it is never published with gaps."""
    from web.core import privacy as notice

    if not notice.is_published():
        raise Http404("The privacy notice has not been published yet.")
    return render(request, "core/privacy.html", {"facts": notice.FACTS})
