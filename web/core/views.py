"""BiteTracker page views.

Views stay thin: they validate input with forms, call the framework-independent
services, and render templates. No forecasting, feature-engineering or
decision-support logic lives here, and no model is ever trained in a request.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from urllib.parse import urlencode

from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from config import settings as ml_settings
from services import ServiceError, data_service, forecast_service, model_service, performance_service
from web.core import charts
from web.core.errors import user_message
from web.core.forecast_records import record_forecast_run, stored_days
from web.core.forms import ForecastForm, HistoryFilterForm
from web.schools.models import ForecastRun

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_DAYS = 90
HISTORY_PAGE_SIZE = 25
HISTORY_TABLE_COLUMNS = [
    ("meals_served", "Meals served"),
    ("students_present", "Students present"),
    ("enrolled_students", "Enrolled (dataset)"),
    ("rainfall_mm", "Rainfall (mm)"),
    ("temperature_c", "Temp (°C)"),
]


def dashboard(request):
    context = {"recent_runs": ForecastRun.objects.select_related("user")[:5]}
    try:
        summary = data_service.recent_demand_summary()
        date_range = data_service.get_date_range()
        latest = date_range["end"]
        context.update(
            schools=summary,
            date_range=date_range,
            latest_day_total=sum(s["latest_meals_served"] for s in summary if s["latest_date"] == latest),
            recent_daily_total=round(sum(s["recent_mean_meals"] for s in summary)),
            history_chart=charts.all_schools_history_chart(data_service.load_processed_data()),
            quick_form=ForecastForm(schools=summary),
        )
    except ServiceError as exc:
        context["data_error"] = user_message(exc)
    try:
        context["model_metadata"] = model_service.load_model_metadata()
    except ServiceError as exc:
        context["model_error"] = user_message(exc)
    return render(request, "core/dashboard.html", context)


def history(request):
    try:
        schools = data_service.list_schools()
    except ServiceError as exc:
        return render(request, "core/history.html", {"data_error": user_message(exc)})

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
    context.update(form=form, school=school, available=available, start=start, end=end, records=len(history_df))

    if not history_df.empty:
        meals = history_df["meals_served"]
        context["stats"] = {
            "mean": round(float(meals.mean()), 1), "min": int(meals.min()),
            "max": int(meals.max()), "total": int(meals.sum()),
        }
        context["chart"] = charts.school_history_chart(history_df, school["school_name"])
        rows = []
        for _, row in history_df.iloc[::-1].iterrows():
            rows.append({
                "date": row[ml_settings.DATE_COLUMN].date(),
                "values": [row[col] if col in history_df else None for col, _ in HISTORY_TABLE_COLUMNS],
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
        schools = data_service.list_schools()
        defaults = {s["school_id"]: forecast_service.default_assumptions(s["school_id"]) for s in schools}
    except ServiceError as exc:
        return render(request, "core/forecast.html", {"data_error": user_message(exc)})

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
                form.add_error(None, user_message(exc))
            else:
                run = record_forecast_run(result, request.user)
                return redirect("core:forecast_run", pk=run.pk)
    else:
        initial = {k: request.GET[k] for k in ("school", "horizon") if k in request.GET}
        form = ForecastForm(initial=initial, schools=schools)

    defaults_json = {
        school_id: {**d, "origin_date": d["origin_date"].isoformat()} for school_id, d in defaults.items()
    }
    return render(request, "core/forecast.html", {"form": form, "defaults": defaults_json})


def forecast_run(request, pk: int):
    run = get_object_or_404(ForecastRun.objects.select_related("user"), pk=pk)
    days = stored_days(run)
    try:
        recent = data_service.get_school_history(run.school_code, end=run.origin_date).tail(21)
    except ServiceError:
        recent = None  # chart still shows the stored forecast
    summary = run.result_summary
    return render(request, "core/forecast_result.html", {
        "run": run,
        "days": days,
        "summary": summary,
        "ds": summary.get("decision_support", {}),
        "chart": charts.forecast_chart(recent, days, run.school_name or run.school_code),
    })


def forecast_history(request):
    runs = ForecastRun.objects.select_related("user")
    school = request.GET.get("school")
    if school:
        runs = runs.filter(school_code=school)
    page = Paginator(runs, 20).get_page(request.GET.get("page"))
    schools = ForecastRun.objects.order_by().values_list("school_code", "school_name").distinct()
    return render(request, "core/forecast_history.html", {
        "page": page,
        "selected_school": school,
        "schools": sorted(set(schools)),
        "querystring": urlencode({"school": school}) if school else "",
    })


def performance(request):
    try:
        perf = performance_service.get_model_performance()
    except ServiceError as exc:
        return render(request, "core/performance.html", {"error": user_message(exc)})
    context = {"perf": perf, "baselines_chart": charts.model_vs_baselines_chart(perf)}
    if perf["validation_ranking"]:
        context["ranking_chart"] = charts.validation_ranking_chart(perf["validation_ranking"])
    return render(request, "core/performance.html", context)
