"""School calendar: month and list views, and event create/edit/delete within the user's school scope.

Calendar events are operational reference only; the forecasting model does not use them.
"""

from __future__ import annotations

import calendar as pycal
from datetime import date, timedelta
from urllib.parse import urlencode

from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from web.accounts import access
from web.audit.models import AuditLog
from web.audit.services import log_action
from web.notifications.models import Notification
from web.notifications.services import notify, users_for_school
from web.schoolcalendar.forms import CalendarEventForm
from web.schoolcalendar.models import CalendarEvent

FILTER_KEYS = ("view", "school", "year", "type")


def visible_events(user):
    schools = access.accessible_schools(user)
    return CalendarEvent.objects.filter(Q(school__isnull=True) | Q(school__in=schools)).select_related(
        "school", "created_by", "updated_by")


def can_edit(user, event: CalendarEvent) -> bool:
    """School Managers edit their schools' events; Resource Planning Stakeholders are read-only."""
    if event.school_id is None:
        return access.is_system_admin(user)
    return access.can_edit_school_data(user, event.school.school_id)


def _parse_month(value: str | None, fallback: date) -> date:
    try:
        year, month = (int(part) for part in (value or "").split("-"))
        return date(year, month, 1)
    except ValueError:
        return fallback.replace(day=1)


def calendar_view(request):
    user, today = request.user, date.today()
    events = visible_events(user)
    schools = access.accessible_schools(user).order_by("name", "school_id")

    school_id = request.GET.get("school", "")
    if school_id:
        if not school_id.isdigit() or not schools.filter(pk=school_id).exists():
            raise PermissionDenied
        events = events.filter(Q(school_id=school_id) | Q(school__isnull=True))
    event_type = request.GET.get("type", "")
    if event_type in CalendarEvent.EventType.values:
        events = events.filter(event_type=event_type)
    else:
        event_type = ""

    years = sorted({d.year for d in visible_events(user).dates("start_date", "year")} | {today.year}, reverse=True)
    year = request.GET.get("year", "")
    year = int(year) if year.isdigit() and int(year) in years else None
    view = "list" if request.GET.get("view") == "list" else "month"

    filters = {"school": school_id, "type": event_type, "year": year or ""}
    context = {
        "view": view, "schools": schools, "years": years, "event_types": CalendarEvent.EventType.choices,
        "filters": filters, "today": today, "can_add": access.can_edit_school_data(user),
        "feeding_statuses": CalendarEvent.FeedingStatus.choices,
    }

    if view == "list":
        if year:
            events = events.filter(start_date__lte=date(year, 12, 31), end_date__gte=date(year, 1, 1))
        page = Paginator(events.order_by("start_date", "title"), 25).get_page(request.GET.get("page"))
        query = {k: v for k, v in {**filters, "view": "list"}.items() if v}
        context.update(page=page, querystring=urlencode(query))
        return render(request, "schoolcalendar/calendar.html", context)

    default_month = date(year, today.month if year == today.year else 1, 1) if year else today
    month = _parse_month(request.GET.get("month"), default_month)
    weeks = pycal.Calendar(firstweekday=0).monthdatescalendar(month.year, month.month)
    first, last = weeks[0][0], weeks[-1][-1]
    in_range = list(events.filter(start_date__lte=last, end_date__gte=first))
    grid = [[{"date": d, "in_month": d.month == month.month, "is_today": d == today,
              "events": [e for e in in_range if e.start_date <= d <= e.end_date]} for d in week] for week in weeks]
    prev_month = (month - timedelta(days=1)).replace(day=1)
    next_month = (month + timedelta(days=32)).replace(day=1)
    base = {k: v for k, v in filters.items() if v and k != "year"}
    context.update(
        month=month, grid=grid, weekdays=[pycal.day_abbr[i] for i in range(7)],
        month_events=[e for e in in_range if e.end_date >= month and e.start_date <= next_month - timedelta(days=1)],
        prev_query=urlencode({**base, "month": f"{prev_month:%Y-%m}"}),
        next_query=urlencode({**base, "month": f"{next_month:%Y-%m}"}),
        today_query=urlencode({**base, "month": f"{today:%Y-%m}"}),
    )
    return render(request, "schoolcalendar/calendar.html", context)


def event_detail(request, pk: int):
    event = get_object_or_404(visible_events(request.user), pk=pk)
    return render(request, "schoolcalendar/event_detail.html", {"event": event, "can_edit": can_edit(request.user, event)})


def _notify_change(event: CalendarEvent, actor, verb: str, url: str):
    notify(
        users_for_school(event.school), Notification.Type.CALENDAR,
        f"Calendar event {verb}: {event.title}",
        f"{event.get_event_type_display()} · {event.scope_label} · {event.start_date:%d %b %Y}"
        + (f" – {event.end_date:%d %b %Y}" if event.is_multi_day else "")
        + f" · {event.get_feeding_status_display()}",
        url=url, school=event.school, actor=actor,
    )


def _event_details(event: CalendarEvent) -> dict:
    return {
        "title": event.title, "type": event.event_type, "school": event.school.school_id if event.school else "all",
        "start": event.start_date.isoformat(), "end": event.end_date.isoformat(), "feeding_status": event.feeding_status,
    }


@require_http_methods(["GET", "POST"])
def event_create(request):
    if not access.can_edit_school_data(request.user):
        raise PermissionDenied
    initial = {}
    try:
        initial["start_date"] = initial["end_date"] = date.fromisoformat(request.GET.get("date", ""))
    except ValueError:
        pass
    if request.GET.get("school", "").isdigit():
        initial["school"] = request.GET["school"]
    form = CalendarEventForm(request.POST or None, user=request.user, initial=initial)
    if request.method == "POST" and form.is_valid():
        event = form.save(commit=False)
        event.created_by = event.updated_by = request.user
        event.save()
        log_action(request.user, AuditLog.Action.CALENDAR_CREATED, target=event, school=event.school,
                   details=_event_details(event))
        url = reverse("schoolcalendar:event_detail", args=[event.pk])
        _notify_change(event, request.user, "added", url)
        _overlap_message(request, form)
        messages.success(request, f"“{event.title}” was added to the calendar.")
        return redirect(f"{reverse('schoolcalendar:calendar')}?month={event.start_date:%Y-%m}")
    return render(request, "schoolcalendar/event_form.html", {"form": form, "creating": True})


@require_http_methods(["GET", "POST"])
def event_update(request, pk: int):
    event = get_object_or_404(visible_events(request.user), pk=pk)
    if not can_edit(request.user, event):
        raise PermissionDenied
    before = _event_details(event)
    form = CalendarEventForm(request.POST or None, instance=event, user=request.user)
    if request.method == "POST" and form.is_valid():
        event = form.save(commit=False)
        event.updated_by = request.user
        event.save()
        after = _event_details(event)
        changes = {k: {"from": before[k], "to": after[k]} for k in after if before[k] != after[k]}
        if changes:
            log_action(request.user, AuditLog.Action.CALENDAR_UPDATED, target=event, school=event.school,
                       details={"changes": changes})
            _notify_change(event, request.user, "updated", reverse("schoolcalendar:event_detail", args=[event.pk]))
        _overlap_message(request, form)
        messages.success(request, f"“{event.title}” was updated.")
        return redirect("schoolcalendar:event_detail", pk=event.pk)
    return render(request, "schoolcalendar/event_form.html", {"form": form, "event": event, "creating": False})


@require_http_methods(["GET", "POST"])
def event_delete(request, pk: int):
    event = get_object_or_404(visible_events(request.user), pk=pk)
    if not can_edit(request.user, event):
        raise PermissionDenied
    if request.method == "POST":
        details, school, title = _event_details(event), event.school, event.title
        month = event.start_date
        log_action(request.user, AuditLog.Action.CALENDAR_DELETED, target=event, school=school, details=details)
        _notify_change(event, request.user, "deleted", reverse("schoolcalendar:calendar") + f"?month={month:%Y-%m}")
        event.delete()
        messages.success(request, f"“{title}” was deleted.")
        return redirect(f"{reverse('schoolcalendar:calendar')}?month={month:%Y-%m}")
    return render(request, "schoolcalendar/event_confirm_delete.html", {"event": event})


def _overlap_message(request, form):
    if form.overlaps:
        names = ", ".join(e.title for e in form.overlaps[:3])
        messages.warning(request, f"Note: these dates overlap {len(form.overlaps)} other event(s): {names}.")
