"""Administration: registration requests, users, schools, upload verification, activity logs,
support issues and announcements.

System Administrators only (every view is decorated). Lookups still go through the scope
helpers in web.accounts.access, so a crafted id never reaches a record outside scope (404).
"""

from __future__ import annotations

from datetime import datetime, time
from urllib.parse import urlencode

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import SetPasswordForm
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Count, Prefetch, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from web.accounts import access
from web.accounts.models import AccessRequest, Role, SchoolAssignment
from web.administration.forms import (
    AccessApproveForm,
    AccessRejectForm,
    AnnouncementForm,
    AssignUserForm,
    LogFilterForm,
    SchoolForm,
    UploadReviewForm,
    UserCreateForm,
    UserEditForm,
)
from web.schools import eligibility
from web.audit.models import AuditLog
from web.audit.services import log_action, visible_logs
from web.notifications import emails
from web.notifications.models import Notification
from web.notifications.services import notify
from web.schools.models import DatasetUpload
from web.support.forms import IssueStatusForm
from web.support.models import SupportIssue

User = get_user_model()
PAGE_SIZE = 25


def _tabs(request, active: str) -> list[dict]:
    pending = AccessRequest.objects.filter(status=AccessRequest.Status.PENDING).count()
    tabs = [("access", "Access requests", "bi-person-plus", "administration:access_requests", pending),
            ("users", "User management", "bi-people", "administration:users", 0),
            ("schools", "School management", "bi-buildings", "administration:schools", 0),
            ("uploads", "Upload reviews", "bi-clipboard-check", "administration:uploads", 0),
            ("logs", "Activity logs", "bi-journal-text", "administration:logs", 0),
            ("issues", "Support issues", "bi-life-preserver", "administration:issues", 0),
            ("announce", "Announcements", "bi-megaphone", "administration:announce", 0)]
    return [{"key": k, "label": label, "icon": icon, "url": reverse(name), "active": k == active, "count": count}
            for k, label, icon, name, count in tabs]


def _render(request, template: str, tab: str, context: dict):
    return render(request, template, {"admin_tabs": _tabs(request, tab), **context})


def _school_codes(schools) -> list[str]:
    return sorted(s.school_id for s in schools)


# --- Registration (access) requests ---------------------------------------------------------------


@access.system_admin_required
def access_requests(request):
    status = request.GET.get("status", AccessRequest.Status.PENDING)
    if status not in AccessRequest.Status.values:
        status = ""
    qs = AccessRequest.objects.select_related("user", "decided_by")
    if status:
        qs = qs.filter(status=status)
    counts = dict(AccessRequest.objects.order_by().values_list("status").annotate(n=Count("pk")))
    page = Paginator(qs, PAGE_SIZE).get_page(request.GET.get("page"))
    return _render(request, "administration/access_requests.html", "access", {
        "page": page, "status": status, "statuses": AccessRequest.Status.choices, "counts": counts,
        "querystring": urlencode({"status": status}),
    })


@access.system_admin_required
@require_http_methods(["GET", "POST"])
def access_request_detail(request, pk: int):
    access_request = get_object_or_404(AccessRequest.objects.select_related("user", "decided_by"), pk=pk)
    action = request.POST.get("action") if request.method == "POST" else None
    approve_form = AccessApproveForm(request.POST if action == "approve" else None, admin=request.user,
                                     initial={"role": access_request.requested_role or Role.SCHOOL_MANAGER})
    reject_form = AccessRejectForm(request.POST if action == "reject" else None)

    if request.method == "POST":
        if access_request.status != AccessRequest.Status.PENDING:
            messages.info(request, f"This request was already {access_request.get_status_display().lower()}; nothing was changed.")
            return redirect("administration:access_request_detail", pk=pk)
        if action == "approve" and approve_form.is_valid():
            if _approve(request, access_request, approve_form.cleaned_data):
                messages.success(request, f"{access_request.email} was approved and notified.")
            else:
                messages.info(request, "This request was decided by another administrator; nothing was changed.")
            return redirect("administration:access_request_detail", pk=pk)
        if action == "reject" and reject_form.is_valid():
            if _reject(request, access_request, reject_form.cleaned_data["reason"].strip()):
                messages.success(request, f"{access_request.email} was rejected. They keep seeing a blocked status.")
            else:
                messages.info(request, "This request was decided by another administrator; nothing was changed.")
            return redirect("administration:access_request_detail", pk=pk)
        if action not in ("approve", "reject"):
            return redirect("administration:access_request_detail", pk=pk)
        messages.error(request, "The decision was not saved. Check the highlighted fields.")

    return _render(request, "administration/access_request_detail.html", "access", {
        "req": access_request, "applicant": access_request.user,
        "approve_form": approve_form, "reject_form": reject_form,
        "logs": visible_logs(request.user).filter(target_user=access_request.user)[:10],
    })


def _claim_decision(request, access_request, status: str, **fields) -> bool:
    """Record a decision only if the request is still pending (an atomic conditional update), so
    two administrators, or a double-submitted form, can never decide the same request twice."""
    return bool(AccessRequest.objects.filter(pk=access_request.pk, status=AccessRequest.Status.PENDING).update(
        status=status, decided_by=request.user, decided_at=timezone.now(), **fields))


@transaction.atomic
def _approve(request, access_request, data) -> bool:
    role, schools, note = data["role"], list(data["schools"]), data["note"].strip()
    if not _claim_decision(request, access_request, AccessRequest.Status.APPROVED, granted_role=role, decision_note=note):
        return False
    user = access_request.user
    access.set_role(user, role)
    access.set_school_assignments(user, schools, request.user)
    user.is_active = True
    user.save(update_fields=["is_active"])
    log_action(request.user, AuditLog.Action.ACCESS_APPROVED, target=access_request, target_user=user,
               target_label=access_request.email,
               details={"role": role, "schools": _school_codes(schools), "note": note})
    school_text = f" for {_school_list(schools)}" if schools else ""
    # The applicant's own status notice: recipient is the applicant only.
    notify([user], Notification.Type.ACCOUNT, "Your registration was approved",
           f"Welcome to BiteTracker. Your role is {Role(role).label}{school_text}.",
           url=reverse("core:dashboard"), event_key=f"access_request:{access_request.pk}:approved")
    # The approver's confirmation: recipient is the administrator who approved, nobody else.
    notify([request.user], Notification.Type.ACCESS_REQUEST, "Registration approved",
           f"You approved {_applicant_label(access_request)} as {Role(role).label}{school_text}.",
           url=reverse("administration:access_request_detail", args=[access_request.pk]), include_actor=True,
           event_key=f"access_request:{access_request.pk}:approved_by")
    emails.registration_approved(access_request, role, schools)
    return True


def _school_list(schools) -> str:
    return ", ".join(f"{s.name or s.school_id} ({s.school_id})" for s in schools)


def _applicant_label(access_request) -> str:
    name = access_request.full_name.strip()
    return f"{name} ({access_request.email})" if name else access_request.email


@transaction.atomic
def _reject(request, access_request, reason: str) -> bool:
    if not _claim_decision(request, access_request, AccessRequest.Status.REJECTED, decision_note=reason):
        return False
    user = access_request.user
    if user.is_active:
        user.is_active = False
        user.save(update_fields=["is_active"])
    log_action(request.user, AuditLog.Action.ACCESS_REJECTED, target=access_request, target_user=user,
               target_label=access_request.email, details={"reason": reason})
    # Kept for the record: the applicant cannot sign in, so they see the outcome on the status page.
    notify([user], Notification.Type.ACCOUNT, "Your registration was not approved",
           "Contact your System Administrator if you think this is a mistake.",
           event_key=f"access_request:{access_request.pk}:rejected", include_inactive=True)
    notify([request.user], Notification.Type.ACCESS_REQUEST, "Registration rejected",
           f"You rejected {_applicant_label(access_request)}.",
           url=reverse("administration:access_request_detail", args=[access_request.pk]), include_actor=True,
           event_key=f"access_request:{access_request.pk}:rejected_by")
    emails.registration_rejected(access_request)
    return True


# --- Users ------------------------------------------------------------------------------------


def _users_queryset(admin):
    return access.manageable_users(admin).select_related("profile", "access_request").prefetch_related(
        Prefetch("school_assignments", queryset=SchoolAssignment.objects.select_related("school")))


# Users-page status filter -> queryset filter.
USER_STATUS_FILTERS = {
    access.APPROVED: Q(is_active=True),
    access.SUSPENDED: Q(is_active=False) & (Q(access_request__isnull=True) | Q(access_request__status=AccessRequest.Status.APPROVED)),
    access.PENDING: Q(access_request__status=AccessRequest.Status.PENDING),
    access.REJECTED: Q(access_request__status=AccessRequest.Status.REJECTED),
}


@access.admin_required
def users(request):
    qs = _users_queryset(request.user)
    q, role, status = request.GET.get("q", "").strip(), request.GET.get("role", ""), request.GET.get("status", "")
    if q:
        qs = qs.filter(Q(username__icontains=q) | Q(first_name__icontains=q) | Q(last_name__icontains=q) | Q(email__icontains=q))
    if role == Role.SYSTEM_ADMIN:
        qs = qs.filter(Q(is_superuser=True) | Q(profile__role=role))
    elif role in Role.values:
        qs = qs.filter(profile__role=role, is_superuser=False)
    if status in USER_STATUS_FILTERS:
        qs = qs.filter(USER_STATUS_FILTERS[status])
    page = Paginator(qs.order_by("-is_active", "first_name", "last_name", "username"), PAGE_SIZE).get_page(request.GET.get("page"))
    for user in page:
        user.role_label = access.role_label(user)
        user.account_status = access.account_status(user)
        user.account_status_label = access.STATUS_LABELS[user.account_status]
    return _render(request, "administration/users.html", "users", {
        "page": page, "q": q, "role": role, "status": status, "roles": Role.choices,
        "statuses": list(access.STATUS_LABELS.items()),
        "querystring": urlencode({k: v for k, v in {"q": q, "role": role, "status": status}.items() if v}),
        "can_create": bool(access.assignable_roles(request.user)),
    })


@access.admin_required
@require_http_methods(["GET", "POST"])
def user_create(request):
    form = UserCreateForm(request.POST or None, admin=request.user)
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data
        with transaction.atomic():
            user = User.objects.create_user(
                username=data["username"], email=data["email"], password=data["password1"],
                first_name=data["first_name"], last_name=data["last_name"],
            )
            access.set_role(user, data["role"])
            profile = access.get_profile(user)
            profile.phone = data["phone"]
            profile.save(update_fields=["phone", "updated_at"])
            access.set_school_assignments(user, data["schools"], request.user)
            log_action(request.user, AuditLog.Action.USER_CREATED, target=user, target_user=user,
                       target_label=user.get_username(),
                       details={"role": data["role"], "schools": _school_codes(data["schools"])})
        messages.success(request, f"Account “{user.username}” was created. Share the temporary password privately.")
        return redirect("administration:user_detail", pk=user.pk)
    return _render(request, "administration/user_form.html", "users", {"form": form, "creating": True})


def _assignment_rows(target) -> list[dict]:
    """The user's assignments, each with why it couldn't be made today (None when it still could)."""
    codes = eligibility.active_dataset_school_codes()
    return [{"school": a.school, "assigned_at": a.assigned_at, "unavailable": eligibility.unavailable_reason(a.school, codes)}
            for a in target.school_assignments.select_related("school").order_by("school__name", "school__school_id")]


def _managed_user(request, pk: int):
    return get_object_or_404(_users_queryset(request.user), pk=pk)


def _decided_user(request, pk: int):
    """A managed user whose registration (if any) is approved; others are handled on Access requests."""
    target = _managed_user(request, pk)
    if access.account_status(target) in (access.PENDING, access.REJECTED):
        messages.info(request, "Decide this registration on Access requests first.")
        return target, redirect("administration:access_request_detail", pk=target.access_request.pk)
    return target, None


@access.admin_required
def user_detail(request, pk: int):
    target = _managed_user(request, pk)
    status = access.account_status(target)
    return _render(request, "administration/user_detail.html", "users", {
        "target": target,
        "target_profile": access.get_profile(target),
        "target_role": access.role_label(target),
        "target_status": status,
        "target_status_label": access.STATUS_LABELS[status],
        "access_request": access.access_request_of(target),
        "google_linked": target.socialaccount_set.filter(provider="google").exists(),
        "assignments": (rows := _assignment_rows(target)),
        "has_unavailable": any(r["unavailable"] for r in rows),
        "logs": visible_logs(request.user).filter(Q(target_user=target) | Q(actor=target))[:10],
        "is_self": target.pk == request.user.pk,
    })


@access.admin_required
@require_http_methods(["GET", "POST"])
def user_edit(request, pk: int):
    target, response = _decided_user(request, pk)
    if response:
        return response
    profile = access.get_profile(target)
    scope = None if access.is_system_admin(request.user) else access.accessible_schools(request.user)
    current_schools = [a.school for a in target.school_assignments.all()
                       if scope is None or scope.filter(pk=a.school_id).exists()]
    initial = {"first_name": target.first_name, "last_name": target.last_name, "email": target.email,
               "phone": profile.phone, "role": access.role_of(target), "schools": current_schools}
    form = UserEditForm(request.POST or None, admin=request.user, target=target, initial=initial)
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data
        with transaction.atomic():
            changed = [f for f in ("first_name", "last_name", "email") if getattr(target, f) != data[f]]
            for f in changed:
                setattr(target, f, data[f])
            if changed:
                target.save(update_fields=changed)
            if profile.phone != data["phone"]:
                profile.phone = data["phone"]
                profile.save(update_fields=["phone", "updated_at"])
                changed.append("phone")
            if changed:
                log_action(request.user, AuditLog.Action.USER_UPDATED, target=target, target_user=target,
                           target_label=target.get_username(), details={"fields": changed})

            old_role, access_event = access.role_of(target), None
            role_changed = "role" in data and data["role"] != old_role
            if role_changed:
                access.set_role(target, data["role"])
                access_event = log_action(request.user, AuditLog.Action.ROLE_CHANGED, target=target, target_user=target,
                                          target_label=target.get_username(), details={"from": old_role, "to": data["role"]})
                notify([target], Notification.Type.ACCOUNT, "Your role has changed",
                       f"Your BiteTracker role is now {Role(data['role']).label}.", url=reverse("accounts:profile"))

            added, removed = access.set_school_assignments(target, form.assigned_schools(), request.user, scope=scope)
            if added or removed:
                schools_event = log_action(request.user, AuditLog.Action.SCHOOLS_ASSIGNED, target=target,
                                           target_user=target, target_label=target.get_username(),
                                           details={"added": _school_codes(added), "removed": _school_codes(removed)})
                access_event = access_event or schools_event
                notify([target], Notification.Type.ACCOUNT, "Your school assignments have changed",
                       _assignment_message(added, removed), url=reverse("accounts:profile"))
            if access_event is not None:
                # One email for the whole edit, sent only after it commits.
                emails.access_changed(target, access_event.pk, old_role=old_role if role_changed else None,
                                      added=added, removed=removed)
        messages.success(request, f"Account “{target.username}” was updated.")
        return redirect("administration:user_detail", pk=target.pk)
    return _render(request, "administration/user_form.html", "users", {"form": form, "target": target, "creating": False})


def _assignment_message(added, removed) -> str:
    parts = []
    if added:
        parts.append("Added: " + ", ".join(str(s) for s in sorted(added, key=str)))
    if removed:
        parts.append("Removed: " + ", ".join(str(s) for s in sorted(removed, key=str)))
    return ". ".join(parts) + "."


@access.admin_required
@require_POST
def user_toggle_active(request, pk: int):
    """Suspend or reactivate an approved account. Registrations are decided on Access requests instead."""
    target = _managed_user(request, pk)
    if target.pk == request.user.pk:
        messages.error(request, "You cannot suspend your own account.")
        return redirect("administration:user_detail", pk=pk)
    if access.account_status(target) in (access.PENDING, access.REJECTED):
        messages.error(request, "This account has a registration request that is not approved. "
                                "Approve it from Access requests to give access.")
        return redirect("administration:user_detail", pk=pk)
    # The form says which change was intended, so a resubmitted or stale form never flips the account back.
    intended = request.POST.get("action")
    activate = {"reactivate": True, "suspend": False}.get(intended, not target.is_active)
    if activate == target.is_active:
        messages.info(request, f"Account “{target.username}” is already {'active' if activate else 'suspended'}; nothing was changed.")
        return redirect("administration:user_detail", pk=pk)
    if (not activate and access.role_of(target) == Role.SYSTEM_ADMIN and access.active_system_admin_count() <= 1):
        messages.error(request, "This is the only active System Administrator, so the account cannot be suspended.")
        return redirect("administration:user_detail", pk=pk)
    with transaction.atomic():
        # Conditional update: two concurrent submits can't both change (and email about) the account.
        if not User.objects.filter(pk=target.pk, is_active=not activate).update(is_active=activate):
            messages.info(request, f"Account “{target.username}” was already changed; nothing was changed.")
            return redirect("administration:user_detail", pk=pk)
        target.is_active = activate
        event = log_action(request.user, AuditLog.Action.USER_ACTIVATED if activate else AuditLog.Action.USER_DEACTIVATED,
                           target=target, target_user=target, target_label=target.get_username())
        if activate:
            notify([target], Notification.Type.ACCOUNT, "Your account has been reactivated",
                   "You can use BiteTracker again.", url=reverse("accounts:profile"))
            emails.account_reactivated(target, event.pk)
        else:
            emails.account_suspended(target, event.pk)
    messages.success(request, f"Account “{target.username}” was {'reactivated' if activate else 'suspended'}.")
    return redirect("administration:user_detail", pk=pk)


@access.admin_required
@require_http_methods(["GET", "POST"])
def user_password(request, pk: int):
    """Set a new temporary password. The password itself is never shown, logged or stored in plain text."""
    target, response = _decided_user(request, pk)
    if response:
        return response
    form = SetPasswordForm(target, request.POST or None)
    for field in form.fields.values():
        field.widget.attrs["class"] = "form-control"
    if request.method == "POST" and form.is_valid():
        form.save()  # also signs the user out of their other sessions
        log_action(request.user, AuditLog.Action.PASSWORD_RESET, target=target, target_user=target,
                   target_label=target.get_username())
        notify([target], Notification.Type.SECURITY, "Your password was reset by an administrator",
               "Please change it to one only you know in Settings → Security.",
               url=reverse("accounts:settings") + "?tab=security", actor=request.user)
        messages.success(request, f"A new temporary password was set for “{target.username}”. Share it privately.")
        return redirect("administration:user_detail", pk=pk)
    return _render(request, "administration/user_password.html", "users", {"form": form, "target": target})


# --- Schools ----------------------------------------------------------------------------------


@access.admin_required
def schools(request):
    qs = access.accessible_schools(request.user).annotate(n_users=Count("assignments", distinct=True))
    q, status = request.GET.get("q", "").strip(), request.GET.get("status", "")
    if q:
        qs = qs.filter(Q(school_id__icontains=q) | Q(name__icontains=q) | Q(county__icontains=q) | Q(sub_county__icontains=q))
    if status in ("active", "inactive"):
        qs = qs.filter(is_active=status == "active")
    page = Paginator(qs.order_by("name", "school_id"), PAGE_SIZE).get_page(request.GET.get("page"))
    return _render(request, "administration/schools.html", "schools", {
        "page": page, "q": q, "status": status, "dataset_codes": eligibility.active_dataset_school_codes(),
        "querystring": urlencode({k: v for k, v in {"q": q, "status": status}.items() if v}),
    })


def _scoped_school(request, pk: int):
    return get_object_or_404(access.accessible_schools(request.user), pk=pk)


@access.admin_required
@require_http_methods(["GET", "POST"])
def school_detail(request, pk: int):
    school = _scoped_school(request, pk)
    manageable_ids = set(access.manageable_users(request.user).values_list("pk", flat=True))
    form = AssignUserForm(request.POST or None, admin=request.user, school=school)
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "assign" and form.is_valid():  # form refuses schools not eligible for new assignments
            target = form.cleaned_data["user"]
            with transaction.atomic():
                _, created = SchoolAssignment.objects.get_or_create(user=target, school=school,
                                                                    defaults={"assigned_by": request.user})
                if created:
                    _log_assignment(request, target, school, added=True)
            messages.success(request, f"{target.get_full_name() or target.username} was assigned to {school}.")
            return redirect("administration:school_detail", pk=pk)
        if action == "unassign":
            target = get_object_or_404(User, pk=request.POST.get("user"))
            if target.pk not in manageable_ids:
                raise PermissionDenied
            with transaction.atomic():
                removed = SchoolAssignment.objects.filter(user=target, school=school).delete()[0]
                if removed:
                    _log_assignment(request, target, school, added=False)
            if removed:
                messages.success(request, f"{target.get_full_name() or target.username} was removed from {school}.")
            return redirect("administration:school_detail", pk=pk)
    assignments = school.assignments.select_related("user", "user__profile", "assigned_by").order_by(
        "user__first_name", "user__username")
    rows = [{"assignment": a, "role": access.role_label(a.user), "manageable": a.user_id in manageable_ids}
            for a in assignments]
    return _render(request, "administration/school_detail.html", "schools", {
        "school": school, "rows": rows, "form": form, "unavailable_reason": form.unavailable_reason,
        "logs": visible_logs(request.user).filter(school=school)[:10],
        "forecast_count": school.forecast_runs.count(),
        "event_count": school.calendar_events.count(),
    })


def _log_assignment(request, target, school, added: bool):
    event = log_action(request.user, AuditLog.Action.SCHOOLS_ASSIGNED, target=target, target_user=target, school=school,
               target_label=target.get_username(),
               details={"added": [school.school_id]} if added else {"removed": [school.school_id]})
    notify([target], Notification.Type.ACCOUNT, "Your school assignments have changed",
           f"{'Added' if added else 'Removed'}: {school}.", url=reverse("accounts:profile"))
    emails.access_changed(target, event.pk, added=[school] if added else (), removed=() if added else [school])


@access.admin_required
@require_http_methods(["GET", "POST"])
def school_edit(request, pk: int):
    school = _scoped_school(request, pk)
    form = SchoolForm(request.POST or None, instance=school)
    if request.method == "POST" and form.is_valid():
        if form.has_changed():
            changes = {f: {"from": form.initial.get(f), "to": form.cleaned_data[f]} for f in form.changed_data}
            form.save()
            log_action(request.user, AuditLog.Action.SCHOOL_UPDATED, target=school, school=school,
                       details={"changes": changes})
        messages.success(request, f"{school} was updated.")
        return redirect("administration:school_detail", pk=pk)
    return _render(request, "administration/school_form.html", "schools", {"form": form, "school": school})


# --- Activity logs ----------------------------------------------------------------------------


@access.admin_required
def logs(request):
    qs = visible_logs(request.user)
    form = LogFilterForm(request.GET or None)
    if form.is_valid():
        data = form.cleaned_data
        if data["q"]:
            qs = qs.filter(actor_name__icontains=data["q"])
        if data["action"]:
            qs = qs.filter(action=data["action"])
        tz = timezone.get_current_timezone()
        if data["start"]:
            qs = qs.filter(created_at__gte=datetime.combine(data["start"], time.min, tzinfo=tz))
        if data["end"]:
            qs = qs.filter(created_at__lte=datetime.combine(data["end"], time.max, tzinfo=tz))
    query = request.GET.copy()
    query.pop("page", None)
    page = Paginator(qs, PAGE_SIZE).get_page(request.GET.get("page"))
    return _render(request, "administration/logs.html", "logs", {"form": form, "page": page, "querystring": query.urlencode()})


# --- Dataset upload reviews -----------------------------------------------------------------------


@access.admin_required
def uploads(request):
    qs = access.reviewable_uploads(request.user)
    status = request.GET.get("status", DatasetUpload.ReviewStatus.PENDING)
    if status in DatasetUpload.ReviewStatus.values:
        qs = qs.filter(review_status=status)
    else:
        status = ""
    page = Paginator(qs, PAGE_SIZE).get_page(request.GET.get("page"))
    counts = dict(access.reviewable_uploads(request.user).order_by().values_list("review_status").annotate(n=Count("pk")))
    return _render(request, "administration/uploads.html", "uploads", {
        "page": page, "status": status, "statuses": DatasetUpload.ReviewStatus.choices, "counts": counts,
        "querystring": urlencode({"status": status}),
    })


@access.admin_required
@require_http_methods(["GET", "POST"])
def upload_review(request, pk: int):
    record = get_object_or_404(access.reviewable_uploads(request.user), pk=pk)
    can_decide = access.can_decide_upload(request.user, record)
    form = UploadReviewForm(request.POST or None)
    if request.method == "POST":
        if not can_decide:
            raise PermissionDenied
        if form.is_valid():
            record.review_status = form.cleaned_data["decision"]
            record.review_comment = form.cleaned_data["comment"].strip()
            record.reviewed_by, record.reviewed_at = request.user, timezone.now()
            record.save(update_fields=["review_status", "review_comment", "reviewed_by", "reviewed_at"])
            log_action(request.user, AuditLog.Action.DATASET_REVIEWED, target=record, target_label=record.original_filename,
                       target_user=record.uploaded_by, details={"decision": record.review_status})
            if record.uploaded_by:
                notify([record.uploaded_by], Notification.Type.DATASET_REVIEW,
                       f"Upload {record.get_review_status_display().lower()}: {record.original_filename}",
                       record.review_comment or "No comment was added.",
                       url=reverse("schools:upload_detail", args=[record.pk]), actor=request.user)
            messages.success(request, f"{record.original_filename} {record.get_review_status_display().lower()}. The uploader has been notified.")
            return redirect("administration:uploads")
    return _render(request, "administration/upload_review.html", "uploads", {
        "upload": record, "summary": record.validation_summary, "form": form, "can_decide": can_decide,
    })


# --- Support issues and announcements (System Administrators) --------------------------------------------


@access.system_admin_required
def issues(request):
    qs = SupportIssue.objects.select_related("reporter")
    status = request.GET.get("status", "")
    if status in SupportIssue.Status.values:
        qs = qs.filter(status=status)
    page = Paginator(qs, PAGE_SIZE).get_page(request.GET.get("page"))
    counts = dict(SupportIssue.objects.values_list("status").annotate(n=Count("pk")))
    return _render(request, "administration/issues.html", "issues", {
        "page": page, "status": status, "statuses": SupportIssue.Status.choices, "counts": counts,
        "querystring": urlencode({"status": status}) if status else "",
    })


@access.system_admin_required
@require_http_methods(["GET", "POST"])
def issue_detail(request, pk: int):
    issue = get_object_or_404(SupportIssue.objects.select_related("reporter"), pk=pk)
    old_status = issue.status
    form = IssueStatusForm(request.POST or None, instance=issue)
    if request.method == "POST" and form.is_valid():
        issue = form.save()
        if issue.status != old_status:
            log_action(request.user, AuditLog.Action.ISSUE_STATUS_CHANGED, target=issue,
                       details={"from": old_status, "to": issue.status})
            if issue.reporter:
                notify([issue.reporter], Notification.Type.SUPPORT, f"Your report #{issue.pk} is now {issue.get_status_display().lower()}",
                       issue.subject, url=reverse("support:issue_detail", args=[issue.pk]), actor=request.user)
        messages.success(request, f"Report #{issue.pk} updated.")
        return redirect("administration:issue_detail", pk=pk)
    return _render(request, "administration/issue_detail.html", "issues", {"issue": issue, "form": form})


@access.system_admin_required
@require_http_methods(["GET", "POST"])
def announce(request):
    form = AnnouncementForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data
        sent = notify(User.objects.filter(is_active=True).select_related("profile"), Notification.Type.ANNOUNCEMENT,
                      data["title"], data["message"], actor=request.user)
        log_action(request.user, AuditLog.Action.ANNOUNCEMENT_SENT, target_label=data["title"],
                   details={"recipients": len(sent)})
        messages.success(request, f"Announcement sent to {len(sent)} user(s). Users who turned announcements off were skipped.")
        return redirect("administration:announce")
    recent = AuditLog.objects.filter(action=AuditLog.Action.ANNOUNCEMENT_SENT)[:10]
    return _render(request, "administration/announce.html", "announce", {"form": form, "recent": recent})
