"""Administration (users, schools, logs, issues, announcements) and Help & Support.

What this file checks:
- School Managers and Resource Planning Stakeholders cannot open, or POST to, any
  Administration page: administration is System Administrator only.
- System Administrators: create users with any role (kept in sync with is_superuser),
  cannot demote or suspend themselves or the last System Administrator.
- Suspension blocks login; password reset uses Django's validators and never logs
  the password; every change is audited and the affected user is notified.
- School records: ID is read-only, edits are audited, assignments respect scope.
- Activity logs are filtered by scope and by user/action/date.
- Help page content, report-a-problem storage, screenshot validation and privacy,
  issue status workflow with reporter notification, announcements.
"""

from __future__ import annotations

import json

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from tests.support.web_fixtures import PASSWORD, operator
from web.accounts import access
from web.accounts.models import Role, SchoolAssignment, UserProfile
from web.audit.models import AuditLog
from web.audit.services import log_action
from web.notifications.models import Notification
from web.schools.models import School
from web.support.models import SupportIssue

pytestmark = pytest.mark.django_db
User = get_user_model()
NEW_PASSWORD = "Temp-passw0rd-for-new-user!"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

ADMIN_PAGES = ["administration:users", "administration:user_create", "administration:schools", "administration:logs",
               "administration:access_requests", "administration:uploads"]
SYSTEM_PAGES = ["administration:issues", "administration:announce"]


@pytest.fixture
def media_root(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "uploads"
    return settings.MEDIA_ROOT


@pytest.fixture
def sysadmin(db):
    return User.objects.create_superuser("root", "root@example.com", PASSWORD)


@pytest.fixture
def stakeholder(db):
    """Resource Planning Stakeholder assigned SCH001 and SCH002 (read-only)."""
    user = operator("pa", codes=("SCH001", "SCH002"))
    access.set_role(user, Role.PLANNING_STAKEHOLDER)
    return user


@pytest.fixture
def in_scope(db):
    return operator("in_scope", codes=("SCH001",))


@pytest.fixture
def out_of_scope(db):
    return operator("out_of_scope", codes=("SCH001", "SCH003"))


def _create_payload(**overrides) -> dict:
    data = {"username": "newbie", "first_name": "New", "last_name": "Person", "email": "newbie@example.com",
            "phone": "", "role": Role.SCHOOL_MANAGER,
            "schools": [School.objects.get_or_create(school_id="SCH001")[0].pk],
            "password1": NEW_PASSWORD, "password2": NEW_PASSWORD}
    data.update(overrides)
    return data


# --- Access ------------------------------------------------------------------------------------


@pytest.mark.parametrize("page", ADMIN_PAGES + SYSTEM_PAGES)
def test_school_operators_cannot_open_administration(client, in_scope, page):
    client.force_login(in_scope)
    assert client.get(reverse(page)).status_code == 403


@pytest.mark.parametrize("page", ADMIN_PAGES + SYSTEM_PAGES)
def test_stakeholders_cannot_open_administration(client, stakeholder, page):
    client.force_login(stakeholder)
    assert client.get(reverse(page)).status_code == 403


def test_sidebar_shows_administration_only_to_system_admins(client, in_scope, stakeholder, sysadmin):
    for user in (in_scope, stakeholder):
        client.force_login(user)
        assert reverse("administration:users") not in client.get(reverse("accounts:profile")).content.decode()
    client.force_login(sysadmin)
    assert reverse("administration:users") in client.get(reverse("accounts:profile")).content.decode()


# --- Read-only and administrator rights ---------------------------------------------------------------


def test_stakeholder_cannot_change_accounts_schools_or_registrations(client, stakeholder, in_scope, sysadmin):
    """Read-only role: crafted POSTs to every administration action are refused and change nothing."""
    from web.accounts.models import AccessRequest
    applicant = User.objects.create_user("applicant", "applicant@example.com", is_active=False)
    request_obj = AccessRequest.objects.create(user=applicant, email="applicant@example.com")
    school = School.objects.get(school_id="SCH001")
    client.force_login(stakeholder)
    attempts = [
        (reverse("administration:user_create"), _create_payload(role=Role.SYSTEM_ADMIN)),
        (reverse("administration:user_edit", args=[in_scope.pk]), {"first_name": "x", "last_name": "y",
                                                                   "email": "x@example.com", "role": Role.SYSTEM_ADMIN}),
        (reverse("administration:user_toggle_active", args=[in_scope.pk]), {}),
        (reverse("administration:user_password", args=[in_scope.pk]), {"new_password1": NEW_PASSWORD,
                                                                       "new_password2": NEW_PASSWORD}),
        (reverse("administration:school_edit", args=[school.pk]), {"name": "Hacked"}),
        (reverse("administration:school_detail", args=[school.pk]), {"action": "assign", "user": stakeholder.pk}),
        (reverse("administration:access_request_detail", args=[request_obj.pk]),
         {"action": "approve", "role": Role.SYSTEM_ADMIN}),
    ]
    for url, data in attempts:
        assert client.post(url, data).status_code == 403, url
    in_scope.refresh_from_db()
    applicant.refresh_from_db()
    request_obj.refresh_from_db()
    assert in_scope.is_active and in_scope.check_password(PASSWORD) and access.role_of(in_scope) == Role.SCHOOL_MANAGER
    assert not User.objects.filter(username="newbie").exists()
    assert School.objects.get(pk=school.pk).name != "Hacked"
    assert request_obj.status == AccessRequest.Status.PENDING and not applicant.is_active
    assert access.role_of(stakeholder) == Role.PLANNING_STAKEHOLDER


def test_system_admin_creates_school_manager(client, sysadmin, stub_artifacts):
    client.force_login(sysadmin)
    response = client.post(reverse("administration:user_create"), _create_payload())
    assert response.status_code == 302
    user = User.objects.get(username="newbie")
    assert access.role_of(user) == Role.SCHOOL_MANAGER and not user.is_superuser and not user.is_staff
    assert access.is_approved(user)  # administrator-created accounts need no registration approval
    assert user.check_password(NEW_PASSWORD)
    log = AuditLog.objects.get(action=AuditLog.Action.USER_CREATED)
    assert log.details == {"role": Role.SCHOOL_MANAGER, "schools": ["SCH001"]}
    assert NEW_PASSWORD not in json.dumps(log.details)


def test_system_admin_changes_school_assignments(client, sysadmin, in_scope, stub_artifacts):
    sch2 = School.objects.get_or_create(school_id="SCH002")[0]
    client.force_login(sysadmin)
    response = client.post(reverse("administration:user_edit", args=[in_scope.pk]), {
        "first_name": "In", "last_name": "Scope", "email": "in@example.com", "phone": "", "role": Role.SCHOOL_MANAGER,
        "schools": [sch2.pk],
    })
    assert response.status_code == 302
    assert set(in_scope.school_assignments.values_list("school__school_id", flat=True)) == {"SCH002"}
    assert Notification.objects.filter(recipient=in_scope, type=Notification.Type.ACCOUNT).exists()
    log = AuditLog.objects.get(action=AuditLog.Action.SCHOOLS_ASSIGNED)
    assert log.details == {"added": ["SCH002"], "removed": ["SCH001"]}


# --- System Admin -----------------------------------------------------------------------------------


def test_system_admin_role_change_syncs_superuser_flag_and_notifies(client, sysadmin, in_scope):
    client.force_login(sysadmin)
    client.post(reverse("administration:user_edit", args=[in_scope.pk]), {
        "first_name": "x", "last_name": "y", "email": "x@example.com", "role": Role.SYSTEM_ADMIN, "schools": [],
    })
    in_scope.refresh_from_db()
    assert in_scope.is_superuser and in_scope.is_staff and in_scope.profile.role == Role.SYSTEM_ADMIN
    assert AuditLog.objects.get(action=AuditLog.Action.ROLE_CHANGED).details == {"from": "school_manager", "to": "system_admin"}
    assert Notification.objects.filter(recipient=in_scope, title="Your role has changed").exists()


def test_admin_cannot_change_own_role_or_deactivate_self(client, sysadmin):
    client.force_login(sysadmin)
    form = client.get(reverse("administration:user_edit", args=[sysadmin.pk])).content.decode()
    assert 'name="role"' not in form
    client.post(reverse("administration:user_edit", args=[sysadmin.pk]), {
        "first_name": "R", "last_name": "T", "email": "root@example.com", "role": Role.SCHOOL_MANAGER,
    })
    sysadmin.refresh_from_db()
    assert sysadmin.is_superuser
    client.post(reverse("administration:user_toggle_active", args=[sysadmin.pk]))
    sysadmin.refresh_from_db()
    assert sysadmin.is_active


def test_last_system_admin_guard(sysadmin, monkeypatch):
    """Defence in depth: the form refuses to demote the only active System Admin."""
    from web.administration.forms import UserEditForm
    other_admin = User.objects.create_superuser("root2", "r2@example.com", PASSWORD)
    data = {"first_name": "a", "last_name": "b", "email": "c@example.com", "role": Role.SCHOOL_MANAGER}
    assert UserEditForm(data, admin=other_admin, target=sysadmin).is_valid()
    monkeypatch.setattr(access, "active_system_admin_count", lambda: 1)
    form = UserEditForm(data, admin=other_admin, target=sysadmin)
    assert not form.is_valid() and "only active System Administrator" in str(form.errors)


def test_deactivation_blocks_login_and_is_audited(client, sysadmin, in_scope):
    client.force_login(sysadmin)
    client.post(reverse("administration:user_toggle_active", args=[in_scope.pk]))
    in_scope.refresh_from_db()
    assert not in_scope.is_active
    assert AuditLog.objects.filter(action=AuditLog.Action.USER_DEACTIVATED, target_user=in_scope).exists()
    client.logout()
    client.post(reverse("accounts:login"), {"username": "in_scope", "password": PASSWORD})
    assert "_auth_user_id" not in client.session
    assert client.get(reverse("administration:user_toggle_active", args=[in_scope.pk])).status_code in (302, 405)


def test_password_reset_validates_and_never_logs_password(client, sysadmin, in_scope):
    client.force_login(sysadmin)
    client.post(reverse("administration:user_password", args=[in_scope.pk]),
                {"new_password1": "short", "new_password2": "short"})
    in_scope.refresh_from_db()
    assert in_scope.check_password(PASSWORD)
    client.post(reverse("administration:user_password", args=[in_scope.pk]),
                {"new_password1": NEW_PASSWORD, "new_password2": NEW_PASSWORD})
    in_scope.refresh_from_db()
    assert in_scope.check_password(NEW_PASSWORD)
    for log in AuditLog.objects.all():
        assert NEW_PASSWORD not in json.dumps(log.details) + log.target_label
    assert Notification.objects.filter(recipient=in_scope, type=Notification.Type.SECURITY).exists()


def test_duplicate_username_and_email_rejected(client, sysadmin, in_scope):
    in_scope.email = "taken@example.com"
    in_scope.save()
    client.force_login(sysadmin)
    html = client.post(reverse("administration:user_create"),
                       _create_payload(username="IN_SCOPE", email="Taken@example.com")).content.decode()
    assert "already taken" in html and "already uses this email" in html


def test_user_search_filters(client, sysadmin, in_scope, out_of_scope):
    client.force_login(sysadmin)
    html = client.get(reverse("administration:users"), {"q": "out_of"}).content.decode()
    assert "out_of_scope" in html and ">in_scope<" not in html


# --- Schools -----------------------------------------------------------------------------------------


def test_school_edit_keeps_id_and_is_audited(client, sysadmin, in_scope):
    school = School.objects.get(school_id="SCH001")
    client.force_login(sysadmin)
    response = client.post(reverse("administration:school_edit", args=[school.pk]), {
        "school_id": "HACKED", "name": "Greenfield Primary", "county": "Nakuru", "sub_county": "", "verified_enrolment": 420,
        "is_active": "on",
    })
    assert response.status_code == 302
    school.refresh_from_db()
    assert school.school_id == "SCH001" and school.verified_enrolment == 420
    assert AuditLog.objects.get(action=AuditLog.Action.SCHOOL_UPDATED).school == school


def test_school_management_is_system_admin_only(client, stakeholder, sysadmin):
    school = School.objects.create(school_id="SCH009", name="Elsewhere")
    client.force_login(stakeholder)
    assert client.get(reverse("administration:school_detail", args=[school.pk])).status_code == 403
    client.force_login(sysadmin)
    assert "Elsewhere" in client.get(reverse("administration:schools")).content.decode()
    assert client.get(reverse("administration:school_edit", args=[school.pk])).status_code == 200


def test_school_assign_and_unassign_users(client, sysadmin, in_scope, stub_artifacts):
    sch2 = School.objects.get_or_create(school_id="SCH002")[0]
    client.force_login(sysadmin)
    client.post(reverse("administration:school_detail", args=[sch2.pk]), {"action": "assign", "user": in_scope.pk})
    assert SchoolAssignment.objects.filter(user=in_scope, school=sch2).exists()
    client.post(reverse("administration:school_detail", args=[sch2.pk]), {"action": "unassign", "user": in_scope.pk})
    assert not SchoolAssignment.objects.filter(user=in_scope, school=sch2).exists()


# --- Activity logs ------------------------------------------------------------------------------------


def test_logs_are_admin_only_and_filterable(client, sysadmin, stakeholder, in_scope):
    sch1, sch3 = School.objects.get(school_id="SCH001"), School.objects.get_or_create(school_id="SCH003")[0]
    log_action(sysadmin, AuditLog.Action.SCHOOL_UPDATED, target=sch1, school=sch1)
    log_action(sysadmin, AuditLog.Action.SCHOOL_UPDATED, target=sch3, school=sch3, target_label="Third school")
    client.force_login(stakeholder)
    assert client.get(reverse("administration:logs")).status_code == 403
    client.force_login(sysadmin)
    html = client.get(reverse("administration:logs"), {"action": "school_updated", "q": "root"}).content.decode()
    assert "Third school" in html
    html = client.get(reverse("administration:logs"), {"action": "user_created"}).content.decode()
    assert "Third school" not in html
    assert "start date must be on or before" in client.get(
        reverse("administration:logs"), {"start": "2025-02-01", "end": "2025-01-01"}).content.decode()


def test_audit_details_drop_secret_keys(sysadmin):
    log = log_action(sysadmin, AuditLog.Action.USER_UPDATED, details={"password": "x", "session_key": "y", "fields": ["a"]})
    assert log.details == {"fields": ["a"]}


# --- Help & Support ----------------------------------------------------------------------------------


def test_help_page_reflects_real_configuration(client, in_scope):
    client.force_login(in_scope)
    html = client.get(reverse("support:help")).content.decode()
    for text in ("Getting started", "Forecasting guide", "Next 30 school days", "<code>meals_served</code>",
                 ".csv, .xlsx", "Frequently asked questions", "Report a problem", "About BiteTracker"):
        assert text in html


def test_report_problem_is_stored_and_admins_notified(client, sysadmin, in_scope, media_root):
    client.force_login(in_scope)
    response = client.post(reverse("support:help"), {
        "category": "forecast", "subject": "Forecast page blank", "description": "Nothing loads after submitting.",
        "screenshot": SimpleUploadedFile("shot.png", PNG),
    })
    assert response.status_code == 302
    issue = SupportIssue.objects.get()
    assert issue.reporter == in_scope and issue.status == SupportIssue.Status.OPEN and issue.screenshot
    assert Notification.objects.filter(recipient=sysadmin, type=Notification.Type.SUPPORT).exists()
    assert client.get(reverse("support:issue_screenshot", args=[issue.pk]))["Content-Type"] == "image/png"

    client.force_login(operator("nosy"))
    assert client.get(reverse("support:issue_detail", args=[issue.pk])).status_code == 403
    assert client.get(reverse("support:issue_screenshot", args=[issue.pk])).status_code == 403


def test_report_rejects_non_image_screenshot(client, in_scope, media_root):
    client.force_login(in_scope)
    response = client.post(reverse("support:help"), {
        "category": "other", "subject": "x", "description": "y", "screenshot": SimpleUploadedFile("shot.png", b"<script>"),
    })
    assert response.status_code == 200 and not SupportIssue.objects.exists()


def test_issue_status_workflow_notifies_reporter(client, sysadmin, in_scope):
    issue = SupportIssue.objects.create(reporter=in_scope, category="data", subject="Upload fails", description="...")
    client.force_login(sysadmin)
    assert "Upload fails" in client.get(reverse("administration:issues")).content.decode()
    client.post(reverse("administration:issue_detail", args=[issue.pk]), {"status": "in_progress"})
    issue.refresh_from_db()
    assert issue.status == "in_progress"
    assert Notification.objects.filter(recipient=in_scope, type=Notification.Type.SUPPORT).count() == 1
    assert AuditLog.objects.filter(action=AuditLog.Action.ISSUE_STATUS_CHANGED).exists()


def test_announcement_respects_preferences(client, sysadmin, in_scope, out_of_scope):
    UserProfile.objects.filter(user=out_of_scope).update(notify_announcements=False)
    client.force_login(sysadmin)
    client.post(reverse("administration:announce"), {"title": "Maintenance tonight", "message": "From 22:00."})
    recipients = set(Notification.objects.filter(type=Notification.Type.ANNOUNCEMENT).values_list("recipient__username", flat=True))
    assert recipients == {"in_scope"}  # not the sender, not the opted-out user
    assert AuditLog.objects.get(action=AuditLog.Action.ANNOUNCEMENT_SENT).details == {"recipients": 1}
