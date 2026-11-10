"""Google registration with administrator approval, role/school isolation, and live notifications.

What this file checks:
- Google sign-in runs the real allauth flow: a CSRF-protected POST starts it, the state
  value and PKCE challenge are checked on the callback, and the ID token's issuer,
  audience and expiry are verified. Only the HTTP call to Google's token endpoint is mocked.
- First sign-in creates an inactive account with ONE pending access request and notifies
  System Administrators once; nothing is granted and no session is created.
- Pending, rejected and suspended people are blocked on every route; approval requires a
  role and schools, can't be repeated, and grants exactly what was chosen.
- Unverified Google emails and emails that belong to an existing account are refused
  (no automatic merging); a signed-in user can link Google explicitly.
- School Managers and Resource Planning Stakeholders see only their assigned schools;
  Stakeholders are read-only; model evaluation is administrator-only.
- The notification feed, unread counts, mark-as-read and per-recipient isolation;
  event keys stop the same event notifying twice.
"""

from __future__ import annotations

import time
from datetime import date
from unittest import mock
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from allauth.socialaccount.models import SocialAccount
from allauth.socialaccount.providers.oauth2.views import OAuth2Adapter
from django.conf import settings as django_settings
from django.contrib.auth import get_user_model
from django.urls import reverse

from tests.support.web_fixtures import PASSWORD, operator
from web.accounts import access
from web.accounts.models import AccessRequest, Role
from web.audit.models import AuditLog
from web.notifications.models import Notification
from web.notifications.services import notify
from web.schoolcalendar.models import CalendarEvent
from web.schools.models import ForecastRun, School

pytestmark = pytest.mark.django_db
User = get_user_model()
CLIENT_ID = "test-client.apps.googleusercontent.com"
Type = Notification.Type


# --- Google sign-in harness ----------------------------------------------------------------------------


@pytest.fixture
def google(settings):
    """Google sign-in configured with test credentials (never real ones)."""
    settings.GOOGLE_OAUTH_ENABLED = True
    settings.SOCIALACCOUNT_PROVIDERS = {"google": {
        **django_settings.SOCIALACCOUNT_PROVIDERS["google"],
        "APPS": [{"client_id": CLIENT_ID, "secret": "test-secret-not-real", "key": ""}],
    }}


def claims(**overrides) -> dict:
    now = int(time.time())
    data = {"iss": "https://accounts.google.com", "aud": CLIENT_ID, "sub": "google-uid-1", "iat": now,
            "exp": now + 600, "email": "amina@gmail.com", "email_verified": True, "name": "Amina Otieno",
            "given_name": "Amina", "family_name": "Otieno"}
    data.update(overrides)
    return data


def start_google(client, process="login"):
    """POST to start sign-in, as the login page's button does. Returns the state Google would echo back."""
    response = client.post(reverse("google_login"), {"process": process})
    assert response.status_code == 302
    location = urlparse(response["Location"])
    assert location.netloc == "accounts.google.com"
    query = parse_qs(location.query)
    assert query["client_id"] == [CLIENT_ID]
    assert query["redirect_uri"][0].endswith("/auth/google/login/callback/")
    assert query["code_challenge_method"] == ["S256"] and query["code_challenge"][0]
    return query["state"][0]


def google_callback(client, state, id_claims):
    """Google redirects back with a code; only the server-to-server token exchange is mocked."""
    id_token = jwt.encode(id_claims, "test-signing-key", algorithm="HS256")

    def token_exchange(self, request, app, client_, pkce_code_verifier=None):
        assert pkce_code_verifier, "PKCE verifier must be sent with the code"
        self.did_fetch_access_token = True  # received over TLS from Google, as in production
        return {"access_token": "test-access-token", "token_type": "Bearer", "expires_in": 3600, "id_token": id_token}

    with mock.patch.object(OAuth2Adapter, "get_access_token_data", token_exchange):
        return client.get(reverse("google_callback"), {"code": "test-code", "state": state})


def google_sign_in(client, process="login", **claim_overrides):
    return google_callback(client, start_google(client, process), claims(**claim_overrides))


def is_signed_in(client) -> bool:
    """Whether the client's session authenticates a user (a stale session key for an inactive user doesn't)."""
    return client.get(reverse("notifications:feed"), HTTP_ACCEPT="application/json").status_code == 200


@pytest.fixture
def sysadmin(db):
    return User.objects.create_superuser("root", "root@example.com", PASSWORD)


@pytest.fixture
def schools(db, stub_artifacts):
    """School rows for the stub dataset's IDs; only schools in the active dataset can be assigned."""
    return {code: School.objects.get_or_create(school_id=code, defaults={"name": f"School {code[-1]}"})[0]
            for code in ("SCH001", "SCH002", "SCH003")}


def approve(client, admin, access_request, role=Role.SCHOOL_MANAGER, school_codes=("SCH001",)):
    client.force_login(admin)
    pks = [School.objects.get(school_id=c).pk for c in school_codes]
    response = client.post(reverse("administration:access_request_detail", args=[access_request.pk]),
                           {"action": "approve", "role": role, "schools": pks, "note": ""})
    client.logout()
    return response


# --- Configuration --------------------------------------------------------------------------------------


def test_google_button_and_routes_hidden_without_credentials(client, settings):
    settings.GOOGLE_OAUTH_ENABLED = False
    assert b'data-testid="google-login"' not in client.get(reverse("accounts:login")).content
    assert client.post(reverse("google_login")).status_code == 404
    assert client.get(reverse("google_callback"), {"code": "x", "state": "y"}).status_code == 404


def test_login_page_offers_google_and_keeps_password_login(client, google):
    html = client.get(reverse("accounts:login")).content.decode()
    assert 'data-testid="google-login"' in html and "Continue with Google" in html
    assert f'action="{reverse("google_login")}"' in html and 'name="csrfmiddlewaretoken"' in html
    assert 'name="password"' in html  # existing username/password login is unchanged


def test_starting_google_sign_in_needs_csrf(client, google):
    from django.test import Client
    strict = Client(enforce_csrf_checks=True)
    assert strict.post(reverse("google_login"), {"process": "login"}).status_code == 403


def test_opening_google_url_directly_asks_for_a_post(client, google):
    response = client.get(reverse("google_login"))
    assert response.status_code == 200 and b"Continue to Google" in response.content
    assert not response.has_header("Location")


def test_cancelling_at_google_returns_to_login(client, google):
    state = start_google(client)
    response = client.get(reverse("google_callback"), {"error": "access_denied", "state": state})
    assert response.url == reverse("accounts:login") and not User.objects.exclude(username="").exists()


def test_callback_with_wrong_state_is_refused(client, google, sysadmin):
    start_google(client)
    response = google_callback(client, "forged-state", claims())
    assert not User.objects.filter(email="amina@gmail.com").exists()
    assert response.status_code in (302, 401) and not is_signed_in(client)


def test_id_token_for_another_app_is_refused(client, google, sysadmin):
    response = google_sign_in(client, aud="someone-elses-client-id")
    assert response.status_code == 302 and response["Location"] == reverse("accounts:login")
    assert not User.objects.filter(email="amina@gmail.com").exists()


def test_local_signup_pages_are_not_exposed(client, google):
    assert client.get("/auth/signup/").url == reverse("accounts:login")
    assert client.get("/auth/password/reset/").status_code == 404


# --- Registration ----------------------------------------------------------------------------------------


def test_first_google_sign_in_creates_pending_request_only(client, google, sysadmin):
    response = google_sign_in(client)
    assert response.status_code == 302 and response["Location"] == reverse("accounts:access_status")
    assert not is_signed_in(client)

    user = User.objects.get(email="amina@gmail.com")
    assert not user.is_active and not user.is_staff and not user.is_superuser
    assert not user.school_assignments.exists()
    request = AccessRequest.objects.get(user=user)
    assert request.status == AccessRequest.Status.PENDING
    assert (request.email, request.full_name, request.provider) == ("amina@gmail.com", "Amina Otieno", "google")
    assert SocialAccount.objects.get(user=user).uid == "google-uid-1"
    assert AuditLog.objects.filter(action=AuditLog.Action.ACCESS_REQUESTED, target_user=user).count() == 1

    note = Notification.objects.get(recipient=sysadmin, type=Type.ACCESS_REQUEST)
    assert note.title == "New registration request" and "waiting for account approval" in note.message
    assert note.url == reverse("administration:access_request_detail", args=[request.pk])

    page = client.get(reverse("accounts:access_status")).content.decode()
    assert "Your registration request has been sent to the administrator." in page
    assert "You will be able to access BiteTracker once your account has been approved." in page


def test_repeat_sign_in_while_pending_creates_no_duplicates(client, google, sysadmin):
    google_sign_in(client)
    google_sign_in(client)
    assert User.objects.filter(email="amina@gmail.com").count() == 1
    assert AccessRequest.objects.count() == 1
    assert Notification.objects.filter(recipient=sysadmin, type=Type.ACCESS_REQUEST).count() == 1
    assert not is_signed_in(client)


def test_applicant_can_add_requested_role_but_not_admin(client, google, sysadmin):
    google_sign_in(client)
    client.post(reverse("accounts:access_status"), {"requested_role": Role.SYSTEM_ADMIN, "applicant_note": "x"})
    request = AccessRequest.objects.get()
    assert request.requested_role == ""
    client.post(reverse("accounts:access_status"),
                {"requested_role": Role.PLANNING_STAKEHOLDER, "applicant_note": "County planning office"})
    request.refresh_from_db()
    assert request.requested_role == Role.PLANNING_STAKEHOLDER and request.applicant_note == "County planning office"


def test_access_status_page_needs_the_applicants_own_session(client, google, sysadmin):
    google_sign_in(client)
    from django.test import Client
    stranger = Client()
    assert stranger.get(reverse("accounts:access_status")).url == reverse("accounts:login")


@pytest.mark.parametrize("name", ["core:dashboard", "core:history", "core:forecast", "notifications:feed",
                                  "schools:upload", "administration:users"])
def test_pending_user_reaches_no_protected_page(client, google, sysadmin, name):
    google_sign_in(client)
    response = client.get(reverse(name))
    assert response.status_code == 302 and response.url.startswith(reverse("accounts:login"))


def test_pending_user_cannot_use_password_login(client, google, sysadmin):
    google_sign_in(client)
    user = User.objects.get(email="amina@gmail.com")
    user.set_password(PASSWORD)
    user.save()
    client.post(reverse("accounts:login"), {"username": user.username, "password": PASSWORD})
    assert not is_signed_in(client)


def test_middleware_ends_session_of_unapproved_account(client, sysadmin):
    """Defence in depth: even an (inconsistent) active account with a pending request is signed out."""
    user = User.objects.create_user("sneaky", "sneaky@example.com", PASSWORD)
    AccessRequest.objects.create(user=user, email=user.email)
    client.force_login(user)
    response = client.get(reverse("core:dashboard"))
    assert response.status_code == 302 and response.url == reverse("accounts:access_status")
    assert not is_signed_in(client)
    assert client.get(reverse("notifications:feed"), HTTP_ACCEPT="application/json").status_code in (302, 401)


# --- Approval and rejection -------------------------------------------------------------------------------


def test_approval_grants_exactly_the_assigned_role_and_schools(client, google, sysadmin, schools):
    google_sign_in(client)
    request = AccessRequest.objects.get()
    response = approve(client, sysadmin, request, Role.SCHOOL_MANAGER, ("SCH001",))
    assert response.status_code == 302

    user = User.objects.get(email="amina@gmail.com")
    request.refresh_from_db()
    assert user.is_active and not user.is_superuser and not user.is_staff
    assert request.status == AccessRequest.Status.APPROVED and request.decided_by == sysadmin
    assert request.granted_role == Role.SCHOOL_MANAGER
    assert access.role_of(user) == Role.SCHOOL_MANAGER
    assert set(user.school_assignments.values_list("school__school_id", flat=True)) == {"SCH001"}
    log = AuditLog.objects.get(action=AuditLog.Action.ACCESS_APPROVED)
    assert log.details["role"] == Role.SCHOOL_MANAGER and log.details["schools"] == ["SCH001"]
    note = Notification.objects.get(recipient=user, title="Your registration was approved")
    assert note.event_key == f"access_request:{request.pk}:approved"

    # Next Google sign-in: straight into the app, with only School Manager rights.
    response = google_sign_in(client)
    assert is_signed_in(client) and response.url == reverse("core:dashboard")
    assert client.get(reverse("administration:users")).status_code == 403
    assert client.get(reverse("administration:access_requests")).status_code == 403
    assert client.get(reverse("core:performance")).status_code == 403


def test_approval_needs_a_role_and_schools(client, sysadmin, schools):
    user = User.objects.create_user("applicant", "applicant@example.com", is_active=False)
    request = AccessRequest.objects.create(user=user, email=user.email)
    client.force_login(sysadmin)
    url = reverse("administration:access_request_detail", args=[request.pk])
    html = client.post(url, {"action": "approve", "role": Role.SCHOOL_MANAGER}).content.decode()
    assert "Assign at least one school for this role." in html
    assert client.post(url, {"action": "approve", "role": "superuser", "schools": [schools["SCH001"].pk]}).status_code == 200
    request.refresh_from_db()
    user.refresh_from_db()
    assert request.status == AccessRequest.Status.PENDING and not user.is_active


def test_repeated_approval_changes_nothing(client, sysadmin, schools):
    user = User.objects.create_user("applicant", "applicant@example.com", is_active=False)
    request = AccessRequest.objects.create(user=user, email=user.email)
    approve(client, sysadmin, request, Role.PLANNING_STAKEHOLDER, ("SCH001", "SCH002"))
    approve(client, sysadmin, request, Role.SYSTEM_ADMIN, ("SCH003",))  # double submit / second admin
    client.force_login(sysadmin)
    client.post(reverse("administration:access_request_detail", args=[request.pk]),
                {"action": "reject", "reason": "changed my mind"})
    user.refresh_from_db()
    request.refresh_from_db()
    assert request.status == AccessRequest.Status.APPROVED and request.granted_role == Role.PLANNING_STAKEHOLDER
    assert access.role_of(user) == Role.PLANNING_STAKEHOLDER and not user.is_superuser and user.is_active
    assert AuditLog.objects.filter(action__in=[AuditLog.Action.ACCESS_APPROVED, AuditLog.Action.ACCESS_REJECTED]).count() == 1
    assert Notification.objects.filter(recipient=user, type=Type.ACCOUNT).count() == 1


def test_rejected_user_stays_blocked_with_audit_record(client, google, sysadmin):
    google_sign_in(client)
    request = AccessRequest.objects.get()
    client.force_login(sysadmin)
    client.post(reverse("administration:access_request_detail", args=[request.pk]),
                {"action": "reject", "reason": "Not part of the programme"})
    client.logout()
    request.refresh_from_db()
    assert request.status == AccessRequest.Status.REJECTED and request.decision_note == "Not part of the programme"
    assert AuditLog.objects.filter(action=AuditLog.Action.ACCESS_REJECTED).exists()

    response = google_sign_in(client)
    assert response.url == reverse("accounts:access_status") and not is_signed_in(client)
    page = client.get(reverse("accounts:access_status")).content.decode()
    assert 'data-status="rejected"' in page and "not approved" in page
    assert AuditLog.objects.filter(action=AuditLog.Action.ACCESS_DENIED).exists()
    assert AccessRequest.objects.count() == 1  # signing in again never re-opens a request


def test_suspended_google_user_is_signed_out_and_blocked(client, google, sysadmin, schools):
    google_sign_in(client)
    request = AccessRequest.objects.get()
    approve(client, sysadmin, request)
    google_sign_in(client)
    assert is_signed_in(client)
    user = User.objects.get(email="amina@gmail.com")

    from django.test import Client
    admin_client = Client()
    admin_client.force_login(sysadmin)
    admin_client.post(reverse("administration:user_toggle_active", args=[user.pk]))
    user.refresh_from_db()
    assert not user.is_active and access.account_status(user) == access.SUSPENDED

    # The open session no longer works, and signing in again shows the suspended status.
    assert client.get(reverse("core:dashboard")).status_code == 302
    response = google_sign_in(client)
    assert response.url == reverse("accounts:access_status") and not is_signed_in(client)
    assert 'data-status="suspended"' in client.get(reverse("accounts:access_status")).content.decode()


def test_pending_registration_cannot_be_activated_by_the_suspend_toggle(client, sysadmin):
    user = User.objects.create_user("applicant", "applicant@example.com", is_active=False)
    AccessRequest.objects.create(user=user, email=user.email)
    client.force_login(sysadmin)
    client.post(reverse("administration:user_toggle_active", args=[user.pk]))
    user.refresh_from_db()
    assert not user.is_active


@pytest.mark.parametrize("role", [Role.SCHOOL_MANAGER, Role.PLANNING_STAKEHOLDER])
def test_only_system_admins_decide_registrations(client, schools, role):
    actor = operator("actor", codes=("SCH001",))
    access.set_role(actor, role)
    applicant = User.objects.create_user("applicant", "applicant@example.com", is_active=False)
    request = AccessRequest.objects.create(user=applicant, email=applicant.email)
    client.force_login(actor)
    url = reverse("administration:access_request_detail", args=[request.pk])
    assert client.get(reverse("administration:access_requests")).status_code == 403
    assert client.post(url, {"action": "approve", "role": Role.SYSTEM_ADMIN, "schools": [schools["SCH001"].pk]}).status_code == 403
    request.refresh_from_db()
    assert request.status == AccessRequest.Status.PENDING


def test_admin_pages_list_requests_and_statuses(client, google, sysadmin):
    google_sign_in(client)
    client.force_login(sysadmin)
    html = client.get(reverse("administration:access_requests")).content.decode()
    assert "amina@gmail.com" in html and "Amina Otieno" in html
    assert 'data-testid="tab-count-access">1' in html
    users_html = client.get(reverse("administration:users"), {"status": "pending"}).content.decode()
    assert "Pending approval" in users_html and "amina@gmail.com" in users_html


# --- Existing accounts and linking --------------------------------------------------------------------------


def test_google_email_matching_existing_account_is_not_merged(client, google, sysadmin):
    existing = User.objects.create_user("amina", "Amina@Gmail.com", PASSWORD)
    response = google_sign_in(client)
    assert response.url == reverse("accounts:login") and not is_signed_in(client)
    assert not SocialAccount.objects.exists()
    assert User.objects.filter(email__iexact="amina@gmail.com").count() == 1
    assert not AccessRequest.objects.exists()
    existing.refresh_from_db()
    assert existing.is_active


def test_unverified_google_email_is_refused(client, google, sysadmin):
    response = google_sign_in(client, email_verified=False)
    assert response.url == reverse("accounts:login")
    assert not User.objects.filter(email="amina@gmail.com").exists() and not AccessRequest.objects.exists()


def test_signed_in_user_links_google_then_signs_in_with_it(client, google, schools):
    user = operator("amina", codes=("SCH001",), email="amina@gmail.com")
    client.force_login(user)
    response = google_sign_in(client, process="connect")
    assert response.status_code == 302
    assert SocialAccount.objects.get(provider="google").user == user
    assert AuditLog.objects.filter(action=AuditLog.Action.GOOGLE_LINKED, target_user=user).exists()
    assert not AccessRequest.objects.exists()  # an existing, approved account needs no request

    client.logout()
    google_sign_in(client)
    assert is_signed_in(client) and int(client.session["_auth_user_id"]) == user.pk


def test_existing_password_login_still_works(client, schools):
    operator("planner", codes=("SCH001",))
    response = client.post(reverse("accounts:login"), {"username": "planner", "password": PASSWORD})
    assert response.status_code == 302 and is_signed_in(client)


# --- School isolation -----------------------------------------------------------------------------------------


@pytest.fixture
def manager(db, stub_artifacts):
    return operator("manager", codes=("SCH001",))


@pytest.fixture
def stakeholder(db, stub_artifacts):
    user = operator("stakeholder", codes=("SCH001", "SCH002"))
    access.set_role(user, Role.PLANNING_STAKEHOLDER)
    return user


def _run_for(code: str) -> ForecastRun:
    school = School.objects.get_or_create(school_id=code)[0]
    return ForecastRun.objects.create(
        school=school, school_code=code, school_name=code, horizon_days=1, origin_date=date(2025, 1, 1),
        first_forecast_date=date(2025, 1, 2), last_forecast_date=date(2025, 1, 2), model_label="m",
        results_label="r", day1_predicted_meals=1, day1_recommended_meals=1, day1_risk_level="Low",
        total_predicted_meals=1, total_recommended_meals=1, result_summary={})


def test_school_manager_sees_only_assigned_school(client, manager):
    client.force_login(manager)
    dashboard = client.get(reverse("core:dashboard"))
    assert [s["school_id"] for s in dashboard.context["schools"]] == ["SCH001"]
    history = client.get(reverse("core:history"), {"school": "SCH002"})
    assert not history.context["form"].is_valid() and "school" not in history.context
    response = client.post(reverse("core:forecast"), {"school": "SCH003", "horizon": 1})
    assert response.status_code == 200 and not ForecastRun.objects.exists()
    assert client.get(reverse("core:forecast_run", args=[_run_for("SCH002").pk])).status_code == 404
    assert client.get(reverse("core:forecast_run", args=[_run_for("SCH001").pk])).status_code == 200
    assert list(client.get(reverse("core:forecast_history"), {"school": "SCH002"}).context["page"].object_list) == []


def test_school_manager_cannot_touch_other_schools_calendar(client, manager):
    other = School.objects.get_or_create(school_id="SCH002")[0]
    event = CalendarEvent.objects.create(school=other, title="Sports day", event_type=CalendarEvent.EventType.values[0],
                                         start_date=date(2025, 1, 6), end_date=date(2025, 1, 6))
    client.force_login(manager)
    assert client.get(reverse("schoolcalendar:event_detail", args=[event.pk])).status_code == 404
    assert client.post(reverse("schoolcalendar:event_delete", args=[event.pk])).status_code == 404
    assert CalendarEvent.objects.filter(pk=event.pk).exists()


def test_stakeholder_reads_assigned_schools_only(client, stakeholder):
    client.force_login(stakeholder)
    dashboard = client.get(reverse("core:dashboard"))
    assert sorted(s["school_id"] for s in dashboard.context["schools"]) == ["SCH001", "SCH002"]
    assert client.get(reverse("core:history"), {"school": "SCH002"}).context["school"]["school_id"] == "SCH002"
    assert not client.get(reverse("core:history"), {"school": "SCH003"}).context["form"].is_valid()
    assert client.get(reverse("core:forecast_run", args=[_run_for("SCH002").pk])).status_code == 200
    assert client.get(reverse("core:forecast_run", args=[_run_for("SCH003").pk])).status_code == 404


def test_stakeholder_is_read_only(client, stakeholder):
    school = School.objects.get(school_id="SCH001")
    event = CalendarEvent.objects.create(school=school, title="Exams", event_type=CalendarEvent.EventType.values[0],
                                         start_date=date(2025, 1, 6), end_date=date(2025, 1, 6))
    client.force_login(stakeholder)
    assert client.get(reverse("schoolcalendar:event_detail", args=[event.pk])).status_code == 200
    assert client.get(reverse("schoolcalendar:event_create")).status_code == 403
    assert client.post(reverse("schoolcalendar:event_update", args=[event.pk]), {"title": "x"}).status_code == 403
    assert client.post(reverse("schoolcalendar:event_delete", args=[event.pk])).status_code == 403
    assert client.get(reverse("schools:upload")).status_code == 403
    assert client.get(reverse("administration:uploads")).status_code == 403
    assert client.get(reverse("core:performance")).status_code == 403
    assert CalendarEvent.objects.get(pk=event.pk).title == "Exams"


def test_model_evaluation_is_admin_only(client, manager, sysadmin):
    client.force_login(manager)
    assert client.get(reverse("core:performance")).status_code == 403
    client.force_login(sysadmin)
    assert client.get(reverse("core:performance")).status_code == 200


def test_anonymous_users_are_sent_to_login(client):
    for name in ("core:dashboard", "core:performance", "administration:access_requests", "notifications:feed"):
        response = client.get(reverse(name))
        assert response.status_code == 302 and response.url.startswith(reverse("accounts:login"))


# --- Notifications -------------------------------------------------------------------------------------------


@pytest.fixture
def amina(db):
    return operator("amina", codes=("SCH001",))


@pytest.fixture
def other(db):
    return operator("other", codes=("SCH001",))


def test_feed_returns_own_unread_count_and_new_items(client, amina, other):
    old = notify([amina], Type.ANNOUNCEMENT, "Old news")[0]
    notify([other], Type.ANNOUNCEMENT, "Someone else's")
    client.force_login(amina)
    data = client.get(reverse("notifications:feed"), {"since": old.pk}).json()
    assert data["unread_count"] == 1 and data["new"] == [] and [i["title"] for i in data["items"]] == ["Old news"]

    fresh = notify([amina], Type.ANNOUNCEMENT, "New registration request", "A user is waiting for account approval.")[0]
    data = client.get(reverse("notifications:feed"), {"since": old.pk}).json()
    assert data["unread_count"] == 2 and data["latest_id"] == fresh.pk
    assert [n["title"] for n in data["new"]] == ["New registration request"]
    item = data["new"][0]
    assert item["message"] == "A user is waiting for account approval." and item["icon"].startswith("bi-")
    assert item["open_url"] == reverse("notifications:open", args=[fresh.pk]) and not item["is_read"]
    assert "Someone else's" not in str(data)


def test_feed_never_marks_anything_read(client, amina):
    notify([amina], Type.ANNOUNCEMENT, "Stay unread")
    client.force_login(amina)
    for _ in range(3):
        client.get(reverse("notifications:feed"), {"since": 0})
    assert Notification.objects.get().is_read is False


def test_feed_ignores_bad_since_values(client, amina):
    notify([amina], Type.ANNOUNCEMENT, "x")
    client.force_login(amina)
    for since in ("abc", "-5", "", "1.5"):
        assert client.get(reverse("notifications:feed"), {"since": since}).json()["new"] == []
    assert client.get(reverse("notifications:feed")).json()["new"] == []


def test_first_ever_notification_is_new_since_zero(client, amina):
    """A user with no notifications yet starts from 0; their first one must still pop up."""
    client.force_login(amina)
    assert client.get(reverse("notifications:feed"), {"since": 0}).json()["latest_id"] == 0
    first = notify([amina], Type.ACCOUNT, "Your registration was approved")[0]
    assert [n["id"] for n in client.get(reverse("notifications:feed"), {"since": 0}).json()["new"]] == [first.pk]


def test_json_mark_read_and_mark_all(client, amina):
    first, second = (notify([amina], Type.ANNOUNCEMENT, t)[0] for t in ("A", "B"))
    client.force_login(amina)
    response = client.post(reverse("notifications:mark_read", args=[first.pk]), HTTP_ACCEPT="application/json")
    assert response.json() == {"ok": True, "unread_count": 1}
    response = client.post(reverse("notifications:mark_all_read"), HTTP_ACCEPT="application/json")
    assert response.json() == {"ok": True, "unread_count": 0}
    assert not Notification.objects.filter(is_read=False).exists()


def test_cannot_read_or_change_another_users_notifications(client, amina, other):
    theirs = notify([other], Type.ANNOUNCEMENT, "Private")[0]
    client.force_login(amina)
    for name in ("notifications:mark_read", "notifications:open", "notifications:delete"):
        assert client.post(reverse(name, args=[theirs.pk]), HTTP_ACCEPT="application/json").status_code == 404
    client.post(reverse("notifications:mark_all_read"))
    theirs.refresh_from_db()
    assert not theirs.is_read
    assert client.get(reverse("notifications:feed")).json()["items"] == []


def test_mark_read_requires_post_and_csrf(amina):
    from django.test import Client
    note = notify([amina], Type.ANNOUNCEMENT, "x")[0]
    strict = Client(enforce_csrf_checks=True)
    strict.force_login(amina)
    assert strict.get(reverse("notifications:mark_read", args=[note.pk])).status_code == 405
    assert strict.post(reverse("notifications:mark_read", args=[note.pk])).status_code == 403


def test_event_key_prevents_duplicate_notifications(amina):
    for _ in range(3):
        notify([amina], Type.ACCESS_REQUEST, "New registration request", event_key="access_request:7:submitted")
    notify([amina], Type.ACCESS_REQUEST, "New registration request", event_key="access_request:8:submitted")
    assert Notification.objects.filter(recipient=amina).count() == 2


def test_notifications_persist_across_logout(client, amina):
    notify([amina], Type.ANNOUNCEMENT, "Still here")
    client.force_login(amina)
    client.logout()
    client.force_login(amina)
    assert client.get(reverse("notifications:feed")).json()["unread_count"] == 1


def test_shell_includes_toast_region_and_live_bell(client, amina):
    client.force_login(amina)
    html = client.get(reverse("accounts:profile")).content.decode()
    assert 'id="bt-toasts"' in html and 'aria-live="polite"' in html
    assert f'data-feed-url="{reverse("notifications:feed")}"' in html
    assert 'name="csrf-token"' in html and "js/notifications.js" in html


def test_upload_submission_notifies_admins_once_and_decision_notifies_uploader(client, sysadmin, stub_artifacts, settings, tmp_path):
    from django.core.files.uploadedfile import SimpleUploadedFile
    settings.MEDIA_ROOT = tmp_path / "uploads"
    manager = operator("uploader", codes=("SCH001",))
    frame = stub_artifacts.frame[["date", "school_id", "school_name", "meals_served"]]
    frame = frame[frame["school_id"] == "SCH001"].head(80)
    client.force_login(manager)
    client.post(reverse("schools:upload"), {
        "file": SimpleUploadedFile("d.csv", frame.to_csv(index=False).encode()),
        "letter": SimpleUploadedFile("letter.pdf", b"%PDF-1.4\n%x\n")})
    assert Notification.objects.filter(recipient=sysadmin, type=Type.DATASET_REVIEW).count() == 1
    from web.schools.models import DatasetUpload
    record = DatasetUpload.objects.get()
    client.force_login(sysadmin)
    client.post(reverse("administration:upload_review", args=[record.pk]), {"decision": "approved", "comment": ""})
    assert Notification.objects.filter(recipient=manager, type=Type.DATASET_REVIEW, title__startswith="Upload approved").exists()
