"""Regression tests for the approved-Google-user dashboard fault and notification targeting.

The fault seen in local testing (2026-11-10): a user approved as School Manager for a legacy
synthetic school (SCH001) that has no records in the active Food4Education dataset saw
- "No schools are assigned to your account yet" although the assignment was saved,
- the model-evaluation card (metrics and a Model performance link), and
- an old synthetic forecast run listed as if it were a current result.

These tests pin the fixed behaviour:
- an accurate message names the assigned schools that have no dataset records;
- non-administrators never see other schools' runs, and never see synthetic development runs
  while the active model is real (lists and direct URLs);
- model evaluation is System Administrator-only in the UI as well as the backend;
- the approver gets "You approved ..." and nobody else does; the applicant's own notice stays
  the applicant's; notifications are isolated between two administrators and two users.
"""

from __future__ import annotations

from datetime import date

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

from services import model_service
from src.feature_engineering import FOOD4EDUCATION_SOURCE
from tests.support.web_fixtures import PASSWORD, operator
from web.accounts import access
from web.accounts.adapters import _announce_access_request
from web.accounts.models import AccessRequest, Role
from web.notifications.models import Notification
from web.notifications.services import notify
from web.schools.models import ForecastRun, School

pytestmark = pytest.mark.django_db
User = get_user_model()
Type = Notification.Type

REAL_LABEL = "FOOD4EDUCATION RESEARCH RESULTS"
SYNTHETIC_LABEL = "SYNTHETIC DEVELOPMENT RESULTS"


@pytest.fixture
def real_model(stub_artifacts, make_metadata):
    """The active model is a Food4Education (real-data) model; the stub dataset has SCH001-SCH003."""
    stub_artifacts.write_metadata(make_metadata(results_label=REAL_LABEL, data_source=FOOD4EDUCATION_SOURCE))
    model_service.clear_model_cache()
    return stub_artifacts


@pytest.fixture
def legacy_school(db):
    """A school row with no records in the active dataset (like the legacy synthetic SCH001 locally)."""
    return School.objects.create(school_id="LEGACY1", name="Greenfield Legacy", data_source="SYNTHETIC DEVELOPMENT DATA")


@pytest.fixture
def admins(db):
    return (User.objects.create_superuser("admin_a", "a@example.com", PASSWORD),
            User.objects.create_superuser("admin_b", "b@example.com", PASSWORD))


def _run(code: str, label: str) -> ForecastRun:
    school = School.objects.get_or_create(school_id=code)[0]
    return ForecastRun.objects.create(
        school=school, school_code=code, school_name=code, horizon_days=1, origin_date=date(2025, 1, 1),
        first_forecast_date=date(2025, 1, 2), last_forecast_date=date(2025, 1, 2), model_label="m",
        results_label=label, day1_predicted_meals=1, day1_recommended_meals=1, day1_risk_level="Low",
        total_predicted_meals=1, total_recommended_meals=1, result_summary={})


def _applicant(username="applicant", email="ada@example.com", full_name="Ada Lovelace") -> AccessRequest:
    user = User.objects.create_user(username=username, email=email, password=PASSWORD, is_active=False)
    return AccessRequest.objects.create(user=user, email=email, full_name=full_name)


def _approve(client, admin, req, school_codes, role=Role.SCHOOL_MANAGER):
    client.force_login(admin)
    pks = [School.objects.get(school_id=c).pk for c in school_codes]
    response = client.post(reverse("administration:access_request_detail", args=[req.pk]),
                           {"action": "approve", "role": role, "schools": pks, "note": ""}, follow=True)
    client.logout()
    return response


# --- School assignment vs dataset identifiers ------------------------------------------------------------


def test_school_without_dataset_records_gets_accurate_message_not_no_schools(client, real_model, legacy_school):
    user = operator("legacy_manager", codes=("LEGACY1",))
    assert access.accessible_school_codes(user) == {"LEGACY1"}  # the assignment is saved
    client.force_login(user)
    for name in ("core:dashboard", "core:history", "core:forecast"):
        message = client.get(reverse(name)).context["data_error"]
        assert "No schools are assigned" not in message
        assert "Greenfield Legacy (LEGACY1)" in message and "no records in the active dataset" in message


def test_user_with_no_assignment_still_told_no_schools(client, real_model):
    client.force_login(operator("nobody", codes=()))
    assert "No schools are assigned" in client.get(reverse("core:dashboard")).context["data_error"]


def test_approved_google_user_sees_exactly_their_assigned_dataset_school(client, real_model, admins):
    for code in ("SCH001", "SCH002", "SCH003"):
        School.objects.get_or_create(school_id=code)
    req = _applicant()
    _approve(client, admins[0], req, ("SCH002",))
    user = User.objects.get(pk=req.user_id)
    assert user.is_active and access.role_of(user) == Role.SCHOOL_MANAGER
    assert access.accessible_school_codes(user) == {"SCH002"}
    client.force_login(user)
    response = client.get(reverse("core:dashboard"))
    assert [s["school_id"] for s in response.context["schools"]] == ["SCH002"]
    assert "data_error" not in response.context


# --- Forecast runs: school scope and synthetic runs ---------------------------------------------------------


@pytest.fixture
def runs(db):
    return {
        "synthetic_own": _run("SCH001", SYNTHETIC_LABEL),
        "real_own": _run("SCH001", REAL_LABEL),
        "real_other": _run("SCH002", REAL_LABEL),
        "synthetic_other": _run("SCH002", SYNTHETIC_LABEL),
    }


def test_non_admin_sees_only_real_runs_for_assigned_schools_when_model_is_real(client, real_model, runs):
    client.force_login(operator("manager", codes=("SCH001",)))
    assert [r.pk for r in client.get(reverse("core:dashboard")).context["recent_runs"]] == [runs["real_own"].pk]
    listed = client.get(reverse("core:forecast_history")).context["page"].object_list
    assert [r.pk for r in listed] == [runs["real_own"].pk]
    for key in ("synthetic_own", "real_other", "synthetic_other"):  # direct URLs
        assert client.get(reverse("core:forecast_run", args=[runs[key].pk])).status_code == 404, key
    assert client.get(reverse("core:forecast_run", args=[runs["real_own"].pk])).status_code == 200
    assert list(client.get(reverse("core:forecast_history"), {"school": "SCH002"}).context["page"].object_list) == []


def test_admin_sees_all_runs_with_synthetic_ones_clearly_labelled(client, real_model, runs, admins):
    client.force_login(admins[0])
    html = client.get(reverse("core:forecast_history")).content.decode()
    assert html.count('data-testid="run-data-synthetic"') == 2 and "Synthetic · development only" in html
    assert html.count('data-testid="run-data-real"') == 2
    page = client.get(reverse("core:forecast_run", args=[runs["synthetic_own"].pk]))
    assert page.status_code == 200 and 'data-testid="run-synthetic-note"' in page.content.decode()
    real = client.get(reverse("core:forecast_run", args=[runs["real_own"].pk])).content.decode()
    assert 'data-testid="run-synthetic-note"' not in real


def test_synthetic_runs_stay_visible_to_assigned_users_when_active_model_is_synthetic(client, stub_artifacts, runs):
    client.force_login(operator("manager", codes=("SCH001",)))
    listed = {r.pk for r in client.get(reverse("core:forecast_history")).context["page"].object_list}
    assert listed == {runs["synthetic_own"].pk, runs["real_own"].pk}


# --- Model evaluation is administrator-only -----------------------------------------------------------------


@pytest.mark.parametrize("role", [Role.SCHOOL_MANAGER, Role.PLANNING_STAKEHOLDER])
def test_model_evaluation_hidden_and_blocked_for_non_admins(client, real_model, role):
    user = operator("viewer", codes=("SCH001",))
    access.set_role(user, role)
    client.force_login(user)
    dashboard = client.get(reverse("core:dashboard"))
    html = dashboard.content.decode()
    assert 'data-testid="model-card"' not in html and "Test MAE" not in html
    assert reverse("core:performance") not in html and "model_metadata" not in dashboard.context
    assert reverse("core:performance") not in client.get(reverse("support:help")).content.decode()
    assert client.get(reverse("core:performance")).status_code == 403


def test_model_evaluation_visible_to_system_admin(client, real_model, admins):
    client.force_login(admins[0])
    html = client.get(reverse("core:dashboard")).content.decode()
    assert 'data-testid="model-card"' in html and reverse("core:performance") in html
    assert client.get(reverse("core:performance")).status_code == 200


# --- Notification targeting and isolation -------------------------------------------------------------------


@pytest.fixture
def people(db, admins):
    user_1, user_2 = operator("user_1", codes=("SCH001",)), operator("user_2", codes=("SCH002",))
    return {"admin_a": admins[0], "admin_b": admins[1], "user_1": user_1, "user_2": user_2}


def _titles(user) -> list[str]:
    return list(Notification.objects.filter(recipient=user).values_list("title", flat=True))


def test_registration_request_goes_only_to_active_system_admins(people):
    req = _applicant()
    _announce_access_request(req)
    for key in ("admin_a", "admin_b"):
        assert _titles(people[key]) == ["New registration request"]
    for key in ("user_1", "user_2"):
        assert _titles(people[key]) == []
    assert _titles(req.user) == []


def test_approval_confirmation_goes_only_to_the_approving_admin(client, real_model, people):
    School.objects.filter(school_id="SCH001").update(name="Greenfield Primary")
    req = _applicant()
    _announce_access_request(req)
    _approve(client, people["admin_a"], req, ("SCH001",))

    confirmation = Notification.objects.get(recipient=people["admin_a"], title="Registration approved")
    assert confirmation.message == "You approved Ada Lovelace (ada@example.com) as School Manager for Greenfield Primary (SCH001)."
    assert confirmation.type == Type.ACCESS_REQUEST
    assert "Your registration was approved" not in _titles(people["admin_a"])
    assert "Registration approved" not in _titles(people["admin_b"])
    for key in ("user_1", "user_2"):
        assert _titles(people[key]) == []

    applicant_notes = Notification.objects.filter(title="Your registration was approved")
    assert [n.recipient_id for n in applicant_notes] == [req.user_id]
    assert applicant_notes.get().message.startswith("Welcome to BiteTracker. Your role is School Manager")


def test_rejection_confirmation_goes_only_to_the_rejecting_admin(client, people):
    req = _applicant()
    client.force_login(people["admin_b"])
    client.post(reverse("administration:access_request_detail", args=[req.pk]), {"action": "reject", "reason": "Unknown"})
    assert Notification.objects.get(recipient=people["admin_b"], title="Registration rejected").message == \
        "You rejected Ada Lovelace (ada@example.com)."
    assert "Registration rejected" not in _titles(people["admin_a"])


def test_notifications_isolated_between_two_admins_and_two_users(client, people):
    own = {key: notify([user], Type.ANNOUNCEMENT, f"For {key}", url="/")[0] for key, user in people.items()}
    feed = reverse("notifications:feed")
    for key, user in people.items():
        client.force_login(user)
        data = client.get(feed, {"since": "0"}, HTTP_ACCEPT="application/json").json()
        assert data["unread_count"] == 1
        assert [i["title"] for i in data["items"]] == [f"For {key}"]
        assert [i["title"] for i in data["new"]] == [f"For {key}"]
        listed = [n.title for n in client.get(reverse("notifications:list")).context["page"].object_list]
        assert listed == [f"For {key}"]
        page = client.get(reverse("core:dashboard")).content.decode()
        others = [f"For {k}" for k in people if k != key]
        assert not any(title in page for title in others)

        for other_key, other in own.items():
            if other_key == key:
                continue
            for name in ("notifications:mark_read", "notifications:open", "notifications:delete"):
                response = client.post(reverse(name, args=[other.pk]), HTTP_ACCEPT="application/json")
                assert response.status_code == 404, (key, other_key, name)
        client.post(reverse("notifications:mark_all_read"))
        client.post(reverse("notifications:clear_all"))
        client.logout()

    # Each user's mark-all/clear-all touched only their own row, so every other row was untouched until
    # its owner acted; at the end each user cleared exactly their own.
    assert not Notification.objects.exists()


def test_mark_all_read_affects_only_the_requesting_user(client, people):
    for user in people.values():
        notify([user], Type.ANNOUNCEMENT, "Hello", url="/")
    client.force_login(people["admin_a"])
    assert client.post(reverse("notifications:mark_all_read"), HTTP_ACCEPT="application/json").json()["unread_count"] == 0
    unread = dict(Notification.objects.filter(is_read=False).values_list("recipient__username", "title"))
    assert set(unread) == {"admin_b", "user_1", "user_2"}
