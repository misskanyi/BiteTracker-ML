"""Only schools in the active dataset can be chosen for new assignments and forecasts.

Scenario: the active dataset is Food4Education (school IDs SCHOOL_1..SCHOOL_3 here), while legacy
development school rows SCH001-SCH003 still exist with assignments, forecasts and audit logs.

- Approval, user create/edit, the school page's "assign user" and the forecast form offer, and
  accept on the server, only schools whose exact ID is in the active dataset (never by name).
- Existing assignments to unavailable schools are shown and kept until an administrator unticks
  them; nothing is reassigned or deleted silently. School rows, forecasts and logs are preserved.
- "Not in the active dataset" and "no records for the selected dates" are different messages.
- If the dataset can't be read, nothing can be newly assigned (fail closed).
"""

from __future__ import annotations

from datetime import date

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

from config import settings as ml_settings
from services import data_service
from tests.support.web_fixtures import PASSWORD, operator
from web.accounts import access
from web.accounts.models import AccessRequest, Role, SchoolAssignment
from web.audit.models import AuditLog
from web.schools import eligibility
from web.schools.models import ForecastRun, School

pytestmark = pytest.mark.django_db
User = get_user_model()
RENAME = {"SCH001": "SCHOOL_1", "SCH002": "SCHOOL_2", "SCH003": "SCHOOL_3"}


@pytest.fixture
def f4e(stub_artifacts):
    """Active dataset uses SCHOOL_1..3; legacy rows SCH001..3 exist alongside the new ones."""
    frame = stub_artifacts.frame.copy()
    frame[ml_settings.SCHOOL_ID_COLUMN] = frame[ml_settings.SCHOOL_ID_COLUMN].replace(RENAME)
    stub_artifacts.write_processed(frame)
    data_service.clear_data_cache()
    for legacy, new in RENAME.items():
        School.objects.create(school_id=legacy, name=f"Legacy {legacy}", data_source="SYNTHETIC DEVELOPMENT DATA")
        School.objects.create(school_id=new, name=f"F4E {new}")
    return stub_artifacts


@pytest.fixture
def admin(db):
    return User.objects.create_superuser("root", "root@example.com", PASSWORD)


def pk(code: str) -> int:
    return School.objects.get(school_id=code).pk


def codes_of(user) -> set[str]:
    return set(SchoolAssignment.objects.filter(user=user).values_list("school__school_id", flat=True))


def choice_codes(form_field) -> set[str]:
    return set(form_field.queryset.values_list("school_id", flat=True))


# --- Eligibility rule ---------------------------------------------------------------------------------------


def test_eligibility_uses_exact_dataset_ids_not_names_and_skips_inactive(f4e):
    dataset_name = data_service.list_schools()[0]["school_name"]
    School.objects.create(school_id="DEV9", name=dataset_name)  # same name as a dataset school, different ID
    School.objects.filter(school_id="SCHOOL_3").update(is_active=False)
    eligible = set(eligibility.eligible_schools(School.objects.all()).values_list("school_id", flat=True))
    assert eligible == {"SCHOOL_1", "SCHOOL_2"}


def test_unreadable_dataset_makes_nothing_eligible(f4e, client, admin):
    f4e.processed_path.unlink()
    data_service.clear_data_cache()
    assert eligibility.active_dataset_school_codes() is None
    assert not eligibility.eligible_schools(School.objects.all()).exists()
    req = _applicant()
    client.force_login(admin)
    response = client.post(reverse("administration:access_request_detail", args=[req.pk]),
                           {"action": "approve", "role": Role.SCHOOL_MANAGER, "schools": [pk("SCHOOL_1")], "note": ""})
    assert "could not be read" in response.content.decode()
    assert AccessRequest.objects.get(pk=req.pk).status == AccessRequest.Status.PENDING


# --- Approval ----------------------------------------------------------------------------------------------


def _applicant() -> AccessRequest:
    user = User.objects.create_user("applicant", "ada@example.com", PASSWORD, is_active=False)
    return AccessRequest.objects.create(user=user, email="ada@example.com", full_name="Ada")


def test_approval_offers_and_accepts_only_dataset_schools(f4e, client, admin):
    req = _applicant()
    client.force_login(admin)
    page = client.get(reverse("administration:access_request_detail", args=[req.pk]))
    assert choice_codes(page.context["approve_form"].fields["schools"]) == {"SCHOOL_1", "SCHOOL_2", "SCHOOL_3"}
    assert "(SCH001)" not in page.content.decode()

    crafted = client.post(reverse("administration:access_request_detail", args=[req.pk]),
                          {"action": "approve", "role": Role.SCHOOL_MANAGER, "schools": [pk("SCH001")], "note": ""})
    assert crafted.status_code == 200 and "Select a valid choice" in crafted.content.decode()
    assert AccessRequest.objects.get(pk=req.pk).status == AccessRequest.Status.PENDING and not codes_of(req.user)

    client.post(reverse("administration:access_request_detail", args=[req.pk]),
                {"action": "approve", "role": Role.SCHOOL_MANAGER, "schools": [pk("SCHOOL_1")], "note": ""})
    assert codes_of(req.user) == {"SCHOOL_1"}


# --- User create / edit -------------------------------------------------------------------------------------


def test_user_create_rejects_legacy_school(f4e, client, admin):
    client.force_login(admin)
    payload = {"username": "newbie", "first_name": "N", "last_name": "P", "email": "n@example.com", "phone": "",
               "role": Role.SCHOOL_MANAGER, "password1": "An0ther-str0ng-pass!", "password2": "An0ther-str0ng-pass!"}
    response = client.post(reverse("administration:user_create"), {**payload, "schools": [pk("SCH002")]})
    assert response.status_code == 200 and not User.objects.filter(username="newbie").exists()
    client.post(reverse("administration:user_create"), {**payload, "schools": [pk("SCHOOL_2")]})
    assert codes_of(User.objects.get(username="newbie")) == {"SCHOOL_2"}


@pytest.fixture
def legacy_user(f4e):
    user = operator("sharon2", codes=("SCH001",))
    ForecastRun.objects.create(
        school=School.objects.get(school_id="SCH001"), school_code="SCH001", school_name="Legacy", horizon_days=1,
        origin_date=date(2025, 1, 1), first_forecast_date=date(2025, 1, 2), last_forecast_date=date(2025, 1, 2),
        model_label="m", results_label="SYNTHETIC DEVELOPMENT RESULTS", day1_predicted_meals=1,
        day1_recommended_meals=1, day1_risk_level="Low", total_predicted_meals=1, total_recommended_meals=1,
        result_summary={})
    return user


def _edit(client, user, **fields):
    data = {"first_name": "S", "last_name": "K", "email": "s2@example.com", "phone": "", "role": Role.SCHOOL_MANAGER}
    return client.post(reverse("administration:user_edit", args=[user.pk]), {**data, **fields})


def test_edit_form_shows_unavailable_assignment_kept_by_default(f4e, client, admin, legacy_user):
    client.force_login(admin)
    page = client.get(reverse("administration:user_edit", args=[legacy_user.pk]))
    form = page.context["form"]
    assert choice_codes(form.fields["schools"]) == {"SCHOOL_1", "SCHOOL_2", "SCHOOL_3"}
    assert choice_codes(form.fields["keep_unavailable"]) == {"SCH001"}
    html = page.content.decode()
    assert "Legacy SCH001 (SCH001) · not in the active dataset" in html and 'data-testid="keep-unavailable"' in html

    # Saving other details with the legacy school still ticked changes no assignment.
    assert _edit(client, legacy_user, keep_unavailable=[pk("SCH001")]).status_code == 302
    assert codes_of(legacy_user) == {"SCH001"}
    assert not AuditLog.objects.filter(action=AuditLog.Action.SCHOOLS_ASSIGNED).exists()


def test_admin_replaces_legacy_assignment_explicitly(f4e, client, admin, legacy_user):
    """The sharon2 case: untick SCH001, tick SCHOOL_1."""
    client.force_login(admin)
    assert _edit(client, legacy_user, schools=[pk("SCHOOL_1")]).status_code == 302
    assert codes_of(legacy_user) == {"SCHOOL_1"}
    log = AuditLog.objects.get(action=AuditLog.Action.SCHOOLS_ASSIGNED)
    assert log.details == {"added": ["SCHOOL_1"], "removed": ["SCH001"]}
    # Records are preserved: the legacy school row and its forecast stay.
    assert School.objects.filter(school_id="SCH001").exists()
    assert ForecastRun.objects.filter(school_code="SCH001").count() == 1

    client.force_login(legacy_user)
    dashboard = client.get(reverse("core:dashboard"))
    assert [s["school_id"] for s in dashboard.context["schools"]] == ["SCHOOL_1"]


def test_keep_field_cannot_add_a_legacy_school_the_user_did_not_have(f4e, client, admin, legacy_user):
    client.force_login(admin)
    response = _edit(client, legacy_user, schools=[pk("SCHOOL_1")], keep_unavailable=[pk("SCH001"), pk("SCH002")])
    assert response.status_code == 200 and codes_of(legacy_user) == {"SCH001"}


def test_user_detail_flags_unavailable_assignment(f4e, client, admin, legacy_user):
    client.force_login(admin)
    html = client.get(reverse("administration:user_detail", args=[legacy_user.pk])).content.decode()
    assert 'data-testid="unavailable-assignment"' in html and "Unavailable · not in the active dataset" in html
    assert 'data-testid="unavailable-hint"' in html


# --- School page "assign user" --------------------------------------------------------------------------------


def test_school_page_refuses_new_users_for_legacy_school_but_allows_removal(f4e, client, admin, legacy_user):
    other = operator("other", codes=("SCHOOL_2",))
    legacy = School.objects.get(school_id="SCH001")
    client.force_login(admin)
    page = client.get(reverse("administration:school_detail", args=[legacy.pk])).content.decode()
    assert 'data-testid="school-unavailable"' in page
    client.post(reverse("administration:school_detail", args=[legacy.pk]), {"action": "assign", "user": other.pk})
    assert codes_of(other) == {"SCHOOL_2"}
    client.post(reverse("administration:school_detail", args=[legacy.pk]), {"action": "unassign", "user": legacy_user.pk})
    assert codes_of(legacy_user) == set()

    school_1 = School.objects.get(school_id="SCHOOL_1")
    client.post(reverse("administration:school_detail", args=[school_1.pk]), {"action": "assign", "user": other.pk})
    assert codes_of(other) == {"SCHOOL_1", "SCHOOL_2"}


def test_schools_list_marks_schools_not_in_dataset(f4e, client, admin):
    client.force_login(admin)
    html = client.get(reverse("administration:schools")).content.decode()
    assert html.count('data-testid="school-not-in-dataset"') == 3


# --- Forecast form and messages -----------------------------------------------------------------------------


def test_forecast_form_offers_and_accepts_only_dataset_schools(f4e, client):
    user = operator("mixed", codes=("SCHOOL_1", "SCH001"))
    client.force_login(user)
    form = client.get(reverse("core:forecast")).context["form"]
    assert {v for v, _ in form.fields["school"].choices} == {"SCHOOL_1"}
    response = client.post(reverse("core:forecast"), {"school": "SCH001", "horizon": 1})
    assert response.status_code == 200 and not ForecastRun.objects.exists()
    # Out-of-scope dataset school is refused too (role scope still applies).
    client.post(reverse("core:forecast"), {"school": "SCHOOL_2", "horizon": 1})
    assert not ForecastRun.objects.exists()


def test_not_in_dataset_and_no_records_for_dates_are_distinct(f4e, client):
    client.force_login(operator("legacy_only", codes=("SCH001",)))
    message = client.get(reverse("core:history")).context["data_error"]
    assert "no records in the active dataset" in message

    client.force_login(operator("f4e_user", codes=("SCHOOL_1",)))
    response = client.get(reverse("core:history"), {"school": "SCHOOL_1", "start": "2020-01-01", "end": "2020-01-31"})
    html = response.content.decode()
    assert "data_error" not in response.context and 'data-testid="history-empty"' in html
    assert "No records for" in html and "active dataset" not in html


def test_dashboard_notes_assigned_schools_outside_dataset(f4e, client):
    client.force_login(operator("mixed", codes=("SCHOOL_1", "SCH001")))
    response = client.get(reverse("core:dashboard"))
    assert [s["school_id"] for s in response.context["schools"]] == ["SCHOOL_1"]
    assert 'data-testid="unavailable-schools"' in response.content.decode()
    assert [s.school_id for s in response.context["unavailable_schools"]] == ["SCH001"]
    assert access.accessible_school_codes(User.objects.get(username="mixed")) == {"SCHOOL_1", "SCH001"}  # not removed
