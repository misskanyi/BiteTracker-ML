"""Stage 3: School / DatasetUpload / ForecastRun models, Django Admin and import_schools.

What this file checks:
- School model: the ID is unique and no enrolment numbers are made up.
- Upload filenames are sanitised.
- import_schools command: safe to run twice, keeps manual edits, reports schools missing
  from the dataset, and supports a dry run.
- Admin pages load; admin is staff-only, and audit records can't be added by hand.
"""

from __future__ import annotations

from datetime import date
from io import StringIO

import pytest
from django.contrib.auth import get_user_model
from django.core.management import CommandError, call_command
from django.db import IntegrityError
from django.urls import reverse

from web.schools.models import DatasetUpload, ForecastRun, School, dataset_upload_path

pytestmark = pytest.mark.django_db


@pytest.fixture
def admin_client_logged_in(client):
    admin = get_user_model().objects.create_superuser("admin", "admin@example.com", "Adm1n-pass-word!")
    client.force_login(admin)
    return client


def _run_import(*args) -> str:
    out = StringIO()
    call_command("import_schools", *args, stdout=out)
    return out.getvalue()


# --- Models ---------------------------------------------------------------------------


def test_school_id_is_unique():
    School.objects.create(school_id="SCH001", name="A")
    with pytest.raises(IntegrityError):
        School.objects.create(school_id="SCH001", name="B")


def test_school_defaults_do_not_invent_enrolment():
    school = School.objects.create(school_id="SCH009")
    assert school.verified_enrolment is None
    assert school.is_active is True
    assert str(school) == "SCH009"


def test_forecast_run_stores_summary_without_school_row():
    run = ForecastRun.objects.create(
        school_code="SCH001", school_name="Greenfield Primary", horizon_days=7,
        origin_date=date(2024, 12, 20), first_forecast_date=date(2024, 12, 23), last_forecast_date=date(2024, 12, 31),
        model_label="Random Forest", results_label="SYNTHETIC DEVELOPMENT RESULTS",
        day1_predicted_meals=4, day1_recommended_meals=5, day1_risk_level="High",
        total_predicted_meals=181, total_recommended_meals=193, result_summary={"days": []},
    )
    assert run.school is None
    assert run.get_horizon_days_display() == "Next 7 school days"
    assert "SCH001" in str(run)


def test_dataset_upload_defaults_to_not_processed():
    upload = DatasetUpload.objects.create(
        original_filename="x.csv", file_size=10, validation_status=DatasetUpload.ValidationStatus.INVALID
    )
    assert upload.processing_status == DatasetUpload.ProcessingStatus.NOT_PROCESSED
    assert "Invalid" in str(upload)


def test_upload_path_sanitises_filename():
    path = dataset_upload_path(None, "../../etc/My File (final).CSV")
    assert path.startswith("datasets/")
    assert ".." not in path and " " not in path
    assert path.endswith(".csv")


# --- import_schools -----------------------------------------------------------------------


def test_import_schools_creates_then_is_idempotent(stub_artifacts):
    first = _run_import()
    assert "created: 3, updated: 0, unchanged (skipped): 0" in first
    assert School.objects.count() == 3

    second = _run_import()
    assert "created: 0, updated: 0, unchanged (skipped): 3" in second
    assert School.objects.count() == 3


def test_import_schools_copies_dataset_metadata_only(stub_artifacts):
    _run_import()
    school = School.objects.get(school_id="SCH001")
    assert school.name == "Greenfield Primary"
    assert school.county == "Nakuru" and school.sub_county == "Naivasha"
    assert school.data_source == "SYNTHETIC DEVELOPMENT DATA"
    assert school.verified_enrolment is None  # dataset enrolment is not verified enrolment


def test_import_schools_updates_changed_fields_and_keeps_manual_data(stub_artifacts):
    _run_import()
    School.objects.filter(school_id="SCH002").update(name="Old name", verified_enrolment=600)
    output = _run_import()
    assert "created: 0, updated: 1, unchanged (skipped): 2" in output
    school = School.objects.get(school_id="SCH002")
    assert school.name == "Riverside Primary"
    assert school.verified_enrolment == 600


def test_import_schools_reports_schools_missing_from_dataset(stub_artifacts):
    School.objects.create(school_id="OLD001", name="Closed school")
    output = _run_import()
    assert "1 existing school record(s) are not in the active dataset" in output
    assert School.objects.filter(school_id="OLD001").exists()


def test_import_schools_dry_run_saves_nothing(stub_artifacts):
    output = _run_import("--dry-run")
    assert "[dry run]" in output and "created: 3" in output
    assert School.objects.count() == 0


def test_import_schools_without_data_fails_cleanly(stub_artifacts):
    stub_artifacts.processed_path.unlink()
    with pytest.raises(CommandError, match="Processed dataset not found"):
        _run_import()


# --- Admin ---------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["school", "datasetupload", "forecastrun"])
def test_admin_changelists_load(admin_client_logged_in, model):
    response = admin_client_logged_in.get(reverse(f"admin:schools_{model}_changelist"))
    assert response.status_code == 200


def test_admin_can_add_school(admin_client_logged_in):
    response = admin_client_logged_in.get(reverse("admin:schools_school_add"))
    assert response.status_code == 200


@pytest.mark.parametrize("model", ["datasetupload", "forecastrun"])
def test_admin_cannot_add_audit_records(admin_client_logged_in, model):
    response = admin_client_logged_in.get(reverse(f"admin:schools_{model}_add"))
    assert response.status_code == 403


def test_admin_user_management_available(admin_client_logged_in):
    assert admin_client_logged_in.get(reverse("admin:auth_user_changelist")).status_code == 200


def test_non_staff_user_cannot_use_admin(client):
    user = get_user_model().objects.create_user("planner", password="Str0ng-test-passw0rd!")
    client.force_login(user)
    response = client.get(reverse("admin:schools_school_changelist"))
    assert response.status_code == 302
    assert reverse("admin:login") in response.url
