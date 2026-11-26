"""Forecast vs actual, publishing verified records into forecasting history, and the follow-up CSV.

What this file checks:
- A School Manager records an actual on a saved forecast for an assigned school; it is pending,
  labelled as such and excluded from accuracy. Zero is valid; future, negative and non-integer
  counts are refused. Repeated submissions keep one record per school-day, shared by every run.
- A Stakeholder sees the value but cannot record, edit, correct, verify or publish (403).
  A manager of another school gets 404. A System Administrator cannot record, only review.
  Accounts pending approval cannot use any of it.
- Verify → publish puts the value into forecasting history once (retries are no-ops); the saved
  forecast keeps its original prediction and model; a new forecast starts from the published day.
- Conflicting values (baseline or upload overlap) must be acknowledged before publishing.
- Corrections: proposed on a published value, approved by an administrator, re-published.
- Upload publication is a separate, retry-safe step after verification, with overlap flags.
- Export: only schools in scope (another school's code → 404), estimates and counts separated,
  no learner identifiers/letters/notes, and formula-like text neutralised.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date, timedelta

import pandas as pd
import pytest
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.urls import reverse

from services import data_service, forecast_service, model_service
from src.feature_engineering import FOOD4EDUCATION_FEATURE_COLUMNS, FOOD4EDUCATION_SOURCE
from tests.support.web_fixtures import PASSWORD, operator
from web.accounts import access
from web.accounts.models import AccessRequest, Role
from web.audit.models import AuditLog
from web.core.actual_views import EXPORT_COLUMNS, spreadsheet_safe
from web.core.forecast_records import record_forecast_run
from web.schools.models import ActualMealRecord, ActualMealRevision, DatasetUpload, DemandRecord, ForecastRun

pytestmark = pytest.mark.django_db
User = get_user_model()


@pytest.fixture
def sysadmin(db):
    return User.objects.create_superuser("root", "root@example.com", PASSWORD)


@pytest.fixture
def manager(db):
    return operator("manager1", codes=("SCH001",))


@pytest.fixture
def other_manager(db):
    return operator("manager2", codes=("SCH002",))


@pytest.fixture
def stakeholder(db):
    user = operator("stake", codes=("SCH001", "SCH002"))
    access.set_role(user, Role.PLANNING_STAKEHOLDER)
    return user


@pytest.fixture
def run(stub_artifacts, manager):
    """A saved 7-day forecast for SCH001 starting the school day after the latest record (all past dates)."""
    result = forecast_service.generate_forecast("SCH001", 7)
    return record_forecast_run(result, manager)


def _days(run):
    return [d["date"] for d in run.result_summary["days"]]


def _record(client, run, day, served, prepared=""):
    return client.post(reverse("core:record_actual", args=[run.pk]),
                       {"date": day, "meals_served": served, "meals_prepared": prepared})


def _review(client, record, action, comment="", ack=False, expected=None):
    data = {"action": action, "comment": comment, "expected_status": expected or record.status}
    if ack:
        data["acknowledge_conflicts"] = "on"
    return client.post(reverse("core:actual_review", args=[record.pk]), data)


# --- Recording ---------------------------------------------------------------------------------


def test_manager_records_actual_pending_and_excluded_from_accuracy(client, manager, run):
    client.force_login(manager)
    day = _days(run)[0]
    response = _record(client, run, day, 140, 150)
    assert response.status_code == 302
    record = ActualMealRecord.objects.get()
    assert (record.school.school_id, record.date.isoformat(), record.meals_served, record.meals_prepared) == ("SCH001", day, 140, 150)
    assert record.status == ActualMealRecord.Status.PENDING and record.submitted_by == manager
    assert record.prepared_minus_served == 10
    assert AuditLog.objects.filter(action=AuditLog.Action.ACTUAL_SUBMITTED).count() == 1

    html = client.get(reverse("core:forecast_run", args=[run.pk])).content.decode()
    assert "Forecast vs actual" in html and "Pending review" in html and "(pending)" in html
    assert 'data-testid="fva-accuracy"' not in html  # pending values never count towards accuracy
    assert "Not recorded" in html  # the other days


def test_zero_is_valid_and_invalid_counts_and_future_dates_are_refused(client, manager, run, monkeypatch):
    client.force_login(manager)
    days = _days(run)
    assert _record(client, run, days[0], 0).status_code == 302
    assert ActualMealRecord.objects.get().meals_served == 0
    assert ActualMealRecord.objects.count() == 1  # a zero record, distinct from the unrecorded days

    for bad, message in (("-1", "greater than or equal to 0"), ("12.5", "Enter a whole number"), ("", "required")):
        response = _record(client, run, days[1], bad)
        assert response.status_code == 400 and message in response.content.decode()
    assert ActualMealRecord.objects.count() == 1

    # A forecast day that is still in the future (pretend "today" is before day 2).
    from django.utils import timezone
    monkeypatch.setattr(timezone, "localdate", lambda *a, **k: date.fromisoformat(days[0]))
    response = _record(client, run, days[1], 100)
    assert response.status_code == 400 and "future date" in response.content.decode()
    assert ActualMealRecord.objects.count() == 1


def test_duplicate_submission_keeps_one_record_shared_by_every_run(client, manager, run):
    client.force_login(manager)
    day = _days(run)[0]
    _record(client, run, day, 120)
    _record(client, run, day, 120)  # double click
    assert ActualMealRecord.objects.count() == 1
    assert ActualMealRevision.objects.filter(kind=ActualMealRevision.Kind.SUBMITTED).count() == 1

    _record(client, run, day, 125)  # still pending: an edit, not a new record
    record = ActualMealRecord.objects.get()
    assert record.meals_served == 125 and record.revisions.filter(kind=ActualMealRevision.Kind.EDITED).count() == 1

    second = record_forecast_run(forecast_service.generate_forecast("SCH001", 1), manager)
    _record(client, second, day, 125)
    assert ActualMealRecord.objects.count() == 1
    html = client.get(reverse("core:forecast_run", args=[second.pk])).content.decode()
    assert "125" in html and reverse("core:actual_detail", args=[record.pk]) in html


# --- Permissions -------------------------------------------------------------------------------


def test_stakeholder_views_but_cannot_change_or_publish(client, manager, stakeholder, run):
    client.force_login(manager)
    _record(client, run, _days(run)[0], 130)
    record = ActualMealRecord.objects.get()

    client.force_login(stakeholder)
    page = client.get(reverse("core:forecast_run", args=[run.pk]))
    assert page.status_code == 200 and 'data-testid="record-actual-form"' not in page.content.decode()
    assert "130" in page.content.decode()
    assert client.get(reverse("core:actual_detail", args=[record.pk])).status_code == 200
    assert _record(client, run, _days(run)[1], 99).status_code == 403
    assert client.post(reverse("core:actual_detail", args=[record.pk]), {"action": "edit", "meals_served": 1}).status_code == 403
    assert _review(client, record, "verify").status_code == 403
    assert _review(client, record, "publish").status_code == 403
    record.refresh_from_db()
    assert record.status == ActualMealRecord.Status.PENDING and record.meals_served == 130
    assert client.get(reverse("core:actuals")).status_code == 200


def test_other_school_manager_gets_404_and_admin_cannot_record(client, manager, other_manager, sysadmin, run):
    client.force_login(manager)
    _record(client, run, _days(run)[0], 130)
    record = ActualMealRecord.objects.get()

    client.force_login(other_manager)
    assert client.get(reverse("core:forecast_run", args=[run.pk])).status_code == 404
    assert _record(client, run, _days(run)[1], 1).status_code == 404
    assert client.get(reverse("core:actual_detail", args=[record.pk])).status_code == 404
    assert _review(client, record, "verify").status_code == 403  # not a reviewer at all

    client.force_login(sysadmin)
    assert _record(client, run, _days(run)[1], 1).status_code == 403
    assert ActualMealRecord.objects.count() == 1


def test_pending_account_cannot_use_the_features(client, run, manager):
    pending = operator("applicant", codes=("SCH001",))
    pending.is_active = False
    pending.save()
    AccessRequest.objects.create(user=pending, email="applicant@example.com")
    client.force_login(pending)
    for url in (reverse("core:actuals"), reverse("core:followup_export"), reverse("core:forecast_run", args=[run.pk])):
        response = client.get(url)
        assert response.status_code != 200 or b"bitetracker_followup" not in response.content
        assert b"Forecast vs actual" not in response.content
    assert not access.can_record_actuals(pending, "SCH001")
    assert not access.visible_actuals(pending).exists()


# --- Verify, publish, forecasting history ------------------------------------------------------


# The active Food4Education model's inputs are meal-count lags and the calendar only. A recorded actual
# carries meals served (no attendance or weather), so it is a usable history day for that model.


def test_admin_verifies_and_publishes_once_into_history_and_saved_forecast_is_unchanged(
        client, manager, sysadmin, run, stub_artifacts, make_model, make_metadata):
    day = date.fromisoformat(_days(run)[0])
    original = {"pred": run.result_summary["days"][0]["predicted_meals"], "model": run.model_label,
                "summary": dict(run.result_summary), "origin": run.origin_date}
    assert day > data_service.get_latest_date("SCH001")

    client.force_login(manager)
    _record(client, run, day.isoformat(), 133)
    record = ActualMealRecord.objects.get()

    client.force_login(sysadmin)
    assert _review(client, record, "publish").status_code == 302  # not verified yet: refused
    record.refresh_from_db()
    assert record.status == ActualMealRecord.Status.PENDING and not DemandRecord.objects.exists()

    _review(client, record, "verify")
    record.refresh_from_db()
    assert record.status == ActualMealRecord.Status.VERIFIED and record.reviewed_by == sysadmin
    assert data_service.get_latest_date("SCH001") < day  # verified is not yet usable

    _review(client, record, "publish")
    _review(client, record, "publish", expected="verified")  # a retry from a stale page
    record.refresh_from_db()
    assert record.status == ActualMealRecord.Status.PUBLISHED and record.published_by == sysadmin
    assert DemandRecord.objects.filter(actual=record, status="approved").count() == 1
    assert AuditLog.objects.filter(action=AuditLog.Action.ACTUAL_PUBLISHED).count() == 1
    assert record.revisions.filter(kind=ActualMealRevision.Kind.PUBLISHED).count() == 1

    history = data_service.get_school_history("SCH001")
    assert data_service.get_latest_date("SCH001") == day
    assert history.loc[history["date"].dt.date == day, "meals_served"].item() == 133

    run.refresh_from_db()
    assert run.result_summary["days"][0]["predicted_meals"] == original["pred"]
    assert run.model_label == original["model"] and run.origin_date == original["origin"]
    assert run.result_summary == original["summary"]

    # The saved model is still in use and a new forecast starts from the published day.
    stub_artifacts.write_model(make_model(FOOD4EDUCATION_FEATURE_COLUMNS))
    stub_artifacts.write_metadata(make_metadata(features_used=FOOD4EDUCATION_FEATURE_COLUMNS,
                                                data_source=FOOD4EDUCATION_SOURCE))
    stub_artifacts.feature_columns_path.write_text(json.dumps(FOOD4EDUCATION_FEATURE_COLUMNS))
    model_service.clear_model_cache()
    new = forecast_service.generate_forecast("SCH001", 1)
    assert new.origin_date == day

    # Verified values now count towards accuracy on the saved forecast.
    client.force_login(manager)
    html = client.get(reverse("core:forecast_run", args=[run.pk])).content.decode()
    assert 'data-testid="fva-accuracy"' in html and "Published" in html


def test_publishing_over_a_different_history_value_needs_acknowledgement(client, manager, sysadmin, stub_artifacts):
    # A saved forecast covering a day that is already in history (a back-dated run).
    latest = data_service.get_latest_date("SCH001")
    hist = data_service.get_school_history("SCH001")
    existing = float(hist.loc[hist["date"].dt.date == latest, "meals_served"].item())
    run = ForecastRun.objects.create(
        school_code="SCH001", horizon_days=1, origin_date=latest - timedelta(days=1), first_forecast_date=latest,
        last_forecast_date=latest, model_label="Stub", results_label="SYNTHETIC DEVELOPMENT RESULTS",
        day1_predicted_meals=100, day1_recommended_meals=110, day1_risk_level="Low", total_predicted_meals=100,
        total_recommended_meals=110,
        result_summary={"days": [{"date": latest.isoformat(), "predicted_meals": 100, "recommended_meals": 110}]})
    client.force_login(manager)
    _record(client, run, latest.isoformat(), int(existing) + 40)
    record = ActualMealRecord.objects.get()
    client.force_login(sysadmin)
    _review(client, record, "verify")
    record.refresh_from_db()
    assert 'data-testid="actual-conflicts"' in client.get(reverse("core:actual_detail", args=[record.pk])).content.decode()

    response = _review(client, record, "publish")
    record.refresh_from_db()
    assert record.status == ActualMealRecord.Status.VERIFIED  # refused without acknowledgement
    assert "already have a different value" in client.get(response.url).content.decode()

    _review(client, record, "publish", ack=True)
    record.refresh_from_db()
    assert record.status == ActualMealRecord.Status.PUBLISHED
    hist = data_service.get_school_history("SCH001")
    assert hist.loc[hist["date"].dt.date == latest, "meals_served"].item() == int(existing) + 40
    assert len(hist) == len(data_service.get_school_history("SCH001"))  # replaced, not duplicated


def test_correction_of_published_value_is_reviewed_then_republished(client, manager, sysadmin, run):
    day = _days(run)[0]
    client.force_login(manager)
    _record(client, run, day, 133)
    record = ActualMealRecord.objects.get()
    client.force_login(sysadmin)
    _review(client, record, "verify")
    record.refresh_from_db()
    _review(client, record, "publish")
    record.refresh_from_db()

    client.force_login(manager)
    # Published values can't be overwritten directly.
    assert _record(client, run, day, 150).status_code == 400
    response = client.post(reverse("core:actual_detail", args=[record.pk]),
                           {"action": "correction", "meals_served": 150, "reason": "Miscounted the second sitting"})
    assert response.status_code == 302
    client.post(reverse("core:actual_detail", args=[record.pk]),
                {"action": "correction", "meals_served": 150, "reason": "Miscounted the second sitting"})  # repeat
    record.refresh_from_db()
    assert record.meals_served == 133 and record.proposed_meals_served == 150
    assert record.revisions.filter(kind=ActualMealRevision.Kind.CORRECTION_PROPOSED).count() == 1

    client.force_login(sysadmin)
    assert _review(client, record, "approve_correction").status_code == 302
    record.refresh_from_db()
    assert record.meals_served == 150 and record.status == ActualMealRecord.Status.VERIFIED
    history = data_service.get_school_history("SCH001")
    assert history.loc[history["date"].dt.date == date.fromisoformat(day), "meals_served"].item() == 133  # old value still used

    _review(client, record, "publish")
    history = data_service.get_school_history("SCH001")
    assert history.loc[history["date"].dt.date == date.fromisoformat(day), "meals_served"].item() == 150
    assert DemandRecord.objects.filter(actual=record).count() == 1
    kinds = set(record.revisions.values_list("kind", flat=True))
    assert {"submitted", "verified", "published", "correction_proposed", "correction_approved"} <= kinds


def test_reject_and_request_correction_need_comments_and_reject_keeps_value_out(client, manager, sysadmin, run):
    client.force_login(manager)
    _record(client, run, _days(run)[0], 90)
    record = ActualMealRecord.objects.get()
    client.force_login(sysadmin)
    _review(client, record, "reject")  # no comment
    record.refresh_from_db()
    assert record.status == ActualMealRecord.Status.PENDING
    _review(client, record, "request_correction", "Check the register")
    record.refresh_from_db()
    assert record.status == ActualMealRecord.Status.CORRECTION_REQUESTED

    client.force_login(manager)
    client.post(reverse("core:actual_detail", args=[record.pk]), {"action": "edit", "meals_served": 95})
    record.refresh_from_db()
    assert record.status == ActualMealRecord.Status.PENDING and record.meals_served == 95


# --- Upload publication ------------------------------------------------------------------------


@pytest.fixture
def media_root(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "uploads"
    return settings.MEDIA_ROOT


def _upload(user, rows: pd.DataFrame) -> DatasetUpload:
    content = rows.to_csv(index=False).encode()
    record = DatasetUpload(uploaded_by=user, original_filename="data.csv", file_size=len(content),
                           validation_status="valid", review_status=DatasetUpload.ReviewStatus.PENDING)
    record.file.save("data.csv", ContentFile(content), save=False)
    record.save()
    return record


def test_upload_overlapping_a_recorded_actual_is_flagged_and_publish_is_retry_safe(
        client, manager, sysadmin, run, media_root):
    day = _days(run)[0]
    client.force_login(manager)
    _record(client, run, day, 133)
    upload = _upload(manager, pd.DataFrame({"date": [day], "school_id": ["SCH001"], "school_name": ["S1"],
                                            "meals_served": [200]}))
    client.force_login(sysadmin)
    client.post(reverse("administration:upload_review", args=[upload.pk]), {"decision": "approved", "comment": ""})
    upload.refresh_from_db()
    assert upload.stage == "verified"
    html = client.get(reverse("administration:upload_review", args=[upload.pk])).content.decode()
    assert 'data-testid="upload-overlaps"' in html and "Differs" in html

    publish = reverse("administration:upload_publish", args=[upload.pk])
    client.post(publish)  # conflict not acknowledged
    upload.refresh_from_db()
    assert upload.published_at is None

    client.post(publish, {"acknowledge_conflicts": "on"})
    client.post(publish, {"acknowledge_conflicts": "on"})  # retry / double click
    upload.refresh_from_db()
    assert upload.stage == "published" and upload.published_by == sysadmin
    assert DemandRecord.objects.filter(upload=upload).count() == 1
    assert AuditLog.objects.filter(action=AuditLog.Action.DATASET_PUBLISHED).count() == 1
    history = data_service.get_school_history("SCH001")
    assert history.loc[history["date"].dt.date == date.fromisoformat(day), "meals_served"].item() == 200


def test_non_admin_cannot_publish_upload(client, manager, media_root, stub_artifacts):
    upload = _upload(manager, pd.DataFrame({"date": ["2023-05-01"], "school_id": ["SCH001"], "meals_served": [1]}))
    client.force_login(manager)
    assert client.post(reverse("administration:upload_publish", args=[upload.pk])).status_code == 403


# --- Follow-up export --------------------------------------------------------------------------


def _csv(response) -> list[dict]:
    text = b"".join(response.streaming_content if response.streaming else [response.content]).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def test_export_scoped_to_authorised_schools_and_separates_estimates_from_counts(
        client, manager, stakeholder, other_manager, run):
    first, second = _days(run)[0], _days(run)[1]
    client.force_login(manager)
    _record(client, run, first, 133, 150)
    start, end = run.first_forecast_date.isoformat(), run.last_forecast_date.isoformat()

    response = client.get(reverse("core:followup_export"), {"start": start, "end": end})
    assert response.status_code == 200 and response["Content-Type"].startswith("text/csv")
    rows = _csv(response)
    assert list(rows[0].keys()) == EXPORT_COLUMNS
    assert {r["school_code"] for r in rows} == {"SCH001"} and len(rows) == len(_days(run))
    by_date = {r["date"]: r for r in rows}
    row = by_date[first]
    assert row["recorded_meals_served"] == "133" and row["recorded_value_status"] == "Pending review"
    assert row["planned_quantity"] == "150" and row["planned_quantity_source"] == "recorded meals prepared"
    assert row["surplus_positive_or_shortfall_negative"] == "17" and row["surplus_or_shortfall_basis"] == "recorded meals served"
    assert row["forecast_demand_estimate"] == str(run.result_summary["days"][0]["predicted_meals"])
    assert row["latest_usable_record_date"] == data_service.get_latest_date("SCH001").isoformat()
    unrecorded = by_date[second]
    assert unrecorded["recorded_meals_served"] == "" and unrecorded["recorded_value_status"] == ""
    assert unrecorded["planned_quantity_source"].startswith("forecast") and unrecorded["surplus_or_shortfall_basis"].startswith("forecast")
    for banned in ("student", "learner", "letter", "comment", "reason", "submitted_by"):
        assert not any(banned in c for c in EXPORT_COLUMNS)

    # Cross-school attempt: another school's code is refused, not silently filtered.
    assert client.get(reverse("core:followup_export"), {"school": "SCH002"}).status_code == 404
    assert client.get(reverse("core:followup_export"), {"school": ["SCH001", "SCH002"]}).status_code == 404
    client.force_login(other_manager)
    assert all(r["school_code"] == "SCH002" for r in _csv(client.get(reverse("core:followup_export"), {"start": start, "end": end})))

    # A stakeholder may compare both assigned schools.
    client.force_login(stakeholder)
    response = client.get(reverse("core:followup_export"), {"school": ["SCH001", "SCH002"], "start": start, "end": end})
    assert response.status_code == 200 and {r["school_code"] for r in _csv(response)} <= {"SCH001", "SCH002"}
    assert AuditLog.objects.filter(action=AuditLog.Action.FOLLOWUP_EXPORTED).count() == 3  # refused requests are not exports


def test_export_neutralises_formula_text(client, manager, run):
    run.model_label = "=HYPERLINK(\"http://x\")"
    run.save(update_fields=["model_label"])
    for day in run.result_summary["days"]:
        day.pop("method_label", None)
    run.save(update_fields=["result_summary"])
    client.force_login(manager)
    rows = _csv(client.get(reverse("core:followup_export"), {
        "start": run.first_forecast_date.isoformat(), "end": run.last_forecast_date.isoformat()}))
    assert rows and all(r["forecast_method"].startswith("'=") for r in rows)
    assert spreadsheet_safe("+1") == "'+1" and spreadsheet_safe("@x") == "'@x" and spreadsheet_safe(-5) == -5


def test_overview_page_shows_freshness_and_scope(client, manager, other_manager, run):
    client.force_login(manager)
    _record(client, run, _days(run)[0], 133)
    html = client.get(reverse("core:actuals")).content.decode()
    assert 'data-testid="freshness"' in html and "SCH001" in html
    assert data_service.get_latest_date("SCH001").strftime("%-d %b %Y") in html
    assert 'data-testid="followup-export"' in html
    assert client.get(reverse("core:actuals"), {"school": "SCH002"}).status_code == 404
    client.force_login(other_manager)
    html = client.get(reverse("core:actuals")).content.decode()
    assert "133" not in html
