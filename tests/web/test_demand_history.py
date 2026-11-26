"""Demand history: verified and then published uploads feed forecasting data; other decisions keep them out.

What this file checks:
- Verifying an upload stores its rows without using them; publishing adds them to
  `data_service.load_processed_data`.
- An upload row replaces the baseline row for the same school-day; blank school details are filled.
- Rejecting or asking for clarification after approval takes the rows back out.
- Asking for clarification needs a comment.
- An unreadable approved file leaves the decision unchanged.
- Seeding the baseline copies the processed CSV once (again only with --replace).
- `approved_training_frame` returns the training schema with approved upload rows included.
"""

from __future__ import annotations

import pandas as pd
import pytest
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.management import call_command
from django.urls import reverse

from services import data_service
from src.synthetic_data import SCHEMA_COLUMNS
from tests.support.web_fixtures import PASSWORD, operator
from web.schools import demand_history
from web.schools.models import DatasetUpload, DemandRecord

pytestmark = pytest.mark.django_db
User = get_user_model()


@pytest.fixture
def media_root(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "uploads"
    return settings.MEDIA_ROOT


@pytest.fixture
def sysadmin(db):
    return User.objects.create_superuser("root", "root@example.com", PASSWORD)


@pytest.fixture
def history(stub_artifacts):
    data_service.clear_data_cache()
    yield stub_artifacts
    data_service.clear_data_cache()


def _upload_csv(baseline: pd.DataFrame) -> tuple[bytes, str, pd.Timestamp, pd.Timestamp]:
    """Two rows for the first school: one overriding its last baseline day, one new day after it."""
    school = baseline["school_id"].iloc[0]
    last = baseline.loc[baseline["school_id"] == school, "date"].max()
    last = pd.Timestamp(last)
    new_day = last + pd.Timedelta(days=1)
    rows = pd.DataFrame({
        "date": [last.date().isoformat(), new_day.date().isoformat()],
        "school_id": [school, school],
        "meals_served": [999, 321],
        "students_present": [1000, 330],
    })
    return rows.to_csv(index=False).encode(), school, last, new_day


def _upload(user, content: bytes) -> DatasetUpload:
    record = DatasetUpload(uploaded_by=user, original_filename="data.csv", file_size=len(content),
                           validation_status="valid", review_status=DatasetUpload.ReviewStatus.PENDING)
    record.file.save("data.csv", ContentFile(content), save=False)
    record.save()
    return record


def _decide(client, record, decision, comment=""):
    return client.post(reverse("administration:upload_review", args=[record.pk]),
                       {"decision": decision, "comment": comment})


def _publish(client, record, acknowledge=True):
    data = {"acknowledge_conflicts": "on"} if acknowledge else {}
    return client.post(reverse("administration:upload_publish", args=[record.pk]), data)


def test_approval_adds_rows_to_forecasting_data_and_rejection_removes_them(client, sysadmin, media_root, history):
    content, school, last, new_day = _upload_csv(history.frame)
    uploader = operator("officer", codes=(school,))
    record = _upload(uploader, content)
    before = data_service.load_processed_data()
    client.force_login(sysadmin)

    assert _decide(client, record, "approved").status_code == 302
    record.refresh_from_db()
    # Verified only: stored, not yet used.
    assert record.processing_status == DatasetUpload.ProcessingStatus.NOT_PROCESSED
    assert DemandRecord.objects.filter(upload=record, status="verified").count() == 2
    assert new_day not in set(data_service.get_school_history(school)["date"])

    assert _publish(client, record).status_code == 302
    record.refresh_from_db()
    assert record.processing_status == DatasetUpload.ProcessingStatus.PROCESSED
    assert record.published_by == sysadmin and record.published_at is not None
    assert DemandRecord.objects.filter(upload=record, status="approved").count() == 2

    data = data_service.get_school_history(school)
    assert len(data) == len(before[before["school_id"] == school]) + 1
    assert data.loc[data["date"] == last, "meals_served"].item() == 999
    added = data[data["date"] == new_day].iloc[0]
    assert added["meals_served"] == 321
    assert added["school_name"] == before.loc[before["school_id"] == school, "school_name"].iloc[-1]

    assert _decide(client, record, "rejected", "Wrong school.").status_code == 302
    record.refresh_from_db()
    assert record.processing_status == DatasetUpload.ProcessingStatus.NOT_PROCESSED
    assert set(DemandRecord.objects.filter(upload=record).values_list("status", flat=True)) == {"rejected"}
    after = data_service.load_processed_data()
    assert len(after) == len(before) and new_day not in set(after.loc[after["school_id"] == school, "date"])

    assert record.published_at is None  # rejection withdraws the publication

    # Re-verifying reuses the stored rows rather than writing them twice.
    _decide(client, record, "approved")
    assert DemandRecord.objects.filter(upload=record).count() == 2
    assert set(DemandRecord.objects.filter(upload=record).values_list("status", flat=True)) == {"verified"}


def test_clarification_needs_comment_and_keeps_rows_out(client, sysadmin, media_root, history):
    content, school, _, new_day = _upload_csv(history.frame)
    record = _upload(operator("officer", codes=(school,)), content)
    client.force_login(sysadmin)

    response = _decide(client, record, "clarification")
    assert b"Say what the uploader needs to clarify" in response.content
    assert _decide(client, record, "clarification", "Which term is this?").status_code == 302
    record.refresh_from_db()
    assert record.review_status == "clarification"
    assert not DemandRecord.objects.filter(upload=record).exists()
    assert new_day not in set(data_service.get_school_history(school)["date"])


def test_unreadable_file_leaves_decision_unchanged(client, sysadmin, media_root, history):
    record = _upload(operator("officer", codes=("SCH001",)), b"\x00\x01 not a csv")
    record.original_filename = "data.xlsx"
    record.save(update_fields=["original_filename"])
    client.force_login(sysadmin)

    response = _decide(client, record, "approved")
    assert response.status_code == 200 and b"could not be read" in response.content
    record.refresh_from_db()
    assert record.review_status == "pending" and record.reviewed_by is None
    assert not DemandRecord.objects.exists()


def test_seed_baseline_once_then_replace(history):
    call_command("seed_demand_history")
    count = DemandRecord.objects.filter(upload__isnull=True).count()
    assert count == len(history.frame.drop_duplicates(["school_id", "date"]))
    assert demand_history.seed_baseline() == 0
    call_command("seed_demand_history", "--replace")
    assert DemandRecord.objects.filter(upload__isnull=True).count() == count

    data = data_service.load_processed_data()
    assert len(data) == count and set(data["school_id"]) == set(history.frame["school_id"])


def test_training_frame_includes_approved_uploads(client, sysadmin, media_root, history):
    content, school, _, new_day = _upload_csv(history.frame)
    record = _upload(operator("officer", codes=(school,)), content)
    client.force_login(sysadmin)
    _decide(client, record, "approved")
    assert len(demand_history.approved_training_frame()[lambda f: (f["school_id"] == school) & (f["date"] == new_day)]) == 0
    _publish(client, record)

    frame = demand_history.approved_training_frame()
    assert list(frame.columns) == SCHEMA_COLUMNS
    row = frame[(frame["school_id"] == school) & (frame["date"] == new_day)]
    assert len(row) == 1 and row["meals_served"].item() == 321
