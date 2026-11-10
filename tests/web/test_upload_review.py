"""Upload templates, supporting letters and the administrator review workflow.

What this file checks:
- The Excel data template has exactly the validator's columns, and its example rows validate.
- The Word letter template is a real .docx that the letter validator accepts.
- A supporting letter is required; its content must match its extension.
- Valid uploads store the letter, wait for review and notify System Administrators (once);
  invalid ones store nothing.
- A School Manager can only submit records for their assigned schools.
- Data files, letters and upload records are visible only to the uploader and System Administrators.
- Review queue: System Administrators only; School Managers and Stakeholders get 403.
- Rejecting or requesting clarification needs a comment; every decision is audited and the uploader notified.
- Nobody decides on their own upload.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from services import upload_service, upload_templates
from src.data_validation import OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from tests.support.web_fixtures import PASSWORD, operator
from web.accounts import access
from web.accounts.models import Role
from web.audit.models import AuditLog
from web.core.uploads import validate_document
from web.notifications.models import Notification
from web.schools.models import DatasetUpload

pytestmark = pytest.mark.django_db
User = get_user_model()
PDF = b"%PDF-1.4\n%signed cover letter\n"


@pytest.fixture
def media_root(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "uploads"
    return settings.MEDIA_ROOT


@pytest.fixture
def sysadmin(db):
    return User.objects.create_superuser("root", "root@example.com", PASSWORD)


@pytest.fixture
def uploader_user(db):
    """A School Manager (uploads need that role or System Administrator)."""
    return operator("data_officer", codes=("SCH001",))


@pytest.fixture
def stakeholder(db):
    user = operator("pa", codes=("SCH001", "SCH002"))
    access.set_role(user, Role.PLANNING_STAKEHOLDER)
    return user


def _pending_upload(user, media_root) -> DatasetUpload:
    record = DatasetUpload(uploaded_by=user, original_filename="data.csv", file_size=10, validation_status="valid",
                           n_records=80, quality_score=90, review_status=DatasetUpload.ReviewStatus.PENDING,
                           letter_filename="letter.pdf", validation_summary={"status": "valid", "errors": [], "warnings": []})
    record.file.save("data.csv", ContentFile(b"date,school_id\n"), save=False)
    record.supporting_letter.save("letter.pdf", ContentFile(PDF), save=False)
    record.save()
    return record


# --- Templates ---------------------------------------------------------------------------------


def test_data_template_matches_validator_columns_and_example_validates(client, uploader_user):
    client.force_login(uploader_user)
    response = client.get(reverse("schools:data_template"))
    assert response.status_code == 200
    assert response["Content-Type"] == upload_templates.XLSX_CONTENT_TYPE
    assert "attachment" in response["Content-Disposition"]

    sheets = pd.read_excel(io.BytesIO(response.content), sheet_name=None)
    assert list(sheets) == ["Data", "Instructions", "Example"]
    assert list(sheets["Data"].columns) == REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    assert sheets["Data"].empty

    buffer = io.BytesIO()
    sheets["Example"].to_excel(buffer, index=False)
    summary = upload_service.validate_dataset(buffer.getvalue(), "example.xlsx")
    assert summary["status"] == upload_service.VALID, summary["errors"]


def test_letter_template_is_a_valid_docx(client, uploader_user):
    client.force_login(uploader_user)
    response = client.get(reverse("schools:letter_template"))
    assert response.status_code == 200 and response["Content-Type"] == upload_templates.DOCX_CONTENT_TYPE
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "SUBMISSION OF SCHOOL MEAL SERVICE DATA" in archive.read("word/document.xml").decode()
    validate_document(SimpleUploadedFile("letter.docx", response.content), 5 * 1024 * 1024)


def test_upload_page_links_templates(client, uploader_user):
    client.force_login(uploader_user)
    html = client.get(reverse("schools:upload")).content.decode()
    assert reverse("schools:data_template") in html and reverse("schools:letter_template") in html


# --- Uploading with a letter -------------------------------------------------------------------


def _post(client, stub_artifacts, letter):
    frame = stub_artifacts.frame[["date", "school_id", "school_name", "meals_served"]]
    frame = frame[frame["school_id"] == "SCH001"].head(80)  # the uploader's own school only
    data = {"file": SimpleUploadedFile("school_data.csv", frame.to_csv(index=False).encode())}
    if letter is not None:
        data["letter"] = letter
    return client.post(reverse("schools:upload"), data)


def test_letter_is_required(client, uploader_user, stub_artifacts, media_root):
    client.force_login(uploader_user)
    response = _post(client, stub_artifacts, None)
    assert b'data-testid="letter-form-error"' in response.content
    assert DatasetUpload.objects.count() == 0


@pytest.mark.parametrize("name,content", [
    ("letter.pdf", b"<html>not a pdf</html>"),
    ("letter.exe", PDF),
    ("letter.docx", b"PK\x03\x04 not really a word file"),
])
def test_letter_content_must_match_type(client, uploader_user, stub_artifacts, media_root, name, content):
    client.force_login(uploader_user)
    response = _post(client, stub_artifacts, SimpleUploadedFile(name, content))
    assert b'data-testid="letter-form-error"' in response.content
    assert DatasetUpload.objects.count() == 0


def test_valid_upload_stores_letter_and_notifies_reviewers(client, uploader_user, sysadmin, stakeholder,
                                                            stub_artifacts, media_root):
    client.force_login(uploader_user)
    _post(client, stub_artifacts, SimpleUploadedFile("Head Teacher letter.pdf", PDF))
    record = DatasetUpload.objects.get()
    assert record.review_status == DatasetUpload.ReviewStatus.PENDING
    assert record.letter_filename == "Head Teacher letter.pdf"
    assert record.supporting_letter.name.startswith("letters/") and "Head" not in record.supporting_letter.name
    notified = set(Notification.objects.filter(type=Notification.Type.DATASET_REVIEW).values_list("recipient__username", flat=True))
    assert notified == {"root"}  # System Administrators verify; Stakeholders are read-only
    assert Notification.objects.get(recipient=sysadmin, type=Notification.Type.DATASET_REVIEW).event_key == \
        f"upload:{record.pk}:submitted"


def test_upload_for_unassigned_school_is_refused(client, stub_artifacts, media_root):
    """Changing school_id in the file can't submit data for someone else's school."""
    manager = operator("manager_sch2", codes=("SCH002",))
    client.force_login(manager)
    frame = stub_artifacts.frame[["date", "school_id", "school_name", "meals_served"]]
    frame = frame[frame["school_id"] == "SCH001"].head(80)
    client.post(reverse("schools:upload"), {
        "file": SimpleUploadedFile("school_data.csv", frame.to_csv(index=False).encode()),
        "letter": SimpleUploadedFile("letter.pdf", PDF)})
    record = DatasetUpload.objects.get()
    assert record.validation_status == "invalid" and record.review_status == "" and not record.file
    assert any("not assigned to you: SCH001" in e for e in record.validation_summary["errors"])
    assert not Notification.objects.filter(type=Notification.Type.DATASET_REVIEW).exists()


def test_invalid_upload_stores_no_letter_and_is_not_queued(client, uploader_user, stub_artifacts, media_root):
    client.force_login(uploader_user)
    client.post(reverse("schools:upload"), {"file": SimpleUploadedFile("bad.csv", b"a,b\n1,2\n"),
                                            "letter": SimpleUploadedFile("letter.pdf", PDF)})
    record = DatasetUpload.objects.get()
    assert record.validation_status == "invalid" and not record.supporting_letter and record.review_status == ""


# --- Downloads ---------------------------------------------------------------------------------


def test_downloads_limited_to_uploader_and_reviewers(client, uploader_user, sysadmin, media_root):
    record = _pending_upload(uploader_user, media_root)
    letter_url = reverse("schools:upload_letter", args=[record.pk])

    client.force_login(uploader_user)
    response = client.get(letter_url)
    assert response.status_code == 200 and b"".join(response.streaming_content) == PDF
    assert "attachment" in response["Content-Disposition"]

    client.force_login(sysadmin)
    assert client.get(reverse("schools:upload_file", args=[record.pk])).status_code == 200

    client.force_login(operator("someone_else", codes=("SCH001",)))
    assert client.get(letter_url).status_code == 404


# --- Review queue and decisions -------------------------------------------------------------------


def test_operators_cannot_open_review_pages(client, uploader_user, media_root):
    record = _pending_upload(uploader_user, media_root)
    client.force_login(uploader_user)
    assert client.get(reverse("administration:uploads")).status_code == 403
    assert client.get(reverse("administration:upload_review", args=[record.pk])).status_code == 403


def test_stakeholders_cannot_verify_or_open_uploads(client, uploader_user, stakeholder, media_root):
    record = _pending_upload(uploader_user, media_root)
    client.force_login(stakeholder)
    assert client.get(reverse("administration:uploads")).status_code == 403
    assert client.post(reverse("administration:upload_review", args=[record.pk]),
                       {"decision": "approved"}).status_code == 403
    # Someone else's upload record and files are hidden (404), even with a guessed id.
    assert client.get(reverse("schools:upload_detail", args=[record.pk])).status_code == 404
    assert client.get(reverse("schools:upload_letter", args=[record.pk])).status_code == 404
    record.refresh_from_db()
    assert record.review_status == DatasetUpload.ReviewStatus.PENDING


def test_upload_list_shows_only_own_uploads(client, uploader_user, media_root):
    mine = _pending_upload(uploader_user, media_root)
    _pending_upload(operator("elsewhere", codes=("SCH003",)), media_root)
    client.force_login(uploader_user)
    response = client.get(reverse("schools:upload"))
    assert list(response.context["recent_uploads"]) == [mine]


def test_clarification_request_needs_comment_and_notifies_uploader(client, uploader_user, sysadmin, media_root):
    record = _pending_upload(uploader_user, media_root)
    client.force_login(sysadmin)
    response = client.post(reverse("administration:upload_review", args=[record.pk]), {"decision": "clarification"})
    assert b"Say what the uploader needs to clarify" in response.content
    client.post(reverse("administration:upload_review", args=[record.pk]),
                {"decision": "clarification", "comment": "Which term do these dates cover?"})
    record.refresh_from_db()
    assert record.review_status == DatasetUpload.ReviewStatus.CLARIFICATION
    note = Notification.objects.get(recipient=uploader_user, type=Notification.Type.DATASET_REVIEW)
    assert "clarification requested" in note.title and note.message == "Which term do these dates cover?"


def test_reject_requires_comment(client, uploader_user, sysadmin, media_root):
    record = _pending_upload(uploader_user, media_root)
    client.force_login(sysadmin)
    response = client.post(reverse("administration:upload_review", args=[record.pk]), {"decision": "rejected"})
    assert b"Explain why the upload is rejected" in response.content
    record.refresh_from_db()
    assert record.review_status == DatasetUpload.ReviewStatus.PENDING


def test_decision_is_saved_audited_and_notified(client, uploader_user, sysadmin, media_root):
    record = _pending_upload(uploader_user, media_root)
    client.force_login(sysadmin)
    response = client.post(reverse("administration:upload_review", args=[record.pk]),
                           {"decision": "rejected", "comment": "Letter is not signed."})
    assert response.status_code == 302
    record.refresh_from_db()
    assert record.review_status == "rejected" and record.reviewed_by == sysadmin and record.reviewed_at
    assert AuditLog.objects.filter(action=AuditLog.Action.DATASET_REVIEWED, target_id=str(record.pk)).exists()
    note = Notification.objects.get(recipient=uploader_user, type=Notification.Type.DATASET_REVIEW)
    assert "rejected" in note.title and note.message == "Letter is not signed."

    client.force_login(uploader_user)
    html = client.get(reverse("schools:upload_detail", args=[record.pk])).content.decode()
    assert 'data-testid="review-status"' in html and "Letter is not signed." in html


def test_nobody_decides_on_their_own_upload(client, sysadmin, media_root):
    record = _pending_upload(sysadmin, media_root)
    client.force_login(sysadmin)
    page = client.get(reverse("administration:upload_review", args=[record.pk]))
    assert page.status_code == 200 and b'data-testid="upload-review-form"' not in page.content
    response = client.post(reverse("administration:upload_review", args=[record.pk]), {"decision": "approved"})
    assert response.status_code == 403
