"""Dataset upload and validation.

Uploading validates a file and records the attempt. It never replaces the
active dataset, never retrains the model and never changes saved artifacts.

School Managers upload records for their assigned schools only; System Administrators
verify them. Users see only their own uploads (administrators see all).
"""

from __future__ import annotations

from django.conf import settings
from django.core.files.base import ContentFile
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from services import upload_service, upload_templates
from web.accounts import access
from web.audit.models import AuditLog
from web.audit.services import log_action
from web.core.forms import DatasetUploadForm
from web.core.uploads import download_response
from web.notifications.models import Notification
from web.notifications.services import notify, system_admins
from web.schools.models import DatasetUpload


def _check_school_scope(user, summary: dict) -> dict:
    """A valid file is downgraded to invalid if it contains schools the uploader isn't assigned to."""
    if summary["status"] != upload_service.VALID:
        return summary
    codes = access.accessible_school_codes(user)
    outside = [] if codes is None else [s for s in summary.get("school_ids", []) if s not in codes]
    if outside:
        shown = ", ".join(outside[:5]) + (f" and {len(outside) - 5} more" if len(outside) > 5 else "")
        summary = {**summary, "status": upload_service.INVALID,
                   "errors": [*summary.get("errors", []),
                              f"The file contains records for schools not assigned to you: {shown}. "
                              "Upload records only for your assigned schools."]}
    return summary


@require_http_methods(["GET", "POST"])
@access.upload_required
def upload(request):
    is_post = request.method == "POST"
    form = DatasetUploadForm(
        request.POST if is_post else None, request.FILES if is_post else None,
        max_size=settings.MAX_UPLOAD_SIZE_BYTES, allowed_extensions=upload_service.ALLOWED_EXTENSIONS,
        letter_max_size=settings.MAX_LETTER_SIZE_BYTES,
    )
    if is_post and form.is_valid():
        uploaded = form.cleaned_data["file"]
        content = uploaded.read()
        summary = _check_school_scope(request.user, upload_service.validate_dataset(content, uploaded.name))
        record = DatasetUpload(
            uploaded_by=request.user,
            original_filename=uploaded.name[:255],
            file_size=uploaded.size,
            validation_status=summary["status"],
            forecast_readiness=summary.get("forecast_readiness", ""),
            n_records=summary.get("n_records"),
            quality_score=summary.get("quality_score"),
            validation_summary=summary,
        )
        if summary["status"] == upload_service.VALID:
            # Only valid uploads are stored, with their letter, and queued for an administrator.
            letter = form.cleaned_data["letter"]
            record.file.save(uploaded.name, ContentFile(content), save=False)
            record.supporting_letter.save(letter.name, letter, save=False)
            record.letter_filename = letter.name[:255]
            record.review_status = DatasetUpload.ReviewStatus.PENDING
        record.save()
        _record_upload_events(request.user, record)
        return redirect("schools:upload_detail", pk=record.pk)

    return render(request, "schools/upload.html", {
        "form": form,
        "recent_uploads": access.visible_uploads(request.user)[:10],
        "max_size_mb": settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024),
        "letter_max_size_mb": settings.MAX_LETTER_SIZE_BYTES // (1024 * 1024),
    })


def upload_detail(request, pk: int):
    # Someone else's upload 404s rather than 403s, so its existence is not revealed.
    record = get_object_or_404(access.visible_uploads(request.user), pk=pk)
    return render(request, "schools/upload_detail.html", {
        "upload": record, "summary": record.validation_summary,
        "can_download": access.can_download_upload(request.user, record),
    })


# --- Files ---------------------------------------------------------------------------------------


def _downloadable_upload(request, pk: int) -> DatasetUpload:
    record = get_object_or_404(DatasetUpload, pk=pk)
    if not access.can_download_upload(request.user, record):
        raise Http404  # do not reveal that the upload exists
    return record


def upload_file(request, pk: int):
    record = _downloadable_upload(request, pk)
    return download_response(record.file, record.original_filename)


def upload_letter(request, pk: int):
    record = _downloadable_upload(request, pk)
    return download_response(record.supporting_letter, record.letter_filename)


def data_template(request):
    response = HttpResponse(upload_templates.data_template_xlsx(), content_type=upload_templates.XLSX_CONTENT_TYPE)
    response["Content-Disposition"] = f'attachment; filename="{upload_templates.DATA_TEMPLATE_FILENAME}"'
    return response


def letter_template(request):
    response = HttpResponse(upload_templates.letter_template_docx(), content_type=upload_templates.DOCX_CONTENT_TYPE)
    response["Content-Disposition"] = f'attachment; filename="{upload_templates.LETTER_TEMPLATE_FILENAME}"'
    return response


def _reviewers(record: DatasetUpload):
    """System Administrators who may decide on this upload (not the uploader)."""
    return [admin for admin in system_admins() if access.can_decide_upload(admin, record)]


def _record_upload_events(user, record: DatasetUpload) -> None:
    """Audit every upload and tell the uploader the outcome (subject to their notification settings)."""
    log_action(user, AuditLog.Action.DATASET_UPLOADED, target=record, target_label=record.original_filename,
               details={"status": record.validation_status, "records": record.n_records,
                        "quality_score": record.quality_score})
    url = reverse("schools:upload_detail", args=[record.pk])
    if record.validation_status == DatasetUpload.ValidationStatus.VALID:
        notify([user], Notification.Type.DATASET_UPLOADED, f"Dataset validated: {record.original_filename}",
               f"{record.n_records or 0} records checked. Quality score {record.quality_score}/100. "
               "The file and letter were sent to an administrator for review; they do not replace the active dataset.",
               url=url, include_actor=True)
        notify(_reviewers(record), Notification.Type.DATASET_REVIEW, f"Upload awaiting review: {record.original_filename}",
               f"Submitted by {user.get_username()} with a supporting letter. "
               f"{record.n_records or 0} records, quality score {record.quality_score}/100.",
               url=reverse("administration:upload_review", args=[record.pk]), actor=user,
               event_key=f"upload:{record.pk}:submitted")
    else:
        errors = record.validation_summary.get("errors") or []
        notify([user], Notification.Type.DATASET_INVALID, f"Dataset failed validation: {record.original_filename}",
               f"{len(errors)} problem(s) found" + (f", e.g. {errors[0]}" if errors else "") + ".",
               url=url, include_actor=True)
