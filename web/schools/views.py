"""Dataset upload and validation.

Uploading validates a file and records the attempt. It never replaces the
active dataset, never retrains the model and never changes saved artifacts.
"""

from __future__ import annotations

from django.conf import settings
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from services import upload_service
from web.core.forms import DatasetUploadForm
from web.schools.models import DatasetUpload


@require_http_methods(["GET", "POST"])
@permission_required("schools.add_datasetupload", raise_exception=True)
def upload(request):
    is_post = request.method == "POST"
    form = DatasetUploadForm(
        request.POST if is_post else None, request.FILES if is_post else None,
        max_size=settings.MAX_UPLOAD_SIZE_BYTES, allowed_extensions=upload_service.ALLOWED_EXTENSIONS,
    )
    if is_post and form.is_valid():
        uploaded = form.cleaned_data["file"]
        content = uploaded.read()
        summary = upload_service.validate_dataset(content, uploaded.name)
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
            record.file.save(uploaded.name, ContentFile(content), save=False)
        record.save()
        return redirect("schools:upload_detail", pk=record.pk)

    return render(request, "schools/upload.html", {
        "form": form,
        "recent_uploads": DatasetUpload.objects.select_related("uploaded_by")[:10],
        "max_size_mb": settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024),
    })


@permission_required("schools.view_datasetupload", raise_exception=True)
def upload_detail(request, pk: int):
    record = get_object_or_404(DatasetUpload.objects.select_related("uploaded_by"), pk=pk)
    return render(request, "schools/upload_detail.html", {"upload": record, "summary": record.validation_summary})
