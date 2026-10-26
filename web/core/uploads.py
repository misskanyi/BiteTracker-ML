"""Image and document uploads (profile photos, support screenshots, supporting letters):
validation and private serving.

Files are checked by their first bytes, not only their name, so a renamed file of
another type is rejected. Nothing under MEDIA_ROOT is routed publicly; images are
returned by views that check who is asking.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from django import forms
from django.http import FileResponse, Http404

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _sniff(head: bytes) -> str | None:
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_image(upload, max_bytes: int):
    """Raise ValidationError unless `upload` is a PNG, JPEG or WebP image within `max_bytes`."""
    suffix = Path(upload.name).suffix.lower()
    if suffix not in IMAGE_TYPES:
        raise forms.ValidationError("Unsupported file type. Upload a PNG, JPEG or WebP image.")
    if upload.size > max_bytes:
        raise forms.ValidationError(f"The image is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    upload.seek(0)
    head = upload.read(16)
    upload.seek(0)
    if _sniff(head) != IMAGE_TYPES[suffix]:
        raise forms.ValidationError("The file content is not a valid PNG, JPEG or WebP image.")
    return upload


def image_response(field_file) -> FileResponse:
    if not field_file:
        raise Http404
    try:
        handle = field_file.open("rb")
    except (FileNotFoundError, OSError):
        raise Http404
    content_type = _sniff(handle.read(16)) or "application/octet-stream"
    handle.seek(0)
    response = FileResponse(handle, content_type=content_type)
    response["X-Content-Type-Options"] = "nosniff"
    response["Cache-Control"] = "private, max-age=300"
    return response


# Supporting letters: a signed PDF, a Word document, or a photo/scan of the paper letter.
DOCUMENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _sniff_document(upload) -> str | None:
    upload.seek(0)
    head = upload.read(16)
    upload.seek(0)
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"PK\x03\x04"):
        # Any zip starts like this; a Word document must contain its main document part.
        try:
            with zipfile.ZipFile(upload) as archive:
                is_docx = "word/document.xml" in archive.namelist()
        except zipfile.BadZipFile:
            is_docx = False
        finally:
            upload.seek(0)
        return DOCUMENT_TYPES[".docx"] if is_docx else None
    return _sniff(head)


def validate_document(upload, max_bytes: int):
    """Raise ValidationError unless `upload` is a PDF, DOCX, PNG or JPEG file within `max_bytes`."""
    suffix = Path(upload.name).suffix.lower()
    if suffix not in DOCUMENT_TYPES:
        raise forms.ValidationError("Unsupported file type. Upload the letter as PDF, Word (.docx), PNG or JPEG.")
    if upload.size > max_bytes:
        raise forms.ValidationError(f"The letter is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    if _sniff_document(upload) != DOCUMENT_TYPES[suffix]:
        raise forms.ValidationError("The file content does not match its type. Upload a real PDF, Word, PNG or JPEG file.")
    return upload


def download_response(field_file, filename: str) -> FileResponse:
    """Serve a stored private file as a download (never rendered inline by the browser)."""
    if not field_file:
        raise Http404
    try:
        handle = field_file.open("rb")
    except (FileNotFoundError, OSError):
        raise Http404
    response = FileResponse(handle, as_attachment=True, filename=filename or Path(field_file.name).name)
    response["X-Content-Type-Options"] = "nosniff"
    response["Cache-Control"] = "private, no-store"
    return response
