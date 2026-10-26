"""Help & Support: guides, FAQ, About, and the Report a Problem form."""

from __future__ import annotations

from django.conf import settings
from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from services import upload_service
from services.forecast_service import HORIZONS
from src.data_validation import OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from web.accounts import access
from web.audit.models import AuditLog
from web.audit.services import log_action
from web.core.about import about_context
from web.core.uploads import image_response
from web.notifications.models import Notification
from web.notifications.services import notify, system_admins
from web.support.forms import SupportIssueForm
from web.support.models import SupportIssue


@require_http_methods(["GET", "POST"])
def help_page(request):
    form = SupportIssueForm(request.POST or None, request.FILES or None)
    if request.method == "POST":
        if form.is_valid():
            issue = form.save(commit=False)
            issue.reporter = request.user
            issue.save()
            log_action(request.user, AuditLog.Action.ISSUE_REPORTED, target=issue,
                       details={"category": issue.category})
            notify(system_admins(), Notification.Type.SUPPORT, f"New problem reported: {issue.subject}",
                   f"{issue.get_category_display()} · reported by {request.user.get_username()}",
                   url=reverse("administration:issue_detail", args=[issue.pk]), actor=request.user)
            messages.success(request, f"Thank you. Your report #{issue.pk} has been submitted.")
            return redirect(f"{reverse('support:help')}#my-reports")
        messages.error(request, "Your report was not submitted. Check the highlighted fields.")
    return render(request, "support/help.html", {
        "form": form,
        "my_issues": SupportIssue.objects.filter(reporter=request.user)[:10],
        "about": about_context(),
        "horizons": list(HORIZONS.values()),
        "upload_extensions": ", ".join(upload_service.ALLOWED_EXTENSIONS),
        "required_columns": REQUIRED_COLUMNS,
        "optional_columns": OPTIONAL_COLUMNS,
        "max_upload_mb": settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024),
        "can_upload": request.user.has_perm("schools.add_datasetupload"),
        "is_admin": access.is_admin(request.user),
    })


def _visible_issue(request, pk: int) -> SupportIssue:
    issue = get_object_or_404(SupportIssue.objects.select_related("reporter"), pk=pk)
    if issue.reporter_id != request.user.pk and not access.is_system_admin(request.user):
        raise PermissionDenied
    return issue


def issue_detail(request, pk: int):
    issue = _visible_issue(request, pk)
    if access.is_system_admin(request.user):
        return redirect("administration:issue_detail", pk=pk)
    return render(request, "support/issue_detail.html", {"issue": issue})


def issue_screenshot(request, pk: int):
    return image_response(_visible_issue(request, pk).screenshot)
