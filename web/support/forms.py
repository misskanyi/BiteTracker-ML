from __future__ import annotations

from django import forms
from django.conf import settings

from web.core.uploads import validate_image
from web.support.models import SupportIssue


class SupportIssueForm(forms.ModelForm):
    class Meta:
        model = SupportIssue
        fields = ["category", "subject", "description", "screenshot"]
        labels = {"category": "Issue category", "screenshot": "Screenshot (optional)"}
        help_texts = {
            "description": "What were you doing, what did you expect, and what happened instead? "
                           "Do not include passwords or learners' personal information.",
            "screenshot": "PNG, JPEG or WebP, up to 5 MB.",
        }
        widgets = {
            "category": forms.Select(attrs={"class": "form-select"}),
            "subject": forms.TextInput(attrs={"class": "form-control", "maxlength": 200}),
            "description": forms.Textarea(attrs={"class": "form-control", "rows": 5}),
            "screenshot": forms.ClearableFileInput(attrs={"class": "form-control", "accept": "image/png,image/jpeg,image/webp"}),
        }

    def clean_screenshot(self):
        upload = self.cleaned_data.get("screenshot")
        return validate_image(upload, settings.MAX_SCREENSHOT_SIZE_BYTES) if upload else upload


class IssueStatusForm(forms.ModelForm):
    class Meta:
        model = SupportIssue
        fields = ["status"]
        widgets = {"status": forms.Select(attrs={"class": "form-select"})}
