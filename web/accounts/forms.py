"""Profile and personal-settings forms. Role, email and school assignments are not editable here."""

from __future__ import annotations

import re

from django import forms
from django.conf import settings

from web.accounts.models import AccessRequest, Role, UserProfile
from web.core.uploads import validate_image

PHONE_RE = re.compile(r"^\+?[0-9 ()-]{7,20}$")


class ProfileForm(forms.Form):
    first_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs={"class": "form-control", "autocomplete": "given-name"}))
    last_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs={"class": "form-control", "autocomplete": "family-name"}))
    phone = forms.CharField(
        label="Phone number", max_length=20, required=False,
        help_text="Optional. Digits, spaces, brackets, dashes and a leading +.",
        widget=forms.TextInput(attrs={"class": "form-control", "autocomplete": "tel", "inputmode": "tel"}),
    )
    avatar = forms.FileField(
        label="Profile photo", required=False,
        help_text="PNG, JPEG or WebP, up to 2 MB.",
        widget=forms.ClearableFileInput(attrs={"class": "form-control", "accept": "image/png,image/jpeg,image/webp"}),
    )
    remove_avatar = forms.BooleanField(label="Remove current photo", required=False,
                                       widget=forms.CheckboxInput(attrs={"class": "form-check-input"}))

    def clean_first_name(self):
        return self.cleaned_data["first_name"].strip()

    def clean_last_name(self):
        return self.cleaned_data["last_name"].strip()

    def clean_phone(self):
        phone = self.cleaned_data["phone"].strip()
        if phone and not PHONE_RE.match(phone):
            raise forms.ValidationError("Enter a valid phone number, e.g. +254 712 345 678.")
        return phone

    def clean_avatar(self):
        upload = self.cleaned_data.get("avatar")
        return validate_image(upload, settings.MAX_AVATAR_SIZE_BYTES) if upload else upload


def _switch(field):
    field.widget = forms.CheckboxInput(attrs={"class": "form-check-input", "role": "switch"})
    return field


class AppearanceForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ["theme", "compact_tables", "reduce_motion", "landing_page"]
        labels = {"compact_tables": "Compact table view", "reduce_motion": "Reduce animations",
                  "landing_page": "Default landing page"}
        help_texts = {
            "compact_tables": "Tighter row spacing in data tables, so more rows fit on screen.",
            "reduce_motion": "Turns off page, chart and loader animations. Your device's reduced-motion setting is always respected.",
            "landing_page": "The page you see after logging in.",
        }
        widgets = {"theme": forms.RadioSelect, "landing_page": forms.Select(attrs={"class": "form-select"})}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("compact_tables", "reduce_motion"):
            _switch(self.fields[name])


class NotificationPreferencesForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ["notify_forecast", "notify_dataset_upload", "notify_dataset_validation", "notify_calendar",
                  "notify_announcements"]
        help_texts = {
            "notify_forecast": "When a forecast you generate finishes or fails.",
            "notify_dataset_upload": "When a dataset you upload passes validation.",
            "notify_dataset_validation": "When a dataset you upload fails validation.",
            "notify_calendar": "When someone else adds, changes or deletes an event for one of your schools.",
            "notify_announcements": "Announcements sent by a System Administrator.",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            _switch(field)


class ThemeForm(forms.Form):
    theme = forms.ChoiceField(choices=UserProfile.Theme.choices)



class AccessRequestDetailsForm(forms.ModelForm):
    """What an applicant may add to their own pending request. Administrators decide the actual role."""

    class Meta:
        model = AccessRequest
        fields = ["requested_role", "applicant_note"]
        labels = {"requested_role": "Role you need", "applicant_note": "School(s) or organisation"}
        widgets = {
            "requested_role": forms.Select(attrs={"class": "form-select"}),
            "applicant_note": forms.TextInput(attrs={"class": "form-control",
                                                     "placeholder": "e.g. Head teacher, Kibera Primary (SCH004)"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Applicants can ask for a non-administrator role only.
        self.fields["requested_role"].choices = [("", "Not sure"), *[
            (value, label) for value, label in Role.choices if value != Role.SYSTEM_ADMIN]]
