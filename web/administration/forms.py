"""Administration forms (System Administrators only). Role and school choices still come from
the acting user's scope in web.accounts.access, so a crafted POST can't submit anything else."""

from __future__ import annotations

from django import forms
from django.contrib.auth import get_user_model, password_validation

from web.accounts import access
from web.accounts.forms import PHONE_RE
from web.accounts.models import Role
from web.audit.models import AuditLog
from web.schools.models import DatasetUpload, School

_text = {"class": "form-control"}


class _UserFieldsMixin:
    """Shared validation for create and edit."""

    def _setup_scope_fields(self, admin, target=None):
        self.admin, self.target = admin, target
        self.fields["role"].choices = access.assignable_roles(admin)
        self.fields["schools"].queryset = access.accessible_schools(admin).order_by("name", "school_id")

    def clean_email(self):
        email = self.cleaned_data["email"].strip()
        others = get_user_model().objects.filter(email__iexact=email)
        if self.target is not None:
            others = others.exclude(pk=self.target.pk)
        if email and others.exists():
            # Email login needs unique addresses (see web/accounts/backends.py).
            raise forms.ValidationError("Another account already uses this email address.")
        return email

    def clean_phone(self):
        phone = self.cleaned_data.get("phone", "").strip()
        if phone and not PHONE_RE.match(phone):
            raise forms.ValidationError("Enter a valid phone number, e.g. +254 712 345 678.")
        return phone


class UserCreateForm(_UserFieldsMixin, forms.Form):
    username = forms.CharField(max_length=150, widget=forms.TextInput(attrs={**_text, "autocomplete": "off"}),
                               help_text="Letters, digits and @ . + - _ only.")
    first_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs=_text))
    last_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs=_text))
    email = forms.EmailField(widget=forms.EmailInput(attrs=_text))
    phone = forms.CharField(label="Phone number", max_length=20, required=False, widget=forms.TextInput(attrs=_text))
    role = forms.ChoiceField(widget=forms.Select(attrs={"class": "form-select"}))
    schools = forms.ModelMultipleChoiceField(queryset=School.objects.none(), required=False,
                                             widget=forms.CheckboxSelectMultiple)
    password1 = forms.CharField(label="Temporary password", strip=False,
                                widget=forms.PasswordInput(attrs={**_text, "autocomplete": "new-password"}),
                                help_text="Share it with the user privately; they can change it in Settings → Security.")
    password2 = forms.CharField(label="Confirm temporary password", strip=False,
                                widget=forms.PasswordInput(attrs={**_text, "autocomplete": "new-password"}))

    def __init__(self, *args, admin, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_scope_fields(admin)
        self.fields["role"].initial = Role.SCHOOL_MANAGER

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        User = get_user_model()
        User.username_validator(username)
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError("That username is already taken.")
        return username

    def clean(self):
        cleaned = super().clean()
        p1, p2 = cleaned.get("password1"), cleaned.get("password2")
        if p1 and p2 and p1 != p2:
            self.add_error("password2", "The two passwords do not match.")
        elif p1:
            probe = get_user_model()(username=cleaned.get("username", ""), email=cleaned.get("email", ""),
                                     first_name=cleaned.get("first_name", ""), last_name=cleaned.get("last_name", ""))
            try:
                password_validation.validate_password(p1, probe)
            except forms.ValidationError as exc:
                self.add_error("password1", exc)
        return cleaned


class UserEditForm(_UserFieldsMixin, forms.Form):
    first_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs=_text))
    last_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs=_text))
    email = forms.EmailField(widget=forms.EmailInput(attrs=_text))
    phone = forms.CharField(label="Phone number", max_length=20, required=False, widget=forms.TextInput(attrs=_text))
    role = forms.ChoiceField(widget=forms.Select(attrs={"class": "form-select"}))
    schools = forms.ModelMultipleChoiceField(queryset=School.objects.none(), required=False,
                                             widget=forms.CheckboxSelectMultiple)

    def __init__(self, *args, admin, target, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_scope_fields(admin, target)
        if target.pk == admin.pk:
            # Nobody changes their own role.
            del self.fields["role"]

    def clean_role(self):
        role = self.cleaned_data["role"]
        was_sysadmin = access.role_of(self.target) == Role.SYSTEM_ADMIN
        if was_sysadmin and role != Role.SYSTEM_ADMIN and self.target.is_active and access.active_system_admin_count() <= 1:
            raise forms.ValidationError("This is the only active System Administrator, so their role cannot be changed.")
        return role


class SchoolForm(forms.ModelForm):
    class Meta:
        model = School
        fields = ["name", "county", "sub_county", "verified_enrolment", "is_active"]
        labels = {"is_active": "Active school record"}
        help_texts = {"is_active": "Inactive schools keep their history and forecasts but cannot be picked for new calendar events."}
        widgets = {
            "name": forms.TextInput(attrs=_text),
            "county": forms.TextInput(attrs=_text),
            "sub_county": forms.TextInput(attrs=_text),
            "verified_enrolment": forms.NumberInput(attrs={**_text, "min": 0}),
            "is_active": forms.CheckboxInput(attrs={"class": "form-check-input", "role": "switch"}),
        }


class AssignUserForm(forms.Form):
    user = forms.ModelChoiceField(queryset=get_user_model().objects.none(), label="Add a user to this school",
                                  widget=forms.Select(attrs={"class": "form-select"}), empty_label="Select a user")

    def __init__(self, *args, admin, school, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["user"].queryset = (
            access.manageable_users(admin).filter(is_active=True).exclude(school_assignments__school=school)
            .order_by("first_name", "last_name", "username")
        )
        self.fields["user"].label_from_instance = lambda u: f"{u.get_full_name() or u.username} ({u.username})"


class AnnouncementForm(forms.Form):
    title = forms.CharField(max_length=200, widget=forms.TextInput(attrs=_text))
    message = forms.CharField(widget=forms.Textarea(attrs={**_text, "rows": 4}), max_length=2000)


class LogFilterForm(forms.Form):
    q = forms.CharField(label="User", required=False,
                        widget=forms.TextInput(attrs={**_text, "placeholder": "Username", "class": "form-control form-control-sm"}))
    action = forms.ChoiceField(label="Action", required=False, choices=[("", "All actions"), *AuditLog.Action.choices],
                               widget=forms.Select(attrs={"class": "form-select form-select-sm"}))
    start = forms.DateField(label="From", required=False,
                            widget=forms.DateInput(attrs={"type": "date", "class": "form-control form-control-sm"}))
    end = forms.DateField(label="To", required=False,
                          widget=forms.DateInput(attrs={"type": "date", "class": "form-control form-control-sm"}))

    def clean(self):
        cleaned = super().clean()
        if cleaned.get("start") and cleaned.get("end") and cleaned["start"] > cleaned["end"]:
            raise forms.ValidationError("The start date must be on or before the end date.")
        return cleaned


class UploadReviewForm(forms.Form):
    decision = forms.ChoiceField(
        choices=[(DatasetUpload.ReviewStatus.APPROVED, "Approve"),
                 (DatasetUpload.ReviewStatus.CLARIFICATION, "Request clarification"),
                 (DatasetUpload.ReviewStatus.REJECTED, "Reject")],
        widget=forms.RadioSelect(attrs={"class": "form-check-input"}),
    )
    comment = forms.CharField(
        label="Comment to the uploader", required=False, max_length=2000,
        help_text="Required when rejecting or asking for clarification: say what is wrong or what you need.",
        widget=forms.Textarea(attrs={**_text, "rows": 3}),
    )

    def clean(self):
        cleaned = super().clean()
        decision = cleaned.get("decision")
        if not cleaned.get("comment", "").strip():
            if decision == DatasetUpload.ReviewStatus.REJECTED:
                self.add_error("comment", "Explain why the upload is rejected.")
            elif decision == DatasetUpload.ReviewStatus.CLARIFICATION:
                self.add_error("comment", "Say what the uploader needs to clarify.")
        return cleaned


# --- Registration (access) requests -------------------------------------------------------------


class AccessApproveForm(forms.Form):
    """Approval must grant exactly one role and, for non-administrators, at least one school."""

    role = forms.ChoiceField(choices=Role.choices, widget=forms.Select(attrs={"class": "form-select"}))
    schools = forms.ModelMultipleChoiceField(
        queryset=School.objects.none(), required=False, widget=forms.CheckboxSelectMultiple,
        help_text="School Managers and Resource Planning Stakeholders see only these schools.",
    )
    note = forms.CharField(label="Note for the audit record", required=False, max_length=1000,
                           widget=forms.TextInput(attrs=_text))

    def __init__(self, *args, admin, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["role"].choices = access.assignable_roles(admin)
        self.fields["schools"].queryset = access.accessible_schools(admin).filter(is_active=True).order_by(
            "name", "school_id")

    def clean(self):
        cleaned = super().clean()
        role = cleaned.get("role")
        if role and role != Role.SYSTEM_ADMIN and not cleaned.get("schools"):
            self.add_error("schools", "Assign at least one school for this role.")
        if role == Role.SYSTEM_ADMIN:
            cleaned["schools"] = []  # System Administrators see every school
        return cleaned


class AccessRejectForm(forms.Form):
    reason = forms.CharField(
        label="Reason (kept in the audit record, not shown to the applicant)", max_length=1000,
        widget=forms.Textarea(attrs={**_text, "rows": 2}),
    )
