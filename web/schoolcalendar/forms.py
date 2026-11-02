from __future__ import annotations

from django import forms
from django.db.models import Q

from web.accounts import access
from web.schoolcalendar.models import CalendarEvent


class CalendarEventForm(forms.ModelForm):
    confirm_overlap = forms.BooleanField(
        label="Save anyway: I have checked the overlapping events", required=False,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )

    class Meta:
        model = CalendarEvent
        fields = ["title", "event_type", "school", "start_date", "end_date", "feeding_status", "description"]
        labels = {"description": "Notes"}
        help_texts = {
            "feeding_status": "Whether learners are expected to be fed on these days. For planning reference only.",
            "description": "Optional details for other planners.",
        }
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control", "maxlength": 200}),
            "event_type": forms.Select(attrs={"class": "form-select"}),
            "school": forms.Select(attrs={"class": "form-select"}),
            "start_date": forms.DateInput(attrs={"type": "date", "class": "form-control"}, format="%Y-%m-%d"),
            "end_date": forms.DateInput(attrs={"type": "date", "class": "form-control"}, format="%Y-%m-%d"),
            "feeding_status": forms.Select(attrs={"class": "form-select"}),
            "description": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
        }

    def __init__(self, *args, user, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user
        self.overlaps: list[CalendarEvent] = []
        schools = access.accessible_schools(user).filter(is_active=True)
        if self.instance.pk and self.instance.school_id:
            schools = (schools | access.accessible_schools(user).filter(pk=self.instance.school_id)).distinct()
        field = self.fields["school"]
        field.queryset = schools.order_by("name", "school_id")
        if access.is_system_admin(user):
            field.required, field.empty_label = False, "All schools"
            field.help_text = "Leave as “All schools” for programme-wide dates such as public holidays."
        else:
            field.required, field.empty_label = True, "Select a school"

    def clean(self):
        cleaned = super().clean()
        start, end = cleaned.get("start_date"), cleaned.get("end_date")
        if start and end and end < start:
            self.add_error("end_date", "The end date cannot be before the start date.")
            return cleaned
        if not start or not end or self.errors:
            return cleaned
        school = cleaned.get("school")
        overlapping = CalendarEvent.objects.filter(start_date__lte=end, end_date__gte=start).select_related("school")
        if school is not None:
            overlapping = overlapping.filter(Q(school=school) | Q(school__isnull=True))
        if self.instance.pk:
            overlapping = overlapping.exclude(pk=self.instance.pk)
        self.overlaps = list(overlapping[:10])
        conflicting = [e for e in self.overlaps if e.feeding_status != cleaned.get("feeding_status")]
        if conflicting and not cleaned.get("confirm_overlap"):
            names = "; ".join(f"{e.title} ({e.start_date:%d %b}–{e.end_date:%d %b %Y}, {e.get_feeding_status_display()})"
                              for e in conflicting[:3])
            raise forms.ValidationError(
                f"These dates overlap events with a different feeding status: {names}. "
                "Check them, then tick the confirmation box to save anyway.", code="overlap",
            )
        return cleaned

    @property
    def needs_overlap_confirmation(self) -> bool:
        return any(e.code == "overlap" for e in self.non_field_errors().as_data())
