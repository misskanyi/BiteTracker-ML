"""Forms for the BiteTracker pages.

Field limits mirror the service-layer validation in services/forecast_service,
which remains the final check.
"""

from __future__ import annotations

import datetime as _dt

from django import forms
from django.utils import timezone

from services import date_forecast_service, planning_service
from services.forecast_service import HORIZONS, TEMPERATURE_RANGE_C
from web.core.uploads import validate_document


def _number(label: str, help_text: str, **kwargs) -> forms.FloatField:
    return forms.FloatField(
        label=label, required=False, help_text=help_text,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "any"}), **kwargs,
    )


class SchoolChoiceMixin:
    def __init__(self, *args, schools: list[dict], **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["school"].choices = [(s["school_id"], f"{s['school_name']} ({s['school_id']})") for s in schools]


class HistoryFilterForm(SchoolChoiceMixin, forms.Form):
    school = forms.ChoiceField(label="School", widget=forms.Select(attrs={"class": "form-select"}))
    start = forms.DateField(label="From", required=False, widget=forms.DateInput(attrs={"type": "date", "class": "form-control"}))
    end = forms.DateField(label="To", required=False, widget=forms.DateInput(attrs={"type": "date", "class": "form-control"}))

    def clean(self):
        cleaned = super().clean()
        start, end = cleaned.get("start"), cleaned.get("end")
        if start and end and start > end:
            raise forms.ValidationError("The start date must be on or before the end date.")
        return cleaned


def _date(label: str) -> forms.DateField:
    return forms.DateField(label=label, required=False, widget=forms.DateInput(attrs={"type": "date", "class": "form-control"}))


class ForecastForm(SchoolChoiceMixin, forms.Form):
    MODE_HORIZON, MODE_SINGLE, MODE_RANGE = "horizon", date_forecast_service.SINGLE_DAY, date_forecast_service.DATE_RANGE

    school = forms.ChoiceField(label="School", widget=forms.Select(attrs={"class": "form-select"}))
    # Blank = "horizon", so requests that predate date selection behave as before.
    mode = forms.ChoiceField(
        label="Dates", required=False, initial=MODE_HORIZON,
        choices=[(MODE_HORIZON, "Next school days"), (MODE_SINGLE, "Single day"), (MODE_RANGE, "Date range")],
        widget=forms.RadioSelect(attrs={"class": "btn-check"}),
    )
    horizon = forms.TypedChoiceField(
        label="Forecast horizon", coerce=int, choices=list(HORIZONS.items()), initial=1, required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    target_date = _date("School day")
    # Set by the "Forecast anyway" button of the likely-closed popup; must match the selection it confirmed.
    confirm_closed = forms.CharField(required=False, widget=forms.HiddenInput)
    range_start = _date("From")
    range_end = _date("To")
    expected_attendance = _number(
        "Expected students present on day 1",
        "Optional. A day's own attendance is not a model input, so this only affects day 2 onwards.",
        min_value=0,
    )
    expected_rainfall_mm = _number("Expected rainfall (mm)", "Optional. Blank = recent 14-day average.", min_value=0)
    expected_temperature_c = _number(
        "Expected temperature (°C)", "Optional. Blank = recent 14-day average.",
        min_value=TEMPERATURE_RANGE_C[0], max_value=TEMPERATURE_RANGE_C[1],
    )
    available_capacity = _number(
        "Planned meal capacity for day 1",
        "Optional. Compares day-1 predicted demand with the meals you currently plan to prepare.",
        min_value=0,
    )
    # Planning assumption only: it prices the forecast and never reaches the model.
    cost_per_meal_kes = forms.DecimalField(
        label="Cost per meal (KES)", required=False, max_digits=7, decimal_places=2,
        min_value=planning_service.MIN_COST_PER_MEAL, max_value=planning_service.MAX_COST_PER_MEAL,
        help_text="Planning assumption used to estimate the meal budget. Blank = the default.",
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01", "inputmode": "decimal"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["cost_per_meal_kes"].initial = planning_service.default_cost_per_meal()
        # One month ahead at most: the date picker greys out later dates and clean() rejects them.
        self.today = timezone.localdate()
        self.max_date = date_forecast_service.latest_selectable_date(self.today)
        for name in ("target_date", "range_start", "range_end"):
            self.fields[name].widget.attrs["max"] = self.max_date.isoformat()

    def clean_cost_per_meal_kes(self):
        cost = self.cleaned_data.get("cost_per_meal_kes")
        return planning_service.default_cost_per_meal() if cost is None else cost

    def closure_signature(self) -> str:
        """Identifies the exact selection a likely-closed confirmation applies to."""
        c = self.cleaned_data
        single = c.get("mode") == self.MODE_SINGLE
        start = c.get("target_date") if single else c.get("range_start")
        end = start if single else c.get("range_end")
        return f"{c.get('school')}|{c.get('mode')}|{start}|{end}"

    def clean_mode(self):
        return self.cleaned_data.get("mode") or self.MODE_HORIZON

    def clean(self):
        cleaned = super().clean()
        mode = cleaned.get("mode") or self.MODE_HORIZON
        if mode == self.MODE_HORIZON:
            if cleaned.get("horizon") in (None, ""):
                self.add_error("horizon", "Choose a forecast horizon.")
        elif mode == self.MODE_SINGLE:
            if not cleaned.get("target_date"):
                self.add_error("target_date", "Choose a school day.")
            elif cleaned["target_date"] > self.max_date:
                self.add_error("target_date", date_forecast_service.max_date_message(self.today))
        else:
            start, end = cleaned.get("range_start"), cleaned.get("range_end")
            if not start:
                self.add_error("range_start", "Choose the first date.")
            if not end:
                self.add_error("range_end", "Choose the last date.")
            if start and end and start > end:
                self.add_error("range_end", "The last date must be on or after the first date.")
            for name in ("range_start", "range_end"):
                if cleaned.get(name) and cleaned[name] > self.max_date and name not in self.errors:
                    self.add_error(name, date_forecast_service.max_date_message(self.today))
        return cleaned


class DatasetUploadForm(forms.Form):
    file = forms.FileField(
        label="Dataset file (.csv or .xlsx)",
        widget=forms.ClearableFileInput(attrs={"class": "form-control", "accept": ".csv,.xlsx"}),
    )
    letter = forms.FileField(
        label="Signed supporting letter (.pdf, .docx, .png or .jpg)",
        widget=forms.ClearableFileInput(attrs={"class": "form-control", "accept": ".pdf,.docx,.png,.jpg,.jpeg"}),
    )

    def __init__(self, *args, max_size: int, allowed_extensions: tuple[str, ...], letter_max_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size, self.allowed_extensions, self.letter_max_size = max_size, allowed_extensions, letter_max_size

    def clean_letter(self):
        return validate_document(self.cleaned_data["letter"], self.letter_max_size)

    def clean_file(self):
        upload = self.cleaned_data["file"]
        name = upload.name.lower()
        if not name.endswith(self.allowed_extensions):
            raise forms.ValidationError("Unsupported file type. Upload a CSV (.csv) or Excel (.xlsx) file.")
        if upload.size > self.max_size:
            raise forms.ValidationError(f"The file is larger than the {self.max_size // (1024 * 1024)} MB limit.")
        return upload


# --- Forecast vs actual ------------------------------------------------------------------------

MAX_MEAL_COUNT = 100_000


def _count(label: str, required: bool, help_text: str = "") -> forms.IntegerField:
    # IntegerField rejects "12.5" ("Enter a whole number."); min_value rejects negatives. Zero is valid.
    return forms.IntegerField(
        label=label, required=required, min_value=0, max_value=MAX_MEAL_COUNT, help_text=help_text,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1", "min": "0", "inputmode": "numeric"}),
    )


class ActualMealsForm(forms.Form):
    """Record meals actually served on one day of a saved forecast."""

    date = forms.TypedChoiceField(label="Day", coerce=_dt.date.fromisoformat,
                                  widget=forms.Select(attrs={"class": "form-select"}))
    meals_served = _count("Meals served", True, "Count of meals actually served. Enter 0 if no meals were served.")
    meals_prepared = _count("Meals prepared (optional)", False,
                            "Leave blank if unknown. Used only for the prepared-vs-served planning figure.")

    def __init__(self, *args, days, today, **kwargs):
        super().__init__(*args, **kwargs)
        self.today = today
        self.fields["date"].choices = [(d.isoformat(), f"{d:%a} {d.day} {d:%b %Y}" + (" (future)" if d > today else ""))
                                       for d in days]

    def clean_date(self):
        day = self.cleaned_data["date"]
        if day > self.today:
            raise forms.ValidationError("You can't record meals served for a future date.")
        return day


class ActualCorrectionForm(forms.Form):
    meals_served = _count("Corrected meals served", True)
    meals_prepared = _count("Corrected meals prepared (optional)", False)
    reason = forms.CharField(label="Reason for the correction", max_length=1000,
                             widget=forms.Textarea(attrs={"class": "form-control", "rows": 2}))


class ActualEditForm(forms.Form):
    meals_served = _count("Meals served", True)
    meals_prepared = _count("Meals prepared (optional)", False)


class ActualReviewForm(forms.Form):
    action = forms.ChoiceField(choices=[(a, a) for a in (
        "verify", "reject", "request_correction", "publish", "approve_correction", "decline_correction")])
    comment = forms.CharField(required=False, max_length=2000,
                              widget=forms.Textarea(attrs={"class": "form-control", "rows": 2}))
    expected_status = forms.CharField(required=False, max_length=24, widget=forms.HiddenInput)
    acknowledge_conflicts = forms.BooleanField(required=False)


class FollowUpFilterForm(forms.Form):
    """School and date selection for the forecast-vs-actual page and its CSV export.

    The school choices are the user's own scope; the view still rejects anything outside it (404).
    """

    MAX_DAYS = 366

    school = forms.MultipleChoiceField(label="Schools", required=False, widget=forms.CheckboxSelectMultiple,
                                       help_text="None ticked = all your schools.")
    start = _date("From")
    end = _date("To")
    status = forms.ChoiceField(label="Status", required=False, widget=forms.Select(attrs={"class": "form-select"}))

    def __init__(self, *args, schools: list[tuple[str, str]], statuses, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["school"].choices = schools
        self.fields["status"].choices = [("", "Any status"), *statuses]

    def clean(self):
        cleaned = super().clean()
        start, end = cleaned.get("start"), cleaned.get("end")
        if start and end:
            if start > end:
                raise forms.ValidationError("The start date must be on or before the end date.")
            if (end - start).days >= self.MAX_DAYS:
                raise forms.ValidationError(f"Choose a period of at most {self.MAX_DAYS} days.")
        return cleaned
