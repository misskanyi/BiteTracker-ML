"""Forms for the BiteTracker pages.

Field limits mirror the service-layer validation in services/forecast_service,
which remains the final check.
"""

from __future__ import annotations

from django import forms

from services import planning_service
from services.forecast_service import HORIZONS, TEMPERATURE_RANGE_C


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


class ForecastForm(SchoolChoiceMixin, forms.Form):
    school = forms.ChoiceField(label="School", widget=forms.Select(attrs={"class": "form-select"}))
    horizon = forms.TypedChoiceField(
        label="Forecast horizon", coerce=int, choices=list(HORIZONS.items()), initial=1,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
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

    def clean_cost_per_meal_kes(self):
        cost = self.cleaned_data.get("cost_per_meal_kes")
        return planning_service.default_cost_per_meal() if cost is None else cost


class DatasetUploadForm(forms.Form):
    file = forms.FileField(
        label="Dataset file (.csv or .xlsx)",
        widget=forms.ClearableFileInput(attrs={"class": "form-control", "accept": ".csv,.xlsx"}),
    )

    def __init__(self, *args, max_size: int, allowed_extensions: tuple[str, ...], **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size, self.allowed_extensions = max_size, allowed_extensions

    def clean_file(self):
        upload = self.cleaned_data["file"]
        name = upload.name.lower()
        if not name.endswith(self.allowed_extensions):
            raise forms.ValidationError("Unsupported file type. Upload a CSV (.csv) or Excel (.xlsx) file.")
        if upload.size > self.max_size:
            raise forms.ValidationError(f"The file is larger than the {self.max_size // (1024 * 1024)} MB limit.")
        return upload
