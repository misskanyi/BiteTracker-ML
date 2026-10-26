"""School calendar events for operational planning.

These events are reference information for planners. They are NOT model
features: the forecasting pipeline does not read this table.
"""

from __future__ import annotations

from django.conf import settings
from django.db import models
from django.db.models import Q

from web.schools.models import School


class CalendarEvent(models.Model):
    class EventType(models.TextChoices):
        TERM_OPENING = "term_opening", "Term opening"
        TERM_CLOSING = "term_closing", "Term closing"
        PUBLIC_HOLIDAY = "public_holiday", "Public holiday"
        MID_TERM_BREAK = "mid_term_break", "Mid-term break"
        SCHOOL_CLOSURE = "school_closure", "School closure"
        EXAMINATION = "examination", "Examination period"
        SPECIAL_EVENT = "special_event", "Special school event"
        OTHER = "other", "Other"

    class FeedingStatus(models.TextChoices):
        MEALS_EXPECTED = "meals_expected", "Meals expected"
        NO_MEALS = "no_meals", "No meals expected"
        UNKNOWN = "unknown", "Unknown"

    title = models.CharField(max_length=200)
    event_type = models.CharField(max_length=30, choices=EventType.choices)
    # Blank school = applies to every school (only System Admins can create these).
    school = models.ForeignKey(School, null=True, blank=True, on_delete=models.PROTECT, related_name="calendar_events")
    start_date = models.DateField()
    end_date = models.DateField()
    feeding_status = models.CharField(max_length=20, choices=FeedingStatus.choices, default=FeedingStatus.UNKNOWN)
    description = models.TextField(blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["start_date", "end_date", "title"]
        constraints = [
            models.CheckConstraint(condition=Q(end_date__gte=models.F("start_date")), name="calendar_event_end_after_start"),
        ]
        indexes = [models.Index(fields=["start_date", "end_date"], name="calendar_event_dates")]

    def __str__(self) -> str:
        return f"{self.title} ({self.start_date:%d %b %Y})"

    @property
    def scope_label(self) -> str:
        return str(self.school) if self.school else "All schools"

    @property
    def academic_year(self) -> int:
        # Kenyan primary schools run January–November, so the academic year is the calendar year.
        return self.start_date.year

    @property
    def is_multi_day(self) -> bool:
        return self.end_date != self.start_date
