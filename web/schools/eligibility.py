"""Which school records may be chosen for NEW assignments and forecasts.

A school is eligible when its exact `school_id` appears in the active dataset (the same IDs the
dashboard, history and forecasts use) and its record is active. Names are never compared and no
IDs are hard-coded, so legacy development schools (e.g. SCH001-SCH003 while the Food4Education
dataset is active) drop out automatically, and come back if a dataset containing them is active.

Nothing here deletes or changes records: existing assignments, forecasts and audit logs stay.
If the active dataset can't be read, no school is eligible (fail closed) and every existing
assignment is treated as unavailable-but-kept, so nothing is silently granted or removed.
"""

from __future__ import annotations

from django.db.models import QuerySet

from web.schools.models import School

NOT_IN_DATASET = "not in the active dataset"
INACTIVE = "school record is inactive"
DATASET_UNAVAILABLE = "active dataset could not be read"


def active_dataset_school_codes() -> set[str] | None:
    """Exact school IDs present in the active dataset, or None if it can't be read."""
    from services import ServiceError, data_service

    try:
        return {str(s["school_id"]) for s in data_service.list_schools()}
    except ServiceError:
        return None


def eligible_schools(schools: QuerySet[School], codes: set[str] | None = ...) -> QuerySet[School]:
    """Restrict `schools` (already scoped to the acting user) to those eligible for new assignments."""
    if codes is ...:
        codes = active_dataset_school_codes()
    if codes is None:
        return schools.none()
    return schools.filter(is_active=True, school_id__in=codes)


def unavailable_reason(school: School, codes: set[str] | None) -> str | None:
    """Why an (existing) assignment to `school` can't be made today, or None if it can."""
    if codes is None:
        return DATASET_UNAVAILABLE
    if school.school_id not in codes:
        return NOT_IN_DATASET
    if not school.is_active:
        return INACTIVE
    return None


def school_label(school: School) -> str:
    return f"{school.name or school.school_id} ({school.school_id})"
