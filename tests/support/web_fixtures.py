"""Account helpers for Django tests.

Since roles were introduced, School Operators only see their assigned schools. Page
tests that exercise school data log in as an operator assigned to the stub dataset's
schools (SCH001–SCH003), which matches what every user could see before roles.
"""

from __future__ import annotations

from django.contrib.auth import get_user_model

from web.accounts.models import SchoolAssignment
from web.schools.models import School

PASSWORD = "Str0ng-test-passw0rd!"
STUB_SCHOOLS = ("SCH001", "SCH002", "SCH003")


def assign_schools(user, codes=STUB_SCHOOLS):
    for code in codes:
        school, _ = School.objects.get_or_create(school_id=code)
        SchoolAssignment.objects.get_or_create(user=user, school=school)
    return user


def operator(username: str = "planner", codes=STUB_SCHOOLS, **extra):
    """A School Operator assigned to `codes`."""
    return assign_schools(get_user_model().objects.create_user(username=username, password=PASSWORD, **extra), codes)
