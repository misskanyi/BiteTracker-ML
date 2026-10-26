"""Server-side authorisation: roles, school scope and who may manage whom.

Every view that shows school data or changes an account goes through these
functions. Hiding a link in the sidebar is never the access check.
"""

from __future__ import annotations

from functools import wraps

from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.db.models import Q, QuerySet

from web.accounts.models import Role, SchoolAssignment, UserProfile
from web.schools.models import DatasetUpload, School


def get_profile(user) -> UserProfile:
    """The user's profile, created on first use for accounts that predate profiles."""
    try:
        return user.profile
    except UserProfile.DoesNotExist:
        profile, _ = UserProfile.objects.get_or_create(
            user=user, defaults={"role": Role.SYSTEM_ADMIN if user.is_superuser else Role.SCHOOL_OPERATOR}
        )
        user.profile = profile
        return profile


def role_of(user) -> str:
    if user.is_superuser:
        return Role.SYSTEM_ADMIN
    return get_profile(user).role


def role_label(user) -> str:
    return Role(role_of(user)).label


def is_system_admin(user) -> bool:
    return user.is_authenticated and user.is_active and role_of(user) == Role.SYSTEM_ADMIN


def is_programme_admin(user) -> bool:
    return user.is_authenticated and user.is_active and role_of(user) == Role.PROGRAMME_ADMIN


def is_admin(user) -> bool:
    """Programme Admins and System Admins may open Administration."""
    return is_system_admin(user) or is_programme_admin(user)


# --- School scope ---------------------------------------------------------------------------


def accessible_schools(user) -> QuerySet[School]:
    """School records the user may see and work with."""
    if is_system_admin(user):
        return School.objects.all()
    if not user.is_authenticated:
        return School.objects.none()
    return School.objects.filter(assignments__user=user).distinct()


def accessible_school_codes(user) -> set[str] | None:
    """Dataset school IDs the user may see, or None for every school (System Admins).

    Historical demand data lives in the pipeline CSV keyed by school ID, so this is
    the filter applied to service-layer results.
    """
    if is_system_admin(user):
        return None
    return set(accessible_schools(user).values_list("school_id", flat=True))


def can_access_school_code(user, school_code: str) -> bool:
    codes = accessible_school_codes(user)
    return codes is None or school_code in codes


def filter_school_records(user, records: list[dict], key: str = "school_id") -> list[dict]:
    codes = accessible_school_codes(user)
    return list(records) if codes is None else [r for r in records if r.get(key) in codes]


# --- Account management ---------------------------------------------------------------------


def assignable_roles(admin) -> list[tuple[str, str]]:
    """Roles an administrator may give. Programme Admins can never grant a higher role."""
    if is_system_admin(admin):
        return list(Role.choices)
    if is_programme_admin(admin):
        return [(Role.SCHOOL_OPERATOR.value, Role.SCHOOL_OPERATOR.label)]
    return []


def manageable_users(admin) -> QuerySet:
    """Accounts the administrator may view and change.

    Programme Admins manage School Operators whose every assigned school lies within
    their own schools (operators with no schools yet are unclaimed, so included).
    """
    User = get_user_model()
    if is_system_admin(admin):
        return User.objects.all()
    if not is_programme_admin(admin):
        return User.objects.none()
    scope = accessible_schools(admin)
    return (
        User.objects.filter(is_superuser=False, profile__role=Role.SCHOOL_OPERATOR)
        .exclude(school_assignments__school__in=School.objects.exclude(pk__in=scope.values("pk")))
        .exclude(pk=admin.pk)
        .distinct()
    )


def can_manage_user(admin, target) -> bool:
    return manageable_users(admin).filter(pk=target.pk).exists()


# --- Dataset upload review ------------------------------------------------------------------


def reviewable_uploads(admin) -> QuerySet[DatasetUpload]:
    """Stored uploads (and their letters) the administrator may review.

    System Admins see every upload; Programme Admins see uploads from the operators
    they manage. `can_decide_upload` stops anyone deciding on their own upload.
    """
    uploads = DatasetUpload.objects.exclude(review_status="").select_related("uploaded_by", "reviewed_by")
    if is_system_admin(admin):
        return uploads
    if is_programme_admin(admin):
        return uploads.filter(uploaded_by__in=manageable_users(admin))
    return uploads.none()


def can_decide_upload(admin, upload: DatasetUpload) -> bool:
    return upload.uploaded_by_id != admin.pk and reviewable_uploads(admin).filter(pk=upload.pk).exists()


def can_download_upload(user, upload: DatasetUpload) -> bool:
    """The uploader and the administrators who may review it can download its files."""
    if not user.is_authenticated:
        return False
    return upload.uploaded_by_id == user.pk or reviewable_uploads(user).filter(pk=upload.pk).exists()


def active_system_admin_count() -> int:
    return (
        get_user_model()
        .objects.filter(is_active=True)
        .filter(Q(is_superuser=True) | Q(profile__role=Role.SYSTEM_ADMIN))
        .distinct()
        .count()
    )


def set_role(user, role: str) -> None:
    """Store a role and keep Django's superuser/staff flags consistent with it."""
    profile = get_profile(user)
    profile.role = role
    profile.save(update_fields=["role", "updated_at"])
    is_sysadmin = role == Role.SYSTEM_ADMIN
    if user.is_superuser != is_sysadmin or user.is_staff != is_sysadmin:
        user.is_superuser = user.is_staff = is_sysadmin
        user.save(update_fields=["is_superuser", "is_staff"])


def set_school_assignments(user, schools, assigned_by, scope: QuerySet[School] | None = None) -> tuple[set, set]:
    """Make the user's assignments within `scope` (None = all schools) equal `schools`.

    Assignments outside the scope are left alone, so a Programme Admin can never
    remove another programme's schools. Returns (added, removed) School sets.
    """
    wanted = set(schools)
    current_qs = SchoolAssignment.objects.filter(user=user).select_related("school")
    if scope is not None:
        current_qs = current_qs.filter(school__in=scope)
    current = {a.school for a in current_qs}
    added, removed = wanted - current, current - wanted
    SchoolAssignment.objects.bulk_create(
        [SchoolAssignment(user=user, school=s, assigned_by=assigned_by) for s in added], ignore_conflicts=True
    )
    if removed:
        SchoolAssignment.objects.filter(user=user, school__in=removed).delete()
    return added, removed


# --- View decorators ------------------------------------------------------------------------


def admin_required(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not is_admin(request.user):
            raise PermissionDenied
        return view(request, *args, **kwargs)
    return wrapper


def system_admin_required(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not is_system_admin(request.user):
            raise PermissionDenied
        return view(request, *args, **kwargs)
    return wrapper
