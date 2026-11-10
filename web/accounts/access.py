"""Server-side authorisation: roles, school scope, account status and who may manage whom.

Every view that shows school data or changes an account goes through these
functions. Hiding a link in the sidebar is never the access check.

- School Manager: assigned schools; may change data for them (uploads, calendar).
- Resource Planning Stakeholder: assigned schools, read-only.
- System Administrator: everything, and the only role with administration rights.
"""

from __future__ import annotations

from functools import wraps

from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.db.models import Q, QuerySet

from web.accounts.models import AccessRequest, Role, SchoolAssignment, UserProfile
from web.schools.models import DatasetUpload, School


def get_profile(user) -> UserProfile:
    """The user's profile, created on first use for accounts that predate profiles."""
    try:
        return user.profile
    except UserProfile.DoesNotExist:
        profile, _ = UserProfile.objects.get_or_create(
            user=user, defaults={"role": Role.SYSTEM_ADMIN if user.is_superuser else Role.SCHOOL_MANAGER}
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


def is_school_manager(user) -> bool:
    return user.is_authenticated and user.is_active and role_of(user) == Role.SCHOOL_MANAGER


def is_planning_stakeholder(user) -> bool:
    return user.is_authenticated and user.is_active and role_of(user) == Role.PLANNING_STAKEHOLDER


def is_admin(user) -> bool:
    """Who may open Administration: System Administrators only."""
    return is_system_admin(user)


def can_view_model_evaluation(user) -> bool:
    """Model evaluation results are an administrator function."""
    return is_system_admin(user)


# --- Account status ---------------------------------------------------------------------------

APPROVED, PENDING, REJECTED, SUSPENDED = "approved", "pending", "rejected", "suspended"
STATUS_LABELS = {APPROVED: "Active", PENDING: "Pending approval", REJECTED: "Rejected", SUSPENDED: "Suspended"}


def access_request_of(user) -> AccessRequest | None:
    try:
        return user.access_request
    except AccessRequest.DoesNotExist:
        return None


def account_status(user) -> str:
    """approved, pending, rejected or suspended.

    Accounts without an access request were created by an administrator and are
    approved; an approved account that is inactive is suspended.
    """
    request = access_request_of(user)
    if request is not None and request.status != AccessRequest.Status.APPROVED:
        return PENDING if request.status == AccessRequest.Status.PENDING else REJECTED
    return APPROVED if user.is_active else SUSPENDED


def is_approved(user) -> bool:
    return account_status(user) == APPROVED


# --- School scope ---------------------------------------------------------------------------


def accessible_schools(user) -> QuerySet[School]:
    """School records the user may see and work with."""
    if is_system_admin(user):
        return School.objects.all()
    if not user.is_authenticated:
        return School.objects.none()
    return School.objects.filter(assignments__user=user).distinct()


def accessible_school_codes(user) -> set[str] | None:
    """Dataset school IDs the user may see, or None for every school (System Administrators).

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


def can_edit_school_data(user, school_code: str | None = None) -> bool:
    """School Managers (for their assigned schools) and System Administrators.

    Resource Planning Stakeholders are read-only. With no school code: may the user
    edit data for at least one school?
    """
    if is_system_admin(user):
        return True
    if not is_school_manager(user):
        return False
    if school_code is None:
        return accessible_schools(user).exists()
    return can_access_school_code(user, school_code)


def can_upload(user) -> bool:
    """Submitting school records for verification."""
    return can_edit_school_data(user)


# --- Account management ---------------------------------------------------------------------


def assignable_roles(admin) -> list[tuple[str, str]]:
    """Roles an administrator may give: every role, for System Administrators only."""
    return list(Role.choices) if is_system_admin(admin) else []


def manageable_users(admin) -> QuerySet:
    """Accounts the user may view and change: all of them for System Administrators, else none."""
    User = get_user_model()
    return User.objects.all() if is_system_admin(admin) else User.objects.none()


def can_manage_user(admin, target) -> bool:
    return manageable_users(admin).filter(pk=target.pk).exists()


# --- Dataset upload review ------------------------------------------------------------------


def reviewable_uploads(admin) -> QuerySet[DatasetUpload]:
    """Stored uploads (and their letters) awaiting or past verification: System Administrators only.

    `can_decide_upload` stops anyone deciding on their own upload.
    """
    uploads = DatasetUpload.objects.exclude(review_status="").select_related("uploaded_by", "reviewed_by")
    return uploads if is_system_admin(admin) else uploads.none()


def visible_uploads(user) -> QuerySet[DatasetUpload]:
    """Upload records a user may open: their own, or every upload for System Administrators."""
    uploads = DatasetUpload.objects.select_related("uploaded_by", "reviewed_by")
    if is_system_admin(user):
        return uploads
    if not user.is_authenticated:
        return uploads.none()
    return uploads.filter(uploaded_by=user)


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

    Assignments outside the scope are left alone, so a partial update can never
    remove schools outside the scope. Returns (added, removed) School sets.
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


def system_admin_required(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not is_system_admin(request.user):
            raise PermissionDenied
        return view(request, *args, **kwargs)
    return wrapper


# Administration is System Administrator only; the old name is kept for existing views.
admin_required = system_admin_required


def model_evaluation_required(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not can_view_model_evaluation(request.user):
            raise PermissionDenied
        return view(request, *args, **kwargs)
    return wrapper


def upload_required(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not can_upload(request.user):
            raise PermissionDenied
        return view(request, *args, **kwargs)
    return wrapper
