"""My Profile, Settings, roles migration behaviour and school scoping of existing pages.

What this file checks:
- New accounts get a profile; superusers are System Admins.
- Profile edits persist, invalid phone numbers and non-image files are rejected, and
  role/school assignments cannot be changed through the profile form.
- Profile photos are private: other operators cannot load them.
- Theme, compact tables, reduced motion and landing page persist and reach the page.
- The header theme toggle endpoint saves the theme and rejects bad values.
- Password change verifies the current password, applies Django's validators,
  keeps the session, and is audited without storing the password.
- Notification preferences persist.
- Existing pages only show schools assigned to the user, and URL tampering is refused.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from tests.support.web_fixtures import PASSWORD, operator
from web.accounts import access
from web.accounts.models import Role, SchoolAssignment, UserProfile
from web.audit.models import AuditLog
from web.notifications.models import Notification
from web.schools.models import ForecastRun

pytestmark = pytest.mark.django_db

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture
def media_root(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "uploads"
    return settings.MEDIA_ROOT


@pytest.fixture
def planner(client):
    user = operator(first_name="Amina", last_name="Otieno", email="amina@example.com")
    client.force_login(user)
    return user


# --- Profiles and roles ------------------------------------------------------------------------


def test_new_users_get_a_profile_with_least_privilege():
    user = get_user_model().objects.create_user("new", password=PASSWORD)
    assert user.profile.role == Role.SCHOOL_OPERATOR
    assert access.accessible_school_codes(user) == set()
    admin = get_user_model().objects.create_superuser("root", "root@example.com", PASSWORD)
    assert access.role_of(admin) == Role.SYSTEM_ADMIN
    assert access.accessible_school_codes(admin) is None


def test_profile_page_shows_real_account_details(client, planner):
    html = client.get(reverse("accounts:profile")).content.decode()
    assert "Amina Otieno" in html and "amina@example.com" in html
    assert "School Operator" in html and "Active" in html
    assert 'data-testid="profile-schools"' in html and "SCH001" in html


def test_profile_edit_persists_and_is_audited(client, planner):
    response = client.post(reverse("accounts:profile_edit"),
                           {"first_name": "Amina", "last_name": "Wanjiru", "phone": "+254 712 345 678"})
    assert response.status_code == 302
    planner.refresh_from_db()
    assert planner.last_name == "Wanjiru" and planner.profile.phone == "+254 712 345 678"
    log = AuditLog.objects.get(action=AuditLog.Action.PROFILE_UPDATED)
    assert set(log.details["fields"]) == {"last_name", "phone"}


def test_profile_edit_rejects_invalid_phone_and_blank_name(client, planner):
    response = client.post(reverse("accounts:profile_edit"), {"first_name": "", "last_name": "X", "phone": "call me"})
    assert response.status_code == 200
    planner.refresh_from_db()
    assert planner.first_name == "Amina" and planner.profile.phone == ""
    assert "Enter a valid phone number" in response.content.decode()


def test_profile_form_cannot_change_role_email_or_schools(client, planner):
    client.post(reverse("accounts:profile_edit"), {
        "first_name": "Amina", "last_name": "Otieno", "phone": "", "role": Role.SYSTEM_ADMIN,
        "email": "evil@example.com", "is_superuser": "on", "schools": ["999"],
    })
    planner.refresh_from_db()
    assert access.role_of(planner) == Role.SCHOOL_OPERATOR and not planner.is_superuser
    assert planner.email == "amina@example.com"
    assert SchoolAssignment.objects.filter(user=planner).count() == 3


def test_avatar_upload_validation_and_privacy(client, planner, media_root):
    fake = SimpleUploadedFile("photo.png", b"not an image at all", content_type="image/png")
    response = client.post(reverse("accounts:profile_edit"), {"first_name": "A", "last_name": "B", "avatar": fake})
    assert "not a valid PNG" in response.content.decode()
    assert not UserProfile.objects.get(user=planner).avatar

    exe = SimpleUploadedFile("photo.exe", PNG)
    assert "Unsupported file type" in client.post(
        reverse("accounts:profile_edit"), {"first_name": "A", "last_name": "B", "avatar": exe}).content.decode()

    client.post(reverse("accounts:profile_edit"),
                {"first_name": "A", "last_name": "B", "avatar": SimpleUploadedFile("me.png", PNG)})
    profile = UserProfile.objects.get(user=planner)
    assert profile.avatar.name.startswith(f"avatars/user_{planner.pk}_")
    own = client.get(reverse("accounts:avatar", args=[planner.pk]))
    assert own.status_code == 200 and own["Content-Type"] == "image/png"

    client.force_login(operator("someone_else"))
    assert client.get(reverse("accounts:avatar", args=[planner.pk])).status_code == 403

    client.force_login(planner)
    client.post(reverse("accounts:profile_edit"), {"first_name": "A", "last_name": "B", "remove_avatar": "on"})
    assert not UserProfile.objects.get(user=planner).avatar


def test_avatar_too_large_is_rejected(client, planner, media_root, settings):
    settings.MAX_AVATAR_SIZE_BYTES = 32
    big = SimpleUploadedFile("me.png", PNG)
    response = client.post(reverse("accounts:profile_edit"), {"first_name": "A", "last_name": "B", "avatar": big})
    assert "larger than" in response.content.decode()


# --- Appearance ---------------------------------------------------------------------------------


def test_appearance_preferences_persist_and_reach_the_page(client, planner):
    response = client.post(reverse("accounts:settings"), {
        "section": "appearance", "theme": "dark", "compact_tables": "on", "reduce_motion": "on", "landing_page": "forecast",
    })
    assert response.status_code == 302
    profile = UserProfile.objects.get(user=planner)
    assert (profile.theme, profile.compact_tables, profile.reduce_motion, profile.landing_page) == ("dark", True, True, "forecast")
    html = client.get(reverse("notifications:list")).content.decode()
    assert 'data-theme-pref="dark"' in html
    assert "bt-compact-tables" in html and "bt-reduce-motion" in html


def test_theme_init_prefers_saved_account_theme(client, planner):
    html = client.get(reverse("accounts:profile")).content.decode()
    head = html.split("</head>")[0]
    assert head.index('getAttribute("data-theme-pref")') < head.index("bootstrap.min.css")


def test_theme_toggle_endpoint_saves_and_validates(client, planner):
    response = client.post(reverse("accounts:theme"), {"theme": "light"})
    assert response.status_code == 200 and json.loads(response.content)["theme"] == "light"
    assert UserProfile.objects.get(user=planner).theme == "light"
    assert client.post(reverse("accounts:theme"), {"theme": "neon"}).status_code == 400
    assert client.get(reverse("accounts:theme")).status_code == 405


def test_login_redirects_to_preferred_landing_page(client, db):
    user = operator("lander")
    UserProfile.objects.filter(user=user).update(landing_page="forecast")
    response = client.post(reverse("accounts:login"), {"username": "lander", "password": PASSWORD})
    assert response.url == reverse("core:forecast")


def test_login_next_parameter_still_wins_over_landing_page(client, db):
    user = operator("lander")
    UserProfile.objects.filter(user=user).update(landing_page="forecast")
    response = client.post(reverse("accounts:login") + "?next=/calendar/",
                           {"username": "lander", "password": PASSWORD, "next": "/calendar/"})
    assert response.url == "/calendar/"


# --- Notifications preferences and security ----------------------------------------------------


def test_notification_preferences_persist(client, planner):
    client.post(reverse("accounts:settings"), {"section": "notifications", "notify_forecast": "on"})
    profile = UserProfile.objects.get(user=planner)
    assert profile.notify_forecast and not profile.notify_calendar and not profile.notify_announcements


def test_password_change_requires_current_password(client, planner):
    response = client.post(reverse("accounts:settings"), {
        "section": "password", "old_password": "wrong", "new_password1": "N3w-sufficiently-long!", "new_password2": "N3w-sufficiently-long!",
    })
    assert response.status_code == 200
    planner.refresh_from_db()
    assert planner.check_password(PASSWORD)


def test_password_change_applies_validators(client, planner):
    client.post(reverse("accounts:settings"), {
        "section": "password", "old_password": PASSWORD, "new_password1": "12345678", "new_password2": "12345678",
    })
    planner.refresh_from_db()
    assert planner.check_password(PASSWORD)


def test_password_change_succeeds_keeps_session_and_is_audited(client, planner):
    new = "N3w-sufficiently-long!"
    response = client.post(reverse("accounts:settings"), {
        "section": "password", "old_password": PASSWORD, "new_password1": new, "new_password2": new,
    })
    assert response.status_code == 302
    planner.refresh_from_db()
    assert planner.check_password(new)
    assert client.get(reverse("accounts:profile")).status_code == 200  # still signed in
    log = AuditLog.objects.get(action=AuditLog.Action.PASSWORD_CHANGED)
    assert new not in json.dumps(log.details)
    assert Notification.objects.filter(recipient=planner, type=Notification.Type.SECURITY).exists()


def test_settings_about_uses_saved_metadata(client, planner, stub_artifacts):
    html = client.get(reverse("accounts:settings") + "?tab=about").content.decode()
    assert "Linear Regression" in html  # stub metadata's selected model
    assert "1.0.0" in html


def test_unknown_settings_section_is_rejected(client, planner):
    assert client.post(reverse("accounts:settings"), {"section": "role"}).status_code == 400


# --- School scope on existing pages --------------------------------------------------------------


def test_operator_sees_only_assigned_schools(client, db, stub_artifacts):
    client.force_login(operator("one_school", codes=("SCH002",)))
    html = client.get(reverse("core:forecast")).content.decode()
    assert 'value="SCH002"' in html and 'value="SCH001"' not in html
    response = client.get(reverse("core:history"), {"school": "SCH001"})
    assert "Select a valid choice" in response.content.decode()
    response = client.post(reverse("core:forecast"), {"school": "SCH001", "horizon": 1})
    assert response.status_code == 200 and not ForecastRun.objects.exists()


def test_operator_without_schools_gets_a_clear_message(client, db, stub_artifacts):
    client.force_login(operator("nobody", codes=()))
    for page in ("core:dashboard", "core:history", "core:forecast", "core:donors"):
        response = client.get(reverse(page))
        assert response.status_code == 200
        assert "No schools are assigned to your account" in response.content.decode()


def _run(code: str) -> ForecastRun:
    return ForecastRun.objects.create(
        school_code=code, school_name=code, horizon_days=1, origin_date=date(2024, 1, 1),
        first_forecast_date=date(2024, 1, 2), last_forecast_date=date(2024, 1, 2), model_label="LR",
        results_label="SYNTHETIC DEVELOPMENT RESULTS", day1_predicted_meals=1, day1_recommended_meals=1,
        day1_risk_level="Low", total_predicted_meals=1, total_recommended_meals=1, result_summary={"days": []},
    )


def test_forecast_runs_of_other_schools_are_hidden(client, db, stub_artifacts):
    client.force_login(operator("one_school", codes=("SCH002",)))
    mine, theirs = _run("SCH002"), _run("SCH001")
    assert client.get(reverse("core:forecast_run", args=[theirs.pk])).status_code == 404
    assert client.get(reverse("core:forecast_run", args=[mine.pk])).status_code == 200
    html = client.get(reverse("core:forecast_history")).content.decode()
    assert reverse("core:forecast_run", args=[mine.pk]) in html
    assert reverse("core:forecast_run", args=[theirs.pk]) not in html
