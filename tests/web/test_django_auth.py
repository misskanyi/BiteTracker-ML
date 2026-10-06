"""Stage 1 Django tests: project configuration, login/logout and page protection.

Run with the rest of the suite via `pytest` (pytest-django reads
DJANGO_SETTINGS_MODULE from pyproject.toml).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSWORD = "Str0ng-test-passw0rd!"


@pytest.fixture
def user(db):
    return get_user_model().objects.create_user(username="planner", password=PASSWORD)


# --- Page protection ---------------------------------------------------------


def test_dashboard_redirects_anonymous_user_to_login(client):
    response = client.get(reverse("core:dashboard"))
    assert response.status_code == 302
    assert response.url == f"{reverse('accounts:login')}?next=/"


def test_login_page_is_publicly_accessible(client, db):
    response = client.get(reverse("accounts:login"))
    assert response.status_code == 200
    assert b'name="username"' in response.content
    assert b'name="password"' in response.content


def test_admin_requires_staff_login(client, user):
    client.force_login(user)  # authenticated but not staff
    response = client.get(reverse("admin:index"))
    assert response.status_code == 302
    assert reverse("admin:login") in response.url


# --- Login -------------------------------------------------------------------


def test_valid_login_redirects_to_dashboard(client, user):
    response = client.post(reverse("accounts:login"), {"username": "planner", "password": PASSWORD})
    assert response.status_code == 302
    assert response.url == reverse("core:dashboard")
    assert "_auth_user_id" in client.session


def test_login_honours_next_parameter(client, user):
    response = client.post(
        f"{reverse('accounts:login')}?next=/", {"username": "planner", "password": PASSWORD, "next": "/"}
    )
    assert response.status_code == 302
    assert response.url == "/"


def test_invalid_login_shows_error_and_stays_anonymous(client, user):
    response = client.post(reverse("accounts:login"), {"username": "planner", "password": "wrong-password"})
    assert response.status_code == 200
    assert b'data-testid="login-error"' in response.content
    assert "_auth_user_id" not in client.session


def test_login_with_email_is_case_insensitive(client, db):
    get_user_model().objects.create_user(username="emailer", email="Planner@Example.com", password=PASSWORD)
    response = client.post(reverse("accounts:login"), {"username": "planner@example.COM", "password": PASSWORD})
    assert response.status_code == 302
    assert response.url == reverse("core:dashboard")
    assert "_auth_user_id" in client.session


def test_login_with_email_rejects_wrong_password(client, db):
    get_user_model().objects.create_user(username="emailer", email="planner@example.com", password=PASSWORD)
    response = client.post(reverse("accounts:login"), {"username": "planner@example.com", "password": "nope"})
    assert response.status_code == 200
    assert "_auth_user_id" not in client.session


def test_login_with_shared_email_is_rejected(client, db):
    User = get_user_model()
    User.objects.create_user(username="one", email="shared@example.com", password=PASSWORD)
    User.objects.create_user(username="two", email="shared@example.com", password=PASSWORD)
    response = client.post(reverse("accounts:login"), {"username": "shared@example.com", "password": PASSWORD})
    assert response.status_code == 200
    assert "_auth_user_id" not in client.session


def test_inactive_user_cannot_login_with_email(client, db):
    get_user_model().objects.create_user(
        username="gone", email="gone@example.com", password=PASSWORD, is_active=False
    )
    response = client.post(reverse("accounts:login"), {"username": "gone@example.com", "password": PASSWORD})
    assert "_auth_user_id" not in client.session


def test_authenticated_user_sees_dashboard(client, user):
    client.force_login(user)
    response = client.get(reverse("core:dashboard"))
    assert response.status_code == 200
    assert b"Dashboard" in response.content
    assert b"Signed in as planner" in response.content


# --- Logout ------------------------------------------------------------------


def test_logout_logs_user_out_and_reprotects_pages(client, user):
    client.force_login(user)
    response = client.post(reverse("accounts:logout"))
    assert response.status_code == 302
    assert response.url == reverse("accounts:login")
    assert "_auth_user_id" not in client.session
    assert client.get(reverse("core:dashboard")).status_code == 302


def test_logout_rejects_get_requests(client, user):
    """Logout must be a CSRF-protected POST, not a link anyone can trigger."""
    client.force_login(user)
    response = client.get(reverse("accounts:logout"))
    assert response.status_code == 405
    assert "_auth_user_id" in client.session


# --- Synthetic-data labelling -------------------------------------------------


@pytest.mark.parametrize("authenticated", [False, True])
def test_synthetic_banner_shown_on_every_page(client, user, authenticated):
    if authenticated:
        client.force_login(user)
        response = client.get(reverse("core:dashboard"))
    else:
        response = client.get(reverse("accounts:login"))
    assert response.status_code == 200
    assert b'data-testid="synthetic-banner"' in response.content
    assert b"SYNTHETIC DEVELOPMENT DATA" in response.content


# --- Configuration -------------------------------------------------------------


def test_secret_key_required_when_debug_is_off():
    """With DEBUG off, settings must refuse to load without DJANGO_SECRET_KEY
    rather than fall back to the development key."""
    env = {k: v for k, v in os.environ.items() if k != "DJANGO_SECRET_KEY"}
    env.update({"DJANGO_DEBUG": "False", "DJANGO_SECRET_KEY": "", "DJANGO_SETTINGS_MODULE": "bitetracker_web.settings"})
    result = subprocess.run(
        [sys.executable, "-c", "import django; django.setup()"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "DJANGO_SECRET_KEY must be set" in result.stderr
