"""seed_demo_accounts must never create accounts without an explicitly supplied password."""

import pytest
from django.contrib.auth import get_user_model
from django.core.management import CommandError, call_command

from web.accounts.management.commands import seed_demo_accounts as cmd

pytestmark = pytest.mark.django_db
USERNAMES = [a["username"] for a in cmd.DEMO_ACCOUNTS]


def test_no_builtin_default_password():
    assert not hasattr(cmd, "DEFAULT_PASSWORD")


def test_refuses_without_password(monkeypatch):
    monkeypatch.delenv("DEMO_ACCOUNT_PASSWORD", raising=False)
    with pytest.raises(CommandError, match="No password supplied"):
        call_command("seed_demo_accounts")
    assert not get_user_model().objects.filter(username__in=USERNAMES).exists()


def test_refuses_empty_env_password(monkeypatch):
    monkeypatch.setenv("DEMO_ACCOUNT_PASSWORD", "")
    with pytest.raises(CommandError):
        call_command("seed_demo_accounts")
    assert not get_user_model().objects.filter(username__in=USERNAMES).exists()


def test_explicit_password_creates_accounts(monkeypatch):
    monkeypatch.delenv("DEMO_ACCOUNT_PASSWORD", raising=False)
    secret = "Unit-test-only-Xq7#mPz2"
    call_command("seed_demo_accounts", password=secret)
    users = get_user_model().objects.filter(username__in=USERNAMES)
    assert users.count() == len(USERNAMES)
    assert all(u.check_password(secret) for u in users)


def test_env_password_used(monkeypatch):
    secret = "Unit-test-env-Rk4!vLw9"
    monkeypatch.setenv("DEMO_ACCOUNT_PASSWORD", secret)
    call_command("seed_demo_accounts")
    assert get_user_model().objects.get(username="demo_manager").check_password(secret)
