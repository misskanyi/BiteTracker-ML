"""Keep tests on local fixtures: never read the real Google Sheet configured in .env."""

import os

import pytest

# Child processes started by tests (settings/deployment checks) must not read the developer's
# .env either: it may hold SMTP or OAuth credentials that change how settings load.
os.environ["BITETRACKER_SKIP_DOTENV"] = "1"

from config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def _no_google_sheet(monkeypatch):
    monkeypatch.setattr(settings, "USE_GOOGLE_SHEET", False)
    monkeypatch.setattr(settings, "GOOGLE_SHEET_URL", "")
