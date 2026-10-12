"""Keep tests on local fixtures: never read the real Google Sheet configured in .env."""

import pytest

from config import settings


@pytest.fixture(autouse=True)
def _no_google_sheet(monkeypatch):
    monkeypatch.setattr(settings, "USE_GOOGLE_SHEET", False)
    monkeypatch.setattr(settings, "GOOGLE_SHEET_URL", "")
