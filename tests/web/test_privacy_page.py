"""The public privacy notice (/privacy/).

- It is not served (404), and not linked, until every owner-confirmed and server-verified fact
  is filled in, so it can never be published with gaps or placeholder text.
- Once complete it is public (no sign-in), names the project and owner correctly, describes
  Food4Education only as the source of observed records, and distinguishes simulated records.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date

import pytest
from django.urls import reverse

from tests.support.web_fixtures import operator
from web.core import privacy

pytestmark = pytest.mark.django_db

COMPLETE = privacy.PrivacyFacts(
    contact_email="privacy-test@example.org",
    retention=("Test retention paragraph.",),
    deletion=("Test deletion paragraph.",),
    effective_date=date(2026, 11, 10),
    aws_region="Test Region (xx-test-1)",
    scheduled_backups="Test scheduled-backup sentence.",
    backup_offsite_copy=True,
    backups_kept=14,
    server_log_retention="14 days",
)


@pytest.fixture
def published(monkeypatch):
    monkeypatch.setattr(privacy, "FACTS", COMPLETE)
    return COMPLETE


def test_not_served_or_linked_until_every_fact_is_confirmed(client):
    assert privacy.missing()  # the repository default: decisions and server checks still pending
    assert client.get(reverse("core:privacy")).status_code == 404
    assert 'data-testid="privacy-link"' not in client.get(reverse("accounts:login")).content.decode()
    client.force_login(operator("reader"))
    assert client.get(reverse("core:privacy")).status_code == 404


def test_one_missing_fact_still_blocks_publication(client, monkeypatch):
    monkeypatch.setattr(privacy, "FACTS", replace(COMPLETE, contact_email=None))
    assert privacy.missing() == ["contact_email"]
    assert client.get(reverse("core:privacy")).status_code == 404


def test_published_notice_is_public_and_accurate(client, published):
    response = client.get(reverse("core:privacy"))
    assert response.status_code == 200  # anonymous: no redirect to the login page
    html = response.content.decode()
    assert ("BiteTracker is a final-year university project developed by Sharon Kanyi for the BSc in "
            "Informatics and Computer Science at Strathmore University.") in html
    assert "Food4Education does not own or operate BiteTracker." in html
    assert "Simulated records" in html and "not Food4Education results" in html
    assert 'href="mailto:privacy-test@example.org"' in html
    for fact in ("Test Region (xx-test-1)", "Test retention paragraph.", "Test deletion paragraph.",
                 "Test scheduled-backup sentence.", "14 most recent backups", "kept for 14 days",
                 "copied, encrypted, to Amazon S3", "10 November 2026"):
        assert fact in html
    assert "<code>openid</code>, <code>email</code> and <code>profile</code>" in html
    text = re.sub(r"<[^>]+>", " ", html)
    for placeholder in ("None", "TODO", "TBD", "[", "<your", "example.com"):
        assert placeholder not in text


def test_login_page_links_to_published_notice(client, published):
    assert f'href="{reverse("core:privacy")}"' in client.get(reverse("accounts:login")).content.decode()


def test_optional_backup_details_render_only_as_confirmed(client, monkeypatch):
    monkeypatch.setattr(privacy, "FACTS", replace(COMPLETE, scheduled_backups="", backup_offsite_copy=False))
    html = client.get(reverse("core:privacy")).content.decode()
    assert "Test scheduled-backup sentence." not in html and "Amazon S3" not in html
    assert "kept on the same server" in html


def test_owner_text_is_escaped(client, monkeypatch):
    monkeypatch.setattr(privacy, "FACTS", replace(COMPLETE, retention=("<script>x</script>",)))
    html = client.get(reverse("core:privacy")).content.decode()
    assert "<script>x</script>" not in html and "&lt;script&gt;" in html
