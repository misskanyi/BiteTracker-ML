"""The public privacy notice (/privacy/).

- It is not served (404), and not linked, until every server-verified fact and the publication
  date are filled in, so it can never be published with gaps or guessed values.
- Once complete it is public (no sign-in), names the project and owner correctly, mentions
  Food4Education only as the source of observed records, distinguishes simulated records, and
  states retention, backup expiry and request handling from the configured facts.
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
    effective_date=date(2026, 11, 20),
    aws_region="Test Region (xx-test-1)",  # completes "...backups in ___."
    backup_interval_days=1,
    backups_kept=14,
    backup_offsite_copy=True,
    offsite_backup_retention="deleted after 30 days",
    web_log_retention="kept for 14 days",
    web_log_has_visitor_ip=False,
    app_log_retention="for up to 30 days",
    production_model_uses_simulated_records=True,
    aws_snapshot_copies="",
)


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


@pytest.fixture
def published(monkeypatch):
    monkeypatch.setattr(privacy, "FACTS", COMPLETE)
    return COMPLETE


def test_not_served_or_linked_until_every_fact_is_confirmed(client, monkeypatch):
    monkeypatch.setattr(privacy, "FACTS", replace(privacy.FACTS, effective_date=None))
    assert privacy.missing() == ["effective_date"]
    assert client.get(reverse("core:privacy")).status_code == 404
    assert 'data-testid="privacy-link"' not in client.get(reverse("accounts:login")).content.decode()
    client.force_login(operator("reader"))
    assert client.get(reverse("core:privacy")).status_code == 404


def test_one_missing_fact_still_blocks_publication(client, monkeypatch):
    monkeypatch.setattr(privacy, "FACTS", replace(COMPLETE, effective_date=None))
    assert privacy.missing() == ["effective_date"]
    assert client.get(reverse("core:privacy")).status_code == 404


def test_published_notice_is_public_and_states_the_owner_policy(client, published):
    response = client.get(reverse("core:privacy"))
    assert response.status_code == 200  # anonymous: no redirect to the login page
    text = _text(response.content.decode())
    assert ("BiteTracker is a final-year university project developed by Sharon Kanyi for the BSc in "
            "Informatics and Computer Science at Strathmore University.") in text
    assert "come from Food4Education" in text and "operate" not in text.split("Food4Education")[1][:80]
    assert "simulated records" in text and "are not Food4Education data" in text
    assert "Registering with Google does not by itself give access" in text
    assert 'href="mailto:sharon.kanyi@strathmore.edu"' in response.content.decode()
    assert "within 90 days after the project closes" in text
    assert ("If a specific university or legal requirement requires longer retention, we will explain the reason "
            "and applicable retention period to the requester.") in text
    assert ("Records and files that remain after account deletion will be reviewed when handling a deletion request. "
            "Personal information will be deleted or anonymised where appropriate, with any necessary exceptions "
            "explained.") in text
    assert "Backups are used for recovery, restoration testing and maintaining service continuity." in text
    assert "Your identity is verified before any information is shared or changed" in text
    assert "no automatic deletion feature" in text and "doesn't use your information for advertising" in text
    # Mixed research data is not passed off as observed data, and the synthetic dataset is separate.
    assert "simulated records for 2023 and 2024" in text and "not based on observed data alone" in text
    assert "separate synthetic development dataset" in text
    assert "model currently in use was trained on this combined dataset" in text
    # No guaranteed backup lifetime, and nothing described as anonymous just because it was unlinked.
    assert "no fixed date by which deleted information leaves the backups" in text
    assert "at most" not in text and "unlinked" not in text
    for fact in ("Test Region (xx-test-1)", "20 November 2026", "scheduled every night", "14 most recent",
                 "copies are deleted after 30 days", "kept for 14 days", "requests and errors for up to 30 days",
                 "normally recorded with Cloudflare's IP address", "your own IP address can still appear"):
        assert fact in text, fact
    assert "openid , email and profile" in text
    for placeholder in ("None", "TODO", "TBD", "<your", "example.com", "[pending"):
        assert placeholder not in text


def test_no_scheduled_backups_means_no_fixed_backup_expiry(client, monkeypatch):
    monkeypatch.setattr(privacy, "FACTS", replace(COMPLETE, backup_interval_days=0, backup_offsite_copy=False,
                                                  offsite_backup_retention="", web_log_has_visitor_ip=True))
    text = _text(client.get(reverse("core:privacy")).content.decode())
    assert "taken before each software update" in text and "every night" not in text
    assert "no fixed date" in text and "Amazon S3" not in text
    assert "stored on the same server; this backup configuration does not copy them elsewhere" in text
    assert "and your IP address" in text and "normally recorded with Cloudflare" not in text


def test_repository_notice_is_complete_and_public(client):
    assert privacy.missing() == []
    response = client.get(reverse("core:privacy"))
    assert response.status_code == 200 and "Effective 10 November 2026" in _text(response.content.decode())
    assert 'data-testid="privacy-link"' in client.get(reverse("accounts:login")).content.decode()


def test_aws_snapshot_sentence_and_model_note_follow_the_facts(client, monkeypatch):
    monkeypatch.setattr(privacy, "FACTS", replace(COMPLETE, aws_snapshot_copies="Test snapshot sentence.",
                                                  production_model_uses_simulated_records=False))
    text = _text(client.get(reverse("core:privacy")).content.decode())
    assert "Test snapshot sentence." in text and "model currently in use" not in text


def test_unsupported_log_retention_is_left_out(client, monkeypatch):
    monkeypatch.setattr(privacy, "FACTS", replace(COMPLETE, web_log_retention="", app_log_retention=""))
    text = _text(client.get(reverse("core:privacy")).content.decode())
    assert "These logs are kept for" not in text
    assert "The application also keeps its own log of requests and errors." in text


def test_login_page_links_to_published_notice(client, published):
    assert f'href="{reverse("core:privacy")}"' in client.get(reverse("accounts:login")).content.decode()
