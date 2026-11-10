"""Account emails: who gets them, when, exactly once, and never at the expense of the account action.

What this file checks (all with Django's in-memory test backend; nothing is sent):
- Registration received -> the applicant; new request -> each active System Administrator,
  one message each with a single recipient (no shared To/Cc/Bcc).
- Approval, rejection, suspension, reactivation and role/school changes email the affected
  person only, with status, role and schools, and never internal notes or secrets.
- Repeat Google sign-ins, repeated approvals, resubmitted suspend forms and edits that change
  nothing send no further email.
- Emails go out only after the transaction commits: a rollback leaves no email and no outbox row.
- A failing mail server never undoes the approval; the failure is recorded and logged without
  the address or contents, and `send_pending_emails` retries it exactly once successfully.
- Forecast-ready and other routine notifications stay in-app only.
- Delivery switched off records but never sends, and enabling it later sends nothing old.
- The optional administrator email respects its preference; essential notices ignore preferences.
- User-provided names are escaped in the HTML part.
"""

from __future__ import annotations

import io
import logging
import os
import socket
import subprocess
import sys
from unittest import mock

import pytest
from django.contrib.auth import get_user_model
from django.core import mail
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend
from django.core.management import call_command
from django.db import transaction
from django.urls import reverse

from tests.support.web_fixtures import PASSWORD, operator
from tests.web.test_rbac_google_notifications import approve, google, google_sign_in, schools  # noqa: F401
from web.accounts import access
from web.accounts.adapters import _announce_access_request
from web.accounts.models import AccessRequest, Role
from web.notifications import emails
from web.notifications.models import EmailDelivery, Notification
from web.notifications.services import notify

pytestmark = pytest.mark.django_db
User = get_user_model()
Kind, Status = EmailDelivery.Kind, EmailDelivery.Status


class FailingSendBackend(LocmemBackend):
    """Connects, then the server refuses every message."""

    def send_messages(self, messages):
        raise ConnectionRefusedError("550 rejected for amina@gmail.com")


class UnreachableBackend(LocmemBackend):
    """The mail server can't be reached (connection timeout)."""

    opens = 0

    def open(self):
        UnreachableBackend.opens += 1
        raise socket.timeout("timed out connecting to smtp.example.org")


@pytest.fixture(autouse=True)
def email_on(settings):
    settings.EMAIL_DELIVERY = "smtp"  # the mode under test; the backend itself is Django's in-memory one
    settings.EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
    settings.EMAIL_ALLOWED_RECIPIENTS = []
    settings.EMAIL_REQUEST_BUDGET_SECONDS = 10
    settings.SITE_URL = "https://bitetracker.example.org"
    settings.DEFAULT_FROM_EMAIL = "BiteTracker <no-reply@bitetracker.example.org>"


@pytest.fixture
def commit(django_capture_on_commit_callbacks):
    """Run on-commit callbacks as a real commit would (tests otherwise never commit)."""
    return lambda: django_capture_on_commit_callbacks(execute=True)


@pytest.fixture
def admins(db):
    a = User.objects.create_superuser("admin_a", "admin.a@example.com", PASSWORD)
    b = User.objects.create_superuser("admin_b", "admin.b@example.org", PASSWORD)
    retired = User.objects.create_superuser("admin_old", "old@example.com", PASSWORD)
    retired.is_active = False
    retired.save()
    return a, b


def _applicant(email="ada@example.com", full_name="Ada Lovelace") -> AccessRequest:
    user = User.objects.create_user("applicant", email=email, password=PASSWORD, is_active=False,
                                    first_name=full_name.split()[0], last_name=full_name.split()[-1])
    return AccessRequest.objects.create(user=user, email=email, full_name=full_name)


def _to(message) -> list[str]:
    return message.to + message.cc + message.bcc


def _sent_to(address) -> list:
    return [m for m in mail.outbox if address in _to(m)]


# --- Registration ---------------------------------------------------------------------------------------


def test_first_google_sign_in_emails_applicant_and_each_active_admin_individually(client, google, admins, commit):
    bystander = operator("bystander", email="bystander@example.com")
    with commit():
        google_sign_in(client)

    assert len(mail.outbox) == 3
    assert all(len(_to(m)) == 1 for m in mail.outbox)  # nobody sees anyone else's address
    applicant_mail = _sent_to("amina@gmail.com")
    assert len(applicant_mail) == 1
    assert applicant_mail[0].subject == "We received your BiteTracker registration request"
    assert "Pending approval" in applicant_mail[0].body
    assert "https://bitetracker.example.org/accounts/login/" in applicant_mail[0].body
    for admin in admins:
        (message,) = _sent_to(admin.email)
        assert message.subject == "New BiteTracker registration request"
        assert "Amina Otieno" in message.body and "amina@gmail.com" in message.body
        assert "/administration/" in message.body
        assert message.alternatives and message.alternatives[0][1] == "text/html"
    assert not _sent_to("old@example.com") and not _sent_to(bystander.email)
    # In-app notifications are unchanged.
    assert Notification.objects.filter(type=Notification.Type.ACCESS_REQUEST).count() == 2


def test_repeat_google_sign_in_sends_no_more_email(client, google, admins, commit):
    with commit():
        google_sign_in(client)
    with commit():
        google_sign_in(client)
        google_sign_in(client)
    assert len(mail.outbox) == 3
    assert EmailDelivery.objects.count() == 3


def test_admin_can_switch_off_request_emails_but_keeps_in_app_notice(admins, commit):
    profile = access.get_profile(admins[0])
    profile.email_access_requests = False
    profile.save()
    req = _applicant()
    with commit():
        _announce_access_request(req)
    assert not _sent_to(admins[0].email) and len(_sent_to(admins[1].email)) == 1
    assert Notification.objects.filter(recipient=admins[0], type=Notification.Type.ACCESS_REQUEST).exists()
    assert len(_sent_to("ada@example.com")) == 1  # essential notice to the applicant is unaffected


def test_request_email_preference_shown_to_admins_only(client, admins):
    client.force_login(admins[0])
    assert "email_access_requests" in client.get(reverse("accounts:settings") + "?tab=notifications").content.decode()
    client.force_login(operator("plain"))
    assert "email_access_requests" not in client.get(reverse("accounts:settings") + "?tab=notifications").content.decode()


def test_user_supplied_name_is_escaped_in_html(admins, commit):
    req = _applicant(full_name='<script>alert("x")</script> Eve')
    with commit():
        _announce_access_request(req)
    (message,) = _sent_to(admins[0].email)
    html = message.alternatives[0][0]
    assert "<script>" not in html and "&lt;script&gt;" in html


# --- Decisions --------------------------------------------------------------------------------------------


def test_approval_emails_applicant_role_and_schools_once(client, admins, schools, commit):
    req = _applicant()
    with commit():
        client.force_login(admins[0])
        client.post(reverse("administration:access_request_detail", args=[req.pk]),
                    {"action": "approve", "role": Role.SCHOOL_MANAGER, "schools": [schools["SCH001"].pk],
                     "note": "internal: vouched for by HQ"})
    (message,) = mail.outbox
    assert message.to == ["ada@example.com"]
    assert message.subject == "Your BiteTracker registration was approved"
    assert "School Manager" in message.body and "School 1 (SCH001)" in message.body
    assert "https://bitetracker.example.org/accounts/login/" in message.body
    assert "vouched" not in message.body and "vouched" not in message.alternatives[0][0]

    # Repeating the approval (refresh / double submit / second admin) changes nothing and sends nothing.
    with commit():
        for admin in admins:
            client.force_login(admin)
            client.post(reverse("administration:access_request_detail", args=[req.pk]),
                        {"action": "approve", "role": Role.SYSTEM_ADMIN, "schools": [schools["SCH002"].pk], "note": ""})
    assert len(mail.outbox) == 1


def test_rejection_emails_applicant_without_the_internal_reason(client, admins, commit):
    req = _applicant()
    with commit():
        client.force_login(admins[1])
        client.post(reverse("administration:access_request_detail", args=[req.pk]),
                    {"action": "reject", "reason": "Not on staff list XYZ-42"})
        client.post(reverse("administration:access_request_detail", args=[req.pk]),
                    {"action": "reject", "reason": "again"})
    (message,) = mail.outbox
    assert message.to == ["ada@example.com"] and "not approved" in message.subject
    assert "XYZ-42" not in message.body and "XYZ-42" not in message.alternatives[0][0]


# --- Suspension, reactivation, role and schools ---------------------------------------------------------------


def test_suspend_and_reactivate_email_the_user_once_each(client, admins, commit):
    user = operator("worker", email="worker@example.com")
    client.force_login(admins[0])
    url = reverse("administration:user_toggle_active", args=[user.pk])
    with commit():
        client.post(url, {"action": "suspend"})
        client.post(url, {"action": "suspend"})  # resubmitted form: already suspended, no change
    user.refresh_from_db()
    assert not user.is_active
    (message,) = mail.outbox
    assert message.to == ["worker@example.com"] and "suspended" in message.subject

    with commit():
        client.post(url, {"action": "reactivate"})
        client.post(url, {"action": "reactivate"})
    user.refresh_from_db()
    assert user.is_active
    assert [m.subject for m in mail.outbox] == ["Your BiteTracker account has been suspended",
                                                "Your BiteTracker account has been reactivated"]
    assert all(m.to == ["worker@example.com"] for m in mail.outbox)


def test_role_and_school_change_sends_one_email_and_no_change_sends_none(client, admins, schools, commit):
    user = operator("worker", codes=("SCH001",), email="worker@example.com", first_name="W", last_name="K")
    client.force_login(admins[0])
    url = reverse("administration:user_edit", args=[user.pk])
    data = {"first_name": "W", "last_name": "K", "email": "worker@example.com", "phone": "",
            "role": Role.PLANNING_STAKEHOLDER, "schools": [schools["SCH001"].pk, schools["SCH002"].pk]}
    with commit():
        assert client.post(url, data).status_code == 302
    (message,) = mail.outbox
    assert message.to == ["worker@example.com"] and message.subject == "Your BiteTracker access has changed"
    assert "Resource Planning Stakeholder" in message.body and "School Manager" in message.body
    assert "School 2 (SCH002)" in message.body

    with commit():
        client.post(url, data)  # same values again: nothing changes
        client.post(url, {**data, "phone": "0700000000"})  # contact detail only: not an access change
    assert len(mail.outbox) == 1


def test_school_page_assign_and_unassign_email_the_user(client, admins, schools, commit):
    user = operator("worker", codes=("SCH001",), email="worker@example.com")
    client.force_login(admins[0])
    with commit():
        client.post(reverse("administration:school_detail", args=[schools["SCH002"].pk]),
                    {"action": "assign", "user": user.pk})
        client.post(reverse("administration:school_detail", args=[schools["SCH002"].pk]),
                    {"action": "unassign", "user": user.pk})
        client.post(reverse("administration:school_detail", args=[schools["SCH002"].pk]),
                    {"action": "unassign", "user": user.pk})  # already removed: nothing to report
    assert len(mail.outbox) == 2 and all(m.to == ["worker@example.com"] for m in mail.outbox)
    assert "Schools added" in mail.outbox[0].body and "Schools removed" in mail.outbox[1].body


# --- Transactions and failures ------------------------------------------------------------------------------


def test_rolled_back_transaction_sends_nothing(admins, commit):
    req = _applicant()
    with commit():
        with pytest.raises(RuntimeError):
            with transaction.atomic():
                emails.registration_received(req)
                raise RuntimeError("later step failed")
    assert mail.outbox == [] and not EmailDelivery.objects.exists()


def test_failed_approval_view_rolls_back_and_sends_nothing(client, admins, schools, commit):
    req = _applicant()
    real = emails.registration_approved

    def queue_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("boom after queueing")

    client.force_login(admins[0])
    with commit(), mock.patch.object(emails, "registration_approved", queue_then_fail):
        with pytest.raises(RuntimeError):
            client.post(reverse("administration:access_request_detail", args=[req.pk]),
                        {"action": "approve", "role": Role.SCHOOL_MANAGER, "schools": [schools["SCH001"].pk], "note": ""})
    req.refresh_from_db()
    assert req.status == AccessRequest.Status.PENDING and not req.user.is_active
    assert mail.outbox == [] and not EmailDelivery.objects.exists()


def test_mail_server_failure_keeps_approval_logs_safely_and_retry_sends_once(client, admins, schools, commit,
                                                                              settings, caplog):
    settings.EMAIL_BACKEND = f"{__name__}.FailingSendBackend"
    req = _applicant()
    client.force_login(admins[0])
    with commit(), caplog.at_level(logging.WARNING, logger="web.notifications.emails"):
        response = client.post(reverse("administration:access_request_detail", args=[req.pk]),
                               {"action": "approve", "role": Role.SCHOOL_MANAGER,
                                "schools": [schools["SCH001"].pk], "note": ""})
    assert response.status_code == 302
    req.refresh_from_db()
    assert req.status == AccessRequest.Status.APPROVED and req.user.is_active  # approval stands
    row = EmailDelivery.objects.get()
    assert (row.status, row.attempts, row.last_error) == (Status.FAILED, 1, "ConnectionRefusedError")
    assert "Account email failed" in caplog.text
    for secret in ("ada@example.com", "550", row.subject):
        assert secret not in caplog.text

    settings.EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
    call_command("send_pending_emails", stdout=(out := io.StringIO()))
    assert "1 sent" in out.getvalue() and "ada@" not in out.getvalue()
    call_command("send_pending_emails", stdout=io.StringIO())
    assert len(mail.outbox) == 1 and mail.outbox[0].to == ["ada@example.com"]
    assert EmailDelivery.objects.get().status == Status.SENT


def test_unreachable_server_marks_every_queued_email_failed(admins, commit, settings):
    settings.EMAIL_BACKEND = f"{__name__}.UnreachableBackend"
    UnreachableBackend.opens = 0
    with commit():
        _announce_access_request(_applicant())
    assert set(EmailDelivery.objects.values_list("status", "last_error")) == {(Status.FAILED, "TimeoutError")}
    assert EmailDelivery.objects.count() == 3
    assert UnreachableBackend.opens == 1  # one connection timeout per event, not one per recipient


def test_retries_stop_after_max_attempts_and_ignore_old_rows(admins, commit, settings):
    settings.EMAIL_BACKEND = f"{__name__}.UnreachableBackend"
    settings.EMAIL_MAX_ATTEMPTS = 2
    with commit():
        emails.registration_received(_applicant())
    call_command("send_pending_emails", stdout=io.StringIO())
    assert EmailDelivery.objects.get().attempts == 2
    settings.EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
    call_command("send_pending_emails", stdout=io.StringIO())
    assert mail.outbox == []


# --- What is NOT emailed -------------------------------------------------------------------------------------


def test_forecast_ready_and_routine_notifications_are_in_app_only(client, stub_artifacts, admins, commit):
    user = operator("planner", email="planner@example.com")
    client.force_login(user)
    with commit():
        client.post(reverse("core:forecast"), {"school": "SCH001", "horizon": 1})
        notify([user], Notification.Type.FORECAST_COMPLETED, "Forecast ready", url="/")
        notify(User.objects.all(), Notification.Type.ANNOUNCEMENT, "Maintenance tonight")
        notify([user], Notification.Type.SECURITY, "Password reset", url="/")
    assert Notification.objects.filter(recipient=user, type=Notification.Type.FORECAST_COMPLETED).exists()
    assert mail.outbox == [] and not EmailDelivery.objects.exists()


def test_existing_records_are_never_emailed_retrospectively(admins, commit):
    _applicant()  # an old request: no event happened while emails existed
    operator("old_user", email="old_user@example.com")
    call_command("send_pending_emails", stdout=io.StringIO())
    assert mail.outbox == [] and not EmailDelivery.objects.exists()


def test_disabled_delivery_records_but_never_sends_even_after_enabling(admins, commit, settings):
    settings.EMAIL_DELIVERY = "disabled"
    with commit():
        _announce_access_request(_applicant())
    assert set(EmailDelivery.objects.values_list("status", flat=True)) == {Status.DISABLED}
    settings.EMAIL_DELIVERY = "smtp"
    call_command("send_pending_emails", stdout=io.StringIO())
    assert mail.outbox == []


def test_user_without_email_address_is_skipped(admins, commit):
    user = operator("no_email")
    with commit():
        emails.account_suspended(user, event_id=1)
    assert mail.outbox == [] and not EmailDelivery.objects.exists()


# --- Configuration -------------------------------------------------------------------------------------------


def _load_settings(**env) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("EMAIL_", "BITETRACKER_EMAIL", "DEFAULT_FROM"))}
    return subprocess.run([sys.executable, "-c", "import bitetracker_web.settings as s; print(s.EMAIL_BACKEND)"],
                          env={**clean, "BITETRACKER_SKIP_DOTENV": "1", **env}, capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_settings_default_to_non_sending_backends():
    assert "console" in _load_settings(DJANGO_DEBUG="True").stdout
    prod = {"DJANGO_DEBUG": "False", "DJANGO_SECRET_KEY": "x" * 50, "DJANGO_ALLOWED_HOSTS": "bt.example.org"}
    assert "dummy" in _load_settings(**prod).stdout


def test_settings_refuse_google_client_secret_as_smtp_password():
    result = _load_settings(DJANGO_DEBUG="False", DJANGO_SECRET_KEY="x" * 50, DJANGO_ALLOWED_HOSTS="bt.example.org",
                            BITETRACKER_EMAIL_DELIVERY="smtp", EMAIL_HOST="smtp.example.org",
                            DEFAULT_FROM_EMAIL="BiteTracker <no-reply@example.org>", GOOGLE_OAUTH_CLIENT_ID="id",
                            GOOGLE_OAUTH_CLIENT_SECRET="same-value", EMAIL_HOST_PASSWORD="same-value")
    assert result.returncode != 0 and "Google sign-in client secret" in result.stderr


# --- Delivery mode, concurrency, controlled tests and request time --------------------------------------------


def test_console_mode_rows_are_never_resent_through_smtp(admins, commit, settings):
    settings.EMAIL_DELIVERY = "console"
    req = _applicant()
    with transaction.atomic():
        emails.registration_received(req)  # queued in console mode; its on-commit send never runs here
    row = EmailDelivery.objects.get()
    assert (row.channel, row.status) == ("console", Status.PENDING)
    EmailDelivery.objects.create(recipient=req.user, to_email=req.email, kind=Kind.ACCOUNT_SUSPENDED,
                                 event_key="legacy:1", subject="s", body_text="t", body_html="h",
                                 status=Status.FAILED, channel="")  # a row from before `channel` existed

    settings.EMAIL_DELIVERY = "smtp"
    call_command("send_pending_emails", stdout=io.StringIO())
    emails.send_deliveries(list(EmailDelivery.objects.values_list("pk", flat=True)))
    assert mail.outbox == []
    assert set(EmailDelivery.objects.values_list("status", flat=True)) == {Status.PENDING, Status.FAILED}


def test_a_row_is_claimed_by_exactly_one_sender(admins):
    with transaction.atomic():
        (row,) = emails.registration_received(_applicant())
    assert emails._claim(row.pk) is True
    # The other sender (retry command / second worker) now finds nothing to claim or send.
    assert emails._claim(row.pk) is False
    assert emails.send_deliveries([row.pk]) == {"sent": 0, "failed": 0, "suppressed": 0, "deferred": 0}
    call_command("send_pending_emails", stdout=io.StringIO())
    assert mail.outbox == [] and EmailDelivery.objects.get().attempts == 1


def test_in_request_send_and_retry_command_send_a_row_once(admins, commit):
    with commit():
        emails.registration_received(_applicant())
    call_command("send_pending_emails", stdout=io.StringIO())
    call_command("send_pending_emails", stdout=io.StringIO())
    assert len(mail.outbox) == 1 and EmailDelivery.objects.get().attempts == 1


def test_only_a_stale_claim_is_taken_over(admins):
    from django.utils import timezone

    with transaction.atomic():
        (row,) = emails.registration_received(_applicant())
    EmailDelivery.objects.filter(pk=row.pk).update(status=Status.SENDING, attempts=1,
                                                   last_attempt_at=timezone.now() - emails.STALE_SENDING / 2)
    call_command("send_pending_emails", stdout=io.StringIO())
    assert mail.outbox == []  # another sender may still be working on it
    EmailDelivery.objects.filter(pk=row.pk).update(last_attempt_at=timezone.now() - emails.STALE_SENDING * 2)
    call_command("send_pending_emails", stdout=io.StringIO())
    assert len(mail.outbox) == 1


def test_allowed_recipients_limit_a_controlled_test_to_one_address(admins, commit, settings):
    settings.EMAIL_ALLOWED_RECIPIENTS = ["admin.a@example.com"]
    with commit():
        _announce_access_request(_applicant())
    assert [m.to for m in mail.outbox] == [["admin.a@example.com"]]
    assert EmailDelivery.objects.filter(status=Status.SUPPRESSED).count() == 2
    settings.EMAIL_ALLOWED_RECIPIENTS = []  # lifting the limit later never sends the suppressed ones
    call_command("send_pending_emails", stdout=io.StringIO())
    assert len(mail.outbox) == 1


def test_request_budget_defers_remaining_emails_to_the_retry_command(admins, commit, settings, monkeypatch):
    clock = iter([0.0, 0.0, 0.5, 11.0, 11.0, 11.0, 11.0])
    monkeypatch.setattr(emails.time, "monotonic", lambda: next(clock))
    with commit():
        _announce_access_request(_applicant())  # applicant first, then two administrators
    assert len(mail.outbox) == 2  # the applicant's email sent; first admin email sent within budget
    assert EmailDelivery.objects.filter(status=Status.PENDING).count() == 1
    monkeypatch.undo()
    call_command("send_pending_emails", stdout=io.StringIO())
    assert len(mail.outbox) == 3 and not EmailDelivery.objects.exclude(status=Status.SENT).exists()


def test_batch_survives_a_rolled_back_savepoint(admins, commit):
    req = _applicant()
    worker = operator("worker", email="worker@example.com")
    with commit():
        with transaction.atomic():
            with pytest.raises(RuntimeError), transaction.atomic():
                emails.registration_received(req)
                raise RuntimeError("inner step failed")
            emails.account_suspended(worker, event_id=7)
    assert [m.to for m in mail.outbox] == [["worker@example.com"]]
    assert list(EmailDelivery.objects.values_list("kind", flat=True)) == [Kind.ACCOUNT_SUSPENDED]


def test_local_smtp_requires_an_allow_list_and_a_password():
    base = {"DJANGO_DEBUG": "True", "BITETRACKER_EMAIL_DELIVERY": "smtp", "EMAIL_HOST": "smtp.example.org",
            "DEFAULT_FROM_EMAIL": "BiteTracker <no-reply@example.org>", "EMAIL_HOST_USER": "user"}
    no_password = _load_settings(**base, EMAIL_ALLOWED_RECIPIENTS="me@example.org")
    assert no_password.returncode != 0 and "EMAIL_HOST_PASSWORD is empty" in no_password.stderr
    no_allow_list = _load_settings(**base, EMAIL_HOST_PASSWORD="smtp-test-value")
    assert no_allow_list.returncode != 0 and "EMAIL_ALLOWED_RECIPIENTS" in no_allow_list.stderr
    ok = _load_settings(**base, EMAIL_HOST_PASSWORD="smtp-test-value", EMAIL_ALLOWED_RECIPIENTS="me@example.org")
    assert ok.returncode == 0 and "smtp" in ok.stdout
    assert "smtp-test-value" not in ok.stdout + ok.stderr


# --- Branded header: inline CID logo ------------------------------------------------------------------------


def _parts(message):
    mime = message.message()
    yield mime
    if mime.is_multipart():
        for part in mime.walk():
            yield part


def test_logo_is_an_inline_cid_png_beside_the_html_only(admins, commit):
    with commit():
        emails.registration_received(_applicant())
    (message,) = mail.outbox
    mime = message.message()
    assert mime.get_content_type() == "multipart/alternative"
    text_part, related = mime.get_payload()
    assert text_part.get_content_type() == "text/plain" and "BiteTracker" in text_part.get_payload(decode=True).decode()
    assert related.get_content_type() == "multipart/related"
    html_part, image = related.get_payload()
    html = html_part.get_payload(decode=True).decode()
    assert image.get_content_type() == "image/png" and image["Content-ID"] == "<bitetracker-logo>"
    assert image["Content-Disposition"].startswith("inline")
    assert image.get_payload(decode=True) == emails.LOGO_PATH.read_bytes()
    assert 'src="cid:bitetracker-logo"' in html and 'alt="BiteTracker logo"' in html
    assert 'width="48" height="48"' in html
    # The brand name is real text, so the header still reads correctly with images blocked.
    assert ">Bite</span>" in html and ">Tracker</span>" in html
    assert "localhost" not in html.split("<img", 1)[1].split(">", 1)[0] and "<svg" not in html


def test_missing_logo_file_still_sends_without_an_image(admins, commit, monkeypatch, tmp_path):
    monkeypatch.setattr(emails, "LOGO_PATH", tmp_path / "missing.png")
    emails._logo_bytes.cache_clear()
    try:
        with commit():
            emails.registration_received(_applicant())
    finally:
        emails._logo_bytes.cache_clear()
    (message,) = mail.outbox
    assert [p.get_content_type() for p in message.message().walk()] == ["multipart/alternative", "text/plain", "text/html"]
    assert EmailDelivery.objects.get().status == Status.SENT
