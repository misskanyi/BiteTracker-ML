"""Account emails: the few events important enough to reach people outside BiteTracker.

Emailed (each to the affected person only):
- registration received and awaiting approval -> the applicant
- new registration request -> each active System Administrator, one message each
- registration approved / not approved -> the applicant
- account suspended / reactivated -> the account holder
- role or school assignments changed -> the account holder

Everything else (forecasts, uploads, calendar, announcements, support) stays in-app only;
`web.notifications.services.notify` never sends email.

Delivery
--------
1. The view queues an `EmailDelivery` row inside the same database transaction as the event.
   If the transaction rolls back, the row disappears with it and nothing is sent.
2. `transaction.on_commit` sends the queued rows right after the commit, in the same request,
   over one SMTP connection. Each network operation is limited by EMAIL_TIMEOUT (default 5 s)
   and the request stops starting new sends after EMAIL_REQUEST_BUDGET_SECONDS (default 10 s);
   unsent rows stay `pending`. A failure marks the row `failed` and is logged (delivery id, kind
   and exception class only); the account change is already committed and is never undone.
   Nothing here raises into the view.
3. `python manage.py send_pending_emails` retries `failed` (and stranded) rows, up to
   EMAIL_MAX_ATTEMPTS and only while they are younger than EMAIL_RETRY_MAX_AGE_HOURS. Run it
   from cron or a systemd timer; there is no background worker.

Duplicates are prevented by the unique (recipient, event_key) constraint: the same event never
emails the same person twice. Each row is claimed by one atomic conditional UPDATE before it is
sent, so the in-request send and the retry command never send the same row twice. Each row also
records the delivery mode it was queued under (`channel`) and is only ever sent under that mode.
EMAIL_ALLOWED_RECIPIENTS, when set, limits sending to those addresses (others are `suppressed`). Email content never includes passwords, tokens, internal
decision notes or school data beyond the names of the person's own assigned schools.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from django.core.mail import EmailMultiAlternatives, get_connection
from django.db import IntegrityError, transaction
from django.db.models import F, Q
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import timezone

from web.accounts import access
from web.accounts.models import Role
from web.notifications.models import EmailDelivery

logger = logging.getLogger(__name__)

Kind = EmailDelivery.Kind
Status = EmailDelivery.Status

# Optional emails a person can switch off in Settings -> Notifications. Every other kind is an
# essential notice about the person's own account and is always sent.
OPTIONAL_PREFERENCE = {Kind.ACCESS_REQUEST_ADMIN: "email_access_requests"}

# A row left in `sending` this long (the process died mid-send) may be retried by the command.
STALE_SENDING = timedelta(minutes=15)


def delivery_mode() -> str:
    """console, smtp or disabled (BITETRACKER_EMAIL_DELIVERY)."""
    return getattr(settings, "EMAIL_DELIVERY", "disabled")


def delivery_enabled() -> bool:
    return delivery_mode() != "disabled"


def recipient_allowed(address: str) -> bool:
    """With EMAIL_ALLOWED_RECIPIENTS set (controlled tests), only those addresses are ever emailed."""
    allowed = getattr(settings, "EMAIL_ALLOWED_RECIPIENTS", [])
    return not allowed or address.strip().lower() in allowed


def site_link(path: str) -> str:
    base = getattr(settings, "SITE_URL", "")
    return f"{base}{path}" if base else ""


def _display_name(user) -> str:
    return user.get_full_name().strip() or user.get_username()


def _school_text(schools) -> str:
    names = sorted(f"{s.name or s.school_id} ({s.school_id})" for s in schools)
    return ", ".join(names) if names else "None"


# --- Queueing --------------------------------------------------------------------------------------


def _initial_status(address: str) -> str:
    if not delivery_enabled():
        return Status.DISABLED
    return Status.PENDING if recipient_allowed(address) else Status.SUPPRESSED


def queue(recipients, kind: str, event_key: str, subject: str, context: dict) -> list[EmailDelivery]:
    """Record one email per recipient for this event and send them after the transaction commits.

    Recipients without an email address, who switched an optional email off, or who were already
    emailed about `event_key` are skipped. Returns the new rows.
    """
    created: list[EmailDelivery] = []
    seen: set[int] = set()
    for user in recipients:
        if user.pk in seen:
            continue
        seen.add(user.pk)
        address = (user.email or "").strip()
        if not address:
            logger.info("Account email %s skipped for user %s: no email address", kind, user.pk)
            continue
        field = OPTIONAL_PREFERENCE.get(kind)
        if field and not getattr(access.get_profile(user), field):
            continue
        if EmailDelivery.objects.filter(recipient=user, event_key=event_key).exists():
            continue
        body_context = {**context, "subject": subject, "greeting_name": _display_name(user)}
        try:
            # A savepoint, so losing a race to a concurrent request never breaks the caller's transaction.
            with transaction.atomic():
                row = EmailDelivery.objects.create(
                    recipient=user, to_email=address, kind=kind, event_key=event_key[:120], subject=subject[:200],
                    body_text=render_to_string("emails/account_email.txt", body_context),
                    body_html=render_to_string("emails/account_email.html", body_context),
                    status=_initial_status(address), channel=delivery_mode(),
                )
        except IntegrityError:
            continue
        created.append(row)
    pending = [row.pk for row in created if row.status == Status.PENDING]
    if pending:
        _schedule(pending)
    return created


class _Batch:
    """Every email queued in one transaction, sent together after it commits: one connection and
    one time budget per event, however many recipients or kinds of email it has."""

    def __init__(self):
        self.pks: list[int] = []
        self.done = False

    def __call__(self):
        self.done = True
        send_deliveries(self.pks, budget=getattr(settings, "EMAIL_REQUEST_BUDGET_SECONDS", None))


_local = threading.local()


def _schedule(pks: list[int]) -> None:
    connection = transaction.get_connection()
    batch = getattr(_local, "batch", None)
    # Join the batch only while it is still waiting on this transaction's commit (not already run,
    # and not discarded by a rollback).
    if (batch is not None and not batch.done and connection.in_atomic_block
            and any(entry[1] is batch for entry in connection.run_on_commit)):
        batch.pks.extend(pks)
        return
    batch = _local.batch = _Batch()
    batch.pks.extend(pks)
    transaction.on_commit(batch, robust=True)


# --- Sending ---------------------------------------------------------------------------------------


# The header logo travels inside the email as an inline PNG (Content-ID), so it never depends on a
# public or localhost image URL. A 96px copy of icon-light.png, shown at 48px.
LOGO_CID = "bitetracker-logo"
LOGO_PATH = Path(settings.BASE_DIR) / "static" / "images" / "branding" / "email" / "logo-icon-96.png"


@lru_cache(maxsize=1)
def _logo_bytes() -> bytes | None:
    try:
        return LOGO_PATH.read_bytes()
    except OSError:
        logger.warning("Email logo not found; emails are sent without it (the brand name is text)")
        return None


class BrandedEmail(EmailMultiAlternatives):
    """Plain text + HTML, with the logo attached inline beside the HTML part.

    MIME layout: multipart/alternative [ text/plain, multipart/related [ text/html, image/png ] ].
    The image belongs to the HTML part only, so clients don't list it as a separate attachment.
    """

    def __init__(self, *args, logo: bytes | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logo = logo

    def message(self, *args, **kwargs):
        msg = super().message(*args, **kwargs)
        if not self.logo:
            return msg
        parts = msg.get_payload()
        html_index = next(i for i, part in enumerate(parts) if part.get_content_type() == "text/html")
        image = MIMEImage(self.logo, "png")
        image.add_header("Content-ID", f"<{LOGO_CID}>")
        image.add_header("Content-Disposition", "inline", filename="bitetracker-logo.png")
        related = MIMEMultipart("related")
        related.attach(parts[html_index])
        related.attach(image)
        parts[html_index] = related
        return msg


def build_message(subject: str, body_text: str, body_html: str, to_email: str, *, connection=None) -> BrandedEmail:
    logo = _logo_bytes() if f"cid:{LOGO_CID}" in body_html else None
    message = BrandedEmail(subject, body_text, settings.DEFAULT_FROM_EMAIL, [to_email], connection=connection, logo=logo)
    message.attach_alternative(body_html, "text/html")
    return message


def _unsent(now) -> Q:
    """Rows still to send: waiting, failed, or stranded in `sending` (the process died mid-send).

    Only rows queued under the current delivery mode qualify, so emails recorded while printing to
    the console (or before this field existed) can never be sent later through SMTP.
    """
    return (Q(status__in=[Status.PENDING, Status.FAILED])
            | Q(status=Status.SENDING, last_attempt_at__lt=now - STALE_SENDING)) & Q(channel=delivery_mode())


def _claim(pk: int) -> bool:
    """Move one row to `sending` only if nobody else has.

    A single conditional UPDATE, so when the on-commit send and the retry command (or two retry
    runs) reach the same row, the database lets exactly one of them change it; the other updates
    0 rows and skips it. Rows are claimed one at a time, immediately before their own send, so a
    claim is never older than one send and can't go stale while it waits in a batch.
    """
    now = timezone.now()
    return bool(EmailDelivery.objects.filter(_unsent(now), pk=pk, attempts__lt=settings.EMAIL_MAX_ATTEMPTS).update(
        status=Status.SENDING, attempts=F("attempts") + 1, last_attempt_at=now))


def _mark_failed(row: EmailDelivery, exc: BaseException) -> None:
    EmailDelivery.objects.filter(pk=row.pk).update(status=Status.FAILED, last_error=type(exc).__name__[:100])
    # No address, subject or body in the log: the delivery id identifies the email for an operator.
    logger.warning("Account email failed: delivery=%s kind=%s error=%s", row.pk, row.kind, type(exc).__name__)


def send_deliveries(pks, *, budget: float | None = None) -> dict[str, int]:
    """Send the given queued rows now. Never raises: failures are recorded on the rows and logged.

    `budget` (seconds, used for in-request sends) stops starting new work once that much time has
    passed; rows not started stay `pending` for `send_pending_emails`. It can't interrupt a step
    already in progress, which is bounded separately by EMAIL_TIMEOUT per network operation.
    """
    result = {"sent": 0, "failed": 0, "suppressed": 0, "deferred": 0}
    started = time.monotonic()
    connection, open_error = None, None
    try:
        for index, pk in enumerate(pks):
            if budget is not None and open_error is None and time.monotonic() - started >= budget:
                result["deferred"] = len(pks) - index
                logger.info("Account email time budget used: %s email(s) left for send_pending_emails",
                            result["deferred"])
                break
            if not _claim(pk):
                continue
            row = EmailDelivery.objects.get(pk=pk)
            if not recipient_allowed(row.to_email):
                EmailDelivery.objects.filter(pk=pk).update(status=Status.SUPPRESSED)
                result["suppressed"] += 1
                continue
            if open_error is None and connection is None:
                connection = get_connection(fail_silently=False)
                try:
                    connection.open()
                except Exception as exc:  # unreachable server, bad credentials, timeout ...
                    open_error, connection = exc, None
            if open_error is not None:
                # One failed connection is enough: record the rest as failed without more network waits.
                _mark_failed(row, open_error)
                result["failed"] += 1
                continue
            message = build_message(row.subject, row.body_text, row.body_html, row.to_email, connection=connection)
            try:
                message.send()
            except Exception as exc:
                _mark_failed(row, exc)
                result["failed"] += 1
                continue
            EmailDelivery.objects.filter(pk=pk).update(status=Status.SENT, sent_at=timezone.now(), last_error="")
            result["sent"] += 1
    except Exception as exc:  # never let email break a request whose account change is committed
        logger.exception("Account email sending stopped unexpectedly: error=%s", type(exc).__name__)
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
    return result


def retryable():
    """Rows the retry command may send: unsent, under the attempt limit and recent enough."""
    now = timezone.now()
    return EmailDelivery.objects.filter(
        _unsent(now), attempts__lt=settings.EMAIL_MAX_ATTEMPTS,
        created_at__gte=now - timedelta(hours=settings.EMAIL_RETRY_MAX_AGE_HOURS),
    ).order_by("pk")


# --- The account events ------------------------------------------------------------------------------


def registration_received(access_request) -> list[EmailDelivery]:
    user = access_request.user
    return queue([user], Kind.REGISTRATION_RECEIVED, f"access_request:{access_request.pk}:received",
                 "We received your BiteTracker registration request", {
                     "heading": "Registration received",
                     "paragraphs": ["Thank you for registering for BiteTracker. Your request has been sent to a "
                                    "System Administrator for approval."],
                     "details": [("Status", "Pending approval"), ("Email", access_request.email)],
                     "next_step": "You don't need to do anything now. We'll email you when your request has been "
                                  "decided. Until then, signing in shows your request status.",
                     "action_label": "Check your status", "action_url": site_link(reverse("accounts:login")),
                 })


def access_request_for_admins(access_request) -> list[EmailDelivery]:
    from web.notifications.services import system_admins

    who = access_request.full_name.strip() or access_request.email
    return queue(system_admins(), Kind.ACCESS_REQUEST_ADMIN, f"access_request:{access_request.pk}:submitted",
                 "New BiteTracker registration request", {
                     "heading": "New registration request",
                     "paragraphs": [f"{who} has registered for BiteTracker and is waiting for approval."],
                     "details": [("Name", access_request.full_name.strip() or "Not given"),
                                 ("Email", access_request.email),
                                 ("Signed in with", access_request.provider.title()),
                                 ("Status", "Pending approval")],
                     "next_step": "Review the request, then approve it with a role and schools, or reject it.",
                     "action_label": "Review request",
                     "action_url": site_link(reverse("administration:access_request_detail", args=[access_request.pk])),
                 })


def registration_approved(access_request, role: str, schools) -> list[EmailDelivery]:
    return queue([access_request.user], Kind.REGISTRATION_APPROVED, f"access_request:{access_request.pk}:approved",
                 "Your BiteTracker registration was approved", {
                     "heading": "Registration approved",
                     "paragraphs": ["Good news: a System Administrator has approved your BiteTracker account."],
                     "details": [("Status", "Active"), ("Role", Role(role).label),
                                 ("Assigned schools", _school_text(schools))],
                     "next_step": "Sign in with the same Google account you registered with.",
                     "action_label": "Sign in to BiteTracker", "action_url": site_link(reverse("accounts:login")),
                 })


def registration_rejected(access_request) -> list[EmailDelivery]:
    # The administrator's reason is an internal note and is deliberately not included.
    return queue([access_request.user], Kind.REGISTRATION_REJECTED, f"access_request:{access_request.pk}:rejected",
                 "Your BiteTracker registration was not approved", {
                     "heading": "Registration not approved",
                     "paragraphs": ["A System Administrator has reviewed your BiteTracker registration request "
                                    "and did not approve it."],
                     "details": [("Status", "Not approved")],
                     "next_step": "If you think this is a mistake, contact the Food4Education team member who "
                                  "manages BiteTracker access.",
                     "action_label": "", "action_url": "",
                 })


def account_suspended(user, event_id: int) -> list[EmailDelivery]:
    return queue([user], Kind.ACCOUNT_SUSPENDED, f"account:{user.pk}:suspended:{event_id}",
                 "Your BiteTracker account has been suspended", {
                     "heading": "Account suspended",
                     "paragraphs": ["A System Administrator has suspended your BiteTracker account. You can no "
                                    "longer sign in, with a password or with Google. Your history is kept."],
                     "details": [("Status", "Suspended")],
                     "next_step": "If you think this is a mistake, contact your BiteTracker System Administrator.",
                     "action_label": "", "action_url": "",
                 })


def account_reactivated(user, event_id: int) -> list[EmailDelivery]:
    return queue([user], Kind.ACCOUNT_REACTIVATED, f"account:{user.pk}:reactivated:{event_id}",
                 "Your BiteTracker account has been reactivated", {
                     "heading": "Account reactivated",
                     "paragraphs": ["A System Administrator has reactivated your BiteTracker account."],
                     "details": [("Status", "Active"), ("Role", access.role_label(user))],
                     "next_step": "Sign in with your existing password or Google account.",
                     "action_label": "Sign in to BiteTracker", "action_url": site_link(reverse("accounts:login")),
                 })


def access_changed(user, event_id: int, *, old_role: str | None = None, added=(), removed=()) -> list[EmailDelivery]:
    """Role and/or school assignment change. `old_role` is set only when the role changed."""
    details = [("Role", access.role_label(user))]
    if old_role is not None:
        details.append(("Previous role", Role(old_role).label))
    if added:
        details.append(("Schools added", _school_text(added)))
    if removed:
        details.append(("Schools removed", _school_text(removed)))
    details.append(("Assigned schools now", _school_text(a.school for a in user.school_assignments.select_related("school"))))
    changed = " and ".join(p for p, on in (("role", old_role is not None), ("school assignments", added or removed)) if on)
    return queue([user], Kind.ACCESS_CHANGED, f"account:{user.pk}:access:{event_id}",
                 "Your BiteTracker access has changed", {
                     "heading": "Your access has changed",
                     "paragraphs": [f"A System Administrator changed your BiteTracker {changed or 'access'}."],
                     "details": details,
                     "next_step": "What you can see in BiteTracker now follows these settings. Contact your System "
                                  "Administrator if something looks wrong.",
                     "action_label": "View your profile", "action_url": site_link(reverse("accounts:profile")),
                 })
