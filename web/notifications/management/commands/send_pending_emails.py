"""Retry account emails that could not be sent right after their event.

    python manage.py send_pending_emails            # send what is due
    python manage.py send_pending_emails --dry-run  # only count what is due

Only rows already in the outbox are considered (never older events), each at most
EMAIL_MAX_ATTEMPTS times and only while younger than EMAIL_RETRY_MAX_AGE_HOURS. Safe to run from
cron or a systemd timer while the web app is also sending: each row is claimed atomically.
Output contains counts only, never addresses or message contents.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from web.notifications import emails


class Command(BaseCommand):
    help = "Send account emails that are waiting or failed (bounded retries)."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Count the emails that are due without sending.")
        parser.add_argument("--limit", type=int, default=200, help="Most emails to send in one run (default 200).")

    def handle(self, *args, dry_run=False, limit=200, **options):
        if not emails.delivery_enabled():
            self.stdout.write("Email delivery is disabled (BITETRACKER_EMAIL_DELIVERY); nothing sent.")
            return
        pks = list(emails.retryable().values_list("pk", flat=True)[:max(limit, 0)])
        if dry_run:
            self.stdout.write(f"{len(pks)} email(s) due.")
            return
        result = emails.send_deliveries(pks)
        self.stdout.write(f"{len(pks)} due: {result['sent']} sent, {result['failed']} failed.")
