"""Facts behind the public privacy notice (/privacy/).

Everything the notice says about BiteTracker's own behaviour (data stored, cookies, Google
scopes, email events) is written in the template from the code. The facts below cannot be read
from the code: they are decisions by the project owner or settings on the production server.

Each one stays None until it has been confirmed (owner decisions) or checked on the server
(infrastructure). While any is None the page is not served (404), so the notice is never
published with gaps or placeholder text. Record where each verified value came from.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date


@dataclass(frozen=True)
class PrivacyFacts:
    # --- Owner decisions (confirmed by Sharon Kanyi) -----------------------------------------
    contact_email: str | None = None
    # Paragraphs, exactly as approved: how long account data is kept, and what happens at the
    # end of the project.
    retention: tuple[str, ...] | None = None
    # Paragraphs, exactly as approved: how to ask for access, correction or deletion.
    deletion: tuple[str, ...] | None = None
    effective_date: date | None = None

    # --- Production infrastructure (checked on the server) ---------------------------------------
    aws_region: str | None = None          # e.g. "Africa (Cape Town), af-south-1"
    scheduled_backups: str | None = None   # verified sentence, or "" if none are scheduled
    backup_offsite_copy: bool | None = None  # BACKUP_S3_URI set and uploads confirmed
    backups_kept: int | None = None        # BACKUP_KEEP in the server env file
    server_log_retention: str | None = None  # from /etc/logrotate.d/nginx, e.g. "14 days"

    # Verified 2026-11-10: bitetracker.xyz uses Cloudflare nameservers and responses carry
    # `server: cloudflare` and `cf-ray` headers.
    cloudflare: bool = True


FACTS = PrivacyFacts()


def missing(facts: PrivacyFacts | None = None) -> list[str]:
    """Names of facts not yet confirmed. The notice is only published when this is empty."""
    facts = FACTS if facts is None else facts
    return [f.name for f in fields(facts) if getattr(facts, f.name) is None]


def is_published(facts: PrivacyFacts | None = None) -> bool:
    return not missing(facts)
