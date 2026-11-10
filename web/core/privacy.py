"""Facts behind the public privacy notice (/privacy/).

What the notice says about BiteTracker's own behaviour (data stored, what deleting an account
removes, cookies, Google scopes, email events) and the owner's retention and request policy are
written in templates/core/privacy.html. The facts below come from the production server, plus
the publication date. Each stays None until it has been checked on the server (record how in
the comment), and while any is None the page is not served (404), so the notice is never
published with gaps or guessed values.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date

CONTACT_EMAIL = "sharon.kanyi@strathmore.edu"  # confirmed by Sharon Kanyi, 2026-11-10


@dataclass(frozen=True)
class PrivacyFacts:
    # Set to the day the completed notice first goes live (owner decision, 2026-11-10).
    effective_date: date | None = None

    # --- Production server (each checked read-only on the instance) ------------------------------
    aws_region: str | None = None  # instance metadata placement/region; completes "...backups in ___."
    # Days between scheduled backups (1 = nightly); 0 = none scheduled, only the pre-update backups.
    backup_interval_days: int | None = None
    backups_kept: int | None = None  # BACKUP_KEEP in /etc/bitetracker/bitetracker.env (default 14)
    backup_offsite_copy: bool | None = None  # BACKUP_S3_URI set AND copies seen in the bucket
    # Completes "...copied to Amazon S3, where copies are ___." from the bucket lifecycle rule;
    # backup.sh never deletes them itself. "" when there is no off-site copy.
    offsite_backup_retention: str | None = None  # e.g. "deleted after 30 days" / "kept until deleted by hand"
    # Only what the configuration supports; "" = checked, but no retention statement can be made.
    web_log_retention: str | None = None  # /etc/logrotate.d/nginx, e.g. "14 days"
    # Does Nginx record the visitor's own IP (real_ip / X-Forwarded-For in log_format)? With the
    # repository's config it records Cloudflare's address instead.
    web_log_has_visitor_ip: bool | None = None
    # Completes "...its own log of requests and errors, ___." from journald settings (not disk
    # usage); "" when no retention statement is supported.
    app_log_retention: str | None = None  # e.g. "for up to 30 days (MaxRetentionSec=30day)"

    # True when the model in production was trained on the combined research dataset (observed 2025
    # + simulated 2023-2024 records). Re-check whenever a new model is installed.
    production_model_uses_simulated_records: bool | None = None
    # AWS-level copies outside backup.sh (EBS snapshots, AMIs, Data Lifecycle Manager, AWS Backup):
    # a sentence describing them and how long they're kept, or "" when none exist.
    aws_snapshot_copies: str | None = None

    # Verified 2026-11-10: bitetracker.xyz uses Cloudflare nameservers and responses carry
    # `server: cloudflare` and `cf-ray` headers.
    cloudflare: bool = True


# Server checks reported by Sharon Kanyi on 2026-11-10 (read-only, on the instance).
FACTS = PrivacyFacts(
    effective_date=date(2026, 11, 10),  # first public release; change if the release goes out on a later day
    aws_region="the US East (N. Virginia) region (us-east-1), in the United States",  # instance metadata
    backup_interval_days=1,  # cron at 02:15 server time; successful runs logged 4-10 Nov 2026
    backups_kept=14,  # BACKUP_KEEP=14
    backup_offsite_copy=False,  # BACKUP_S3_URI is empty
    offsite_backup_retention="",
    # /etc/logrotate.d/nginx: daily, rotate 14, notifempty
    web_log_retention="rotated daily when they contain entries, and the 14 most recent rotated files are kept",
    # No real_ip_header / set_real_ip_from / CF-Connecting-IP / custom log_format configured.
    web_log_has_visitor_ip=False,
    # journald: no time-based retention configured, ForwardToSyslog=yes; syslog files: weekly,
    # rotate 4, notifempty (that rotation does not govern the journal itself).
    # Production metadata (checked 2026-11-10): Random Forest, timestamp 2026-10-26T10:27:40.020876+00:00,
    # identical to the model trained on 2023-02-10..2025-04-29 records on the combined sheet, tested on observed 2025 rows.
    production_model_uses_simulated_records=True,
    # AWS console, this account, us-east-1 only (2026-11-10): no owned EBS snapshots or AMIs, no Data
    # Lifecycle Manager policies, no AWS Backup plans or account-created vaults. Not a claim about other
    # regions or accounts; the notice only describes the server's own backups.
    aws_snapshot_copies="",
    app_log_retention=("in the server's system journal, for which no time limit is configured. Copies are also "
                       "written to the server's system log files, which are rotated weekly when they contain "
                       "entries, with the four most recent rotated copies kept"),
)


def missing(facts: PrivacyFacts | None = None) -> list[str]:
    """Names of facts not yet confirmed. The notice is only published when this is empty."""
    facts = FACTS if facts is None else facts
    return [f.name for f in fields(facts) if getattr(facts, f.name) is None]


def is_published(facts: PrivacyFacts | None = None) -> bool:
    return not missing(facts)
