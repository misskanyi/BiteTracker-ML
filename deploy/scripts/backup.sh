#!/usr/bin/env bash
# Back up BiteTracker state: SQLite database (consistent online copy), trained model and
# metadata, processed historical CSV, metrics and user uploads.
#
# Usage: sudo deploy/scripts/backup.sh [label]
# Writes $BACKUP_DIR/<UTC timestamp>[-label]/ and prints that path on the last line.
# Keeps the newest $BACKUP_KEEP backups. If BACKUP_S3_URI is set, also uploads there
# (server-side encrypted). Suggested nightly cron (/etc/cron.d/bitetracker-backup):
#   15 2 * * * root /opt/bitetracker/current/deploy/scripts/backup.sh nightly >>/var/log/bitetracker-backup.log 2>&1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "$SCRIPT_DIR/common.sh"

require_root
load_env_file

BACKUP_DIR="${BACKUP_DIR:-/var/backups/bitetracker}"
BACKUP_KEEP="${BACKUP_KEEP:-14}"
STATE="$BITETRACKER_STATE_DIR"
DB_PATH="${DJANGO_SQLITE_PATH:-$STATE/db.sqlite3}"
label="${1:-manual}"
# Same characters deploy.sh allows in a release id, since it passes "pre-deploy-<release-id>".
[[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || die "label may only contain letters, digits, '.', '-' and '_'."

dest="$BACKUP_DIR/$(date -u +%Y%m%dT%H%M%SZ)-$label"
umask 077
mkdir -p "$dest"

if [ -f "$DB_PATH" ]; then
  # sqlite3's backup API gives a consistent copy even while Gunicorn is writing.
  python3 - "$DB_PATH" "$dest/db.sqlite3" <<'PY'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
with dst:
    src.backup(dst)
dst.close(); src.close()
PY
  log "Database backed up."
else
  log "No database at $DB_PATH yet; skipping."
fi

# Everything else in the state dir except the live database files.
tar -C "$STATE" -czf "$dest/state.tar.gz" \
  --exclude='./db.sqlite3' --exclude='./db.sqlite3-*' \
  --ignore-failed-read .
log "State files backed up."

if [ -n "${BACKUP_S3_URI:-}" ]; then
  aws s3 cp --recursive --only-show-errors --sse AES256 "$dest" "${BACKUP_S3_URI%/}/$(basename "$dest")/"
  log "Uploaded to ${BACKUP_S3_URI%/}/$(basename "$dest")/"
fi

# Retention: keep the newest $BACKUP_KEEP local backups.
ls -1dt "$BACKUP_DIR"/*/ 2>/dev/null | tail -n +"$((BACKUP_KEEP + 1))" | xargs -r rm -rf --

echo "$dest"
