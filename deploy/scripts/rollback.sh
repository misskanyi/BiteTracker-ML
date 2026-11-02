#!/usr/bin/env bash
# Manually switch BiteTracker back to an earlier release.
#
# Usage: sudo deploy/scripts/rollback.sh                 # previous release (newest that isn't active)
#        sudo deploy/scripts/rollback.sh <release-id>    # a specific release in /opt/bitetracker/releases
#        sudo deploy/scripts/rollback.sh <release-id> --restore-db <backup-dir>
#
# Code only by default: the database, model and uploads are left as they are. Use
# --restore-db with a backup from /var/backups/bitetracker only if a newer release
# applied database migrations the older code cannot handle (writes since then are lost).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "$SCRIPT_DIR/common.sh"

require_root
load_env_file

ACTIVE="$(readlink -f "$CURRENT_LINK" 2>/dev/null || true)"
TARGET=""
RESTORE_DB=""
if [ $# -ge 1 ]; then
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid release id."
  TARGET="$RELEASES_DIR/$1"
fi
if [ "${2:-}" = "--restore-db" ]; then
  RESTORE_DB="${3:?--restore-db needs a backup directory}"
  [ -f "$RESTORE_DB/db.sqlite3" ] || die "$RESTORE_DB/db.sqlite3 not found."
fi
if [ -z "$TARGET" ]; then
  TARGET="$(ls -1dt "$RELEASES_DIR"/*/ 2>/dev/null | while read -r r; do
    r="$(readlink -f "$r")"; [ "$r" = "$ACTIVE" ] || { echo "$r"; break; }; done)"
fi
[ -n "$TARGET" ] && [ -x "$TARGET/.venv/bin/gunicorn" ] || die "no usable release to roll back to."

log "Rolling back from ${ACTIVE:-none} to $TARGET"
bash "$SCRIPT_DIR/backup.sh" pre-rollback >/dev/null

if [ -n "$RESTORE_DB" ]; then
  systemctl stop "$SERVICE_NAME"
  DB_PATH="${DJANGO_SQLITE_PATH:-$BITETRACKER_STATE_DIR/db.sqlite3}"
  install -o "$APP_USER" -g "$APP_USER" -m 640 "$RESTORE_DB/db.sqlite3" "$DB_PATH"
  rm -f "$DB_PATH-wal" "$DB_PATH-shm" "$DB_PATH-journal"
  log "Database restored from $RESTORE_DB"
fi

ln -sfn "$TARGET" "$CURRENT_LINK.tmp"
mv -Tf "$CURRENT_LINK.tmp" "$CURRENT_LINK"
systemctl restart "$SERVICE_NAME"
health_check 30 || die "rolled-back release is unhealthy; check: journalctl -u $SERVICE_NAME"
log "Rollback complete."
