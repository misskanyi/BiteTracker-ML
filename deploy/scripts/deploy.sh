#!/usr/bin/env bash
# Deploy one BiteTracker release on the EC2 instance, with automatic rollback.
#
# Usage: sudo deploy/scripts/deploy.sh <release-tarball.tar.gz> <release-id>
#   release-tarball: output of `git archive` (tracked code only: never models, data or databases)
#   release-id:      e.g. the git commit SHA
#
# Steps: check production artifacts exist in the state dir -> unpack release -> refuse
# any bundled model/data/database files -> build venv -> back up state -> collectstatic,
# check --deploy, migrate -> switch /opt/bitetracker/current -> restart -> /health/.
# If the health check fails, the previous release is restored (and the database too,
# if this release applied migrations). The state dir is never overwritten by a deploy.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "$SCRIPT_DIR/common.sh"

[ $# -eq 2 ] || die "usage: $0 <release-tarball.tar.gz> <release-id>"
TARBALL="$1"
RELEASE_ID="$2"
[[ "$RELEASE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "release id may only contain letters, digits, '.', '-' and '_'."
[ -f "$TARBALL" ] || die "tarball $TARBALL not found."

require_root
load_env_file
require_state_artifacts

RELEASE="$RELEASES_DIR/$RELEASE_ID"
PREVIOUS="$(readlink -f "$CURRENT_LINK" 2>/dev/null || true)"
[ "$PREVIOUS" != "$(readlink -f "$RELEASE" 2>/dev/null || echo none)" ] || die "release $RELEASE_ID is already active."

# --- 1. Unpack -------------------------------------------------------------------
log "Unpacking release $RELEASE_ID"
mkdir -p "$RELEASES_DIR"
rm -rf "$RELEASE"
mkdir -p "$RELEASE"
tar -xzf "$TARBALL" -C "$RELEASE"

# --- 2. Never ship trained models, processed data or databases ------------------------
bundled="$(cd "$RELEASE" && find . -type f \( \
    -name '*.joblib' -o -name '*.pkl' -o -name '*.sqlite3*' -o -name '*.db' \
    -o -path './models/*' ! -name '.gitkeep' \
    -o -path './data/processed/*' ! -name '.gitkeep' \
    -o -path './data/uploads/*' ! -name '.gitkeep' \
    -o -path './reports/metrics/*' ! -name '.gitkeep' \) -print)"
if [ -n "$bundled" ]; then
  rm -rf "$RELEASE"
  die "release bundles model/data/database files; refusing to deploy:
$bundled"
fi

# --- 3. Python environment ---------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.11 || command -v python3)}"
log "Building virtualenv with $("$PYTHON_BIN" --version)"
"$PYTHON_BIN" -m venv "$RELEASE/.venv"
"$RELEASE/.venv/bin/pip" install --quiet --upgrade pip
"$RELEASE/.venv/bin/pip" install --quiet -r "$RELEASE/requirements.txt"
chmod -R a+rX "$RELEASE"

# State directory layout, owned by the app user (settings would otherwise create it as root).
for d in "" data data/processed data/raw data/uploads models reports reports/metrics reports/figures; do
  install -d -o "$APP_USER" -g "$APP_USER" -m 750 "$BITETRACKER_STATE_DIR/$d"
done

# --- 4. Back up state before touching the database ---------------------------------------
BACKUP_PATH="$(bash "$SCRIPT_DIR/backup.sh""pre-deploy-$RELEASE_ID" | tail -n 1)"
log "Pre-deploy backup: $BACKUP_PATH"

# --- 5. Django checks, static files, migrations ----------------------------------------
manage "$RELEASE" check --deploy
(cd "$RELEASE" && "$RELEASE/.venv/bin/python" manage.py collectstatic --noinput --verbosity 0)
chmod -R a+rX "$RELEASE/staticfiles"
MIGRATIONS_PENDING=0
manage "$RELEASE" migrate --check >/dev/null 2>&1 || MIGRATIONS_PENDING=1
manage "$RELEASE" migrate --noinput

# --- 6. Switch and restart -----------------------------------------------------------------
switch_to() {
  ln -sfn "$1" "$CURRENT_LINK.tmp"
  mv -Tf "$CURRENT_LINK.tmp" "$CURRENT_LINK"
}

log "Activating release $RELEASE_ID"
switch_to "$RELEASE"
systemctl restart "$SERVICE_NAME"

if health_check 30; then
  log "Deploy of $RELEASE_ID succeeded."
else
  log "Deploy of $RELEASE_ID failed its health check; rolling back."
  if [ -n "$PREVIOUS" ] && [ -d "$PREVIOUS" ]; then
    if [ "$MIGRATIONS_PENDING" -eq 1 ] && [ -f "$BACKUP_PATH/db.sqlite3" ]; then
      log "Restoring the pre-deploy database (this release applied migrations)."
      systemctl stop "$SERVICE_NAME"
      DB_PATH="${DJANGO_SQLITE_PATH:-$BITETRACKER_STATE_DIR/db.sqlite3}"
      install -o "$APP_USER" -g "$APP_USER" -m 640 "$BACKUP_PATH/db.sqlite3" "$DB_PATH"
      rm -f "$DB_PATH-wal" "$DB_PATH-shm" "$DB_PATH-journal"
    fi
    switch_to "$PREVIOUS"
    systemctl restart "$SERVICE_NAME"
    health_check 30 || log "WARNING: previous release is also unhealthy; investigate with: journalctl -u $SERVICE_NAME"
  else
    log "No previous release to roll back to."
  fi
  exit 1
fi

# --- 7. Keep the newest $KEEP_RELEASES releases (never the active or previous one) ------------
ls -1dt "$RELEASES_DIR"/*/ 2>/dev/null | tail -n +"$((KEEP_RELEASES + 1))" | while read -r old; do
  old="$(readlink -f "$old")"
  [ "$old" = "$(readlink -f "$CURRENT_LINK")" ] || [ "$old" = "$PREVIOUS" ] || rm -rf -- "$old"
done
