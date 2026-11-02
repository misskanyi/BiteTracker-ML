#!/usr/bin/env bash
# Shared settings and helpers for deploy.sh, rollback.sh and backup.sh. Sourced, not run.

APP_USER="${APP_USER:-bitetracker}"
APP_ROOT="${APP_ROOT:-/opt/bitetracker}"
RELEASES_DIR="$APP_ROOT/releases"
CURRENT_LINK="$APP_ROOT/current"
ENV_FILE="${ENV_FILE:-/etc/bitetracker/bitetracker.env}"
SERVICE_NAME="${SERVICE_NAME:-bitetracker}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health/}"
KEEP_RELEASES="${KEEP_RELEASES:-5}"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

require_root() {
  [ "$(id -u)" -eq 0 ] || die "run as root (sudo); SSM Run Command already runs as root."
}

# Export KEY=value lines from the env file literally (no shell expansion, so
# secrets containing $, quotes or backticks are passed through unchanged).
load_env_file() {
  [ -r "$ENV_FILE" ] || die "environment file $ENV_FILE not found or not readable."
  local line key value
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|'#'*) continue ;; esac
    key="${line%%=*}"
    value="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    export "$key=$value"
  done < "$ENV_FILE"
  : "${BITETRACKER_STATE_DIR:?BITETRACKER_STATE_DIR must be set in $ENV_FILE}"
  export DJANGO_SETTINGS_MODULE=bitetracker_web.settings
}

# Run manage.py from a release as the app user (so new files in the state dir are owned by it).
manage() {
  local release="$1"; shift
  (cd "$release" && runuser -u "$APP_USER" --preserve-environment -- "$release/.venv/bin/python" manage.py "$@")
}

# Poll /health/ until it answers 200 or the attempts run out.
health_check() {
  local attempts="${1:-30}" host
  host="$(printf '%s' "${DJANGO_ALLOWED_HOSTS:-localhost}" | cut -d, -f1 | tr -d ' ')"
  for _ in $(seq "$attempts"); do
    if curl -fsS --max-time 10 -H "Host: $host" "$HEALTH_URL" >/dev/null 2>&1; then
      log "Health check passed."
      return 0
    fi
    sleep 2
  done
  log "Health check failed. Last response:"
  curl -sS --max-time 10 -H "Host: $host" "$HEALTH_URL" || true
  echo
  return 1
}

# The production model, metadata and processed data must already be in the state dir.
# They are copied there once by hand and never come from git or CI.
require_state_artifacts() {
  local missing=0 f
  for f in \
    "$BITETRACKER_STATE_DIR/models/final_best_model.joblib" \
    "$BITETRACKER_STATE_DIR/models/final_best_model_metadata.json" \
    "$BITETRACKER_STATE_DIR/data/processed/school_meals_processed.csv"; do
    if [ ! -s "$f" ]; then log "Missing production artifact: $f"; missing=1; fi
  done
  [ "$missing" -eq 0 ] || die "copy the trained model, metadata and processed CSV into $BITETRACKER_STATE_DIR first (see deploy/README.md)."
}
