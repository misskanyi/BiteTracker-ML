#!/usr/bin/env bash
# One-time EC2 setup for BiteTracker (Ubuntu 24.04). Safe to run again: it never deletes
# or overwrites the database, models, uploads, backups or an existing environment file.
#
# Usage (as root, e.g. through SSM Run Command, from an unpacked copy of the repository):
#   bash deploy/scripts/bootstrap.sh [extra-allowed-host ...]
#
# What it does:
#   1. Installs python3-venv (needed to build release virtualenvs), nginx and curl if missing.
#   2. Creates the `bitetracker` system user (no login shell).
#   3. Creates /etc/bitetracker/bitetracker.env from deploy/bitetracker.env.example if it does
#      not exist: new random DJANGO_SECRET_KEY (never printed), DJANGO_ALLOWED_HOSTS from the
#      instance's public IP and DNS name (plus any hosts given as arguments), plain-HTTP settings.
#   4. Creates /opt/bitetracker/releases, the state dir (/var/lib/bitetracker) and the backup dir.
#   5. Installs and enables the systemd service (started by the first deploy, not here).
#   6. Installs the Nginx site on port 80 (only if not installed yet, so certbot edits survive)
#      and disables Nginx's default site.
#   7. Installs a nightly backup cron job and reports what is still missing.
# The instance's IAM role is used as is (deploys download releases from S3 with it).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=common.sh
. "$SCRIPT_DIR/common.sh"

require_root
for host in "$@"; do
  [[ "$host" =~ ^[A-Za-z0-9.-]+$ ]] || die "invalid host name: $host"
done

# --- 1. Packages ---------------------------------------------------------------------------
missing_pkgs=()
python3 -c 'import ensurepip' >/dev/null 2>&1 || missing_pkgs+=(python3-venv)
command -v nginx >/dev/null 2>&1 || missing_pkgs+=(nginx)
command -v curl >/dev/null 2>&1 || missing_pkgs+=(curl)
if [ "${#missing_pkgs[@]}" -gt 0 ]; then
  log "Installing: ${missing_pkgs[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q "${missing_pkgs[@]}"
fi
log "Python for releases: $(python3 --version)"

# --- 2. Application user -------------------------------------------------------------------
if id -u "$APP_USER" >/dev/null 2>&1; then
  log "User $APP_USER already exists."
else
  useradd --system --user-group --home-dir /var/lib/bitetracker --no-create-home \
    --shell /usr/sbin/nologin "$APP_USER"
  log "Created system user $APP_USER."
fi

# --- 3. Environment file (secrets) ----------------------------------------------------------
install -d -o root -g "$APP_USER" -m 750 "$(dirname "$ENV_FILE")"
if [ -e "$ENV_FILE" ]; then
  log "Keeping existing $ENV_FILE (not modified)."
  [ $# -eq 0 ] || log "NOTE: host arguments are ignored because $ENV_FILE exists; edit DJANGO_ALLOWED_HOSTS there."
else
  # Instance metadata (IMDSv2) for the public IP and DNS name.
  token="$(curl -fsS -m 5 -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 300' 2>/dev/null || true)"
  metadata() {
    [ -n "$token" ] || return 0
    curl -fsS -m 5 -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null || true
  }
  hosts=()
  public_ip="$(metadata public-ipv4)"
  public_dns="$(metadata public-hostname)"
  [ -n "$public_ip" ] && hosts+=("$public_ip")
  [ -n "$public_dns" ] && hosts+=("$public_dns")
  hosts+=("$@")
  [ "${#hosts[@]}" -gt 0 ] || die "no public IP or DNS name found; pass the host name(s) as arguments."
  allowed="$(IFS=,; echo "${hosts[*]}")"
  origins="$(printf 'http://%s,' "${hosts[@]}")"
  origins="${origins%,}"

  tmp="$(mktemp "$ENV_FILE.XXXXXX")"
  # The secret is generated inside Python and written straight to the file: never on a
  # command line, in the SSM output or in a log.
  python3 - "$DEPLOY_DIR/bitetracker.env.example" "$tmp" "$allowed" "$origins" <<'PY'
import secrets
import sys

template, destination, allowed_hosts, csrf_origins = sys.argv[1:5]
values = {
    "DJANGO_DEBUG": "False",
    "DJANGO_SECRET_KEY": secrets.token_urlsafe(50),
    "DJANGO_ALLOWED_HOSTS": allowed_hosts,
    "DJANGO_CSRF_TRUSTED_ORIGINS": csrf_origins,
    # Plain HTTP until a certificate is installed (see deploy/README.md, HTTPS).
    "DJANGO_SECURE_COOKIES": "False",
    "DJANGO_SECURE_SSL_REDIRECT": "False",
    "DJANGO_SECURE_HSTS_SECONDS": "0",
    "BITETRACKER_STATE_DIR": "/var/lib/bitetracker",
}
lines, seen = [], set()
with open(template) as f:
    for line in f:
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in values:
            lines.append(f"{key}={values[key]}\n")
            seen.add(key)
        else:
            lines.append(line)
lines += [f"{key}={value}\n" for key, value in values.items() if key not in seen]
with open(destination, "w") as f:
    f.writelines(lines)
PY
  chown root:"$APP_USER" "$tmp"
  chmod 640 "$tmp"
  mv "$tmp" "$ENV_FILE"
  log "Created $ENV_FILE (new secret key; DJANGO_ALLOWED_HOSTS=$allowed; plain HTTP)."
fi
load_env_file
BACKUP_DIR="${BACKUP_DIR:-/var/backups/bitetracker}"

# --- 4. Directories (existing contents are left alone) -------------------------------------
install -d -o root -g root -m 755 "$APP_ROOT" "$RELEASES_DIR"
install -d -o root -g root -m 700 "$BACKUP_DIR"
for d in "" data data/processed data/raw data/uploads models reports reports/metrics reports/figures; do
  install -d -o "$APP_USER" -g "$APP_USER" -m 750 "$BITETRACKER_STATE_DIR/$d"
done
log "Directories ready: $APP_ROOT, $BITETRACKER_STATE_DIR, $BACKUP_DIR."

# --- 5. systemd service ---------------------------------------------------------------------
install -o root -g root -m 644 "$DEPLOY_DIR/systemd/bitetracker.service" "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable --quiet "$SERVICE_NAME"
log "systemd service $SERVICE_NAME installed and enabled (the first deploy starts it)."

# --- 6. Nginx --------------------------------------------------------------------------------
site=/etc/nginx/sites-available/bitetracker
if [ -e "$site" ]; then
  log "Keeping existing $site (it may contain certbot's HTTPS settings)."
else
  install -o root -g root -m 644 "$DEPLOY_DIR/nginx/bitetracker.conf" "$site"
  log "Installed $site."
fi
ln -sfn "$site" /etc/nginx/sites-enabled/bitetracker
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx
log "Nginx serves BiteTracker on port 80 (502 until the first deploy)."

# --- 7. Nightly backup ---------------------------------------------------------------------------
cat > /etc/cron.d/bitetracker-backup <<'CRON'
# BiteTracker nightly state backup (installed by deploy/scripts/bootstrap.sh).
15 2 * * * root [ -x /opt/bitetracker/current/deploy/scripts/backup.sh ] && /opt/bitetracker/current/deploy/scripts/backup.sh nightly >>/var/log/bitetracker-backup.log 2>&1
CRON
chmod 644 /etc/cron.d/bitetracker-backup

# --- Summary -----------------------------------------------------------------------------------
log "Disk space:"
df -h "$APP_ROOT" | sed 's/^/    /'
log "Memory:"
free -m | sed 's/^/    /'
missing=0
for f in models/final_best_model.joblib models/final_best_model_metadata.json models/feature_columns.json \
         data/processed/school_meals_processed.csv reports/metrics/naive_baseline_metrics.csv \
         reports/metrics/validation_model_selection.csv; do
  if [ -s "$BITETRACKER_STATE_DIR/$f" ]; then
    log "  present: $f"
  else
    log "  MISSING: $f"
    missing=1
  fi
done
if [ "$missing" -eq 1 ]; then
  log "Next: install the production model and data with deploy/scripts/install-artifacts.sh (see deploy/README.md)."
else
  log "Production artifacts are in place. Next: run the Deploy workflow."
fi
log "Bootstrap complete."
