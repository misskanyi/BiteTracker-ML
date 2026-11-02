#!/usr/bin/env bash
# Install the production model, metadata, feature columns, processed CSV and model metrics
# into the state directory from a tarball made on the machine that trained the model.
# These files never come from git or CI.
#
# Usage (as root): bash deploy/scripts/install-artifacts.sh <bitetracker-state.tar.gz>
#
# Only these paths are taken from the tarball (anything else, such as macOS ._ files,
# is ignored and listed); the database and uploads are never touched:
#   models/final_best_model.joblib             (required)
#   models/final_best_model_metadata.json      (required, must not be a synthetic-data model)
#   models/feature_columns.json                (required)
#   data/processed/school_meals_processed.csv  (required)
#   reports/metrics/naive_baseline_metrics.csv      (Model performance page)
#   reports/metrics/validation_model_selection.csv  (Model performance page)
# Existing files are backed up first (backup.sh), then replaced. Safe to run again.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "$SCRIPT_DIR/common.sh"

[ $# -eq 1 ] || die "usage: $0 <bitetracker-state.tar.gz>"
TARBALL="$1"
[ -f "$TARBALL" ] || die "tarball $TARBALL not found."
require_root
load_env_file

REQUIRED=(
  models/final_best_model.joblib
  models/final_best_model_metadata.json
  models/feature_columns.json
  data/processed/school_meals_processed.csv
)
OPTIONAL=(
  reports/metrics/naive_baseline_metrics.csv
  reports/metrics/validation_model_selection.csv
)

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
# GNU tar strips leading "/" and refuses ".." members; never keep the archive's owners or modes.
tar -xzf "$TARBALL" -C "$work" --no-same-owner --no-same-permissions

for f in "${REQUIRED[@]}"; do
  [ -f "$work/$f" ] && [ ! -L "$work/$f" ] && [ -s "$work/$f" ] || die "tarball is missing $f (nothing was installed)."
done

# Same rule as /health/: a production server must never serve a synthetic-data model.
python3 - "$work/models/final_best_model_metadata.json" <<'PY' || die "the model metadata is not from the Food4Education data (nothing was installed)."
import json, sys
meta = json.load(open(sys.argv[1]))
source = meta.get("data_source", "synthetic")
label = str(meta.get("results_label", ""))
print(f"Model: {meta.get('selected_model')} | data_source={source} | label={label} | trained {meta.get('timestamp')}")
sys.exit(0 if source != "synthetic" and "SYNTHETIC" not in label.upper() else 1)
PY

ignored="$(cd "$work" && find . \( -type f -o -type l \) | sed 's|^\./||' | sort | while read -r f; do
  case " ${REQUIRED[*]} ${OPTIONAL[*]} " in *" $f "*) ;; *) echo "$f" ;; esac
done)"
[ -z "$ignored" ] || log "Ignored (not a production artifact): $(echo "$ignored" | tr '\n' ' ')"

if [ -e "$BITETRACKER_STATE_DIR/models/final_best_model.joblib" ]; then
  log "Backing up the current state before replacing the model."
  bash "$SCRIPT_DIR/backup.sh" pre-artifacts | tail -n 1
fi

for d in data/processed models reports/metrics; do
  install -d -o "$APP_USER" -g "$APP_USER" -m 750 "$BITETRACKER_STATE_DIR/$d"
done
for f in "${REQUIRED[@]}" "${OPTIONAL[@]}"; do
  if [ -f "$work/$f" ] && [ ! -L "$work/$f" ]; then
    install -o "$APP_USER" -g "$APP_USER" -m 640 "$work/$f" "$BITETRACKER_STATE_DIR/$f"
    log "Installed $f"
  else
    log "WARNING: $f not in tarball; the Model performance page needs it."
  fi
done
# The app reloads the model when the files change; no restart needed.
log "Artifacts installed in $BITETRACKER_STATE_DIR."
