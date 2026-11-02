#!/usr/bin/env bash
# Run a Django management command in the active release, as the app user, with the
# production environment loaded (secrets are read from the env file, never put on the
# command line).
#
# Usage (as root):
#   bash /opt/bitetracker/current/deploy/scripts/manage.sh import_schools
#   bash /opt/bitetracker/current/deploy/scripts/manage.sh createsuperuser   # interactive: use a Session Manager shell

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "$SCRIPT_DIR/common.sh"

[ $# -ge 1 ] || die "usage: $0 <manage.py command> [args...]"
require_root
load_env_file
RELEASE="$(readlink -f "$CURRENT_LINK" 2>/dev/null || true)"
[ -n "$RELEASE" ] && [ -x "$RELEASE/.venv/bin/python" ] || die "no active release yet; run the first deploy."
manage "$RELEASE" "$@"
