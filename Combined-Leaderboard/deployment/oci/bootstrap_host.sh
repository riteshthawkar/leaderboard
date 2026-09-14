#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${SCRIPT_DIR}/../azure/bootstrap_host.sh"

cat <<'EOF'
OCI host bootstrap complete.
Before deployment, attach and mount a separate ext4 or XFS block volume at
/mnt/ms-vista-backups, then run ./check_host.sh.
EOF
