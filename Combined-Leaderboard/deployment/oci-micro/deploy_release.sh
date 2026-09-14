#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MS_VISTA_DEPLOY_DIR="${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/../azure/deploy_release.sh" "$@"
