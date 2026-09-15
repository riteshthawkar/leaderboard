#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEPLOY_DIR=${MS_VISTA_DEPLOY_DIR:-${SCRIPT_DIR}}
ENV_FILE=${MS_VISTA_ENV_FILE:-${DEPLOY_DIR}/production.env}
APP_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

bash "${DEPLOY_DIR}/check_host.sh"
docker compose --project-directory "${DEPLOY_DIR}" --env-file "${ENV_FILE}" \
  --file "${DEPLOY_DIR}/compose.yaml" config --format json \
  | python3 "${APP_DIR}/scripts/check_deployment_capacity.py"
