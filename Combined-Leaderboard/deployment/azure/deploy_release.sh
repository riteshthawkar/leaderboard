#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <image-tag>" >&2
  exit 2
fi

TAG=$1
if [[ ! ${TAG} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "Invalid Docker image tag: ${TAG}" >&2
  exit 2
fi

if [[ ${EUID} -ne 0 ]]; then
  exec sudo --preserve-env=PATH "$0" "$@"
fi

DEPLOY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=$(cd "${DEPLOY_DIR}/../.." && pwd)
ENV_FILE="${DEPLOY_DIR}/production.env"
BACKUP_DIR=/srv/ms-vista/deploy-env-backups
MANIFEST_DIR=/srv/ms-vista/deployment-manifests
DEPLOY_LOCK=/run/lock/ms-vista-deployment.lock
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ENV_BACKUP="${BACKUP_DIR}/production.env.${STAMP}"
COMPOSE=(
  docker compose
  --project-directory "${DEPLOY_DIR}"
  --env-file "${ENV_FILE}"
  --file "${DEPLOY_DIR}/compose.yaml"
)

exec 9>"${DEPLOY_LOCK}"
if ! flock -n 9; then
  echo "Another deployment or recovery operation is already running." >&2
  exit 1
fi

if [[ ! -f ${ENV_FILE} ]]; then
  echo "Missing production environment: ${ENV_FILE}" >&2
  exit 1
fi
if [[ $(grep -c '^MS_VISTA_IMAGE_TAG=' "${ENV_FILE}") -ne 1 ]]; then
  echo "Expected exactly one MS_VISTA_IMAGE_TAG entry in ${ENV_FILE}" >&2
  exit 1
fi
if [[ $(grep -c '^MS_VISTA_DOMAIN=' "${ENV_FILE}") -ne 1 ]]; then
  echo "Expected exactly one MS_VISTA_DOMAIN entry in ${ENV_FILE}" >&2
  exit 1
fi

DOMAIN=$(sed -n 's/^MS_VISTA_DOMAIN=//p' "${ENV_FILE}")
if [[ ! ${DOMAIN} =~ ^[A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9]$ ]]; then
  echo "MS_VISTA_DOMAIN must be a hostname without a scheme, path, or port." >&2
  exit 1
fi

REQUIRE_SPATIAL=$(sed -n 's/^REQUIRE_OFFICIAL_SPATIAL=//p' "${ENV_FILE}" | tr '[:upper:]' '[:lower:]')

docker image inspect "ms-vista-api:${TAG}" "ms-vista-frontend:${TAG}" >/dev/null

install -d -m 700 -o root -g root "${BACKUP_DIR}"
install -d -m 700 -o root -g root "${MANIFEST_DIR}"
install -m 600 -o root -g root "${ENV_FILE}" "${ENV_BACKUP}"

OLD_TAG=$(sed -n 's/^MS_VISTA_IMAGE_TAG=//p' "${ENV_FILE}")

wait_for_healthy_containers() {
  local attempt container_id health service
  for attempt in $(seq 1 60); do
    local all_healthy=true
    for service in api frontend caddy; do
      container_id=$("${COMPOSE[@]}" ps -q "${service}")
      if [[ -z ${container_id} ]]; then
        all_healthy=false
        break
      fi
      health=$(docker inspect \
        --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
        "${container_id}")
      if [[ ${health} != healthy ]]; then
        all_healthy=false
        break
      fi
    done
    if [[ ${all_healthy} == true ]]; then
      return 0
    fi
    sleep 3
  done
  echo "Deployment did not become healthy within 180 seconds." >&2
  return 1
}

run_production_smoke() {
  local smoke_args=(
    python3
    "${APP_DIR}/scripts/production_smoke.py"
    --api-url "https://${DOMAIN}"
    --frontend-url "https://${DOMAIN}"
  )
  if [[ ${REQUIRE_SPATIAL} == true ]]; then
    smoke_args+=(--require-spatial)
  fi
  "${smoke_args[@]}"
}

rollback() {
  local exit_code=$?
  local rollback_failed=false
  trap - ERR INT TERM
  set +e
  install -m 600 -o root -g root "${ENV_BACKUP}" "${ENV_FILE}"
  if ! "${COMPOSE[@]}" up -d --no-build --remove-orphans >/dev/null; then
    rollback_failed=true
  elif ! wait_for_healthy_containers; then
    rollback_failed=true
  elif ! run_production_smoke; then
    rollback_failed=true
  fi
  set -e
  if [[ ${rollback_failed} == true ]]; then
    echo "CRITICAL: deployment and automatic rollback both failed." >&2
    exit 1
  fi
  echo "Deployment failed; restored and verified image tag ${OLD_TAG}." >&2
  exit "${exit_code}"
}
trap rollback ERR INT TERM

sed -i "s/^MS_VISTA_IMAGE_TAG=.*/MS_VISTA_IMAGE_TAG=${TAG}/" "${ENV_FILE}"
chown root:root "${ENV_FILE}"
chmod 600 "${ENV_FILE}"
"${COMPOSE[@]}" config -q
"${COMPOSE[@]}" up -d --no-build --remove-orphans

wait_for_healthy_containers
run_production_smoke

trap - ERR INT TERM

API_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "ms-vista-api:${TAG}")
FRONTEND_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "ms-vista-frontend:${TAG}")
SOURCE_CONTROL_AVAILABLE=false
SOURCE_DIRTY=null
if APP_COMMIT=$(git -C "${APP_DIR}" rev-parse HEAD 2>/dev/null); then
  SOURCE_CONTROL_AVAILABLE=true
  SOURCE_DIRTY=false
  if [[ -n $(git -C "${APP_DIR}" status --porcelain) ]]; then
    SOURCE_DIRTY=true
  fi
else
  APP_COMMIT=unknown
fi
MANIFEST="${MANIFEST_DIR}/${TAG}.json"

printf '{\n  "deployed_at_utc": "%s",\n  "tag": "%s",\n  "previous_tag": "%s",\n  "api_image_id": "%s",\n  "frontend_image_id": "%s",\n  "app_commit": "%s",\n  "source_control_available": %s,\n  "source_dirty": %s,\n  "environment_backup": "%s"\n}\n' \
  "${STAMP}" \
  "${TAG}" \
  "${OLD_TAG}" \
  "${API_IMAGE_ID}" \
  "${FRONTEND_IMAGE_ID}" \
  "${APP_COMMIT}" \
  "${SOURCE_CONTROL_AVAILABLE}" \
  "${SOURCE_DIRTY}" \
  "${ENV_BACKUP}" \
  > "${MANIFEST}"
chmod 600 "${MANIFEST}"

find "${BACKUP_DIR}" -maxdepth 1 -type f -name 'production.env.*' -printf '%T@ %p\n' \
  | sort -nr \
  | awk 'NR > 20 {sub(/^[^ ]+ /, ""); print}' \
  | xargs -r rm -f --
find "${MANIFEST_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%T@ %p\n' \
  | sort -nr \
  | awk 'NR > 20 {sub(/^[^ ]+ /, ""); print}' \
  | xargs -r rm -f --

echo "Deployed ${TAG}."
echo "Release manifest: ${MANIFEST}"
"${COMPOSE[@]}" ps
