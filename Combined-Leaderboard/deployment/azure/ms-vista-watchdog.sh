#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
  exec sudo --preserve-env=PATH,MS_VISTA_DEPLOY_DIR,WATCHDOG_FAILURE_THRESHOLD,WATCHDOG_RECOVERY_COOLDOWN_SECONDS \
    "$0" "$@"
fi

DEPLOY_DIR=${MS_VISTA_DEPLOY_DIR:-/srv/ms-vista/app/deployment/azure}
ENV_FILE="${DEPLOY_DIR}/production.env"
STATE_DIR=/var/lib/ms-vista-watchdog
FAILURE_FILE="${STATE_DIR}/consecutive_failures"
RECOVERY_FILE="${STATE_DIR}/last_recovery_epoch"
DEPLOY_LOCK=/run/lock/ms-vista-deployment.lock
FAILURE_THRESHOLD=${WATCHDOG_FAILURE_THRESHOLD:-3}
RECOVERY_COOLDOWN_SECONDS=${WATCHDOG_RECOVERY_COOLDOWN_SECONDS:-900}
COMPOSE=(
  docker compose
  --project-directory "${DEPLOY_DIR}"
  --env-file "${ENV_FILE}"
  --file "${DEPLOY_DIR}/compose.yaml"
)
SERVICES=(api frontend caddy)
STACK_ERRORS=()

log() {
  printf '%s\n' "$*"
}

write_state() {
  local destination=$1
  local value=$2
  local temporary="${destination}.tmp"
  printf '%s\n' "${value}" >"${temporary}"
  chmod 600 "${temporary}"
  mv -f "${temporary}" "${destination}"
}

read_nonnegative_integer() {
  local path=$1
  local value=0
  if [[ -r ${path} ]]; then
    value=$(<"${path}")
  fi
  if [[ ! ${value} =~ ^[0-9]+$ ]]; then
    value=0
  fi
  printf '%s\n' "${value}"
}

service_health() {
  local service=$1
  local container_id status
  container_id=$("${COMPOSE[@]}" ps -q "${service}" 2>/dev/null || true)
  if [[ -z ${container_id} ]]; then
    STACK_ERRORS+=("${service}:missing")
    return 1
  fi
  status=$(docker inspect \
    --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
    "${container_id}" 2>/dev/null || true)
  if [[ ${status} != healthy ]]; then
    STACK_ERRORS+=("${service}:${status:-inspect-failed}")
    return 1
  fi
  return 0
}

https_health() {
  local domain body
  domain=$(sed -n 's/^MS_VISTA_DOMAIN=//p' "${ENV_FILE}")
  if [[ ! ${domain} =~ ^[A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9]$ ]]; then
    STACK_ERRORS+=("https:invalid-domain")
    return 1
  fi
  if ! body=$(curl \
    --fail \
    --silent \
    --show-error \
    --max-time 10 \
    --resolve "${domain}:443:127.0.0.1" \
    "https://${domain}/api/health/live"); then
    STACK_ERRORS+=("https:liveness")
    return 1
  fi
  if ! python3 -c \
    'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("status") == "alive" else 1)' \
    <<<"${body}"; then
    STACK_ERRORS+=("https:invalid-liveness-payload")
    return 1
  fi
  if ! curl \
    --fail \
    --silent \
    --show-error \
    --output /dev/null \
    --max-time 10 \
    --resolve "${domain}:443:127.0.0.1" \
    "https://${domain}/"; then
    STACK_ERRORS+=("https:frontend")
    return 1
  fi
  return 0
}

stack_is_healthy() {
  local healthy=true service
  STACK_ERRORS=()
  for service in "${SERVICES[@]}"; do
    if ! service_health "${service}"; then
      healthy=false
    fi
  done
  if ! https_health; then
    healthy=false
  fi
  [[ ${healthy} == true ]]
}

if [[ ! ${FAILURE_THRESHOLD} =~ ^[1-9][0-9]*$ ]]; then
  echo "WATCHDOG_FAILURE_THRESHOLD must be a positive integer." >&2
  exit 2
fi
if [[ ! ${RECOVERY_COOLDOWN_SECONDS} =~ ^[0-9]+$ ]]; then
  echo "WATCHDOG_RECOVERY_COOLDOWN_SECONDS must be a non-negative integer." >&2
  exit 2
fi
if [[ ! -f ${ENV_FILE} ]]; then
  echo "Missing production environment: ${ENV_FILE}" >&2
  exit 1
fi

install -d -m 700 -o root -g root "${STATE_DIR}"
exec 9>"${DEPLOY_LOCK}"
if ! flock -n 9; then
  log "Skipped health check because a deployment or recovery operation is active."
  exit 0
fi

if stack_is_healthy; then
  write_state "${FAILURE_FILE}" 0
  exit 0
fi

failures=$(read_nonnegative_integer "${FAILURE_FILE}")
failures=$((failures + 1))
write_state "${FAILURE_FILE}" "${failures}"
log "Stack health failure ${failures}/${FAILURE_THRESHOLD}: ${STACK_ERRORS[*]}"

if (( failures < FAILURE_THRESHOLD )); then
  exit 0
fi

now=$(date +%s)
last_recovery=$(read_nonnegative_integer "${RECOVERY_FILE}")
if (( now - last_recovery < RECOVERY_COOLDOWN_SECONDS )); then
  remaining=$((RECOVERY_COOLDOWN_SECONDS - (now - last_recovery)))
  log "Recovery is in cooldown for another ${remaining} seconds."
  exit 0
fi
write_state "${RECOVERY_FILE}" "${now}"

log "Attempting a bounded recovery after ${failures} consecutive failures."
unhealthy_services=()
for service in "${SERVICES[@]}"; do
  container_id=$("${COMPOSE[@]}" ps -q "${service}" 2>/dev/null || true)
  status=""
  if [[ -n ${container_id} ]]; then
    status=$(docker inspect \
      --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
      "${container_id}" 2>/dev/null || true)
  fi
  if [[ ${status} != healthy ]]; then
    unhealthy_services+=("${service}")
  fi
done

"${COMPOSE[@]}" up -d --no-build --remove-orphans

if (( ${#unhealthy_services[@]} > 0 )); then
  # A stopped or missing container may recover from `compose up` alone. Give
  # that path a short window before restarting only the components still bad.
  for _attempt in $(seq 1 6); do
    still_unhealthy=()
    for service in "${unhealthy_services[@]}"; do
      container_id=$("${COMPOSE[@]}" ps -q "${service}" 2>/dev/null || true)
      status=""
      if [[ -n ${container_id} ]]; then
        status=$(docker inspect \
          --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
          "${container_id}" 2>/dev/null || true)
      fi
      if [[ ${status} != healthy ]]; then
        still_unhealthy+=("${service}")
      fi
    done
    unhealthy_services=("${still_unhealthy[@]}")
    if (( ${#unhealthy_services[@]} == 0 )); then
      break
    fi
    sleep 5
  done
  if (( ${#unhealthy_services[@]} > 0 )); then
    "${COMPOSE[@]}" restart "${unhealthy_services[@]}"
  fi
else
  # Internal checks passed but the HTTPS path failed repeatedly.
  "${COMPOSE[@]}" restart caddy
fi

for _attempt in $(seq 1 24); do
  if stack_is_healthy; then
    write_state "${FAILURE_FILE}" 0
    log "Stack recovery succeeded."
    exit 0
  fi
  sleep 5
done

log "Stack recovery failed: ${STACK_ERRORS[*]}"
exit 1
