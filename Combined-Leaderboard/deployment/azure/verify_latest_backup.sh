#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
  exec sudo --preserve-env=PATH,MS_VISTA_DEPLOY_DIR,MS_VISTA_BACKUP_DIR,BACKUP_VERIFY_ALLOWED_FSTYPES,BACKUP_VERIFY_MAX_AGE_HOURS \
    "$0" "$@"
fi

DEPLOY_DIR=${MS_VISTA_DEPLOY_DIR:-/srv/ms-vista/app/deployment/azure}
ENV_FILE="${DEPLOY_DIR}/production.env"
BACKUP_DIR=${MS_VISTA_BACKUP_DIR:-/mnt/ms-vista-backups}
MAX_AGE_HOURS=${BACKUP_VERIFY_MAX_AGE_HOURS:-60}
DEPLOY_LOCK=/run/lock/ms-vista-deployment.lock
COMPOSE=(
  docker compose
  --project-directory "${DEPLOY_DIR}"
  --env-file "${ENV_FILE}"
  --file "${DEPLOY_DIR}/compose.yaml"
)

log() {
  printf '%s\n' "$*"
}

if [[ ! ${MAX_AGE_HOURS} =~ ^[1-9][0-9]*$ ]]; then
  echo "BACKUP_VERIFY_MAX_AGE_HOURS must be a positive integer." >&2
  exit 2
fi
if [[ ! -f ${ENV_FILE} ]]; then
  echo "Missing production environment: ${ENV_FILE}" >&2
  exit 1
fi

ALLOWED_FSTYPES=${BACKUP_VERIFY_ALLOWED_FSTYPES:-}
if [[ -z ${ALLOWED_FSTYPES} ]]; then
  ALLOWED_FSTYPES=$(sed -n 's/^BACKUP_VERIFY_ALLOWED_FSTYPES=//p' "${ENV_FILE}")
fi
ALLOWED_FSTYPES=${ALLOWED_FSTYPES:-cifs}
if [[ ! ${ALLOWED_FSTYPES} =~ ^[A-Za-z0-9._-]+(,[A-Za-z0-9._-]+)*$ ]]; then
  echo "BACKUP_VERIFY_ALLOWED_FSTYPES must be a comma-separated filesystem list." >&2
  exit 2
fi

exec 9>"${DEPLOY_LOCK}"
if ! flock -n 9; then
  log "Skipped backup verification because a deployment or recovery operation is active."
  exit 0
fi

# Accessing the directory activates remote or systemd-managed mounts.
if ! find "${BACKUP_DIR}" -maxdepth 0 -type d >/dev/null 2>&1; then
  log "Off-VM backup directory is unavailable."
  exit 1
fi
filesystems=$(findmnt -n -o FSTYPE --target "${BACKUP_DIR}" 2>/dev/null || true)
filesystem_allowed=false
IFS=',' read -r -a allowed_filesystems <<<"${ALLOWED_FSTYPES}"
for allowed_filesystem in "${allowed_filesystems[@]}"; do
  if grep -Fxq "${allowed_filesystem}" <<<"${filesystems}"; then
    filesystem_allowed=true
    break
  fi
done
if [[ ${filesystem_allowed} != true ]]; then
  filesystem_summary=$(tr '\n' ',' <<<"${filesystems}" | sed 's/,$//')
  log "Backup directory uses an unexpected filesystem (actual=${filesystem_summary:-none}, allowed=${ALLOWED_FSTYPES})."
  exit 1
fi

latest=$(find "${BACKUP_DIR}" -maxdepth 1 -type f -name 'ms-vista-backup-*.zip' \
  -printf '%T@ %f\n' | sort -nr | awk 'NR == 1 {print $2}')
if [[ -z ${latest} ]]; then
  log "No mirrored backup archive exists."
  exit 1
fi

modified_epoch=$(stat -c %Y "${BACKUP_DIR}/${latest}")
now=$(date +%s)
max_age_seconds=$((MAX_AGE_HOURS * 3600))
if (( now - modified_epoch > max_age_seconds )); then
  log "Latest mirrored backup is older than ${MAX_AGE_HOURS} hours: ${latest}"
  exit 1
fi

api_container=$("${COMPOSE[@]}" ps -q api)
if [[ -z ${api_container} ]]; then
  log "API container is unavailable for the offline restore drill."
  exit 1
fi

archive="/backup/${latest}"
docker exec \
  --env "MS_VISTA_VERIFY_ARCHIVE=${archive}" \
  "${api_container}" \
  sh -eu -c '
    restore_dir=$(mktemp -d /tmp/ms-vista-restore-check.XXXXXX)
    trap "rm -rf \"${restore_dir}\"" EXIT
    python -m backend.backup_cli verify "${MS_VISTA_VERIFY_ARCHIVE}" >/dev/null
    python -m backend.backup_cli restore "${MS_VISTA_VERIFY_ARCHIVE}" \
      --destination "${restore_dir}" >/dev/null
  '

log "Verified and restored the latest off-VM backup: ${latest}"
