#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
  exec sudo --preserve-env=PATH,MS_VISTA_DEPLOY_DIR "$0" "$@"
fi

DEPLOY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMON_DIR=$(cd "${DEPLOY_DIR}/../azure" && pwd)
TARGET_DIR=$(cd "${MS_VISTA_DEPLOY_DIR:-${DEPLOY_DIR}}" && pwd)
PROFILE=$(basename "${TARGET_DIR}")
[[ ${PROFILE} == oci || ${PROFILE} == oci-micro ]] || {
  echo "Unsupported OCI deployment profile: ${PROFILE}" >&2
  exit 1
}

install -m 0755 "${COMMON_DIR}/ms-vista-watchdog.sh" \
  /usr/local/sbin/ms-vista-watchdog
install -m 0755 "${COMMON_DIR}/verify_latest_backup.sh" \
  /usr/local/sbin/ms-vista-verify-backup

for unit in ms-vista.service ms-vista-watchdog.service ms-vista-backup-verify.service; do
  install -m 0644 "${DEPLOY_DIR}/${unit}" "/etc/systemd/system/${unit}"
  sed -i "s@/deployment/oci@/deployment/${PROFILE}@g" "/etc/systemd/system/${unit}"
done
install -m 0644 "${COMMON_DIR}/ms-vista-watchdog.timer" \
  /etc/systemd/system/ms-vista-watchdog.timer
install -m 0644 "${COMMON_DIR}/ms-vista-backup-verify.timer" \
  /etc/systemd/system/ms-vista-backup-verify.timer

install -d -m 0755 /etc/systemd/system/docker.service.d
install -m 0644 "${COMMON_DIR}/docker-ms-vista-storage.conf" \
  /etc/systemd/system/docker.service.d/ms-vista-storage.conf

systemctl daemon-reload
systemctl enable ms-vista.service
systemctl enable --now ms-vista-watchdog.timer
systemctl enable --now ms-vista-backup-verify.timer
systemctl reset-failed ms-vista-watchdog.service ms-vista-backup-verify.service

echo "Installed and enabled MS-VISTA OCI startup, watchdog, and backup-verification units."
systemctl list-timers ms-vista-watchdog.timer ms-vista-backup-verify.timer --no-pager
