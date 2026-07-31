#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
  exec sudo --preserve-env=PATH "$0" "$@"
fi

DEPLOY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

install -m 0755 "${DEPLOY_DIR}/ms-vista-watchdog.sh" \
  /usr/local/sbin/ms-vista-watchdog
install -m 0755 "${DEPLOY_DIR}/verify_latest_backup.sh" \
  /usr/local/sbin/ms-vista-verify-backup

install -m 0644 "${DEPLOY_DIR}/ms-vista.service" \
  /etc/systemd/system/ms-vista.service
install -m 0644 "${DEPLOY_DIR}/ms-vista-watchdog.service" \
  /etc/systemd/system/ms-vista-watchdog.service
install -m 0644 "${DEPLOY_DIR}/ms-vista-watchdog.timer" \
  /etc/systemd/system/ms-vista-watchdog.timer
install -m 0644 "${DEPLOY_DIR}/ms-vista-backup-verify.service" \
  /etc/systemd/system/ms-vista-backup-verify.service
install -m 0644 "${DEPLOY_DIR}/ms-vista-backup-verify.timer" \
  /etc/systemd/system/ms-vista-backup-verify.timer

install -d -m 0755 /etc/systemd/system/docker.service.d
install -m 0644 "${DEPLOY_DIR}/docker-ms-vista-storage.conf" \
  /etc/systemd/system/docker.service.d/ms-vista-storage.conf

systemctl daemon-reload
systemctl enable ms-vista.service
systemctl enable --now ms-vista-watchdog.timer
systemctl enable --now ms-vista-backup-verify.timer
systemctl reset-failed ms-vista-watchdog.service ms-vista-backup-verify.service

echo "Installed and enabled MS-VISTA startup, watchdog, and backup-verification units."
systemctl list-timers ms-vista-watchdog.timer ms-vista-backup-verify.timer --no-pager
