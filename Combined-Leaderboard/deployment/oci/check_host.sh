#!/usr/bin/env bash
set -Eeuo pipefail

DATA_DIR=${MS_VISTA_DATA_DIR:-/srv/ms-vista/data}
BACKUP_DIR=${MS_VISTA_BACKUP_DIR:-/mnt/ms-vista-backups}
MIN_MEMORY_KIB=${MIN_MEMORY_KIB:-3670016}
MIN_CPUS=${MIN_CPUS:-2}
MIN_FREE_KIB=${MIN_FREE_KIB:-10485760}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

for command in docker findmnt lsblk; do
  command -v "${command}" >/dev/null 2>&1 || fail "Missing command: ${command}"
done

cpus=$(getconf _NPROCESSORS_ONLN)
memory_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
(( cpus >= MIN_CPUS )) || fail "At least ${MIN_CPUS} CPUs are required; found ${cpus}."
(( memory_kib >= MIN_MEMORY_KIB )) || fail "At least ${MIN_MEMORY_KIB} KiB RAM is required; found ${memory_kib}."

[[ -d ${DATA_DIR} ]] || fail "Data directory is missing: ${DATA_DIR}"
[[ -d ${BACKUP_DIR} ]] || fail "Backup directory is missing: ${BACKUP_DIR}"

data_device=$(findmnt -n -o MAJ:MIN --target "${DATA_DIR}")
backup_device=$(findmnt -n -o MAJ:MIN --target "${BACKUP_DIR}")
backup_fstype=$(findmnt -n -o FSTYPE --target "${BACKUP_DIR}")
[[ -n ${data_device} && -n ${backup_device} ]] || fail "Could not resolve data filesystems."
[[ ${data_device} != "${backup_device}" ]] || fail "Data and backup paths must use different filesystems."
[[ ${backup_fstype} == ext4 || ${backup_fstype} == xfs ]] \
  || fail "Backup filesystem must be ext4 or XFS; found ${backup_fstype:-none}."

free_kib=$(df -Pk "${DATA_DIR}" | awk 'NR == 2 {print $4}')
(( free_kib >= MIN_FREE_KIB )) || fail "Data filesystem has less than 10 GiB free."

docker compose version >/dev/null
systemctl is-active --quiet docker || fail "Docker is not active."

echo "OCI host checks passed."
echo "CPUs: ${cpus}"
echo "Memory KiB: ${memory_kib}"
echo "Data filesystem: $(findmnt -n -o SOURCE,FSTYPE --target "${DATA_DIR}")"
echo "Backup filesystem: $(findmnt -n -o SOURCE,FSTYPE --target "${BACKUP_DIR}")"
