#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
swap_kib=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
if (( swap_kib < 2097148 )); then
  echo "ERROR: The Micro testing profile requires at least 2 GiB of swap." >&2
  exit 1
fi
MIN_MEMORY_KIB=819200 MIN_CPUS=1 "${SCRIPT_DIR}/../oci/check_host.sh"
echo "Micro TESTING profile only; this is not a production capacity check."
