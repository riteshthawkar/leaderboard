#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

GPU_ID="${GPU_ID:-2}"
RESEARCH_DATA_ROOT="${MS_VISTA_RESEARCH_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/ms-vista-research}"
CURRENT_ANALYSIS_ROOT="${CURRENT_ANALYSIS_ROOT:-$RESEARCH_DATA_ROOT/analysis-v2}"
ABLATION_ROOT="${ABLATION_ROOT:-$RESEARCH_DATA_ROOT/qwen35-thinking-v1}"
POLL_SECONDS="${POLL_SECONDS:-120}"

mkdir -p "$ABLATION_ROOT"/{logs,status}
exec 8>"$ABLATION_ROOT/followup-watcher.lock"
if ! flock -n 8; then
  printf 'A Qwen3.5 follow-up watcher is already active.\n' >&2
  exit 2
fi
exec >>"$ABLATION_ROOT/logs/followup-watcher.log" 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

log "Waiting for the controlled analysis queue to release GPU $GPU_ID"
while true; do
  if flock -n "$CURRENT_ANALYSIS_ROOT/analysis-queue.lock" -c true; then
    phase=""
    if [[ -f "$CURRENT_ANALYSIS_ROOT/status/current.tsv" ]]; then
      phase="$(cut -f2 "$CURRENT_ANALYSIS_ROOT/status/current.tsv")"
    fi
    case "$phase" in
      complete)
        log "Controlled analysis completed; starting Qwen3.5 factorial ablation"
        break
        ;;
      failed|stopped)
        log "Controlled analysis ended in phase $phase; follow-up will not start"
        exit 3
        ;;
      *)
        log "Analysis lock is free but completion is not recorded; phase=${phase:-missing}"
        ;;
    esac
  fi
  printf '%s\twaiting_for_analysis\tgpu=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$GPU_ID" \
    >"$ABLATION_ROOT/status/followup.tsv"
  sleep "$POLL_SECONDS"
done

cd "$PROJECT_ROOT"
exec env \
  GPU_ID="$GPU_ID" \
  ABLATION_ROOT="$ABLATION_ROOT" \
  bash evaluation/research/run_qwen35_thinking_ablation.sh
