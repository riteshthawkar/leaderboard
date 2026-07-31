#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TRACK3_ROOT="${TRACK3_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/ms-vista-track3}"
TRACK3_SOURCE_ROOT="${TRACK3_SOURCE_ROOT:-$TRACK3_ROOT}"
TRACK3_RUN_ROOT="${TRACK3_RUN_ROOT:-$TRACK3_ROOT/runs/paper-aligned-v5}"
CLIENT_PYTHON="${TRACK3_PYTHON:-$TRACK3_SOURCE_ROOT/conda-env/bin/python}"
LMUDATA="${LMUDATA:-$TRACK3_SOURCE_ROOT/LMUData}"
CONTRACT_ROOT="${TRACK3_CONTRACT_ROOT:-$TRACK3_RUN_ROOT/contracts/paper-aligned-v5}"
PRIVATE_CONTRACT_ROOT="${TRACK3_PRIVATE_CONTRACT_ROOT:-$TRACK3_RUN_ROOT/private/contracts/paper-aligned-v5}"
BENCHMARK_VERSION="${TRACK3_BENCHMARK_VERSION:-paper-aligned-v5-2026-07}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SLUGS="${TRACK3_MODEL_SLUGS:-OpenGVLab__InternVL3_5-8B,Qwen__Qwen3.6-27B}"

mkdir -p "$TRACK3_RUN_ROOT"/{logs,private}
exec 9>"$TRACK3_RUN_ROOT/package-queue.lock"
if ! flock -n 9; then
  printf 'Another package watcher already holds %s/package-queue.lock\n' \
    "$TRACK3_RUN_ROOT" >&2
  exit 2
fi
exec >>"$TRACK3_RUN_ROOT/logs/package-watcher.log" 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

build_contract_once() {
  if [[ -s "$CONTRACT_ROOT/manifest.json" \
      && -s "$CONTRACT_ROOT/questions.jsonl" \
      && -s "$CONTRACT_ROOT/submission_template.jsonl" ]]; then
    return
  fi
  log "Building the public Track-3 contract"
  nice -n 15 "$CLIENT_PYTHON" -m spatial_harness.build_public_contract \
    --lmudata "$LMUDATA" \
    --output "$CONTRACT_ROOT" \
    --private-ground-truth "$PRIVATE_CONTRACT_ROOT/ground_truth.json" \
    --benchmark-version "$BENCHMARK_VERSION"
}

package_ready_model() {
  local slug="$1"
  local result_root="$TRACK3_RUN_ROOT/results/$slug"
  local package="$result_root/submission_package/spatial_reasoning_submission.zip"
  if [[ -s "$package" ]]; then
    return 0
  fi
  if [[ ! -s "$result_root/judged.jsonl" || ! -s "$result_root/leaderboard.json" ]]; then
    return 1
  fi
  build_contract_once
  log "Packaging completed Track-3 run for $slug"
  nice -n 15 "$CLIENT_PYTHON" -m spatial_harness.package_submission \
    --input "$result_root" \
    --contract "$CONTRACT_ROOT" \
    --output "$result_root/submission_package"
  log "Package ready for $slug: $package"
}

run_prespecified_analysis() {
  local output="$TRACK3_RUN_ROOT/analysis"
  if [[ -s "$output/summary.json" ]]; then
    return
  fi
  log "Running the frozen 11-test Track-3 confirmatory analysis"
  nice -n 15 "$CLIENT_PYTHON" analysis/research/analyze_track3_results.py \
    --model "internvl35-8b=$TRACK3_RUN_ROOT/results/OpenGVLab__InternVL3_5-8B/judged.jsonl" \
    --model "qwen36-27b=$TRACK3_RUN_ROOT/results/Qwen__Qwen3.6-27B/judged.jsonl" \
    --output "$output" \
    --bootstrap-reps 5000 \
    --permutation-reps 20000 \
    --seed 20260723
  log "Prespecified Track-3 analysis complete: $output"
}

[[ -x "$CLIENT_PYTHON" ]] || {
  printf 'Track-3 Python is not executable: %s\n' "$CLIENT_PYTHON" >&2
  exit 2
}
cd "$PROJECT_ROOT"
IFS=',' read -r -a model_slugs <<<"$SLUGS"
log "Watching ${#model_slugs[@]} Track-3 model runs for package readiness"

while true; do
  complete=0
  for slug in "${model_slugs[@]}"; do
    if package_ready_model "$slug"; then
      complete=$((complete + 1))
    fi
  done
  if (( complete == ${#model_slugs[@]} )); then
    run_prespecified_analysis
    log "All Track-3 packages are ready"
    exit 0
  fi
  if [[ -f "$TRACK3_RUN_ROOT/status.tsv" \
      && "$(cut -f2 "$TRACK3_RUN_ROOT/status.tsv")" == "failed" ]]; then
    log "Track-3 queue failed before all packages were available"
    exit 1
  fi
  sleep "$POLL_SECONDS"
done
