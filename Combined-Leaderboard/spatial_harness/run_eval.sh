#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATASETS="BLINK,CV-Bench-2D,CV-Bench-3D,MMVP,RealWorldQA,VStarBench,MMSIBench_wo_circular,3DSRBench,VSR_MCQ,SpatialBench,MindCube,OmniSpatial,SAT-Real"
TRACK3_ROOT="${TRACK3_ROOT:-/share/data/drive_3/track3-v2}"
ENV_PREFIX="${TRACK3_CONDA_ENV:-/share/data/drive_3/conda_envs/track3-v2}"
PYTHON_BIN="${TRACK3_PYTHON:-$ENV_PREFIX/bin/python}"
LMUDATA="${LMUDATA:-$TRACK3_ROOT/LMUData}"

usage() {
  cat <<'EOF'
Usage: run_eval.sh MODEL VLM_ENDPOINTS JUDGE_ENDPOINT [JUDGE_MODEL]

MODEL          Exact model name exposed by every VLM endpoint.
VLM_ENDPOINTS  One or more comma-separated OpenAI-compatible endpoints.
JUDGE_ENDPOINT OpenAI-compatible endpoint used for MCQ and VQA judging.
JUDGE_MODEL    Exact judge model name; defaults to JUDGE_MODEL or MODEL.

Optional environment variables:
  OUT, LMUDATA, TRACK3_ROOT, TRACK3_CONDA_ENV, TRACK3_PYTHON
  VLM_API_KEY, JUDGE_API_KEY, VLM_CONCURRENCY, JUDGE_CONCURRENCY
  CHECKPOINT_EVERY, REQUEST_RETRIES, REQUEST_TIMEOUT, LIMIT
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# < 3 || $# > 4 )); then
  usage >&2
  exit 2
fi

MODEL="$1"
VLM_ENDPOINTS="$2"
JUDGE_ENDPOINT="$3"
JUDGE_MODEL_NAME="${4:-${JUDGE_MODEL:-$MODEL}}"
MODEL_SLUG="${MODEL//\//__}"
OUT="${OUT:-$TRACK3_ROOT/results/$MODEL_SLUG}"

[[ -x "$PYTHON_BIN" ]] || {
  printf 'Track-3 Python is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 2
}

missing=()
IFS=',' read -r -a dataset_names <<<"$DATASETS"
for dataset in "${dataset_names[@]}"; do
  [[ -s "$LMUDATA/$dataset.tsv" ]] || missing+=("$dataset")
done
if (( ${#missing[@]} )); then
  printf 'Track-3 data bundle is incomplete under %s. Missing:\n' "$LMUDATA" >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf 'Run: %s -m spatial_harness.prepare_data --lmudata %s --cache %s/cache\n' \
    "$PYTHON_BIN" "$LMUDATA" "$TRACK3_ROOT" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m spatial_harness.prepare_data \
  --verify-only \
  --lmudata "$LMUDATA" \
  --cache "$TRACK3_ROOT/cache" \
  --datasets "$DATASETS"

runner_args=(
  --model "$MODEL"
  --endpoints "$VLM_ENDPOINTS"
  --api-key "${VLM_API_KEY:-EMPTY}"
  --lmudata "$LMUDATA"
  --out "$OUT"
  --datasets "$DATASETS"
  --modes main noimgpp
  --prompt-modes noncot cot
  --concurrency "${VLM_CONCURRENCY:-4}"
  --checkpoint-every "${CHECKPOINT_EVERY:-25}"
  --request-retries "${REQUEST_RETRIES:-2}"
  --timeout "${REQUEST_TIMEOUT:-900}"
  --max-tokens-noncot 16384
  --max-tokens-cot 16384
)
if [[ "${LIMIT:-0}" != "0" ]]; then
  runner_args+=(--limit "$LIMIT")
fi

"$PYTHON_BIN" -m spatial_harness.run_track3_vllm "${runner_args[@]}"
"$PYTHON_BIN" -m spatial_harness.judge_track3 \
  --input "$OUT" \
  --endpoint "$JUDGE_ENDPOINT" \
  --model "$JUDGE_MODEL_NAME" \
  --api-key "${JUDGE_API_KEY:-EMPTY}" \
  --concurrency "${JUDGE_CONCURRENCY:-16}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  --timeout "${REQUEST_TIMEOUT:-900}"

printf 'Track-3 v2 leaderboard: %s/leaderboard.json\n' "$OUT"