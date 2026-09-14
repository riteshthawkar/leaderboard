#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATASETS="BLINK,CV-Bench-2D,CV-Bench-3D,MMVP,RealWorldQA,VStarBench,MMSIBench_wo_circular,3DSRBench,VSR_MCQ,SpatialBench,MindCube,OmniSpatial,SAT-Real"
PAPER_JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
PAPER_JUDGE_REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
TRACK3_ROOT="${TRACK3_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/ms-vista-track3}"
ENV_PREFIX="${TRACK3_CONDA_ENV:-$TRACK3_ROOT/conda-env}"
PYTHON_BIN="${TRACK3_PYTHON:-$ENV_PREFIX/bin/python}"
LMUDATA="${LMUDATA:-$TRACK3_ROOT/LMUData}"
CONTRACT_ROOT="${TRACK3_CONTRACT_ROOT:-$TRACK3_ROOT/contracts/paper-aligned-v5}"
PRIVATE_CONTRACT_ROOT="${TRACK3_PRIVATE_CONTRACT_ROOT:-$TRACK3_ROOT/private/contracts/paper-aligned-v5}"
BENCHMARK_VERSION="${TRACK3_BENCHMARK_VERSION:-paper-aligned-v5-2026-07}"

usage() {
  cat <<'EOF'
Usage: run_eval.sh MODEL MODEL_REVISION VLM_ENDPOINTS JUDGE_ENDPOINT

MODEL          Exact model name exposed by every VLM endpoint.
MODEL_REVISION Immutable Hugging Face commit used by the model server.
VLM_ENDPOINTS  One or more comma-separated OpenAI-compatible endpoints.
JUDGE_ENDPOINT OpenAI-compatible endpoint used for MCQ and VQA judging.

Optional environment variables:
  OUT, LMUDATA, TRACK3_ROOT, TRACK3_CONDA_ENV, TRACK3_PYTHON
  TRACK3_CONTRACT_ROOT, TRACK3_PRIVATE_CONTRACT_ROOT, TRACK3_BENCHMARK_VERSION
  VLM_API_KEY, JUDGE_API_KEY, VLM_CONCURRENCY, JUDGE_CONCURRENCY
  CHECKPOINT_EVERY, REQUEST_RETRIES, REQUEST_TIMEOUT, LIMIT
  MAX_TOKENS, CHAT_TEMPLATE_KWARGS_NONCOT, CHAT_TEMPLATE_KWARGS_COT
  SERVER_METADATA, ENDPOINT_START_TIMEOUT
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# != 4 )); then
  usage >&2
  exit 2
fi

MODEL="$1"
MODEL_REVISION="$2"
VLM_ENDPOINTS="$3"
JUDGE_ENDPOINT="$4"
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
"$PYTHON_BIN" -m spatial_harness.build_public_contract \
  --lmudata "$LMUDATA" \
  --output "$CONTRACT_ROOT" \
  --private-ground-truth "$PRIVATE_CONTRACT_ROOT/ground_truth.json" \
  --benchmark-version "$BENCHMARK_VERSION"

runner_args=(
  --model "$MODEL"
  --model-revision "$MODEL_REVISION"
  --endpoints "$VLM_ENDPOINTS"
  --api-key "${VLM_API_KEY:-EMPTY}"
  --lmudata "$LMUDATA"
  --out "$OUT"
  --datasets "$DATASETS"
  --modes main noimage noimgpp
  --prompt-modes noncot cot
  --concurrency "${VLM_CONCURRENCY:-4}"
  --checkpoint-every "${CHECKPOINT_EVERY:-25}"
  --request-retries "${REQUEST_RETRIES:-2}"
  --timeout "${REQUEST_TIMEOUT:-900}"
  --endpoint-start-timeout "${ENDPOINT_START_TIMEOUT:-1800}"
  --chat-template-kwargs-noncot "${CHAT_TEMPLATE_KWARGS_NONCOT:-\{\}}"
  --chat-template-kwargs-cot "${CHAT_TEMPLATE_KWARGS_COT:-\{\}}"
  --server-metadata "${SERVER_METADATA:-\{\}}"
  --max-tokens-noncot "${MAX_TOKENS:-0}"
  --max-tokens-cot "${MAX_TOKENS:-0}"
)
if [[ "${LIMIT:-0}" != "0" ]]; then
  runner_args+=(--limit "$LIMIT")
fi

"$PYTHON_BIN" -m spatial_harness.run_track3_vllm "${runner_args[@]}"
"$PYTHON_BIN" -m spatial_harness.judge_track3 \
  --input "$OUT" \
  --endpoint "$JUDGE_ENDPOINT" \
  --model "$PAPER_JUDGE_MODEL" \
  --model-revision "$PAPER_JUDGE_REVISION" \
  --api-key "${JUDGE_API_KEY:-EMPTY}" \
  --concurrency "${JUDGE_CONCURRENCY:-16}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  --request-retries "${REQUEST_RETRIES:-2}" \
  --timeout "${REQUEST_TIMEOUT:-900}"
"$PYTHON_BIN" -m spatial_harness.package_submission \
  --input "$OUT" \
  --contract "$CONTRACT_ROOT" \
  --output "$OUT/submission_package" \
  --public-model-name "${PUBLIC_MODEL_NAME:-$MODEL}"

printf 'Track-3 audit report: %s/leaderboard.json\n' "$OUT"
printf 'Track-3 upload package: %s/submission_package/track3_artifact_submission.zip\n' "$OUT"
