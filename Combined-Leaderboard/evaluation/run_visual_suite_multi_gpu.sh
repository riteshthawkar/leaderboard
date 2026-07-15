#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNNER="$SCRIPT_DIR/run_visual_suite.sh"

GPU_IDS="${GPU_IDS:-}"
MODEL_LIST="${MODEL_LIST:-qwen35-9b,internvl35-8b,qwen3-vl-8b,minicpm-v46}"
BASE_PORT="${BASE_PORT:-8011}"
TRACKS="${TRACKS:-all}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv/visual-suite}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/evaluation/results/.cache}"
DATASET_DIR="${DATASET_DIR:-$CACHE_ROOT/visual-intelligence-dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/evaluation/results/visual_suite}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/_worker_logs}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
KEEP_MODEL_CACHE="${KEEP_MODEL_CACHE:-1}"
FORCE="${FORCE:-0}"
CONTINUE_ON_MODEL_ERROR="${CONTINUE_ON_MODEL_ERROR:-0}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
MIN_FREE_GPU_MEMORY_MIB="${MIN_FREE_GPU_MEMORY_MIB:-22000}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Launch one visual-suite model on each explicitly selected physical GPU.

Usage:
  GPU_IDS=1,3,5,7 bash evaluation/run_visual_suite_multi_gpu.sh

Optional overrides:
  MODEL_LIST=qwen35-9b,internvl35-8b,qwen3-vl-8b,minicpm-v46
  BASE_PORT=8011
  TRACKS=all
  STAGGER_SECONDS=20
  DRY_RUN=1

GPU_IDS and MODEL_LIST must contain the same number of comma-separated values.
The launcher uses one shared Python environment, dataset cache, and output root.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

normalize_csv() {
  printf '%s' "$1" | tr -d '[:space:]'
}

port_is_available() {
  local port="$1"
  "$VENV_DIR/bin/python" - "$port" <<'PY'
import socket
import sys

with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}

validate_flag() {
  local name="$1" value="$2"
  [[ "$value" == "0" || "$value" == "1" ]] || die "$name must be 0 or 1."
}

main() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    return 0
  fi
  [[ $# -eq 0 ]] || die "Unknown argument: $1. Use --help for supported configuration."

  command -v nvidia-smi >/dev/null || die "nvidia-smi is required."
  [[ -x "$RUNNER" ]] || die "Single-GPU runner is not executable: $RUNNER"
  [[ -x "$VENV_DIR/bin/python" && -x "$VENV_DIR/bin/vllm" ]] \
    || die "Shared environment is not ready at $VENV_DIR. Complete the common setup first."
  [[ -n "$GPU_IDS" ]] || die "GPU_IDS is required, for example GPU_IDS=1,3,5,7."
  [[ "$BASE_PORT" =~ ^[0-9]+$ ]] && (( BASE_PORT > 0 && BASE_PORT < 65536 )) \
    || die "BASE_PORT must be between 1 and 65535."
  [[ "$STAGGER_SECONDS" =~ ^[0-9]+$ ]] || die "STAGGER_SECONDS must be a non-negative integer."
  [[ "$MIN_FREE_GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] && (( MIN_FREE_GPU_MEMORY_MIB > 0 )) \
    || die "MIN_FREE_GPU_MEMORY_MIB must be positive."
  validate_flag "KEEP_MODEL_CACHE" "$KEEP_MODEL_CACHE"
  validate_flag "FORCE" "$FORCE"
  validate_flag "CONTINUE_ON_MODEL_ERROR" "$CONTINUE_ON_MODEL_ERROR"
  validate_flag "DRY_RUN" "$DRY_RUN"

  GPU_IDS="$(normalize_csv "$GPU_IDS")"
  MODEL_LIST="$(normalize_csv "$MODEL_LIST")"

  local -a gpus models
  IFS=',' read -r -a gpus <<<"$GPU_IDS"
  IFS=',' read -r -a models <<<"$MODEL_LIST"

  (( ${#gpus[@]} > 0 )) || die "GPU_IDS did not contain any GPU indices."
  (( ${#gpus[@]} == ${#models[@]} )) \
    || die "GPU_IDS contains ${#gpus[@]} values but MODEL_LIST contains ${#models[@]}."
  (( BASE_PORT + ${#gpus[@]} - 1 < 65536 )) || die "The allocated port range exceeds 65535."

  local index gpu model port details total_memory free_memory pid_file existing_pid
  local -A selected_gpus=()
  for index in "${!gpus[@]}"; do
    gpu="${gpus[$index]}"
    model="${models[$index]}"
    port=$((BASE_PORT + index))

    [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU index: $gpu"
    [[ -z "${selected_gpus[$gpu]:-}" ]] || die "GPU $gpu was selected more than once."
    selected_gpus[$gpu]=1
    [[ -n "$model" ]] || die "MODEL_LIST contains an empty model value."

    if ! details="$(nvidia-smi -i "$gpu" \
      --query-gpu=name,memory.total,memory.free \
      --format=csv,noheader,nounits 2>/dev/null)"; then
      die "GPU $gpu does not exist or cannot be queried."
    fi
    total_memory="${details#*, }"
    total_memory="${total_memory%%, *}"
    free_memory="${details##*, }"
    [[ "$total_memory" =~ ^[0-9]+$ && "$free_memory" =~ ^[0-9]+$ ]] \
      || die "Could not parse memory information for GPU $gpu: $details"
    (( free_memory >= MIN_FREE_GPU_MEMORY_MIB )) \
      || die "GPU $gpu has only ${free_memory}/${total_memory} MiB free; choose a free GPU."
    MODELS="$model" DRY_RUN=1 GPU_ID="$gpu" PORT="$port" \
      VENV_DIR="$VENV_DIR" CACHE_ROOT="$CACHE_ROOT" DATASET_DIR="$DATASET_DIR" \
      OUTPUT_ROOT="$OUTPUT_ROOT" bash "$RUNNER" >/dev/null
    port_is_available "$port" || die "Port $port is already in use. Set a different BASE_PORT."

    pid_file="$LOG_DIR/gpu${gpu}_${model}.pid"
    if [[ -f "$pid_file" ]]; then
      existing_pid="$(<"$pid_file")"
      if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        die "$model already appears to be running on GPU $gpu as PID $existing_pid."
      fi
    fi

    printf '  GPU %-3s port %-5s model %-22s %s\n' "$gpu" "$port" "$model" "$details"
  done

  "$VENV_DIR/bin/python" - <<'PY'
import torch
import vllm

if not torch.cuda.is_available():
    raise SystemExit("The shared environment cannot access CUDA.")
print(
    f"Shared environment ready: vLLM {vllm.__version__}, "
    f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}"
)
PY

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry run complete; no workers were started.\n'
    return 0
  fi

  mkdir -p "$LOG_DIR"
  cd "$PROJECT_ROOT"

  local log pid
  for index in "${!gpus[@]}"; do
    gpu="${gpus[$index]}"
    model="${models[$index]}"
    port=$((BASE_PORT + index))
    log="$LOG_DIR/gpu${gpu}_${model}.log"
    pid_file="$LOG_DIR/gpu${gpu}_${model}.pid"

    nohup env \
      VENV_DIR="$VENV_DIR" \
      CACHE_ROOT="$CACHE_ROOT" \
      DATASET_DIR="$DATASET_DIR" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      GPU_ID="$gpu" \
      PORT="$port" \
      MODELS="$model" \
      TRACKS="$TRACKS" \
      VLLM_DTYPE="$VLLM_DTYPE" \
      GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
      KEEP_MODEL_CACHE="$KEEP_MODEL_CACHE" \
      FORCE="$FORCE" \
      CONTINUE_ON_MODEL_ERROR="$CONTINUE_ON_MODEL_ERROR" \
      bash "$RUNNER" >"$log" 2>&1 </dev/null &

    pid=$!
    printf '%s\n' "$pid" >"$pid_file"
    printf 'Started %-22s on GPU %s as PID %s; log: %s\n' "$model" "$gpu" "$pid" "$log"

    if (( index + 1 < ${#gpus[@]} && STAGGER_SECONDS > 0 )); then
      sleep "$STAGGER_SECONDS"
    fi
  done
}

main "$@"
