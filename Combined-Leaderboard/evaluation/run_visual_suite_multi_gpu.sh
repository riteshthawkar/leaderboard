#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNNER="$SCRIPT_DIR/run_visual_suite.sh"

GPU_GROUPS="${GPU_GROUPS:-}"
GPU_IDS="${GPU_IDS:-}"
MODEL_LIST="${MODEL_LIST:-qwen35-9b,internvl35-8b,qwen3-vl-8b,minicpm-v46}"
BASE_PORT="${BASE_PORT:-8011}"
TRACKS="${TRACKS:-all}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv/visual-suite}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/evaluation/results/.cache}"
DATASET_DIR="${DATASET_DIR:-$CACHE_ROOT/visual-intelligence-dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/evaluation/results/visual_suite_bf16}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/_worker_logs}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.84}"
MAX_NUM_SEQS_PER_REPLICA="${MAX_NUM_SEQS_PER_REPLICA:-auto}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
KEEP_MODEL_CACHE="${KEEP_MODEL_CACHE:-0}"
FORCE="${FORCE:-0}"
CONTINUE_ON_MODEL_ERROR="${CONTINUE_ON_MODEL_ERROR:-0}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
MIN_FREE_GPU_MEMORY_MIB="${MIN_FREE_GPU_MEMORY_MIB:-38000}"
MIN_FREE_DISK_GB_PER_MODEL="${MIN_FREE_DISK_GB_PER_MODEL:-32}"
MIN_FREE_DISK_RESERVE_GB="${MIN_FREE_DISK_RESERVE_GB:-64}"
DRY_RUN="${DRY_RUN:-0}"
ANSWER_EXTRACTOR_ENDPOINTS="${ANSWER_EXTRACTOR_ENDPOINTS:-http://127.0.0.1:8035/v1}"

usage() {
  cat <<'EOF'
Launch independent unquantized BF16 visual evaluation workers.

Examples:
  # Two models, each on one 40 GB GPU.
  GPU_GROUPS='0;1' \
  MODEL_LIST=internvl35-8b,minicpm-v46 \
  ANSWER_EXTRACTOR_ENDPOINTS=http://127.0.0.1:8035/v1 \
    bash evaluation/run_visual_suite_multi_gpu.sh

  # Two models, each tensor-parallel across two GPUs.
  GPU_GROUPS='0,1;2,3' \
  MODEL_LIST=internvl35-8b,minicpm-v46 \
  ANSWER_EXTRACTOR_ENDPOINTS=http://127.0.0.1:8035/v1 \
    bash evaluation/run_visual_suite_multi_gpu.sh

Compatibility syntax:
  GPU_IDS=0,1 MODEL_LIST=internvl35-8b,minicpm-v46 ...

`GPU_GROUPS` is semicolon separated. Commas inside a group assign multiple GPUs
to one model using tensor parallelism. `GPU_IDS` retains the legacy one GPU per
model meaning and is used only when `GPU_GROUPS` is not supplied.

All workers use the externally supervised Qwen3-8B endpoint configured through
`ANSWER_EXTRACTOR_ENDPOINTS`; this launcher never starts one extractor per worker.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

normalize() {
  printf '%s' "$1" | tr -d '[:space:]'
}

validate_flag() {
  local name="$1" value="$2"
  [[ "$value" == "0" || "$value" == "1" ]] || die "$name must be 0 or 1."
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

legacy_gpu_groups() {
  local csv="$1" first=1 gpu
  local -a values
  IFS=',' read -r -a values <<<"$csv"
  for gpu in "${values[@]}"; do
    if (( first == 0 )); then
      printf ';'
    fi
    printf '%s' "$gpu"
    first=0
  done
}

main() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    return 0
  fi
  [[ $# -eq 0 ]] || die "Unknown argument: $1. Use --help for supported configuration."

  command -v nvidia-smi >/dev/null || die "nvidia-smi is required."
  [[ -x "$RUNNER" ]] || die "Visual suite runner is not executable: $RUNNER"
  [[ -x "$VENV_DIR/bin/python" && -x "$VENV_DIR/bin/vllm" ]] \
    || die "Shared environment is not ready at $VENV_DIR. Run the single runner once to create it."
  [[ "$VLLM_DTYPE" == "bfloat16" ]] \
    || die "VLLM_DTYPE must remain bfloat16 for the unquantized research profile."
  [[ "$VLLM_KV_CACHE_DTYPE" == "bfloat16" ]] \
    || die "VLLM_KV_CACHE_DTYPE must remain bfloat16 for the unquantized research profile."
  [[ "$BASE_PORT" =~ ^[0-9]+$ ]] && (( BASE_PORT > 0 && BASE_PORT < 65536 )) \
    || die "BASE_PORT must be between 1 and 65535."
  [[ "$STAGGER_SECONDS" =~ ^[0-9]+$ ]] || die "STAGGER_SECONDS must be non-negative."
  [[ "$MIN_FREE_GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] && (( MIN_FREE_GPU_MEMORY_MIB > 0 )) \
    || die "MIN_FREE_GPU_MEMORY_MIB must be positive."
  [[ "$MIN_FREE_DISK_GB_PER_MODEL" =~ ^[0-9]+$ ]] && (( MIN_FREE_DISK_GB_PER_MODEL > 0 )) \
    || die "MIN_FREE_DISK_GB_PER_MODEL must be positive."
  [[ "$MIN_FREE_DISK_RESERVE_GB" =~ ^[0-9]+$ ]] && (( MIN_FREE_DISK_RESERVE_GB > 0 )) \
    || die "MIN_FREE_DISK_RESERVE_GB must be positive."
  validate_flag "KEEP_MODEL_CACHE" "$KEEP_MODEL_CACHE"
  validate_flag "FORCE" "$FORCE"
  validate_flag "CONTINUE_ON_MODEL_ERROR" "$CONTINUE_ON_MODEL_ERROR"
  validate_flag "DRY_RUN" "$DRY_RUN"
  [[ "$ANSWER_EXTRACTOR_ENDPOINTS" =~ ^https?:// ]] \
    || die "ANSWER_EXTRACTOR_ENDPOINTS must be an absolute HTTP(S) URL."

  GPU_GROUPS="$(normalize "$GPU_GROUPS")"
  GPU_IDS="$(normalize "$GPU_IDS")"
  MODEL_LIST="$(normalize "$MODEL_LIST")"
  if [[ -z "$GPU_GROUPS" ]]; then
    [[ -n "$GPU_IDS" ]] || die "Set GPU_GROUPS, for example GPU_GROUPS='0;1'."
    GPU_GROUPS="$(legacy_gpu_groups "$GPU_IDS")"
  elif [[ -n "$GPU_IDS" ]]; then
    die "Set either GPU_GROUPS or GPU_IDS, not both."
  fi

  local -a groups models
  IFS=';' read -r -a groups <<<"$GPU_GROUPS"
  IFS=',' read -r -a models <<<"$MODEL_LIST"
  (( ${#groups[@]} > 0 )) || die "GPU_GROUPS did not contain any assignments."
  (( ${#groups[@]} == ${#models[@]} )) \
    || die "GPU_GROUPS contains ${#groups[@]} groups but MODEL_LIST contains ${#models[@]} models."
  (( BASE_PORT + ${#groups[@]} - 1 < 65536 )) || die "The allocated port range exceeds 65535."

  local free_kib free_gib required_free_gib
  free_kib="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "Could not determine free disk space."
  free_gib=$((free_kib / 1024 / 1024))
  required_free_gib=$((MIN_FREE_DISK_RESERVE_GB + ${#groups[@]} * MIN_FREE_DISK_GB_PER_MODEL))
  (( free_gib >= required_free_gib )) \
    || die "Only ${free_gib} GiB is free; ${#groups[@]} workers require at least ${required_free_gib} GiB (${MIN_FREE_DISK_RESERVE_GB} GiB host reserve plus ${MIN_FREE_DISK_GB_PER_MODEL} GiB per model). Use fewer concurrent workers, free disk space, or set CACHE_ROOT to a larger filesystem."
  printf '  Disk: %s GiB free; required %s GiB (%s GiB host reserve + %s GiB x %s models)\n' \
    "$free_gib" "$required_free_gib" "$MIN_FREE_DISK_RESERVE_GB" \
    "$MIN_FREE_DISK_GB_PER_MODEL" "${#groups[@]}"

  local index group model port gpu details total free tag pid_file existing_pid selected_gpus=","
  local -a group_gpus
  for index in "${!groups[@]}"; do
    group="${groups[$index]}"
    model="${models[$index]}"
    port=$((BASE_PORT + index))
    [[ -n "$group" ]] || die "GPU_GROUPS contains an empty group."
    [[ -n "$model" ]] || die "MODEL_LIST contains an empty model."
    IFS=',' read -r -a group_gpus <<<"$group"
    (( ${#group_gpus[@]} > 0 )) || die "GPU group $group is empty."
    for gpu in "${group_gpus[@]}"; do
      [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU index: $gpu"
      [[ "$selected_gpus" != *",$gpu,"* ]] || die "GPU $gpu appears in more than one worker group."
      selected_gpus+="$gpu,"
      details="$(nvidia-smi -i "$gpu" --query-gpu=name,memory.total,memory.free \
        --format=csv,noheader,nounits 2>/dev/null || true)"
      [[ -n "$details" ]] || die "GPU $gpu does not exist or cannot be queried."
      total="$(printf '%s' "$details" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
      free="$(printf '%s' "$details" | awk -F',' '{gsub(/ /,"",$3); print $3}')"
      [[ "$total" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] \
        || die "Could not parse memory information for GPU $gpu: $details"
      (( total >= 39000 && free >= MIN_FREE_GPU_MEMORY_MIB )) \
        || die "GPU $gpu has ${free}/${total} MiB free; each unquantized worker requires a free 40 GB-class GPU."
    done

    MODELS="$model" DRY_RUN=1 GPU_IDS="$group" \
      TENSOR_PARALLEL_SIZE="${#group_gpus[@]}" PORT="$port" \
      ANSWER_EXTRACTOR_ENDPOINTS="$ANSWER_EXTRACTOR_ENDPOINTS" \
      VENV_DIR="$VENV_DIR" CACHE_ROOT="$CACHE_ROOT" DATASET_DIR="$DATASET_DIR" \
      OUTPUT_ROOT="$OUTPUT_ROOT" VLLM_DTYPE="$VLLM_DTYPE" \
      VLLM_KV_CACHE_DTYPE="$VLLM_KV_CACHE_DTYPE" bash "$RUNNER" >/dev/null
    port_is_available "$port" || die "Port $port is already in use. Set another BASE_PORT."

    tag="$(printf '%s' "$group" | tr ',' '-')"
    pid_file="$LOG_DIR/gpus${tag}_${model}.pid"
    if [[ -f "$pid_file" ]]; then
      existing_pid="$(<"$pid_file")"
      if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        die "$model already appears to be running on GPUs $group as PID $existing_pid."
      fi
    fi
    printf '  GPUs %-7s TP=%-2s port %-5s model %-22s unquantized BF16 (BF16 KV)\n' \
      "$group" "${#group_gpus[@]}" "$port" "$model"
  done

  "$VENV_DIR/bin/python" - <<'PY'
import torch
import vllm

if not torch.cuda.is_available():
    raise SystemExit("The shared environment cannot access CUDA.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The shared environment does not expose native BF16 support.")
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
  for index in "${!groups[@]}"; do
    group="${groups[$index]}"
    model="${models[$index]}"
    port=$((BASE_PORT + index))
    IFS=',' read -r -a group_gpus <<<"$group"
    tag="$(printf '%s' "$group" | tr ',' '-')"
    log="$LOG_DIR/gpus${tag}_${model}.log"
    pid_file="$LOG_DIR/gpus${tag}_${model}.pid"

    nohup env \
      VENV_DIR="$VENV_DIR" CACHE_ROOT="$CACHE_ROOT" DATASET_DIR="$DATASET_DIR" \
      OUTPUT_ROOT="$OUTPUT_ROOT" GPU_IDS="$group" \
      TENSOR_PARALLEL_SIZE="${#group_gpus[@]}" PORT="$port" MODELS="$model" \
      ANSWER_EXTRACTOR_ENDPOINTS="$ANSWER_EXTRACTOR_ENDPOINTS" \
      TRACKS="$TRACKS" VLLM_DTYPE="$VLLM_DTYPE" \
      VLLM_KV_CACHE_DTYPE="$VLLM_KV_CACHE_DTYPE" \
      GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
      MAX_NUM_SEQS_PER_REPLICA="$MAX_NUM_SEQS_PER_REPLICA" \
      PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
      MIN_FREE_GPU_MEMORY_MIB="$MIN_FREE_GPU_MEMORY_MIB" \
      MIN_FREE_DISK_GB="$((MIN_FREE_DISK_RESERVE_GB + MIN_FREE_DISK_GB_PER_MODEL))" \
      KEEP_MODEL_CACHE="$KEEP_MODEL_CACHE" FORCE="$FORCE" \
      CONTINUE_ON_MODEL_ERROR="$CONTINUE_ON_MODEL_ERROR" \
      bash "$RUNNER" >"$log" 2>&1 </dev/null &

    pid=$!
    printf '%s\n' "$pid" >"$pid_file"
    printf 'Started %-22s on GPUs %-7s as PID %s; log: %s\n' "$model" "$group" "$pid" "$log"
    if (( index + 1 < ${#groups[@]} && STAGGER_SECONDS > 0 )); then
      sleep "$STAGGER_SECONDS"
    fi
  done
}

main "$@"
