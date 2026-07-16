#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VLLM_VERSION="0.25.1"
OPENAI_VERSION="2.45.0"
HUGGINGFACE_HUB_VERSION="1.23.0"
PILLOW_VERSION="12.3.0"
UV_VERSION="0.11.28"
DATASET_REPO_ID="amolharsh/visual-intelligence-leaderboard"
DATASET_REVISION="cc41be90e74679a9d3c9dd295834b2cee9100b9d"

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv/visual-suite}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/evaluation/results/visual_suite_bf16}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/evaluation/results/.cache}"
DATASET_DIR="${DATASET_DIR:-$CACHE_ROOT/visual-intelligence-dataset}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-auto}"
PORT="${PORT:-8011}"
MODELS="${MODELS:-all}"
TRACKS="${TRACKS:-all}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-20}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SETUP_ONLY="${SETUP_ONLY:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_MODEL_CACHE="${KEEP_MODEL_CACHE:-0}"
CONTINUE_ON_MODEL_ERROR="${CONTINUE_ON_MODEL_ERROR:-1}"
MAX_EVAL_ATTEMPTS="${MAX_EVAL_ATTEMPTS:-3}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-600}"
MODEL_START_TIMEOUT_SECONDS="${MODEL_START_TIMEOUT_SECONDS:-7200}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
CONCURRENCY="${CONCURRENCY:-1}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-96}"
MIN_SYSTEM_RAM_GB="${MIN_SYSTEM_RAM_GB:-28}"
MIN_FREE_GPU_MEMORY_MIB="${MIN_FREE_GPU_MEMORY_MIB:-38000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-bfloat16}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
BASE_SEED="${BASE_SEED:-0}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:--1}"
SAMPLING_MIN_P="${SAMPLING_MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

# Primary benchmark profiles reconstructed from the papers and released code.
DYS_PROMPT_MODE="${DYS_PROMPT_MODE:-noncot}"
DYS_MAX_TOKENS="${DYS_MAX_TOKENS:-200}"
DYS_TEMPERATURE="${DYS_TEMPERATURE:-1.0}"
DYS_TOP_P="${DYS_TOP_P:-0.95}"
MINDS_EYE_PROMPT_MODE="${MINDS_EYE_PROMPT_MODE:-cot}"
MINDS_EYE_MAX_TOKENS="${MINDS_EYE_MAX_TOKENS:-1000}"
MINDS_EYE_TEMPERATURE="${MINDS_EYE_TEMPERATURE:-0.1}"
MINDS_EYE_TOP_P="${MINDS_EYE_TOP_P:-1.0}"

VLLM_STACKED_WEIGHT_PATCH_ID="vllm-0.25.1-stacked-weight-single-match-v1"
UNPARSEABLE_ANSWER_POLICY_ID="retry-with-deterministic-seed-sequence-then-fail-v1"
PIPELINE_REVISION_ID="unquantized-bf16-paper-aligned-protocol-v2"

# slug|repository|revision|weight loading|max context
MODEL_SPECS=(
  'qwen35-9b|Qwen/Qwen3.5-9B|c202236235762e1c871ad0ccb60c8ee5ba337b9a|unquantized|32768'
  'internvl35-8b|OpenGVLab/InternVL3_5-8B|9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79|unquantized|32768'
  'glm46v-flash|zai-org/GLM-4.6V-Flash|411bb4d77144a3f03accbf4b780f5acb8b7cde4e|unquantized|32768'
  'minicpm-v46|openbmb/MiniCPM-V-4.6|8169864629825dc1d755a5aa1cd8b5935dcbc83f|unquantized|32768'
  'qwen25-vl-7b|Qwen/Qwen2.5-VL-7B-Instruct|cc594898137f460bfe9f0759e9844b3ce807cfb5|unquantized|32768'
  'qwen3-vl-8b|Qwen/Qwen3-VL-8B-Instruct|0c351dd01ed87e9c1b53cbc748cba10e6187ff3b|unquantized|32768'
  'phi4-multimodal|microsoft/Phi-4-multimodal-instruct|93f923e1a7727d1c4f446756212d9d3e8fcc5d81|unquantized|32768'
)

SERVER_PID=""
SERVER_OWNS_PROCESS_GROUP=0
SERVER_LOG=""
PYTHON_BIN=""
VLLM_BIN=""
UV_BIN=""
GPU_NAME=""
GPU_COUNT=0
GPU_ID_LIST=()
SUCCESS_MODELS=()
FAILED_MODELS=()
SKIPPED_MODELS=()
RUNNER_ARGS=()

usage() {
  cat <<'EOF'
Run MS-VISTA visual evaluations with original, unquantized checkpoint weights.

Usage:
  bash evaluation/run_visual_suite.sh

Common overrides:
  MODELS=internvl35-8b,minicpm-v46  Run selected model slugs sequentially
  TRACKS=do_you_see_me,minds_eye     Run selected benchmark tracks only
  GPU_IDS=0                          Run a model on one physical GPU
  GPU_IDS=0,1                        Split one model over two physical GPUs
  TENSOR_PARALLEL_SIZE=2             Must match the number of assigned GPUs
  SMOKE_ONLY=1                       Run strict compatibility checks only
  SETUP_ONLY=1                       Prepare the shared environment and dataset, then exit
  FORCE=1                            Replace outputs from an older run contract
  KEEP_MODEL_CACHE=1                 Retain downloaded model weights
  DRY_RUN=1                          Validate and print the resolved plan

The runner uses the original checkpoint tensors with BF16 compute. It never
loads 4-bit or 8-bit weights, resizes benchmark images, or creates a synthetic
answer. Failed or unparseable responses remain in diagnostics and make the run
fail after the configured retry sequence.
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

is_enabled() {
  local requested="${1// /,}"
  local name="$2"
  [[ "$requested" == "all" || ",$requested," == *",$name,"* ]]
}

track_module() {
  case "$1" in
    do_you_see_me) printf '%s\n' 'evaluation.do_you_see_me.run_vllm' ;;
    minds_eye) printf '%s\n' 'evaluation.minds_eye.run_vllm' ;;
    *) return 1 ;;
  esac
}

track_questions() {
  case "$1" in
    do_you_see_me) printf '%s\n' "$PROJECT_ROOT/tasks/do_you_see_me/questions.jsonl" ;;
    minds_eye) printf '%s\n' "$PROJECT_ROOT/tasks/minds_eye/questions.jsonl" ;;
    *) return 1 ;;
  esac
}

track_prompt_mode() {
  case "$1" in
    do_you_see_me) printf '%s\n' "$DYS_PROMPT_MODE" ;;
    minds_eye) printf '%s\n' "$MINDS_EYE_PROMPT_MODE" ;;
    *) return 1 ;;
  esac
}

track_max_tokens() {
  case "$1" in
    do_you_see_me) printf '%s\n' "$DYS_MAX_TOKENS" ;;
    minds_eye) printf '%s\n' "$MINDS_EYE_MAX_TOKENS" ;;
    *) return 1 ;;
  esac
}

track_temperature() {
  case "$1" in
    do_you_see_me) printf '%s\n' "$DYS_TEMPERATURE" ;;
    minds_eye) printf '%s\n' "$MINDS_EYE_TEMPERATURE" ;;
    *) return 1 ;;
  esac
}

track_top_p() {
  case "$1" in
    do_you_see_me) printf '%s\n' "$DYS_TOP_P" ;;
    minds_eye) printf '%s\n' "$MINDS_EYE_TOP_P" ;;
    *) return 1 ;;
  esac
}

track_chat_kwargs() {
  local slug="$1" track="$2"
  if [[ "$slug" == "qwen35-9b" ]]; then
    if [[ "$track" == "do_you_see_me" ]]; then
      printf '%s\n' '{"enable_thinking":false}'
    else
      printf '%s\n' '{"enable_thinking":true}'
    fi
    return 0
  fi
  printf '%s\n' '{}'
}

selected_tracks() {
  local track
  for track in do_you_see_me minds_eye; do
    if is_enabled "$TRACKS" "$track"; then
      printf '%s\n' "$track"
    fi
  done
}

selected_model_count() {
  local count=0 spec slug model_id revision loading model_max_len
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r slug model_id revision loading model_max_len <<<"$spec"
    if is_enabled "$MODELS" "$slug"; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "$count"
}

validate_flag() {
  local name="$1" value="$2"
  [[ "$value" == "0" || "$value" == "1" ]] || die "$name must be 0 or 1."
}

validate_positive_integer() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 )) || die "$name must be positive."
}

validate_probability() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    && awk -v value="$value" 'BEGIN { exit !(value > 0 && value <= 1) }' \
    || die "$name must be greater than 0 and at most 1."
}

initialize_gpu_topology() {
  GPU_IDS="$(printf '%s' "$GPU_IDS" | tr -d '[:space:]')"
  [[ -n "$GPU_IDS" ]] || die "GPU_IDS must contain at least one GPU index."
  IFS=',' read -r -a GPU_ID_LIST <<<"$GPU_IDS"
  GPU_COUNT="${#GPU_ID_LIST[@]}"
  (( GPU_COUNT > 0 )) || die "GPU_IDS must contain at least one GPU index."

  local gpu seen=","
  for gpu in "${GPU_ID_LIST[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU index in GPU_IDS: $gpu"
    [[ "$seen" != *",$gpu,"* ]] || die "GPU $gpu appears more than once in GPU_IDS."
    seen+="$gpu,"
  done

  if [[ "$TENSOR_PARALLEL_SIZE" == "auto" ]]; then
    TENSOR_PARALLEL_SIZE="$GPU_COUNT"
  fi
  [[ "$TENSOR_PARALLEL_SIZE" =~ ^[0-9]+$ ]] && (( TENSOR_PARALLEL_SIZE > 0 )) \
    || die "TENSOR_PARALLEL_SIZE must be auto or a positive integer."
  (( TENSOR_PARALLEL_SIZE == GPU_COUNT )) \
    || die "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE must match the $GPU_COUNT GPU_IDS entries."
}

validate_settings() {
  local penalty_name penalty_value
  command -v awk >/dev/null || die "awk is required."
  initialize_gpu_topology
  [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) \
    || die "PORT must be between 1 and 65535."
  validate_positive_integer "SMOKE_SAMPLES" "$SMOKE_SAMPLES"
  validate_positive_integer "MAX_EVAL_ATTEMPTS" "$MAX_EVAL_ATTEMPTS"
  validate_positive_integer "CHECKPOINT_EVERY" "$CHECKPOINT_EVERY"
  validate_positive_integer "CONCURRENCY" "$CONCURRENCY"
  validate_positive_integer "MODEL_START_TIMEOUT_SECONDS" "$MODEL_START_TIMEOUT_SECONDS"
  validate_positive_integer "MIN_FREE_DISK_GB" "$MIN_FREE_DISK_GB"
  validate_positive_integer "MIN_SYSTEM_RAM_GB" "$MIN_SYSTEM_RAM_GB"
  validate_positive_integer "MIN_FREE_GPU_MEMORY_MIB" "$MIN_FREE_GPU_MEMORY_MIB"
  validate_positive_integer "HF_HUB_DOWNLOAD_TIMEOUT" "$HF_HUB_DOWNLOAD_TIMEOUT"
  [[ "$REQUEST_TIMEOUT_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    && awk -v value="$REQUEST_TIMEOUT_SECONDS" 'BEGIN { exit !(value > 0) }' \
    || die "REQUEST_TIMEOUT_SECONDS must be positive."
  [[ "$GPU_MEMORY_UTILIZATION" =~ ^0([.][0-9]+)?$ ]] \
    && awk -v value="$GPU_MEMORY_UTILIZATION" 'BEGIN { exit !(value > 0 && value < 1) }' \
    || die "GPU_MEMORY_UTILIZATION must be greater than 0 and less than 1."
  [[ "$MAX_MODEL_LEN" == "auto" || "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] \
    || die "MAX_MODEL_LEN must be auto or an integer of at least 4096."
  if [[ "$MAX_MODEL_LEN" != "auto" ]]; then
    (( MAX_MODEL_LEN >= 4096 )) || die "MAX_MODEL_LEN must be at least 4096."
  fi
  [[ "$VLLM_DTYPE" == "bfloat16" ]] \
    || die "VLLM_DTYPE must remain bfloat16 for this unquantized research profile."
  [[ "$VLLM_KV_CACHE_DTYPE" == "bfloat16" ]] \
    || die "VLLM_KV_CACHE_DTYPE must remain bfloat16 for this unquantized research profile."
  [[ "$DYS_PROMPT_MODE" == "noncot" || "$DYS_PROMPT_MODE" == "cot" ]] \
    || die "DYS_PROMPT_MODE must be noncot or cot."
  [[ "$MINDS_EYE_PROMPT_MODE" == "noncot" || "$MINDS_EYE_PROMPT_MODE" == "cot" ]] \
    || die "MINDS_EYE_PROMPT_MODE must be noncot or cot."
  validate_positive_integer "DYS_MAX_TOKENS" "$DYS_MAX_TOKENS"
  validate_positive_integer "MINDS_EYE_MAX_TOKENS" "$MINDS_EYE_MAX_TOKENS"
  [[ "$DYS_TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "DYS_TEMPERATURE must be non-negative."
  [[ "$MINDS_EYE_TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "MINDS_EYE_TEMPERATURE must be non-negative."
  validate_probability "DYS_TOP_P" "$DYS_TOP_P"
  validate_probability "MINDS_EYE_TOP_P" "$MINDS_EYE_TOP_P"
  [[ "$BASE_SEED" =~ ^[0-9]+$ ]] || die "BASE_SEED must be a non-negative integer."
  [[ "$SAMPLING_TOP_K" == "-1" || "$SAMPLING_TOP_K" =~ ^[1-9][0-9]*$ ]] \
    || die "SAMPLING_TOP_K must be -1 or a positive integer."
  [[ "$SAMPLING_MIN_P" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    && awk -v value="$SAMPLING_MIN_P" 'BEGIN { exit !(value >= 0 && value <= 1) }' \
    || die "SAMPLING_MIN_P must be between 0 and 1."
  for penalty_name in PRESENCE_PENALTY FREQUENCY_PENALTY; do
    penalty_value="${!penalty_name}"
    [[ "$penalty_value" =~ ^-?[0-9]+([.][0-9]+)?$ ]] \
      && awk -v value="$penalty_value" 'BEGIN { exit !(value >= -2 && value <= 2) }' \
      || die "$penalty_name must be between -2 and 2."
  done
  [[ "$REPETITION_PENALTY" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    && awk -v value="$REPETITION_PENALTY" 'BEGIN { exit !(value > 0) }' \
    || die "REPETITION_PENALTY must be positive."
  validate_flag "SMOKE_ONLY" "$SMOKE_ONLY"
  validate_flag "SETUP_ONLY" "$SETUP_ONLY"
  validate_flag "FORCE" "$FORCE"
  validate_flag "DRY_RUN" "$DRY_RUN"
  validate_flag "KEEP_MODEL_CACHE" "$KEEP_MODEL_CACHE"
  validate_flag "CONTINUE_ON_MODEL_ERROR" "$CONTINUE_ON_MODEL_ERROR"
  validate_flag "HF_HUB_DISABLE_XET" "$HF_HUB_DISABLE_XET"
  [[ -n "$(selected_tracks)" ]] || die "TRACKS must select do_you_see_me, minds_eye, or all."
  (( $(selected_model_count) > 0 )) || die "MODELS did not match any configured model slug."
}

print_plan() {
  local track spec slug model_id revision loading model_max_len effective_model_len
  log "Evaluation plan"
  printf '  GPUs: %s (tensor parallel size %s)\n' "$GPU_IDS" "$TENSOR_PARALLEL_SIZE"
  printf '  Weights: original checkpoint tensors, unquantized; compute: %s; KV cache: %s\n' \
    "$VLLM_DTYPE" "$VLLM_KV_CACHE_DTYPE"
  printf '  Image preprocessing: original image bytes, no runner resize or recompression\n'
  printf '  Shared sampling: top_k=%s, min_p=%s, presence=%s, frequency=%s, repetition=%s\n' \
    "$SAMPLING_TOP_K" "$SAMPLING_MIN_P" "$PRESENCE_PENALTY" "$FREQUENCY_PENALTY" "$REPETITION_PENALTY"
  while IFS= read -r track; do
    printf '  Track %-14s prompt=%s, temperature=%s, top_p=%s, max_tokens=%s\n' \
      "$track" "$(track_prompt_mode "$track")" "$(track_temperature "$track")" \
      "$(track_top_p "$track")" "$(track_max_tokens "$track")"
  done < <(selected_tracks)
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r slug model_id revision loading model_max_len <<<"$spec"
    if is_enabled "$MODELS" "$slug"; then
      effective_model_len="$model_max_len"
      if [[ "$MAX_MODEL_LEN" != "auto" ]]; then
        effective_model_len="$MAX_MODEL_LEN"
      fi
      printf '  %-22s %-45s %s, context=%s\n' \
        "$slug" "$model_id" "$loading" "$effective_model_len"
    fi
  done
}

preflight_host() {
  [[ "$(uname -s)" == "Linux" ]] || die "This GPU runner requires Linux."
  command -v python3 >/dev/null || die "python3 is required."
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required."
  command -v curl >/dev/null || die "curl is required."
  python3 -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)' \
    || die "Python 3.10 through 3.14 is required."

  local gpu details name total free compute_major
  local -a gpu_names=()
  for gpu in "${GPU_ID_LIST[@]}"; do
    details="$(nvidia-smi -i "$gpu" \
      --query-gpu=name,memory.total,memory.free,compute_cap \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -n "$details" ]] || die "Could not query GPU $gpu."
    IFS=',' read -r name total free compute_cap <<<"$details"
    name="$(printf '%s' "$name" | xargs)"
    total="$(printf '%s' "$total" | xargs)"
    free="$(printf '%s' "$free" | xargs)"
    compute_cap="$(printf '%s' "$compute_cap" | xargs)"
    [[ "$total" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] \
      || die "Could not parse memory information for GPU $gpu: $details"
    (( total >= 39000 )) \
      || die "GPU $gpu has ${total} MiB. Unquantized runs require a 40 GB-class GPU."
    (( free >= MIN_FREE_GPU_MEMORY_MIB )) \
      || die "GPU $gpu has only ${free}/${total} MiB free; stop other GPU jobs first."
    compute_major="${compute_cap%%.*}"
    [[ "$compute_major" =~ ^[0-9]+$ ]] && (( compute_major >= 8 )) \
      || die "GPU $gpu does not expose native BF16 support (compute capability $compute_cap)."
    gpu_names+=("GPU $gpu: $name (${free}/${total} MiB free)")
  done
  GPU_NAME="$(IFS=';'; printf '%s' "${gpu_names[*]}")"

  local free_kib free_gib ram_kib ram_gib
  free_kib="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
  free_gib=$((free_kib / 1024 / 1024))
  (( free_gib >= MIN_FREE_DISK_GB )) \
    || die "Only ${free_gib} GiB is free; at least ${MIN_FREE_DISK_GB} GiB is required."
  ram_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
  [[ "$ram_kib" =~ ^[0-9]+$ ]] || die "Could not read total system memory."
  ram_gib=$((ram_kib / 1024 / 1024))
  (( ram_gib >= MIN_SYSTEM_RAM_GB )) \
    || die "The host has ${ram_gib} GiB RAM; at least ${MIN_SYSTEM_RAM_GB} GiB is required."
  log "Host preflight passed: $GPU_NAME; ${ram_gib} GiB RAM; ${free_gib} GiB disk free"
}

setup_environment() {
  local marker="$VENV_DIR/.ms-vista-vllm-${VLLM_VERSION}-unquantized-bf16-uv-${UV_VERSION}"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Creating evaluation environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR" \
      || die "Could not create a virtual environment. Install python3-venv and rerun."
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
  VLLM_BIN="$VENV_DIR/bin/vllm"
  UV_BIN="$VENV_DIR/bin/uv"

  if [[ ! -f "$marker" ]]; then
    log "Installing pinned unquantized evaluation dependencies"
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install "uv==$UV_VERSION"
    "$UV_BIN" pip install --python "$PYTHON_BIN" --torch-backend auto \
      "vllm==$VLLM_VERSION" \
      "openai==$OPENAI_VERSION" \
      "huggingface-hub==$HUGGINGFACE_HUB_VERSION" \
      "pillow==$PILLOW_VERSION"
    : >"$marker"
  fi

  "$PYTHON_BIN" - <<'PY'
import torch
import vllm

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the NVIDIA driver and vLLM wheel.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected CUDA runtime does not expose native BF16 support.")
print(f"Environment ready: vLLM {vllm.__version__}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
PY
}

verify_vllm_cli() {
  local help_text option
  help_text="$("$VLLM_BIN" serve --help=all 2>&1)" \
    || die "Could not inspect the installed vLLM serve command."
  for option in \
    --host --port --served-model-name --revision --dtype --gpu-memory-utilization \
    --max-model-len --max-num-seqs --limit-mm-per-prompt --generation-config --kv-cache-dtype \
    --trust-remote-code --tensor-parallel-size; do
    [[ "$help_text" == *"$option"* ]] \
      || die "Installed vLLM $VLLM_VERSION does not expose required option $option."
  done
  log "vLLM serve CLI compatibility preflight passed"
}

prepare_dataset() {
  log "Downloading or validating the pinned public visual dataset"
  HF_TOKEN="${HF_TOKEN:-}" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
    HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" \
    "$PYTHON_BIN" -m evaluation.prepare_visual_data \
      --output "$DATASET_DIR" \
      --repo-id "$DATASET_REPO_ID" \
      --revision "$DATASET_REVISION"
}

apply_model_compatibility_patches() {
  if is_enabled "$MODELS" "minicpm-v46"; then
    log "Applying audited MiniCPM vLLM compatibility patch"
    "$PYTHON_BIN" -m evaluation.common.patch_vllm_weights_mapper
  fi
}

model_compatibility_patch() {
  if [[ "$1" == "minicpm-v46" ]]; then
    printf '%s\n' "$VLLM_STACKED_WEIGHT_PATCH_ID"
  fi
}

port_is_available() {
  "$PYTHON_BIN" - "$PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}

server_serves_model() {
  local expected_model="$1"
  curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | "$PYTHON_BIN" -c '
import json
import sys

expected = sys.argv[1]
payload = json.load(sys.stdin)
model_ids = {
    str(item.get("id") or "")
    for item in payload.get("data", [])
    if isinstance(item, dict)
}
raise SystemExit(0 if expected in model_ids else 1)
' "$expected_model"
}

server_model_names() {
  curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | "$PYTHON_BIN" -c '
import json
import sys

payload = json.load(sys.stdin)
model_ids = [
    str(item.get("id") or "")
    for item in payload.get("data", [])
    if isinstance(item, dict) and item.get("id")
]
print(", ".join(model_ids) if model_ids else "<none>")
'
}

stop_server() {
  if [[ -z "$SERVER_PID" ]]; then
    return 0
  fi
  log "Stopping model server PID $SERVER_PID"
  if [[ "$SERVER_OWNS_PROCESS_GROUP" == "1" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  else
    kill -TERM "$SERVER_PID" 2>/dev/null || true
  fi
  local waited=0
  while kill -0 "$SERVER_PID" 2>/dev/null && (( waited < 60 )); do
    sleep 2
    waited=$((waited + 2))
  done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    if [[ "$SERVER_OWNS_PROCESS_GROUP" == "1" ]]; then
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    else
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  SERVER_OWNS_PROCESS_GROUP=0
  sleep 5
}

report_startup_progress() {
  local elapsed="$1" latest=""
  if [[ -s "$SERVER_LOG" ]]; then
    latest="$(tail -c 16384 "$SERVER_LOG" 2>/dev/null | tr '\r' '\n' | awk 'NF { line = $0 } END { print line }')" || true
  fi
  log "Waiting for model download and startup (${elapsed}s elapsed); log: $SERVER_LOG"
  if [[ -n "$latest" ]]; then
    printf '  Latest vLLM log: %.300s\n' "$latest"
  fi
}

report_startup_failure() {
  if grep -Fq 'No space left on device' "$SERVER_LOG" 2>/dev/null; then
    printf 'Model startup failed because the cache filesystem ran out of space. Free disk space, run fewer models concurrently, or set CACHE_ROOT to a larger mounted filesystem.\n' >&2
  elif grep -Fq 'Disk quota exceeded' "$SERVER_LOG" 2>/dev/null; then
    printf 'Model startup failed because the account disk quota was exceeded. Free quota or set CACHE_ROOT to a filesystem with sufficient quota.\n' >&2
  elif grep -Eq 'CUDA out of memory|torch\.OutOfMemoryError' "$SERVER_LOG" 2>/dev/null; then
    printf 'Model startup failed because the assigned GPU group ran out of memory. Confirm the GPUs are otherwise unused and that the requested tensor-parallel topology matches GPU_IDS.\n' >&2
  fi
  printf 'Model server exited during startup. Last log lines:\n' >&2
  tail -n 80 "$SERVER_LOG" >&2 || true
}

start_server() {
  local slug="$1" model_id="$2" revision="$3" loading="$4" model_cache="$5" model_max_len="$6"
  [[ "$loading" == "unquantized" ]] || die "Refusing unsupported weight loading mode: $loading"
  SERVER_LOG="$OUTPUT_ROOT/$slug/vllm.log"
  mkdir -p "$(dirname "$SERVER_LOG")" "$model_cache"
  : >"$SERVER_LOG"
  port_is_available || die "Port $PORT is already in use. Set PORT to an unused local port."

  local -a command=(
    "$VLLM_BIN" serve "$model_id"
    --host 127.0.0.1
    --port "$PORT"
    --served-model-name "$model_id"
    --revision "$revision"
    --dtype "$VLLM_DTYPE"
    --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$model_max_len"
    --max-num-seqs "$CONCURRENCY"
    --limit-mm-per-prompt '{"image":1}'
    --generation-config vllm
    --trust-remote-code
  )

  log "Starting $model_id at revision ${revision:0:12} (unquantized $VLLM_DTYPE, TP=$TENSOR_PARALLEL_SIZE)"
  log "Model startup log: $SERVER_LOG"
  if command -v setsid >/dev/null; then
    setsid env \
      HF_HOME="$model_cache" HF_TOKEN="${HF_TOKEN:-}" \
      HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
      HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONUNBUFFERED=1 \
      "${command[@]}" >"$SERVER_LOG" 2>&1 &
    SERVER_OWNS_PROCESS_GROUP=1
  else
    env \
      HF_HOME="$model_cache" HF_TOKEN="${HF_TOKEN:-}" \
      HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
      HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONUNBUFFERED=1 \
      "${command[@]}" >"$SERVER_LOG" 2>&1 &
    SERVER_OWNS_PROCESS_GROUP=0
  fi
  SERVER_PID=$!

  local elapsed=0 identity_mismatches=0 served_models=""
  while (( elapsed < MODEL_START_TIMEOUT_SECONDS )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      report_startup_failure
      stop_server
      return 1
    fi
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      if server_serves_model "$model_id"; then
        log "Model server is ready and serves $model_id"
        return 0
      fi
      identity_mismatches=$((identity_mismatches + 1))
      if (( identity_mismatches >= 3 )); then
        served_models="$(server_model_names 2>/dev/null || printf '<unavailable>')"
        printf 'Port %s is healthy but serves [%s], not %s. Another evaluation likely claimed the same PORT. Stop the conflicting runner, assign distinct PORT values, or use evaluation/run_visual_suite_multi_gpu.sh.\n' \
          "$PORT" "$served_models" "$model_id" >&2
        stop_server
        return 1
      fi
    else
      identity_mismatches=0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed % 60 == 0 )); then
      report_startup_progress "$elapsed"
    fi
  done

  printf 'Model server did not become ready within %s seconds. Last log lines:\n' \
    "$MODEL_START_TIMEOUT_SECONDS" >&2
  tail -n 80 "$SERVER_LOG" >&2 || true
  stop_server
  return 1
}

validate_submission() {
  local output="$1" questions="$2"
  "$PYTHON_BIN" - "$output" "$questions" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
questions = Path(sys.argv[2])
if not output.is_file():
    raise SystemExit(f"submission file does not exist: {output}")

expected = [
    str(json.loads(line)["question_id"])
    for line in questions.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
rows = []
for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    if set(row) != {"question_id", "condition", "answer"}:
        raise SystemExit(f"{output.name} line {line_number} has an invalid schema")
    if row["condition"] != "standard" or not str(row["answer"]).strip():
        raise SystemExit(f"{output.name} line {line_number} has an invalid answer")
    rows.append(row)

actual = [str(row["question_id"]) for row in rows]
if actual != expected:
    raise SystemExit(
        f"{output.name} coverage/order mismatch: expected {len(expected)} rows, got {len(actual)}"
    )
print(f"Validated {output}: {len(rows)} canonical rows")
PY
}

runner_base_args() {
  local slug="$1" model_id="$2" track="$3" seed="$4"
  local prompt_mode max_tokens temperature top_p chat_kwargs
  prompt_mode="$(track_prompt_mode "$track")"
  max_tokens="$(track_max_tokens "$track")"
  temperature="$(track_temperature "$track")"
  top_p="$(track_top_p "$track")"
  chat_kwargs="$(track_chat_kwargs "$slug" "$track")"
  RUNNER_ARGS=(
    --model "$model_id"
    --endpoints "http://127.0.0.1:$PORT/v1"
    --api-key EMPTY
    --image-root "$DATASET_DIR"
    --prompt-mode "$prompt_mode"
    --max-tokens "$max_tokens"
    --temperature "$temperature"
    --top-p "$top_p"
    --top-k "$SAMPLING_TOP_K"
    --min-p "$SAMPLING_MIN_P"
    --presence-penalty "$PRESENCE_PENALTY"
    --frequency-penalty "$FREQUENCY_PENALTY"
    --repetition-penalty "$REPETITION_PENALTY"
    --concurrency "$CONCURRENCY"
    --request-timeout "$REQUEST_TIMEOUT_SECONDS"
    --max-retries 3
    --seed "$seed"
    --checkpoint-every "$CHECKPOINT_EVERY"
  )
  if [[ "$chat_kwargs" != "{}" ]]; then
    RUNNER_ARGS+=(--chat-template-kwargs "$chat_kwargs")
  fi
}

run_track() {
  local slug="$1" model_id="$2" track="$3"
  local module questions model_dir output diagnostics smoke_diagnostics attempt seed smoke_passed
  local -a smoke_args
  module="$(track_module "$track")"
  questions="$(track_questions "$track")"
  model_dir="$OUTPUT_ROOT/$slug"
  output="$model_dir/${track}_submission.jsonl"
  diagnostics="$model_dir/${track}.diagnostics.jsonl"
  smoke_diagnostics="$model_dir/${track}.smoke.diagnostics.jsonl"
  mkdir -p "$model_dir"

  if [[ "$FORCE" != "1" ]] && validate_submission "$output" "$questions" >/dev/null 2>&1; then
    log "$slug/$track already has a complete canonical output; skipping"
    return 0
  fi
  if [[ "$FORCE" == "1" ]]; then
    rm -f -- "$output" "$diagnostics" "$smoke_diagnostics"
  elif [[ -f "$output" ]]; then
    mv -- "$output" "$output.invalid.$(date -u '+%Y%m%dT%H%M%SZ')"
  fi

  runner_base_args "$slug" "$model_id" "$track" "$BASE_SEED"
  rm -f -- "$smoke_diagnostics"
  smoke_passed=0
  for ((attempt = 1; attempt <= MAX_EVAL_ATTEMPTS; attempt++)); do
    seed=$((BASE_SEED + attempt - 1))
    runner_base_args "$slug" "$model_id" "$track" "$seed"
    log "Running strict $SMOKE_SAMPLES-sample smoke test for $slug/$track, pass $attempt/$MAX_EVAL_ATTEMPTS (seed=$seed)"
    smoke_args=(
      "${RUNNER_ARGS[@]}"
      --questions "$questions"
      --limit "$SMOKE_SAMPLES"
      --strict-partial
      --diagnostics "$smoke_diagnostics"
    )
    if (( attempt > 1 )); then
      smoke_args+=(--resume)
    fi
    if "$PYTHON_BIN" -m "$module" "${smoke_args[@]}"; then
      smoke_passed=1
      break
    fi
    if [[ -f "$smoke_diagnostics" ]]; then
      cp -- "$smoke_diagnostics" "$model_dir/${track}.smoke.attempt-${attempt}.diagnostics.jsonl"
    fi
    if (( attempt < MAX_EVAL_ATTEMPTS )); then
      log "Smoke pass $attempt left failed or unparseable samples; retrying only those samples with seed $((seed + 1))"
      sleep 15
    fi
  done
  if [[ "$smoke_passed" != "1" ]]; then
    printf 'Evaluation failed: %s/%s smoke test still has failed or unparseable outputs after %s passes. Raw responses remain in %s. The full evaluation was not started.\n' \
      "$slug" "$track" "$MAX_EVAL_ATTEMPTS" "$smoke_diagnostics" >&2
    return 1
  fi
  if [[ "$SMOKE_ONLY" == "1" ]]; then
    return 0
  fi

  for ((attempt = 1; attempt <= MAX_EVAL_ATTEMPTS; attempt++)); do
    seed=$((BASE_SEED + attempt - 1))
    runner_base_args "$slug" "$model_id" "$track" "$seed"
    log "Running $slug/$track full evaluation, pass $attempt/$MAX_EVAL_ATTEMPTS (seed=$seed)"
    if "$PYTHON_BIN" -m "$module" \
      "${RUNNER_ARGS[@]}" \
      --questions "$questions" \
      --resume \
      --out "$output" \
      --diagnostics "$diagnostics" \
      && validate_submission "$output" "$questions"; then
      rm -f -- "$smoke_diagnostics"
      return 0
    fi
    if [[ -f "$diagnostics" ]]; then
      cp -- "$diagnostics" "$model_dir/${track}.attempt-${attempt}.diagnostics.jsonl"
    fi
    if (( attempt < MAX_EVAL_ATTEMPTS )); then
      log "Pass $attempt left failed or unparseable samples; retrying only those samples with seed $((seed + 1))"
      sleep 15
    fi
  done

  printf 'Evaluation failed: %s/%s still has missing, failed, or unparseable outputs after %s passes. Raw responses remain in %s. No submission file was finalized.\n' \
    "$slug" "$track" "$MAX_EVAL_ATTEMPTS" "$diagnostics" >&2
  return 1
}

completed_tracks() {
  local slug="$1" track questions output
  for track in do_you_see_me minds_eye; do
    questions="$(track_questions "$track")"
    output="$OUTPUT_ROOT/$slug/${track}_submission.jsonl"
    if validate_submission "$output" "$questions" >/dev/null 2>&1; then
      printf '%s\n' "$track"
    fi
  done
}

ensure_run_config() {
  local slug="$1" model_id="$2" revision="$3" loading="$4" model_max_len="$5"
  mkdir -p "$OUTPUT_ROOT/$slug"
  RUN_CONFIG_PATH="$OUTPUT_ROOT/$slug/.run_config.json" \
  RUN_CONFIG_MODEL_ID="$model_id" RUN_CONFIG_REVISION="$revision" \
  RUN_CONFIG_LOADING="$loading" RUN_CONFIG_MODEL_MAX_LEN="$model_max_len" \
  RUN_CONFIG_DTYPE="$VLLM_DTYPE" RUN_CONFIG_KV_CACHE_DTYPE="$VLLM_KV_CACHE_DTYPE" \
  RUN_CONFIG_TP_SIZE="$TENSOR_PARALLEL_SIZE" \
  RUN_CONFIG_PROJECT_ROOT="$PROJECT_ROOT" RUN_CONFIG_DATASET_REPO_ID="$DATASET_REPO_ID" \
  RUN_CONFIG_DATASET_REVISION="$DATASET_REVISION" RUN_CONFIG_VLLM_VERSION="$VLLM_VERSION" \
  RUN_CONFIG_COMPATIBILITY_PATCH="$(model_compatibility_patch "$slug")" \
  RUN_CONFIG_UNPARSEABLE_POLICY="$UNPARSEABLE_ANSWER_POLICY_ID" \
  RUN_CONFIG_PIPELINE_REVISION="$PIPELINE_REVISION_ID" RUN_CONFIG_MAX_ATTEMPTS="$MAX_EVAL_ATTEMPTS" \
  RUN_CONFIG_BASE_SEED="$BASE_SEED" RUN_CONFIG_FORCE="$FORCE" \
  RUN_CONFIG_TOP_K="$SAMPLING_TOP_K" RUN_CONFIG_MIN_P="$SAMPLING_MIN_P" \
  RUN_CONFIG_PRESENCE_PENALTY="$PRESENCE_PENALTY" RUN_CONFIG_FREQUENCY_PENALTY="$FREQUENCY_PENALTY" \
  RUN_CONFIG_REPETITION_PENALTY="$REPETITION_PENALTY" \
  RUN_CONFIG_DYS_PROMPT_MODE="$DYS_PROMPT_MODE" RUN_CONFIG_DYS_MAX_TOKENS="$DYS_MAX_TOKENS" \
  RUN_CONFIG_DYS_TEMPERATURE="$DYS_TEMPERATURE" RUN_CONFIG_DYS_TOP_P="$DYS_TOP_P" \
  RUN_CONFIG_DYS_CHAT_KWARGS="$(track_chat_kwargs "$slug" do_you_see_me)" \
  RUN_CONFIG_ME_PROMPT_MODE="$MINDS_EYE_PROMPT_MODE" RUN_CONFIG_ME_MAX_TOKENS="$MINDS_EYE_MAX_TOKENS" \
  RUN_CONFIG_ME_TEMPERATURE="$MINDS_EYE_TEMPERATURE" RUN_CONFIG_ME_TOP_P="$MINDS_EYE_TOP_P" \
  RUN_CONFIG_ME_CHAT_KWARGS="$(track_chat_kwargs "$slug" minds_eye)" \
  "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_CONFIG_PATH"])
root = Path(os.environ["RUN_CONFIG_PROJECT_ROOT"])

def sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()

def protocol(prefix: str, track: str) -> dict:
    return {
        "prompt_mode": os.environ[f"RUN_CONFIG_{prefix}_PROMPT_MODE"],
        "max_tokens": int(os.environ[f"RUN_CONFIG_{prefix}_MAX_TOKENS"]),
        "temperature": float(os.environ[f"RUN_CONFIG_{prefix}_TEMPERATURE"]),
        "top_p": float(os.environ[f"RUN_CONFIG_{prefix}_TOP_P"]),
        "top_k": int(os.environ["RUN_CONFIG_TOP_K"]),
        "min_p": float(os.environ["RUN_CONFIG_MIN_P"]),
        "presence_penalty": float(os.environ["RUN_CONFIG_PRESENCE_PENALTY"]),
        "frequency_penalty": float(os.environ["RUN_CONFIG_FREQUENCY_PENALTY"]),
        "repetition_penalty": float(os.environ["RUN_CONFIG_REPETITION_PENALTY"]),
        "chat_template_kwargs": json.loads(os.environ[f"RUN_CONFIG_{prefix}_CHAT_KWARGS"]),
        "prompt_sha256": sha256(
            root / "evaluation" / track / "prompts" /
            f"{os.environ[f'RUN_CONFIG_{prefix}_PROMPT_MODE']}.txt"
        ),
    }

desired = {
    "schema_version": 4,
    "model_id": os.environ["RUN_CONFIG_MODEL_ID"],
    "model_revision": os.environ["RUN_CONFIG_REVISION"],
    "serving_engine": {"name": "vllm", "version": os.environ["RUN_CONFIG_VLLM_VERSION"]},
    "weight_loading": os.environ["RUN_CONFIG_LOADING"],
    "compute_dtype": os.environ["RUN_CONFIG_DTYPE"],
    "kv_cache_dtype": os.environ["RUN_CONFIG_KV_CACHE_DTYPE"],
    "tensor_parallel_size": int(os.environ["RUN_CONFIG_TP_SIZE"]),
    "max_model_len": int(os.environ["RUN_CONFIG_MODEL_MAX_LEN"]),
    "dataset": {
        "repo_id": os.environ["RUN_CONFIG_DATASET_REPO_ID"],
        "revision": os.environ["RUN_CONFIG_DATASET_REVISION"],
    },
    "generation": {
        "do_you_see_me": protocol("DYS", "do_you_see_me"),
        "minds_eye": protocol("ME", "minds_eye"),
        "base_seed": int(os.environ["RUN_CONFIG_BASE_SEED"]),
        "format_retry_attempts": int(os.environ["RUN_CONFIG_MAX_ATTEMPTS"]),
    },
    "image_preprocessing": "original-bytes-no-runner-resize-or-recompression",
    "answer_extraction": "strict-local-final-answer-parser",
    "unparseable_answers": {"policy": os.environ["RUN_CONFIG_UNPARSEABLE_POLICY"]},
    "pipeline_revision": os.environ["RUN_CONFIG_PIPELINE_REVISION"],
    "source_hashes": {
        "questions": {
            track: sha256(root / "tasks" / track / "questions.jsonl")
            for track in ("do_you_see_me", "minds_eye")
        },
        "runner": {
            "visual_pipeline": sha256(root / "evaluation" / "common" / "visual_pipeline.py"),
            "vllm_runner": sha256(root / "evaluation" / "common" / "vllm_runner.py"),
        },
    },
}
patch = os.environ["RUN_CONFIG_COMPATIBILITY_PATCH"]
if patch:
    desired["compatibility_patches"] = [patch]

force = os.environ["RUN_CONFIG_FORCE"] == "1"
artifacts = (
    list(path.parent.glob("*.diagnostics.jsonl"))
    + list(path.parent.glob("*_submission.jsonl"))
    + list(path.parent.glob("*.invalid.*"))
)
if force:
    for artifact in artifacts:
        artifact.unlink(missing_ok=True)
    (path.parent / "run_manifest.json").unlink(missing_ok=True)
elif path.is_file():
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Run fingerprint is unreadable for {path.parent.name}: {exc}. "
            "Set FORCE=1 to replace its artifacts safely."
        ) from exc
    if existing != desired and artifacts:
        raise SystemExit(
            f"Run configuration changed for {path.parent.name}. Set FORCE=1 to start "
            "a clean run instead of mixing checkpoints from different configurations."
        )
elif artifacts:
    raise SystemExit(
        f"Existing evaluation artifacts for {path.parent.name} have no run fingerprint. "
        "Set FORCE=1 to replace them safely."
    )

temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

write_manifest() {
  local slug="$1" model_id="$2" revision="$3" loading="$4" model_max_len="$5"
  local manifest_tracks
  manifest_tracks="$(completed_tracks "$slug" | paste -sd, -)"
  [[ -n "$manifest_tracks" ]] || die "Cannot write a manifest for $slug without a completed track."
  MANIFEST_PATH="$OUTPUT_ROOT/$slug/run_manifest.json" MANIFEST_PROJECT_ROOT="$PROJECT_ROOT" \
  MANIFEST_MODEL_ID="$model_id" MANIFEST_REVISION="$revision" MANIFEST_LOADING="$loading" \
  MANIFEST_MODEL_MAX_LEN="$model_max_len" MANIFEST_DTYPE="$VLLM_DTYPE" \
  MANIFEST_KV_CACHE_DTYPE="$VLLM_KV_CACHE_DTYPE" \
  MANIFEST_GPU="$GPU_NAME" MANIFEST_GPU_IDS="$GPU_IDS" MANIFEST_TP_SIZE="$TENSOR_PARALLEL_SIZE" \
  MANIFEST_TRACKS="$manifest_tracks" MANIFEST_VLLM_VERSION="$VLLM_VERSION" \
  MANIFEST_OPENAI_VERSION="$OPENAI_VERSION" MANIFEST_COMPATIBILITY_PATCH="$(model_compatibility_patch "$slug")" \
  MANIFEST_PIPELINE_REVISION="$PIPELINE_REVISION_ID" \
  "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["MANIFEST_PATH"])
root = Path(os.environ["MANIFEST_PROJECT_ROOT"])
run_config = json.loads((path.parent / ".run_config.json").read_text(encoding="utf-8"))
tracks = {}
for track in os.environ["MANIFEST_TRACKS"].split(","):
    output = path.parent / f"{track}_submission.jsonl"
    diagnostics = path.parent / f"{track}.diagnostics.jsonl"
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    attempt_files = sorted(path.parent.glob(f"{track}.attempt-*.diagnostics.jsonl"))
    smoke_attempt_files = sorted(path.parent.glob(f"{track}.smoke.attempt-*.diagnostics.jsonl"))
    tracks[track] = {
        "submission_file": output.name,
        "diagnostics_file": diagnostics.name,
        "row_count": len(rows),
        "submission_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "diagnostics_sha256": hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
        "failed_attempt_diagnostics": [
            {
                "file": attempt.name,
                "seed": run_config["generation"]["base_seed"]
                + int(attempt.name.split(".attempt-", 1)[1].split(".", 1)[0])
                - 1,
                "sha256": hashlib.sha256(attempt.read_bytes()).hexdigest(),
            }
            for attempt in attempt_files
        ],
        "failed_smoke_attempt_diagnostics": [
            {
                "file": attempt.name,
                "seed": run_config["generation"]["base_seed"]
                + int(attempt.name.split(".attempt-", 1)[1].split(".", 1)[0])
                - 1,
                "sha256": hashlib.sha256(attempt.read_bytes()).hexdigest(),
            }
            for attempt in smoke_attempt_files
        ],
        "question_bundle_sha256": run_config["source_hashes"]["questions"][track],
        "generation": run_config["generation"][track],
    }

manifest = {
    "schema_version": 4,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_id": os.environ["MANIFEST_MODEL_ID"],
    "model_revision": os.environ["MANIFEST_REVISION"],
    "serving_engine": {"name": "vllm", "version": os.environ["MANIFEST_VLLM_VERSION"]},
    "weight_loading": os.environ["MANIFEST_LOADING"],
    "compute_dtype": os.environ["MANIFEST_DTYPE"],
    "kv_cache_dtype": os.environ["MANIFEST_KV_CACHE_DTYPE"],
    "max_model_len": int(os.environ["MANIFEST_MODEL_MAX_LEN"]),
    "hardware": {
        "assigned_gpu_ids": os.environ["MANIFEST_GPU_IDS"].split(","),
        "description": os.environ["MANIFEST_GPU"],
        "tensor_parallel_size": int(os.environ["MANIFEST_TP_SIZE"]),
    },
    "dataset": run_config["dataset"],
    "image_preprocessing": run_config["image_preprocessing"],
    "answer_extraction": run_config["answer_extraction"],
    "unparseable_answers": run_config["unparseable_answers"],
    "pipeline_revision": os.environ["MANIFEST_PIPELINE_REVISION"],
    "dependencies": {"openai": os.environ["MANIFEST_OPENAI_VERSION"]},
    "tracks": tracks,
}
patch = os.environ["MANIFEST_COMPATIBILITY_PATCH"]
if patch:
    patch_source = root / "evaluation" / "common" / "patch_vllm_weights_mapper.py"
    manifest["compatibility_patches"] = [{
        "id": patch,
        "source_sha256": hashlib.sha256(patch_source.read_bytes()).hexdigest(),
    }]
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

delete_model_cache() {
  local model_cache="$1"
  if [[ "$KEEP_MODEL_CACHE" == "1" ]]; then
    return 0
  fi
  case "$model_cache" in
    "$CACHE_ROOT"/models/*)
      log "Removing model cache to bound disk use"
      rm -rf -- "$model_cache"
      ;;
    *) die "Refusing to remove unexpected model cache path: $model_cache" ;;
  esac
}

model_outputs_complete() {
  local slug="$1" track questions output
  while IFS= read -r track; do
    questions="$(track_questions "$track")"
    output="$OUTPUT_ROOT/$slug/${track}_submission.jsonl"
    validate_submission "$output" "$questions" >/dev/null 2>&1 || return 1
  done < <(selected_tracks)
}

model_was_skipped() {
  local requested="$1" skipped
  for skipped in "${SKIPPED_MODELS[@]}"; do
    [[ "$skipped" == "$requested" ]] && return 0
  done
  return 1
}

run_model() {
  local slug="$1" model_id="$2" revision="$3" loading="$4" model_max_len="$5"
  local model_cache="$CACHE_ROOT/models/$slug" track
  if [[ "$MAX_MODEL_LEN" != "auto" ]]; then
    model_max_len="$MAX_MODEL_LEN"
  fi

  ensure_run_config "$slug" "$model_id" "$revision" "$loading" "$model_max_len" || return 1
  if [[ "$SMOKE_ONLY" != "1" && "$FORCE" != "1" ]] && model_outputs_complete "$slug"; then
    log "$slug is already complete; skipping model startup"
    write_manifest "$slug" "$model_id" "$revision" "$loading" "$model_max_len"
    delete_model_cache "$model_cache"
    SKIPPED_MODELS+=("$slug")
    return 0
  fi

  if ! start_server "$slug" "$model_id" "$revision" "$loading" "$model_cache" "$model_max_len"; then
    delete_model_cache "$model_cache"
    return 1
  fi
  while IFS= read -r track; do
    if ! run_track "$slug" "$model_id" "$track"; then
      stop_server
      delete_model_cache "$model_cache"
      return 1
    fi
  done < <(selected_tracks)
  stop_server

  if [[ "$SMOKE_ONLY" != "1" ]]; then
    write_manifest "$slug" "$model_id" "$revision" "$loading" "$model_max_len"
    delete_model_cache "$model_cache"
  fi
}

main() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    return 0
  fi
  [[ $# -eq 0 ]] || die "Unknown argument: $1. Use --help for supported configuration."
  cd "$PROJECT_ROOT"
  validate_settings
  print_plan
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  preflight_host
  mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT/models"
  setup_environment
  apply_model_compatibility_patches
  verify_vllm_cli
  prepare_dataset
  if [[ "$SETUP_ONLY" == "1" ]]; then
    log "Shared environment and pinned dataset are ready; no model worker was started"
    return 0
  fi

  local selected_count=0 spec slug model_id revision loading model_max_len
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r slug model_id revision loading model_max_len <<<"$spec"
    if ! is_enabled "$MODELS" "$slug"; then
      continue
    fi
    selected_count=$((selected_count + 1))
    if run_model "$slug" "$model_id" "$revision" "$loading" "$model_max_len"; then
      if ! model_was_skipped "$slug"; then
        SUCCESS_MODELS+=("$slug")
      fi
    else
      FAILED_MODELS+=("$slug")
      log "$slug failed; inspect $OUTPUT_ROOT/$slug/vllm.log and diagnostics files"
      if [[ "$CONTINUE_ON_MODEL_ERROR" != "1" ]]; then
        break
      fi
    fi
  done
  (( selected_count > 0 )) || die "MODELS did not match any configured model slug."

  stop_server
  printf '\nEvaluation summary\n'
  printf '  Completed: %s\n' "${SUCCESS_MODELS[*]:-none}"
  printf '  Skipped:   %s\n' "${SKIPPED_MODELS[*]:-none}"
  printf '  Failed:    %s\n' "${FAILED_MODELS[*]:-none}"
  printf '  Outputs:   %s\n' "$OUTPUT_ROOT"
  if (( ${#FAILED_MODELS[@]} > 0 )); then
    return 1
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  trap stop_server EXIT
  trap 'exit 130' INT TERM
  main "$@"
fi
