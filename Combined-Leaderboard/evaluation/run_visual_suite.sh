#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VLLM_VERSION="0.25.1"
BITSANDBYTES_VERSION="0.49.2"
OPENAI_VERSION="2.45.0"
HUGGINGFACE_HUB_VERSION="1.23.0"
PILLOW_VERSION="12.3.0"
UV_VERSION="0.11.28"
DATASET_REPO_ID="amolharsh/visual-intelligence-leaderboard"
DATASET_REVISION="cc41be90e74679a9d3c9dd295834b2cee9100b9d"

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv/visual-suite}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/evaluation/results/visual_suite}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/evaluation/results/.cache}"
DATASET_DIR="${DATASET_DIR:-$CACHE_ROOT/visual-intelligence-dataset}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8011}"
MODELS="${MODELS:-all}"
TRACKS="${TRACKS:-all}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-20}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
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
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-60}"
MIN_SYSTEM_RAM_GB="${MIN_SYSTEM_RAM_GB:-28}"
MIN_FREE_GPU_MEMORY_MIB="${MIN_FREE_GPU_MEMORY_MIB:-22000}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"

MODEL_SPECS=(
  'qwen35-9b|Qwen/Qwen3.5-9B|c202236235762e1c871ad0ccb60c8ee5ba337b9a|bnb4|noncot|256|8192|{"enable_thinking":false}'
  'internvl35-8b|OpenGVLab/InternVL3_5-8B|9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79|full|noncot|256|4096|{}'
  'glm41v-9b-thinking|zai-org/GLM-4.1V-9B-Thinking|9e9a4c5e94f4a095c353f4152d520a2644a553b2|bnb4|cot|4096|8192|{}'
  'minicpm-v46|openbmb/MiniCPM-V-4.6|8169864629825dc1d755a5aa1cd8b5935dcbc83f|bnb4|noncot|256|8192|{}'
  'qwen25-vl-7b|Qwen/Qwen2.5-VL-7B-Instruct|cc594898137f460bfe9f0759e9844b3ce807cfb5|bnb4|noncot|256|8192|{}'
  'qwen3-vl-8b|Qwen/Qwen3-VL-8B-Instruct|0c351dd01ed87e9c1b53cbc748cba10e6187ff3b|bnb4|noncot|256|8192|{}'
  'phi4-multimodal|microsoft/Phi-4-multimodal-instruct|93f923e1a7727d1c4f446756212d9d3e8fcc5d81|full|noncot|256|4096|{}'
)

SERVER_PID=""
SERVER_OWNS_PROCESS_GROUP=0
SERVER_LOG=""
PYTHON_BIN=""
VLLM_BIN=""
UV_BIN=""
GPU_NAME=""
SUCCESS_MODELS=()
FAILED_MODELS=()
SKIPPED_MODELS=()

usage() {
  cat <<'EOF'
Run every supported MS-VISTA visual model sequentially on one NVIDIA GPU.

Usage:
  bash evaluation/run_visual_suite.sh

Common overrides:
  MODELS=internvl35-8b,qwen3-vl-8b   Run selected model slugs only
  TRACKS=do_you_see_me,minds_eye     Run selected benchmark tracks only
  SMOKE_ONLY=1                       Run strict compatibility checks only
  FORCE=1                            Replace existing outputs and checkpoints
  KEEP_MODEL_CACHE=1                 Retain downloaded model weights
  GPU_ID=1                           Use a different physical GPU
  VLLM_DTYPE=float16                 Use FP16 on GPUs without BF16 support
  MAX_MODEL_LEN=4096                 Override every model's context limit
  DRY_RUN=1                          Print the selected plan without setup

The script is resumable. Re-running it keeps validated diagnostic rows and
retries only missing, failed, or unparseable responses.
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

selected_tracks() {
  local track
  for track in do_you_see_me minds_eye; do
    if is_enabled "$TRACKS" "$track"; then
      printf '%s\n' "$track"
    fi
  done
}

selected_model_count() {
  local count=0 spec slug model_id revision quantization prompt_mode max_tokens model_max_len chat_kwargs
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r slug model_id revision quantization prompt_mode max_tokens model_max_len chat_kwargs <<<"$spec"
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

validate_settings() {
  command -v awk >/dev/null || die "awk is required."
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "GPU_ID must be a non-negative integer."
  [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || die "PORT must be between 1 and 65535."
  [[ "$SMOKE_SAMPLES" =~ ^[0-9]+$ ]] && (( SMOKE_SAMPLES > 0 )) || die "SMOKE_SAMPLES must be positive."
  [[ "$MAX_EVAL_ATTEMPTS" =~ ^[0-9]+$ ]] && (( MAX_EVAL_ATTEMPTS > 0 )) || die "MAX_EVAL_ATTEMPTS must be positive."
  [[ "$CHECKPOINT_EVERY" =~ ^[0-9]+$ ]] && (( CHECKPOINT_EVERY > 0 )) || die "CHECKPOINT_EVERY must be positive."
  [[ "$CONCURRENCY" =~ ^[0-9]+$ ]] && (( CONCURRENCY > 0 )) || die "CONCURRENCY must be positive."
  [[ "$MODEL_START_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] && (( MODEL_START_TIMEOUT_SECONDS > 0 )) \
    || die "MODEL_START_TIMEOUT_SECONDS must be positive."
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
  [[ "$MIN_FREE_DISK_GB" =~ ^[0-9]+$ ]] && (( MIN_FREE_DISK_GB > 0 )) \
    || die "MIN_FREE_DISK_GB must be positive."
  [[ "$MIN_SYSTEM_RAM_GB" =~ ^[0-9]+$ ]] && (( MIN_SYSTEM_RAM_GB > 0 )) \
    || die "MIN_SYSTEM_RAM_GB must be positive."
  [[ "$MIN_FREE_GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] && (( MIN_FREE_GPU_MEMORY_MIB > 0 )) \
    || die "MIN_FREE_GPU_MEMORY_MIB must be positive."
  [[ "$VLLM_DTYPE" == "auto" || "$VLLM_DTYPE" == "float16" || "$VLLM_DTYPE" == "bfloat16" ]] \
    || die "VLLM_DTYPE must be auto, float16, or bfloat16."
  validate_flag "SMOKE_ONLY" "$SMOKE_ONLY"
  validate_flag "FORCE" "$FORCE"
  validate_flag "DRY_RUN" "$DRY_RUN"
  validate_flag "KEEP_MODEL_CACHE" "$KEEP_MODEL_CACHE"
  validate_flag "CONTINUE_ON_MODEL_ERROR" "$CONTINUE_ON_MODEL_ERROR"
  [[ -n "$(selected_tracks)" ]] || die "TRACKS must select do_you_see_me, minds_eye, or all."
  (( $(selected_model_count) > 0 )) || die "MODELS did not match any configured model slug."
}

print_plan() {
  local spec slug model_id revision quantization prompt_mode max_tokens model_max_len chat_kwargs effective_model_len
  log "Evaluation plan"
  printf '  Tracks: %s\n' "$(selected_tracks | paste -sd, -)"
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r slug model_id revision quantization prompt_mode max_tokens model_max_len chat_kwargs <<<"$spec"
    if is_enabled "$MODELS" "$slug"; then
      effective_model_len="$model_max_len"
      if [[ "$MAX_MODEL_LEN" != "auto" ]]; then
        effective_model_len="$MAX_MODEL_LEN"
      fi
      printf '  %-22s %-45s %s, %s, max_tokens=%s, context=%s\n' \
        "$slug" "$model_id" "$quantization" "$prompt_mode" "$max_tokens" "$effective_model_len"
    fi
  done
}

preflight_host() {
  [[ "$(uname -s)" == "Linux" ]] || die "This GPU runner requires Linux."
  command -v python3 >/dev/null || die "python3 is required."
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required; install a working NVIDIA driver first."
  command -v curl >/dev/null || die "curl is required."
  python3 -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)' \
    || die "Python 3.10 through 3.14 is required."

  local memory_mib free_memory_mib compute_cap free_kib free_gib ram_kib ram_gib
  memory_mib="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  [[ "$memory_mib" =~ ^[0-9]+$ ]] || die "Could not read GPU $GPU_ID memory."
  (( memory_mib >= 22000 )) || die "GPU $GPU_ID has ${memory_mib} MiB; this suite requires a 24 GB-class GPU."
  free_memory_mib="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  [[ "$free_memory_mib" =~ ^[0-9]+$ ]] || die "Could not read free memory for GPU $GPU_ID."
  (( free_memory_mib >= MIN_FREE_GPU_MEMORY_MIB )) \
    || die "GPU $GPU_ID has only ${free_memory_mib} MiB free. Stop other GPU jobs or lower MIN_FREE_GPU_MEMORY_MIB only after verifying memory requirements."
  GPU_NAME="$(nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"

  compute_cap="$(nvidia-smi -i "$GPU_ID" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | xargs || true)"
  if [[ "$VLLM_DTYPE" == "auto" ]]; then
    if [[ "$compute_cap" =~ ^([0-9]+)\. ]] && (( BASH_REMATCH[1] >= 8 )); then
      VLLM_DTYPE="bfloat16"
    else
      VLLM_DTYPE="float16"
    fi
  fi

  free_kib="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
  free_gib=$(( free_kib / 1024 / 1024 ))
  (( free_gib >= MIN_FREE_DISK_GB )) \
    || die "Only ${free_gib} GiB is free; at least ${MIN_FREE_DISK_GB} GiB is required with cache cleanup enabled."

  ram_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
  [[ "$ram_kib" =~ ^[0-9]+$ ]] || die "Could not read total system memory from /proc/meminfo."
  ram_gib=$(( ram_kib / 1024 / 1024 ))
  (( ram_gib >= MIN_SYSTEM_RAM_GB )) \
    || die "The host has ${ram_gib} GiB RAM; at least ${MIN_SYSTEM_RAM_GB} GiB is required for model loading."

  log "Host preflight passed: $GPU_NAME, ${free_memory_mib}/${memory_mib} MiB VRAM free, ${ram_gib} GiB RAM, dtype=$VLLM_DTYPE, ${free_gib} GiB disk free"
}

setup_environment() {
  local marker="$VENV_DIR/.ms-vista-vllm-${VLLM_VERSION}-bnb-${BITSANDBYTES_VERSION}-uv-${UV_VERSION}"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Creating evaluation environment at $VENV_DIR"
    if ! python3 -m venv "$VENV_DIR"; then
      die "Could not create a virtual environment. Install python3-venv and rerun the script."
    fi
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
  VLLM_BIN="$VENV_DIR/bin/vllm"
  UV_BIN="$VENV_DIR/bin/uv"

  if [[ ! -f "$marker" ]]; then
    log "Installing pinned GPU evaluation dependencies; this may take several minutes"
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install "uv==$UV_VERSION"
    "$UV_BIN" pip install --python "$PYTHON_BIN" --torch-backend auto \
      "vllm==$VLLM_VERSION" \
      "bitsandbytes==$BITSANDBYTES_VERSION" \
      "openai==$OPENAI_VERSION" \
      "huggingface-hub==$HUGGINGFACE_HUB_VERSION" \
      "pillow==$PILLOW_VERSION"
    : >"$marker"
  fi

  "$PYTHON_BIN" - <<'PY'
import torch
import vllm
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the NVIDIA driver and installed vLLM wheel.")
print(f"Environment ready: vLLM {vllm.__version__}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
PY
}

verify_vllm_cli() {
  local help_text option
  if ! help_text="$("$VLLM_BIN" serve --help=all 2>&1)"; then
    die "Could not inspect the installed vLLM serve command. Recreate $VENV_DIR and rerun."
  fi
  for option in \
    --host \
    --port \
    --served-model-name \
    --revision \
    --dtype \
    --gpu-memory-utilization \
    --max-model-len \
    --max-num-seqs \
    --limit-mm-per-prompt \
    --generation-config \
    --trust-remote-code \
    --enforce-eager \
    --quantization \
    --load-format; do
    [[ "$help_text" == *"$option"* ]] \
      || die "Installed vLLM $VLLM_VERSION does not expose required option $option."
  done
  log "vLLM serve CLI compatibility preflight passed"
}

prepare_dataset() {
  log "Downloading or validating the pinned public visual dataset"
  HF_TOKEN="${HF_TOKEN:-}" HF_HUB_DISABLE_TELEMETRY=1 \
    "$PYTHON_BIN" -m evaluation.prepare_visual_data \
      --output "$DATASET_DIR" \
      --repo-id "$DATASET_REPO_ID" \
      --revision "$DATASET_REVISION"
}

port_is_available() {
  "$PYTHON_BIN" - "$PORT" <<'PY'
import socket
import sys
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
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

start_server() {
  local slug="$1" model_id="$2" revision="$3" quantization="$4" model_cache="$5" model_max_len="$6"
  SERVER_LOG="$OUTPUT_ROOT/$slug/vllm.log"
  mkdir -p "$(dirname "$SERVER_LOG")" "$model_cache"
  port_is_available || die "Port $PORT is already in use. Set PORT to an unused local port."

  local -a command=(
    "$VLLM_BIN" serve "$model_id"
    --host 127.0.0.1
    --port "$PORT"
    --served-model-name "$model_id"
    --revision "$revision"
    --dtype "$VLLM_DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$model_max_len"
    --max-num-seqs 1
    --limit-mm-per-prompt '{"image":1}'
    --generation-config vllm
    --trust-remote-code
    --enforce-eager
  )
  if [[ "$quantization" == "bnb4" ]]; then
    command+=(--quantization bitsandbytes --load-format bitsandbytes)
  fi

  log "Starting $model_id at revision ${revision:0:12} ($quantization)"
  if command -v setsid >/dev/null; then
    setsid env \
      HF_HOME="$model_cache" \
      HF_TOKEN="${HF_TOKEN:-}" \
      HF_HUB_DISABLE_TELEMETRY=1 \
      TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      PYTHONUNBUFFERED=1 \
      "${command[@]}" >"$SERVER_LOG" 2>&1 &
    SERVER_OWNS_PROCESS_GROUP=1
  else
    env \
      HF_HOME="$model_cache" \
      HF_TOKEN="${HF_TOKEN:-}" \
      HF_HUB_DISABLE_TELEMETRY=1 \
      TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      PYTHONUNBUFFERED=1 \
      "${command[@]}" >"$SERVER_LOG" 2>&1 &
    SERVER_OWNS_PROCESS_GROUP=0
  fi
  SERVER_PID=$!

  local elapsed=0
  while (( elapsed < MODEL_START_TIMEOUT_SECONDS )); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      log "Model server is ready"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      printf 'Model server exited during startup. Last log lines:\n' >&2
      tail -n 60 "$SERVER_LOG" >&2 || true
      stop_server
      return 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed % 60 == 0 )); then
      log "Waiting for model download and startup (${elapsed}s elapsed)"
    fi
  done

  printf 'Model server did not become ready within %s seconds. Last log lines:\n' \
    "$MODEL_START_TIMEOUT_SECONDS" >&2
  tail -n 60 "$SERVER_LOG" >&2 || true
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

expected = []
for line_number, line in enumerate(questions.read_text(encoding="utf-8").splitlines(), 1):
    if line.strip():
        expected.append(str(json.loads(line)["question_id"]))

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
  local model_id="$1" prompt_mode="$2" max_tokens="$3" chat_kwargs="$4"
  RUNNER_ARGS=(
    --model "$model_id"
    --endpoints "http://127.0.0.1:$PORT/v1"
    --api-key EMPTY
    --image-root "$DATASET_DIR"
    --prompt-mode "$prompt_mode"
    --max-tokens "$max_tokens"
    --concurrency "$CONCURRENCY"
    --request-timeout "$REQUEST_TIMEOUT_SECONDS"
    --max-retries 3
    --seed 0
    --checkpoint-every "$CHECKPOINT_EVERY"
  )
  if [[ "$chat_kwargs" != "{}" ]]; then
    RUNNER_ARGS+=(--chat-template-kwargs "$chat_kwargs")
  fi
}

run_track() {
  local slug="$1" model_id="$2" prompt_mode="$3" max_tokens="$4" chat_kwargs="$5" track="$6"
  local module questions model_dir output diagnostics smoke_diagnostics attempt
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

  runner_base_args "$model_id" "$prompt_mode" "$max_tokens" "$chat_kwargs"
  rm -f -- "$smoke_diagnostics"
  log "Running strict $SMOKE_SAMPLES-sample smoke test for $slug/$track"
  if ! "$PYTHON_BIN" -m "$module" \
    "${RUNNER_ARGS[@]}" \
    --questions "$questions" \
    --limit "$SMOKE_SAMPLES" \
    --strict-partial \
    --diagnostics "$smoke_diagnostics"; then
    return 1
  fi
  if [[ "$SMOKE_ONLY" == "1" ]]; then
    return 0
  fi

  for ((attempt = 1; attempt <= MAX_EVAL_ATTEMPTS; attempt++)); do
    log "Running $slug/$track full evaluation, pass $attempt/$MAX_EVAL_ATTEMPTS"
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
    if (( attempt < MAX_EVAL_ATTEMPTS )); then
      log "Pass $attempt left failed or unparseable samples; retrying only those samples"
      sleep 15
    fi
  done
  return 1
}

write_manifest() {
  local slug="$1" model_id="$2" revision="$3" quantization="$4" prompt_mode="$5" max_tokens="$6" model_max_len="$7" chat_kwargs="$8"
  local manifest_tracks
  manifest_tracks="$(completed_tracks "$slug" | paste -sd, -)"
  [[ -n "$manifest_tracks" ]] || die "Cannot write a manifest for $slug without a valid completed track."
  MANIFEST_PATH="$OUTPUT_ROOT/$slug/run_manifest.json" \
  MANIFEST_PROJECT_ROOT="$PROJECT_ROOT" \
  MANIFEST_MODEL_ID="$model_id" \
  MANIFEST_REVISION="$revision" \
  MANIFEST_QUANTIZATION="$quantization" \
  MANIFEST_PROMPT_MODE="$prompt_mode" \
  MANIFEST_MAX_TOKENS="$max_tokens" \
  MANIFEST_MAX_MODEL_LEN="$model_max_len" \
  MANIFEST_CHAT_KWARGS="$chat_kwargs" \
  MANIFEST_DTYPE="$VLLM_DTYPE" \
  MANIFEST_GPU="$GPU_NAME" \
  MANIFEST_TRACKS="$manifest_tracks" \
  MANIFEST_DATASET_REPO_ID="$DATASET_REPO_ID" \
  MANIFEST_DATASET_REVISION="$DATASET_REVISION" \
  MANIFEST_VLLM_VERSION="$VLLM_VERSION" \
  MANIFEST_BNB_VERSION="$BITSANDBYTES_VERSION" \
  MANIFEST_OPENAI_VERSION="$OPENAI_VERSION" \
  "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["MANIFEST_PATH"])
project_root = Path(os.environ["MANIFEST_PROJECT_ROOT"])
prompt_mode = os.environ["MANIFEST_PROMPT_MODE"]
track_data = {}
for track in os.environ["MANIFEST_TRACKS"].split(","):
    output = path.parent / f"{track}_submission.jsonl"
    diagnostics = path.parent / f"{track}.diagnostics.jsonl"
    questions = project_root / "tasks" / track / "questions.jsonl"
    prompt = project_root / "evaluation" / track / "prompts" / f"{prompt_mode}.txt"
    if not output.is_file():
        continue
    payload = output.read_bytes()
    track_data[track] = {
        "submission_file": output.name,
        "diagnostics_file": diagnostics.name,
        "row_count": len(output.read_text(encoding="utf-8").splitlines()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "question_bundle_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
    }

manifest = {
    "schema_version": 2,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_id": os.environ["MANIFEST_MODEL_ID"],
    "model_revision": os.environ["MANIFEST_REVISION"],
    "serving_engine": {"name": "vllm", "version": os.environ["MANIFEST_VLLM_VERSION"]},
    "weight_loading": os.environ["MANIFEST_QUANTIZATION"],
    "compute_dtype": os.environ["MANIFEST_DTYPE"],
    "dataset": {
        "repo_id": os.environ["MANIFEST_DATASET_REPO_ID"],
        "revision": os.environ["MANIFEST_DATASET_REVISION"],
    },
    "generation": {
        "chat_template_kwargs": json.loads(os.environ["MANIFEST_CHAT_KWARGS"]),
        "max_tokens": int(os.environ["MANIFEST_MAX_TOKENS"]),
        "prompt_mode": prompt_mode,
        "seed": 0,
        "temperature": 0,
    },
    "max_model_len": int(os.environ["MANIFEST_MAX_MODEL_LEN"]),
    "dependencies": {
        "bitsandbytes": os.environ["MANIFEST_BNB_VERSION"],
        "openai": os.environ["MANIFEST_OPENAI_VERSION"],
    },
    "gpu": os.environ["MANIFEST_GPU"],
    "tracks": track_data,
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

ensure_run_config() {
  local slug="$1" model_id="$2" revision="$3" quantization="$4" prompt_mode="$5" max_tokens="$6" model_max_len="$7" chat_kwargs="$8"
  mkdir -p "$OUTPUT_ROOT/$slug"
  RUN_CONFIG_PATH="$OUTPUT_ROOT/$slug/.run_config.json" \
  RUN_CONFIG_MODEL_ID="$model_id" \
  RUN_CONFIG_REVISION="$revision" \
  RUN_CONFIG_QUANTIZATION="$quantization" \
  RUN_CONFIG_PROMPT_MODE="$prompt_mode" \
  RUN_CONFIG_MAX_TOKENS="$max_tokens" \
  RUN_CONFIG_MAX_MODEL_LEN="$model_max_len" \
  RUN_CONFIG_CHAT_KWARGS="$chat_kwargs" \
  RUN_CONFIG_DTYPE="$VLLM_DTYPE" \
  RUN_CONFIG_PROJECT_ROOT="$PROJECT_ROOT" \
  RUN_CONFIG_DATASET_REPO_ID="$DATASET_REPO_ID" \
  RUN_CONFIG_DATASET_REVISION="$DATASET_REVISION" \
  RUN_CONFIG_VLLM_VERSION="$VLLM_VERSION" \
  RUN_CONFIG_FORCE="$FORCE" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_CONFIG_PATH"])
project_root = Path(os.environ["RUN_CONFIG_PROJECT_ROOT"])
prompt_mode = os.environ["RUN_CONFIG_PROMPT_MODE"]


def sha256(path_to_hash: Path) -> str:
    import hashlib

    return hashlib.sha256(path_to_hash.read_bytes()).hexdigest()


source_hashes = {
    track: {
        "questions": sha256(project_root / "tasks" / track / "questions.jsonl"),
        "prompt": sha256(
            project_root / "evaluation" / track / "prompts" / f"{prompt_mode}.txt"
        ),
    }
    for track in ("do_you_see_me", "minds_eye")
}
source_hashes["runner"] = {
    "visual_pipeline": sha256(project_root / "evaluation" / "common" / "visual_pipeline.py"),
    "vllm_runner": sha256(project_root / "evaluation" / "common" / "vllm_runner.py"),
}
desired = {
    "schema_version": 2,
    "model_id": os.environ["RUN_CONFIG_MODEL_ID"],
    "model_revision": os.environ["RUN_CONFIG_REVISION"],
    "serving_engine": {"name": "vllm", "version": os.environ["RUN_CONFIG_VLLM_VERSION"]},
    "weight_loading": os.environ["RUN_CONFIG_QUANTIZATION"],
    "compute_dtype": os.environ["RUN_CONFIG_DTYPE"],
    "prompt_mode": prompt_mode,
    "max_tokens": int(os.environ["RUN_CONFIG_MAX_TOKENS"]),
    "max_model_len": int(os.environ["RUN_CONFIG_MAX_MODEL_LEN"]),
    "chat_template_kwargs": json.loads(os.environ["RUN_CONFIG_CHAT_KWARGS"]),
    "dataset": {
        "repo_id": os.environ["RUN_CONFIG_DATASET_REPO_ID"],
        "revision": os.environ["RUN_CONFIG_DATASET_REVISION"],
    },
    "source_hashes": source_hashes,
}
force = os.environ["RUN_CONFIG_FORCE"] == "1"
artifacts = list(path.parent.glob("*.diagnostics.jsonl")) + list(
    path.parent.glob("*_submission.jsonl")
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
    if existing != desired and not force:
        raise SystemExit(
            f"Run configuration changed for {path.parent.name}. Set FORCE=1 to start "
            "a clean run instead of mixing checkpoints from different configurations."
        )
elif artifacts and not force:
    raise SystemExit(
        f"Existing evaluation artifacts for {path.parent.name} have no run fingerprint. "
        "Set FORCE=1 to replace them safely."
    )

if force or not path.is_file():
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    *)
      die "Refusing to remove unexpected model cache path: $model_cache"
      ;;
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

model_was_skipped() {
  local requested="$1" skipped
  for skipped in "${SKIPPED_MODELS[@]}"; do
    [[ "$skipped" == "$requested" ]] && return 0
  done
  return 1
}

run_model() {
  local slug="$1" model_id="$2" revision="$3" quantization="$4" prompt_mode="$5" max_tokens="$6" model_max_len="$7" chat_kwargs="$8"
  local model_cache="$CACHE_ROOT/models/$slug" track

  if [[ "$MAX_MODEL_LEN" != "auto" ]]; then
    model_max_len="$MAX_MODEL_LEN"
  fi

  if ! ensure_run_config "$slug" "$model_id" "$revision" "$quantization" "$prompt_mode" "$max_tokens" "$model_max_len" "$chat_kwargs"; then
    return 1
  fi

  if [[ "$SMOKE_ONLY" != "1" && "$FORCE" != "1" ]] && model_outputs_complete "$slug"; then
    log "$slug is already complete; skipping model startup"
    write_manifest "$slug" "$model_id" "$revision" "$quantization" "$prompt_mode" "$max_tokens" "$model_max_len" "$chat_kwargs"
    delete_model_cache "$model_cache"
    SKIPPED_MODELS+=("$slug")
    return 0
  fi

  if ! start_server "$slug" "$model_id" "$revision" "$quantization" "$model_cache" "$model_max_len"; then
    delete_model_cache "$model_cache"
    return 1
  fi
  while IFS= read -r track; do
    if ! run_track "$slug" "$model_id" "$prompt_mode" "$max_tokens" "$chat_kwargs" "$track"; then
      stop_server
      delete_model_cache "$model_cache"
      return 1
    fi
  done < <(selected_tracks)
  stop_server

  if [[ "$SMOKE_ONLY" != "1" ]]; then
    write_manifest "$slug" "$model_id" "$revision" "$quantization" "$prompt_mode" "$max_tokens" "$model_max_len" "$chat_kwargs"
    delete_model_cache "$model_cache"
  fi
  return 0
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
  verify_vllm_cli
  prepare_dataset

  local selected_count=0 spec slug model_id revision quantization prompt_mode max_tokens model_max_len chat_kwargs
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r slug model_id revision quantization prompt_mode max_tokens model_max_len chat_kwargs <<<"$spec"
    if ! is_enabled "$MODELS" "$slug"; then
      continue
    fi
    selected_count=$((selected_count + 1))
    if run_model "$slug" "$model_id" "$revision" "$quantization" "$prompt_mode" "$max_tokens" "$model_max_len" "$chat_kwargs"; then
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
