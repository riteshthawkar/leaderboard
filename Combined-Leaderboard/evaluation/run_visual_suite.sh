#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
OPENAI_VERSION="2.45.0"
HUGGINGFACE_HUB_VERSION="${HUGGINGFACE_HUB_VERSION:-1.23.0}"
PILLOW_VERSION="12.3.0"
SCIPY_VERSION="1.15.3"
TIMM_VERSION="1.0.28"
UV_VERSION="0.11.28"
DATASET_REPO_ID="amolharsh/visual-intelligence-leaderboard"
DATASET_REVISION="cc41be90e74679a9d3c9dd295834b2cee9100b9d"

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv/visual-suite}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/evaluation/results/visual_suite_bf16}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/evaluation/results/.cache}"
DATASET_DIR="${DATASET_DIR:-$CACHE_ROOT/visual-intelligence-dataset}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-auto}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
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
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-600}"
MODEL_START_TIMEOUT_SECONDS="${MODEL_START_TIMEOUT_SECONDS:-7200}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.84}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
SERVING_REPLICA_MODE="${SERVING_REPLICA_MODE:-builtin}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_NUM_SEQS_PER_REPLICA="${MAX_NUM_SEQS_PER_REPLICA:-auto}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-96}"
MIN_SYSTEM_RAM_GB="${MIN_SYSTEM_RAM_GB:-28}"
MIN_FREE_GPU_MEMORY_MIB="${MIN_FREE_GPU_MEMORY_MIB:-38000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-bfloat16}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_HUB_ENABLE_HF_TRANSFER="0"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
BASE_SEED="${BASE_SEED:-0}"
ANSWER_EXTRACTOR_SEED="${ANSWER_EXTRACTOR_SEED:-$BASE_SEED}"
ANSWER_EXTRACTOR_MAX_TOKENS="${ANSWER_EXTRACTOR_MAX_TOKENS:-200}"
ANSWER_EXTRACTOR_MODEL="${ANSWER_EXTRACTOR_MODEL:-Qwen/Qwen3-8B}"
ANSWER_EXTRACTOR_REVISION="${ANSWER_EXTRACTOR_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
ANSWER_EXTRACTOR_ENDPOINTS="${ANSWER_EXTRACTOR_ENDPOINTS:-}"
ANSWER_EXTRACTOR_CHAT_TEMPLATE_KWARGS="${ANSWER_EXTRACTOR_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"
ANSWER_EXTRACTOR_GPU_IDS="${ANSWER_EXTRACTOR_GPU_IDS:-}"
ANSWER_EXTRACTOR_PORT="${ANSWER_EXTRACTOR_PORT:-8035}"
ANSWER_EXTRACTOR_GPU_MEMORY_UTILIZATION="${ANSWER_EXTRACTOR_GPU_MEMORY_UTILIZATION:-0.70}"
ANSWER_EXTRACTOR_MAX_MODEL_LEN="${ANSWER_EXTRACTOR_MAX_MODEL_LEN:-32768}"
EXTRACT_EXISTING_ONLY="${EXTRACT_EXISTING_ONLY:-0}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:--1}"
SAMPLING_MIN_P="${SAMPLING_MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
PHI_OFFICIAL_SNAPSHOT_PATH="${PHI_OFFICIAL_SNAPSHOT_PATH:-}"

PHI_OFFICIAL_SNAPSHOT_REPO_ID="microsoft/Phi-4-multimodal-instruct"
PHI_OFFICIAL_SNAPSHOT_COMMIT="7641bf905e6965ee54166808d275266371e28343"
PHI_MODEL_SHARD_1_SHA256="c46bb03332d82f6a3eaf85bd20af388dd4d4d68b198c2203c965c7381a466094"
PHI_MODEL_SHARD_2_SHA256="b3e812c0c8acef4e7f5e34d6c9f77a7640ee4a2b93ea351921365ac62f19918d"
PHI_MODEL_SHARD_3_SHA256="7be96b7339303752634b202d3f377bcf312a03046586eca6cea23347ace1e65a"
PHI_VISION_ADAPTER_SHA256="1620b16722edf701038bf66e3cd46412c7cc5458e58df89e9f92cedb71fcbde8"

# Primary benchmark profiles reconstructed from the papers and released code.
DYS_PROMPT_MODE="${DYS_PROMPT_MODE:-noncot}"
DYS_TEMPERATURE="${DYS_TEMPERATURE:-1.0}"
DYS_TOP_P="${DYS_TOP_P:-0.95}"
MINDS_EYE_PROMPT_MODE="${MINDS_EYE_PROMPT_MODE:-cot}"
MINDS_EYE_TEMPERATURE="${MINDS_EYE_TEMPERATURE:-0.1}"
MINDS_EYE_TOP_P="${MINDS_EYE_TOP_P:-1.0}"
FINAL_ANSWER_MAX_TOKENS="${FINAL_ANSWER_MAX_TOKENS:-200}"
INTERNVL35_MAX_TOKENS="${INTERNVL35_MAX_TOKENS:-8192}"
QWEN36_MAX_TOKENS="${QWEN36_MAX_TOKENS:-8192}"

VLLM_STACKED_WEIGHT_PATCH_ID="vllm-0.25.1-stacked-weight-single-match-v1"
VLLM_PHI4MM_MASK_SUM_PATCH_ID="vllm-0.25.1-phi4mm-fp32-mask-sum-v1"
VLLM_DEEPSEEK_VL2_CONFIG_OVERRIDE_ID="vllm-0.25.1-deepseek-vl2-config-defaults-v1"
ANSWER_EXTRACTION_METHOD_ID="mandatory-gold-blind-text-extractor-v2"
UNPARSEABLE_ANSWER_POLICY_ID="mandatory-extractor-unresolved-token-v1"
PIPELINE_REVISION_ID="unquantized-bf16-mandatory-extraction-v11"

# slug|repository|revision|weight loading|max context
MODEL_SPECS=(
  'qwen35-9b|Qwen/Qwen3.5-9B|c202236235762e1c871ad0ccb60c8ee5ba337b9a|unquantized|32768'
  'qwen36-27b|Qwen/Qwen3.6-27B|6a9e13bd6fc8f0983b9b99948120bc37f49c13e9|unquantized|32768'
  'internvl35-8b|OpenGVLab/InternVL3_5-8B|9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79|unquantized|32768'
  'glm46v-flash|zai-org/GLM-4.6V-Flash|411bb4d77144a3f03accbf4b780f5acb8b7cde4e|unquantized|32768'
  'minicpm-v46|openbmb/MiniCPM-V-4.6|8169864629825dc1d755a5aa1cd8b5935dcbc83f|unquantized|32768'
  'qwen25-vl-7b|Qwen/Qwen2.5-VL-7B-Instruct|cc594898137f460bfe9f0759e9844b3ce807cfb5|unquantized|32768'
  'qwen3-vl-8b|Qwen/Qwen3-VL-8B-Instruct|0c351dd01ed87e9c1b53cbc748cba10e6187ff3b|unquantized|32768'
  'phi4-multimodal|microsoft/Phi-4-multimodal-instruct|93f923e1a7727d1c4f446756212d9d3e8fcc5d81|unquantized|32768'
  'gemma3-12b-it|google/gemma-3-12b-it|96b6f1eccf38110c56df3a15bffe176da04bfd80|unquantized|32768'
  'gemma3-27b-it|google/gemma-3-27b-it|005ad3404e59d6023443cb575daa05336842228a|unquantized|32768'
  'kimi-vl-a3b-instruct|moonshotai/Kimi-VL-A3B-Instruct|398eede0903cd983a2bfa0cc634e9ac1d843f375|unquantized|32768'
  'deepseek-vl2|deepseek-ai/deepseek-vl2|f363772d1c47f4239dd844015b4bd53beb87951b|unquantized|4096'
  'llama32-11b-vision-instruct|meta-llama/Llama-3.2-11B-Vision-Instruct|9eb2daaa8597bf192a8b0e73f848f3a102794df5|unquantized|32768'
)

SERVER_PID=""
SERVER_OWNS_PROCESS_GROUP=0
SERVER_LOG=""
SERVER_ENDPOINTS=""
EXTRACTOR_SERVER_PID=""
EXTRACTOR_SERVER_LOG=""
EXTRACTOR_SERVER_MANAGED=0
EXTRACTOR_SERVER_OWNS_PROCESS_GROUP=0
INDEPENDENT_SERVER_PIDS=()
INDEPENDENT_SERVER_PORTS=()
INDEPENDENT_SERVER_LOGS=()
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
ACTIVE_RUN_MARKER=""

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
  TENSOR_PARALLEL_SIZE=2             Tensor-parallel GPUs per replica
  DATA_PARALLEL_SIZE=4               Run four independent serving replicas
  CONCURRENCY=4                      Keep four data-parallel replicas busy
  SMOKE_ONLY=1                       Run strict compatibility checks only
  SETUP_ONLY=1                       Prepare the shared environment and dataset, then exit
  FORCE=1                            Replace outputs from an older run contract
  KEEP_MODEL_CACHE=1                 Retain downloaded model weights
  DRY_RUN=1                          Validate and print the resolved plan
  INTERNVL35_MAX_TOKENS=8192         Bound InternVL3.5, GLM-4.6V, MiniCPM-V, Qwen2.5-VL, and Qwen3-VL completions
  QWEN36_MAX_TOKENS=8192             Bound Qwen3.6 completions
  ANSWER_EXTRACTOR_MODEL=model-id    Dedicated extractor (default: Qwen/Qwen3-8B)
  ANSWER_EXTRACTOR_REVISION=commit   Immutable dedicated extractor revision
  ANSWER_EXTRACTOR_GPU_IDS=7         Launch the managed extractor on dedicated GPU(s)
  ANSWER_EXTRACTOR_ENDPOINTS=url     Use an already-managed extractor instead
  EXTRACT_EXISTING_ONLY=1            Extract every existing diagnostic; no images

The runner uses the original checkpoint tensors with BF16 compute. It never
loads 4-bit or 8-bit weights, resizes benchmark images, or creates a synthetic
answer. Every visual response is followed by a gold-blind text-only LLM
extraction request. Only that extractor decision is submitted. Missing,
ambiguous, malformed, empty, or failed responses become the standardized token
UNRESOLVED; there are no answer-format reruns or raw-output fallbacks.
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

write_active_run_marker() {
  ACTIVE_RUN_MARKER="$OUTPUT_ROOT/.active-run.json"
  ACTIVE_RUN_MARKER="$ACTIVE_RUN_MARKER" ACTIVE_RUN_PID="$$" \
  ACTIVE_RUN_MODELS="$MODELS" ACTIVE_RUN_TRACKS="$TRACKS" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["ACTIVE_RUN_MARKER"])
temporary = path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(
        {
            "pid": int(os.environ["ACTIVE_RUN_PID"]),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "models": os.environ["ACTIVE_RUN_MODELS"],
            "tracks": os.environ["ACTIVE_RUN_TRACKS"],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
}

cleanup_runner() {
  stop_extractor_server
  stop_server
  if [[ -n "$ACTIVE_RUN_MARKER" ]]; then
    rm -f -- "$ACTIVE_RUN_MARKER"
  fi
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

model_reasoning_profile() {
  case "$1" in
    internvl35-8b) printf '%s\n' 'thinking' ;;
    qwen36-27b) printf '%s\n' 'nonthinking' ;;
    *) printf '%s\n' 'nonthinking' ;;
  esac
}

model_max_tokens() {
  case "$1" in
    qwen36-27b) printf '%s\n' "$QWEN36_MAX_TOKENS" ;;
    qwen35-9b|internvl35-8b|glm46v-flash|minicpm-v46|qwen25-vl-7b|qwen3-vl-8b) printf '%s\n' "$INTERNVL35_MAX_TOKENS" ;;
    *) printf '%s\n' "$FINAL_ANSWER_MAX_TOKENS" ;;
  esac
}

model_max_tokens_policy() {
  case "$1" in
    qwen36-27b) printf '%s\n' 'explicit-model-completion-cap' ;;
    qwen35-9b|internvl35-8b|glm46v-flash|minicpm-v46|qwen25-vl-7b|qwen3-vl-8b) printf '%s\n' 'explicit-model-completion-cap' ;;
    *) printf '%s\n' 'explicit-total-completion-cap' ;;
  esac
}

track_max_tokens() {
  local slug="$1" track="$2"
  if [[ ( "$slug" == gemma3-*-it || "$slug" == "kimi-vl-a3b-instruct" || "$slug" == "llama32-11b-vision-instruct" ) && "$track" == "minds_eye" ]]; then
    printf '%s\n' "$INTERNVL35_MAX_TOKENS"
  else
    model_max_tokens "$slug"
  fi
}

track_max_tokens_policy() {
  local slug="$1" track="$2"
  if [[ ( "$slug" == gemma3-*-it || "$slug" == "kimi-vl-a3b-instruct" || "$slug" == "llama32-11b-vision-instruct" ) && "$track" == "minds_eye" ]]; then
    printf '%s\n' 'explicit-model-completion-cap'
  else
    model_max_tokens_policy "$slug"
  fi
}

track_stop_sequence() {
  case "$(track_prompt_mode "$1")" in
    cot) printf '%s\n' '</answer>' ;;
    noncot) printf '\n' ;;
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
  local slug="$1"
  case "$slug" in
    glm46v-flash|qwen36-27b) printf '%s\n' '{"enable_thinking":false}' ;;
    qwen35-9b) printf '%s\n' '{"enable_thinking":false}' ;;
    *) printf '%s\n' '{}' ;;
  esac
}

model_request_name() {
  local slug="$1" model_id="$2"
  if [[ "$slug" == "phi4-multimodal" ]]; then
    printf '%s\n' 'vision'
  else
    printf '%s\n' "$model_id"
  fi
}

model_adapter_name() {
  if [[ "$1" == "phi4-multimodal" ]]; then
    printf '%s\n' 'vision'
  fi
}

model_adapter_source() {
  if [[ "$1" == "phi4-multimodal" ]]; then
    printf '%s\n' 'vision-lora'
  fi
}

model_source_provider() {
  local slug="$1"
  if [[ "$slug" == "phi4-multimodal" && -n "$PHI_OFFICIAL_SNAPSHOT_PATH" ]]; then
    printf '%s\n' 'modelscope-official-git'
  else
    printf '%s\n' 'huggingface'
  fi
}

model_source_repo_id() {
  local slug="$1" model_id="$2"
  if [[ "$slug" == "phi4-multimodal" && -n "$PHI_OFFICIAL_SNAPSHOT_PATH" ]]; then
    printf '%s\n' "$PHI_OFFICIAL_SNAPSHOT_REPO_ID"
  else
    printf '%s\n' "$model_id"
  fi
}

model_source_revision() {
  local slug="$1" revision="$2"
  if [[ "$slug" == "phi4-multimodal" && -n "$PHI_OFFICIAL_SNAPSHOT_PATH" ]]; then
    printf '%s\n' "$PHI_OFFICIAL_SNAPSHOT_COMMIT"
  else
    printf '%s\n' "$revision"
  fi
}

model_source_objects() {
  local slug="$1"
  if [[ "$slug" == "phi4-multimodal" && -n "$PHI_OFFICIAL_SNAPSHOT_PATH" ]]; then
    printf '{"model-00001-of-00003.safetensors":"%s","model-00002-of-00003.safetensors":"%s","model-00003-of-00003.safetensors":"%s","vision-lora/adapter_model.safetensors":"%s"}\n' \
      "$PHI_MODEL_SHARD_1_SHA256" "$PHI_MODEL_SHARD_2_SHA256" \
      "$PHI_MODEL_SHARD_3_SHA256" "$PHI_VISION_ADAPTER_SHA256"
  else
    printf '%s\n' '{}'
  fi
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
    [[ "$DATA_PARALLEL_SIZE" =~ ^[0-9]+$ ]] && (( DATA_PARALLEL_SIZE > 0 )) \
      || die "DATA_PARALLEL_SIZE must be a positive integer."
    (( GPU_COUNT % DATA_PARALLEL_SIZE == 0 )) \
      || die "$GPU_COUNT GPU_IDS entries cannot be divided into DATA_PARALLEL_SIZE=$DATA_PARALLEL_SIZE replicas."
    TENSOR_PARALLEL_SIZE=$((GPU_COUNT / DATA_PARALLEL_SIZE))
  fi
  [[ "$TENSOR_PARALLEL_SIZE" =~ ^[0-9]+$ ]] && (( TENSOR_PARALLEL_SIZE > 0 )) \
    || die "TENSOR_PARALLEL_SIZE must be auto or a positive integer."
  [[ "$DATA_PARALLEL_SIZE" =~ ^[0-9]+$ ]] && (( DATA_PARALLEL_SIZE > 0 )) \
    || die "DATA_PARALLEL_SIZE must be a positive integer."
  (( TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE == GPU_COUNT )) \
    || die "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE x DATA_PARALLEL_SIZE=$DATA_PARALLEL_SIZE must match the $GPU_COUNT GPU_IDS entries."
}

resolve_max_num_seqs_per_replica() {
  if [[ "$MAX_NUM_SEQS_PER_REPLICA" == "auto" ]]; then
    MAX_NUM_SEQS_PER_REPLICA=$(((CONCURRENCY + DATA_PARALLEL_SIZE - 1) / DATA_PARALLEL_SIZE))
  fi
}

validate_settings() {
  local penalty_name penalty_value extractor_gpu visual_gpu
  local -a extractor_gpu_list=()
  command -v awk >/dev/null || die "awk is required."
  initialize_gpu_topology
  [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) \
    || die "PORT must be between 1 and 65535."
  validate_positive_integer "SMOKE_SAMPLES" "$SMOKE_SAMPLES"
  validate_positive_integer "CHECKPOINT_EVERY" "$CHECKPOINT_EVERY"
  validate_positive_integer "CONCURRENCY" "$CONCURRENCY"
  resolve_max_num_seqs_per_replica
  if [[ "$MAX_NUM_SEQS_PER_REPLICA" != "auto" ]]; then
    validate_positive_integer "MAX_NUM_SEQS_PER_REPLICA" "$MAX_NUM_SEQS_PER_REPLICA"
  fi
  validate_positive_integer "MODEL_START_TIMEOUT_SECONDS" "$MODEL_START_TIMEOUT_SECONDS"
  validate_positive_integer "MIN_FREE_DISK_GB" "$MIN_FREE_DISK_GB"
  validate_positive_integer "MIN_SYSTEM_RAM_GB" "$MIN_SYSTEM_RAM_GB"
  validate_positive_integer "MIN_FREE_GPU_MEMORY_MIB" "$MIN_FREE_GPU_MEMORY_MIB"
  validate_positive_integer "HF_HUB_DOWNLOAD_TIMEOUT" "$HF_HUB_DOWNLOAD_TIMEOUT"
  validate_flag "DISABLE_CUSTOM_ALL_REDUCE" "$DISABLE_CUSTOM_ALL_REDUCE"
  [[ "$SERVING_REPLICA_MODE" == "builtin" || "$SERVING_REPLICA_MODE" == "independent" ]] \
    || die "SERVING_REPLICA_MODE must be builtin or independent."
  if [[ "$SERVING_REPLICA_MODE" == "independent" ]]; then
    (( TENSOR_PARALLEL_SIZE == 1 )) \
      || die "Independent replica serving requires TENSOR_PARALLEL_SIZE=1."
    (( DATA_PARALLEL_SIZE == GPU_COUNT )) \
      || die "Independent replica serving requires one DATA_PARALLEL_SIZE replica per GPU."
    (( PORT + DATA_PARALLEL_SIZE - 1 < 65536 )) \
      || die "Independent replica ports exceed 65535. Choose a lower PORT."
  fi
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
  [[ -z "${COT_MAX_TOKENS:-}" && -z "${THINKING_MAX_TOKENS:-}" ]] \
    || die "COT_MAX_TOKENS and THINKING_MAX_TOKENS must be unset; completion caps are configured per model."
  validate_positive_integer "FINAL_ANSWER_MAX_TOKENS" "$FINAL_ANSWER_MAX_TOKENS"
  validate_positive_integer "INTERNVL35_MAX_TOKENS" "$INTERNVL35_MAX_TOKENS"
  validate_positive_integer "QWEN36_MAX_TOKENS" "$QWEN36_MAX_TOKENS"
  validate_positive_integer "ANSWER_EXTRACTOR_MAX_TOKENS" "$ANSWER_EXTRACTOR_MAX_TOKENS"
  validate_positive_integer "ANSWER_EXTRACTOR_PORT" "$ANSWER_EXTRACTOR_PORT"
  (( ANSWER_EXTRACTOR_PORT < 65536 )) \
    || die "ANSWER_EXTRACTOR_PORT must be below 65536."
  validate_positive_integer "ANSWER_EXTRACTOR_MAX_MODEL_LEN" "$ANSWER_EXTRACTOR_MAX_MODEL_LEN"
  [[ "$ANSWER_EXTRACTOR_GPU_MEMORY_UTILIZATION" =~ ^0([.][0-9]+)?$ ]] \
    && awk -v value="$ANSWER_EXTRACTOR_GPU_MEMORY_UTILIZATION" \
      'BEGIN { exit !(value > 0 && value < 1) }' \
    || die "ANSWER_EXTRACTOR_GPU_MEMORY_UTILIZATION must be greater than 0 and less than 1."
  [[ -n "$ANSWER_EXTRACTOR_MODEL" ]] || die "ANSWER_EXTRACTOR_MODEL must not be empty."
  [[ "$ANSWER_EXTRACTOR_REVISION" =~ ^[0-9a-f]{40}$ ]] \
    || die "ANSWER_EXTRACTOR_REVISION must be a full 40-character commit hash."
  if [[ -n "$ANSWER_EXTRACTOR_GPU_IDS" ]]; then
    [[ -z "$ANSWER_EXTRACTOR_ENDPOINTS" ]] \
      || die "Set either ANSWER_EXTRACTOR_GPU_IDS or ANSWER_EXTRACTOR_ENDPOINTS, not both."
    IFS=',' read -r -a extractor_gpu_list <<<"$ANSWER_EXTRACTOR_GPU_IDS"
    (( ${#extractor_gpu_list[@]} > 0 )) || die "ANSWER_EXTRACTOR_GPU_IDS is empty."
    for extractor_gpu in "${extractor_gpu_list[@]}"; do
      [[ "$extractor_gpu" =~ ^[0-9]+$ ]] \
        || die "ANSWER_EXTRACTOR_GPU_IDS must be comma-separated GPU indices."
      if [[ "$EXTRACT_EXISTING_ONLY" != "1" ]]; then
        for visual_gpu in "${GPU_ID_LIST[@]}"; do
          [[ "$extractor_gpu" != "$visual_gpu" ]] \
            || die "Extractor GPU $extractor_gpu overlaps visual GPU_IDS; use a dedicated GPU."
        done
      fi
    done
    ANSWER_EXTRACTOR_ENDPOINTS="http://127.0.0.1:$ANSWER_EXTRACTOR_PORT/v1"
    EXTRACTOR_SERVER_MANAGED=1
  else
    ANSWER_EXTRACTOR_ENDPOINTS="${ANSWER_EXTRACTOR_ENDPOINTS:-http://127.0.0.1:$ANSWER_EXTRACTOR_PORT/v1}"
  fi
  [[ "$ANSWER_EXTRACTOR_ENDPOINTS" =~ ^https?:// ]] \
    || die "ANSWER_EXTRACTOR_ENDPOINTS must contain absolute HTTP(S) URL(s)."
  [[ "$DYS_TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "DYS_TEMPERATURE must be non-negative."
  [[ "$MINDS_EYE_TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "MINDS_EYE_TEMPERATURE must be non-negative."
  validate_probability "DYS_TOP_P" "$DYS_TOP_P"
  validate_probability "MINDS_EYE_TOP_P" "$MINDS_EYE_TOP_P"
  [[ "$BASE_SEED" =~ ^[0-9]+$ ]] || die "BASE_SEED must be a non-negative integer."
  [[ "$ANSWER_EXTRACTOR_SEED" =~ ^[0-9]+$ ]] \
    || die "ANSWER_EXTRACTOR_SEED must be a non-negative integer."
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
  validate_flag "EXTRACT_EXISTING_ONLY" "$EXTRACT_EXISTING_ONLY"
  if [[ "$EXTRACT_EXISTING_ONLY" == "1" ]]; then
    [[ "$FORCE" != "1" ]] || die "EXTRACT_EXISTING_ONLY cannot be combined with FORCE=1."
    [[ "$SMOKE_ONLY" != "1" ]] || die "EXTRACT_EXISTING_ONLY cannot be combined with SMOKE_ONLY=1."
  fi
  [[ -n "$(selected_tracks)" ]] || die "TRACKS must select do_you_see_me, minds_eye, or all."
  (( $(selected_model_count) > 0 )) || die "MODELS did not match any configured model slug."
  if [[ -n "$PHI_OFFICIAL_SNAPSHOT_PATH" ]]; then
    is_enabled "$MODELS" "phi4-multimodal" \
      || die "PHI_OFFICIAL_SNAPSHOT_PATH is set but phi4-multimodal is not selected."
  fi
}

print_plan() {
  local track max_tokens dys_max_tokens me_max_tokens stop_sequence spec slug model_id revision loading model_max_len effective_model_len adapter_name reasoning_profile
  log "Evaluation plan"
  printf '  GPUs: %s (tensor parallel size %s, data parallel size %s, request concurrency %s)\n' \
    "$GPU_IDS" "$TENSOR_PARALLEL_SIZE" "$DATA_PARALLEL_SIZE" "$CONCURRENCY"
  printf '  Serving replica mode: %s\n' "$SERVING_REPLICA_MODE"
  printf '  Serving admission: at most %s active sequence(s) per data-parallel replica\n' \
    "$MAX_NUM_SEQS_PER_REPLICA"
  printf '  GPU allocation: utilization=%s; PYTORCH_CUDA_ALLOC_CONF=%s\n' \
    "$GPU_MEMORY_UTILIZATION" "$PYTORCH_CUDA_ALLOC_CONF"
  printf '  Weights: original checkpoint tensors, unquantized; compute: %s; KV cache: %s\n' \
    "$VLLM_DTYPE" "$VLLM_KV_CACHE_DTYPE"
  printf '  Serving engine: vLLM %s\n' "$VLLM_VERSION"
  printf '  Image preprocessing: original image bytes, no runner resize or recompression\n'
  printf '  Mandatory answer extraction: model=%s, endpoints=%s, text only, temperature=0, max_tokens=%s, seed=%s\n' \
    "${ANSWER_EXTRACTOR_MODEL:-same-as-inference}" \
    "${ANSWER_EXTRACTOR_ENDPOINTS:-same-as-inference}" \
    "$ANSWER_EXTRACTOR_MAX_TOKENS" "$ANSWER_EXTRACTOR_SEED"
  printf '  Missing/ambiguous response token: UNRESOLVED; extractor transport failures stop finalization\n'
  if [[ "$EXTRACT_EXISTING_ONLY" == "1" ]]; then
    printf '  Mode: extract every existing diagnostic response; no visual requests\n'
  fi
  printf '  Shared sampling: top_k=%s, min_p=%s, presence=%s, frequency=%s, repetition=%s\n' \
    "$SAMPLING_TOP_K" "$SAMPLING_MIN_P" "$PRESENCE_PENALTY" "$FREQUENCY_PENALTY" "$REPETITION_PENALTY"
  while IFS= read -r track; do
    stop_sequence="$(track_stop_sequence "$track")"
    printf '  Track %-14s prompt=%s, temperature=%s, top_p=%s' \
      "$track" "$(track_prompt_mode "$track")" "$(track_temperature "$track")" \
      "$(track_top_p "$track")"
    if [[ -n "$stop_sequence" ]]; then
      printf ', retained_stop=%s' "$stop_sequence"
    fi
    printf '\n'
  done < <(selected_tracks)
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r slug model_id revision loading model_max_len <<<"$spec"
    if is_enabled "$MODELS" "$slug"; then
      effective_model_len="$model_max_len"
      if [[ "$MAX_MODEL_LEN" != "auto" ]]; then
        effective_model_len="$MAX_MODEL_LEN"
      fi
      reasoning_profile="$(model_reasoning_profile "$slug")"
      dys_max_tokens="$(track_max_tokens "$slug" do_you_see_me)"
      me_max_tokens="$(track_max_tokens "$slug" minds_eye)"
      [[ -n "$dys_max_tokens" ]] || dys_max_tokens="uncapped ($(track_max_tokens_policy "$slug" do_you_see_me))"
      [[ -n "$me_max_tokens" ]] || me_max_tokens="uncapped ($(track_max_tokens_policy "$slug" minds_eye))"
      printf '  %-22s %-45s %s, context=%s\n' \
        "$slug" "$model_id" "$loading" "$effective_model_len"
      if [[ "$dys_max_tokens" == "$me_max_tokens" ]]; then
        max_tokens="$dys_max_tokens"
      else
        max_tokens="do_you_see_me=$dys_max_tokens, minds_eye=$me_max_tokens"
      fi
      printf '    Reasoning profile: %s; API max_tokens=%s; final_answer_max_tokens=%s\n' \
        "$reasoning_profile" "$max_tokens" "$FINAL_ANSWER_MAX_TOKENS"
      printf '    Chat template kwargs: %s\n' "$(track_chat_kwargs "$slug")"
      printf '    Engine mode: %s; server KV cache argument: %s\n' \
        "$(model_vllm_engine_mode "$slug")" "$(model_server_kv_cache_dtype "$slug")"
      adapter_name="$(model_adapter_name "$slug")"
      if [[ -n "$adapter_name" ]]; then
        printf '    Request model: %s; bundled adapter: %s\n' \
          "$(model_request_name "$slug" "$model_id")" "$(model_adapter_source "$slug")"
      fi
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
  free_kib="$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')"
  free_gib=$((free_kib / 1024 / 1024))
  (( free_gib >= MIN_FREE_DISK_GB )) \
    || die "Only ${free_gib} GiB is free; at least ${MIN_FREE_DISK_GB} GiB is required."
  ram_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
  [[ "$ram_kib" =~ ^[0-9]+$ ]] || die "Could not read total system memory."
  ram_gib=$((ram_kib / 1024 / 1024))
  (( ram_gib >= MIN_SYSTEM_RAM_GB )) \
    || die "The host has ${ram_gib} GiB RAM; at least ${MIN_SYSTEM_RAM_GB} GiB is required."
  log "Host preflight passed: $GPU_NAME; ${ram_gib} GiB RAM; ${free_gib} GiB cache disk free"
}

setup_environment() {
  local marker="$VENV_DIR/.ms-vista-vllm-${VLLM_VERSION}-scipy-${SCIPY_VERSION}-timm-${TIMM_VERSION}-unquantized-bf16-uv-${UV_VERSION}"
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
      "pillow==$PILLOW_VERSION" \
      "scipy==$SCIPY_VERSION" \
      "timm==$TIMM_VERSION"
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN="$("$PYTHON_BIN" - <<'PY'
from huggingface_hub import get_token

print(get_token() or "")
PY
)"
  fi

  "$PYTHON_BIN" - <<'PY'
from scipy.optimize import linear_sum_assignment
import timm
import torch
import vllm

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the NVIDIA driver and vLLM wheel.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected CUDA runtime does not expose native BF16 support.")
print(
    f"Environment ready: vLLM {vllm.__version__}, timm {timm.__version__}, "
    f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}"
)
PY
  : >"$marker"
}

verify_vllm_cli() {
  local help_text option
  if [[ "$VLLM_VERSION" == 0.10.* ]]; then
    help_text="$("$VLLM_BIN" serve --help 2>&1)" \
      || die "Could not inspect the installed vLLM serve command."
  else
    help_text="$("$VLLM_BIN" serve --help=all 2>&1)" \
      || die "Could not inspect the installed vLLM serve command."
  fi
  for option in \
    --host --port --served-model-name --revision --dtype --gpu-memory-utilization \
    --max-model-len --max-num-seqs --limit-mm-per-prompt --generation-config --kv-cache-dtype \
    --trust-remote-code --tensor-parallel-size; do
    [[ "$help_text" == *"$option"* ]] \
      || die "Installed vLLM $VLLM_VERSION does not expose required option $option."
  done
  if is_enabled "$MODELS" "phi4-multimodal"; then
    for option in --enable-lora --max-lora-rank --max-loras --lora-modules; do
      [[ "$help_text" == *"$option"* ]] \
        || die "Installed vLLM $VLLM_VERSION does not expose Phi-4 requirement $option."
    done
  fi
  log "vLLM serve CLI compatibility preflight passed"
}

prepare_dataset() {
  log "Downloading or validating the pinned public visual dataset"
  HF_TOKEN="${HF_TOKEN:-}" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
    HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
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
  if is_enabled "$MODELS" "phi4-multimodal"; then
    log "Applying audited Phi-4 vLLM mask-sum precision patch"
    "$PYTHON_BIN" -m evaluation.common.patch_vllm_phi4mm
  fi
}

model_compatibility_patch() {
  case "$1" in
    minicpm-v46) printf '%s\n' "$VLLM_STACKED_WEIGHT_PATCH_ID" ;;
    phi4-multimodal) printf '%s\n' "$VLLM_PHI4MM_MASK_SUM_PATCH_ID" ;;
    deepseek-vl2) printf '%s\n' "$VLLM_DEEPSEEK_VL2_CONFIG_OVERRIDE_ID" ;;
  esac
}

model_compatibility_patch_source() {
  case "$1" in
    minicpm-v46) printf '%s\n' 'evaluation/common/patch_vllm_weights_mapper.py' ;;
    phi4-multimodal) printf '%s\n' 'evaluation/common/patch_vllm_phi4mm.py' ;;
    deepseek-vl2) printf '%s\n' 'evaluation/run_visual_suite.sh' ;;
  esac
}

model_hf_overrides() {
  case "$1" in
    deepseek-vl2) printf '%s\n' '{"text_config":{"kv_lora_rank":512,"num_hidden_layers":30}}' ;;
  esac
}

model_vllm_engine_mode() {
  case "$1" in
    llama32-11b-vision-instruct) printf '%s\n' 'legacy-v0' ;;
    *) printf '%s\n' 'v1' ;;
  esac
}

model_server_kv_cache_dtype() {
  case "$1" in
    llama32-11b-vision-instruct) printf '%s\n' 'auto' ;;
    *) printf '%s\n' "$VLLM_KV_CACHE_DTYPE" ;;
  esac
}

prefetch_model_snapshot() {
  local model_id="$1" revision="$2" model_cache="$3"
  log "Prefetching the pinned $model_id snapshot once before launching $DATA_PARALLEL_SIZE data-parallel workers"
  HF_HOME="$model_cache" HF_HUB_CACHE="$model_cache/hub" \
    HUGGINGFACE_HUB_CACHE="$model_cache/hub" TRANSFORMERS_CACHE="$model_cache/hub" \
    HF_XET_CACHE="$model_cache/xet" HF_TOKEN="${HF_TOKEN:-}" \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
    HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
    HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" \
    "$PYTHON_BIN" - "$model_id" "$revision" <<'PY'
import sys

from huggingface_hub import snapshot_download

if sys.argv[1] == "meta-llama/Llama-3.2-11B-Vision-Instruct":
  snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    ignore_patterns=["*consolidated.pth"],
  )
else:
  snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2])
PY
}

download_phi_vision_adapter() {
  local model_id="$1" revision="$2" model_cache="$3"
  HF_HOME="$model_cache" HF_HUB_CACHE="$model_cache/hub" \
    HUGGINGFACE_HUB_CACHE="$model_cache/hub" TRANSFORMERS_CACHE="$model_cache/hub" \
    HF_XET_CACHE="$model_cache/xet" HF_TOKEN="${HF_TOKEN:-}" \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
    HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
    HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" \
    "$PYTHON_BIN" - "$model_id" "$revision" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot = Path(
    snapshot_download(
        repo_id=sys.argv[1],
        revision=sys.argv[2],
        allow_patterns=["vision-lora/*"],
    )
)
adapter = snapshot / "vision-lora"
required = (adapter / "adapter_config.json", adapter / "adapter_model.safetensors")
missing = [path.name for path in required if not path.is_file()]
if missing:
    raise SystemExit("Pinned Phi vision adapter is incomplete: " + ", ".join(missing))
print(adapter)
PY
}

verify_phi_official_snapshot() {
  local snapshot="$PHI_OFFICIAL_SNAPSHOT_PATH" commit
  [[ -n "$snapshot" ]] || die "PHI_OFFICIAL_SNAPSHOT_PATH is not set."
  [[ -d "$snapshot/.git" ]] \
    || die "Phi official snapshot is not a Git checkout: $snapshot"
  command -v git >/dev/null || die "git is required to verify the Phi official snapshot."
  commit="$(git -C "$snapshot" rev-parse HEAD 2>/dev/null || true)"
  [[ "$commit" == "$PHI_OFFICIAL_SNAPSHOT_COMMIT" ]] \
    || die "Phi official snapshot must be commit $PHI_OFFICIAL_SNAPSHOT_COMMIT, found ${commit:-<unreadable>}."

  PHI_SNAPSHOT_PATH="$snapshot" \
  PHI_SHARD_1_SHA256="$PHI_MODEL_SHARD_1_SHA256" \
  PHI_SHARD_2_SHA256="$PHI_MODEL_SHARD_2_SHA256" \
  PHI_SHARD_3_SHA256="$PHI_MODEL_SHARD_3_SHA256" \
  PHI_ADAPTER_SHA256="$PHI_VISION_ADAPTER_SHA256" \
    "$PYTHON_BIN" - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["PHI_SNAPSHOT_PATH"]).expanduser().resolve()

def sha256(file_path: Path) -> str:
  digest = hashlib.sha256()
  with file_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()

expected = {
    "model-00001-of-00003.safetensors": os.environ["PHI_SHARD_1_SHA256"],
    "model-00002-of-00003.safetensors": os.environ["PHI_SHARD_2_SHA256"],
    "model-00003-of-00003.safetensors": os.environ["PHI_SHARD_3_SHA256"],
    "vision-lora/adapter_model.safetensors": os.environ["PHI_ADAPTER_SHA256"],
}
for relative, digest in expected.items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"Phi official snapshot is missing {relative}")
    actual = sha256(path)
    if actual != digest:
        raise SystemExit(
            f"Phi official snapshot hash mismatch for {relative}: {actual} != {digest}"
        )
print(root)
PY
}

port_is_available() {
  local port="${1:-$PORT}"
  "$PYTHON_BIN" - "$port" <<'PY'
import socket
import sys

with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}

wait_for_port_release() {
  local timeout_seconds="$1" port="${2:-$PORT}" waited=0
  while ! port_is_available "$port"; do
    (( waited >= timeout_seconds )) && return 1
    sleep 1
    waited=$((waited + 1))
  done
}

server_serves_model() {
  local expected_model="$1" port="${2:-$PORT}"
  curl -fsS "http://127.0.0.1:$port/v1/models" 2>/dev/null \
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
  local port="${1:-$PORT}"
  curl -fsS "http://127.0.0.1:$port/v1/models" 2>/dev/null \
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
  if (( ${#INDEPENDENT_SERVER_PIDS[@]} > 0 )); then
    local pid port waited=0 alive
    log "Stopping ${#INDEPENDENT_SERVER_PIDS[@]} independent model servers"
    for pid in "${INDEPENDENT_SERVER_PIDS[@]}"; do
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    while (( waited < 60 )); do
      alive=0
      for pid in "${INDEPENDENT_SERVER_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive=1
      done
      (( alive == 0 )) && break
      sleep 2
      waited=$((waited + 2))
    done
    for pid in "${INDEPENDENT_SERVER_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      fi
      wait "$pid" 2>/dev/null || true
    done
    for port in "${INDEPENDENT_SERVER_PORTS[@]}"; do
      wait_for_port_release 30 "$port" \
        || printf 'Independent server port %s was not released.\n' "$port" >&2
    done
    INDEPENDENT_SERVER_PIDS=()
    INDEPENDENT_SERVER_PORTS=()
    INDEPENDENT_SERVER_LOGS=()
    SERVER_ENDPOINTS=""
    SERVER_PID=""
    SERVER_OWNS_PROCESS_GROUP=0
    return 0
  fi
  if [[ -z "$SERVER_PID" ]]; then
    return 0
  fi
  local server_pid="$SERVER_PID" owns_process_group="$SERVER_OWNS_PROCESS_GROUP"
  log "Stopping model server PID $server_pid"
  if [[ "$owns_process_group" == "1" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
  else
    kill -TERM "$server_pid" 2>/dev/null || true
  fi
  local waited=0
  while kill -0 "$server_pid" 2>/dev/null && (( waited < 60 )); do
    sleep 2
    waited=$((waited + 2))
  done
  if kill -0 "$server_pid" 2>/dev/null; then
    if [[ "$owns_process_group" == "1" ]]; then
      kill -KILL -- "-$server_pid" 2>/dev/null || true
    else
      kill -KILL "$server_pid" 2>/dev/null || true
    fi
  fi
  wait "$server_pid" 2>/dev/null || true
  if ! wait_for_port_release 60; then
    log "Port $PORT is still held after graceful model-server shutdown; forcing process-group cleanup"
    if [[ "$owns_process_group" == "1" ]]; then
      kill -KILL -- "-$server_pid" 2>/dev/null || true
    else
      kill -KILL "$server_pid" 2>/dev/null || true
    fi
    if ! wait_for_port_release 10; then
      printf 'Model server stopped, but port %s was not released after 70 seconds.\n' "$PORT" >&2
      SERVER_PID=""
      SERVER_OWNS_PROCESS_GROUP=0
      return 1
    fi
  fi
  SERVER_PID=""
  SERVER_OWNS_PROCESS_GROUP=0
}

stop_extractor_server() {
  [[ -n "$EXTRACTOR_SERVER_PID" ]] || return 0
  local pid="$EXTRACTOR_SERVER_PID" waited=0
  log "Stopping dedicated extractor server PID $pid"
  if [[ "$EXTRACTOR_SERVER_OWNS_PROCESS_GROUP" == "1" ]]; then
    kill -TERM -- "-$pid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  while kill -0 "$pid" 2>/dev/null && (( waited < 60 )); do
    sleep 2
    waited=$((waited + 2))
  done
  if kill -0 "$pid" 2>/dev/null; then
    if [[ "$EXTRACTOR_SERVER_OWNS_PROCESS_GROUP" == "1" ]]; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    else
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
  wait_for_port_release 60 "$ANSWER_EXTRACTOR_PORT" \
    || printf 'Extractor port %s was not released.\n' "$ANSWER_EXTRACTOR_PORT" >&2
  EXTRACTOR_SERVER_PID=""
  EXTRACTOR_SERVER_OWNS_PROCESS_GROUP=0
}

verify_extractor_endpoints() {
  EXTRACTOR_ENDPOINTS="$ANSWER_EXTRACTOR_ENDPOINTS" \
  EXTRACTOR_MODEL="$ANSWER_EXTRACTOR_MODEL" "$PYTHON_BIN" - <<'PY'
import json
import os
from urllib.request import urlopen

expected = os.environ["EXTRACTOR_MODEL"]
for endpoint in os.environ["EXTRACTOR_ENDPOINTS"].split(","):
    endpoint = endpoint.strip().rstrip("/")
    with urlopen(f"{endpoint}/models", timeout=10) as response:
        payload = json.load(response)
    models = {
        str(item.get("id") or "")
        for item in payload.get("data", [])
        if isinstance(item, dict)
    }
    if expected not in models:
        raise SystemExit(
            f"Extractor endpoint {endpoint} serves {sorted(models)}, not {expected}."
        )
PY
}

start_extractor_server() {
  if [[ "$EXTRACTOR_SERVER_MANAGED" != "1" ]]; then
    log "Verifying externally managed extractor endpoint(s) for $ANSWER_EXTRACTOR_MODEL"
    verify_extractor_endpoints
    return
  fi
  port_is_available "$ANSWER_EXTRACTOR_PORT" \
    || die "Extractor port $ANSWER_EXTRACTOR_PORT is already in use."
  local model_cache="$CACHE_ROOT/models/qwen3-8b-extractor"
  local extractor_tp
  local -a extractor_gpu_list=()
  IFS=',' read -r -a extractor_gpu_list <<<"$ANSWER_EXTRACTOR_GPU_IDS"
  extractor_tp="${#extractor_gpu_list[@]}"
  mkdir -p "$model_cache" "$OUTPUT_ROOT"
  EXTRACTOR_SERVER_LOG="$OUTPUT_ROOT/qwen3-8b-extractor.vllm.log"
  local -a command=(
    "$VLLM_BIN" serve "$ANSWER_EXTRACTOR_MODEL"
    --host 127.0.0.1
    --port "$ANSWER_EXTRACTOR_PORT"
    --served-model-name "$ANSWER_EXTRACTOR_MODEL"
    --revision "$ANSWER_EXTRACTOR_REVISION"
    --dtype bfloat16
    --kv-cache-dtype bfloat16
    --tensor-parallel-size "$extractor_tp"
    --data-parallel-size 1
    --gpu-memory-utilization "$ANSWER_EXTRACTOR_GPU_MEMORY_UTILIZATION"
    --max-model-len "$ANSWER_EXTRACTOR_MAX_MODEL_LEN"
    --max-num-seqs "$CONCURRENCY"
    --generation-config vllm
    --default-chat-template-kwargs "$ANSWER_EXTRACTOR_CHAT_TEMPLATE_KWARGS"
    --trust-remote-code
  )
  log "Starting dedicated $ANSWER_EXTRACTOR_MODEL extractor at revision ${ANSWER_EXTRACTOR_REVISION:0:12} on GPU(s) $ANSWER_EXTRACTOR_GPU_IDS"
  log "Extractor startup log: $EXTRACTOR_SERVER_LOG"
  if command -v setsid >/dev/null; then
    setsid env \
      HF_HOME="$model_cache" HF_HUB_CACHE="$model_cache/hub" \
      HUGGINGFACE_HUB_CACHE="$model_cache/hub" TRANSFORMERS_CACHE="$model_cache/hub" \
      HF_XET_CACHE="$model_cache/xet" HF_TOKEN="${HF_TOKEN:-}" \
      HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
      HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES="$ANSWER_EXTRACTOR_GPU_IDS" \
      PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" PYTHONUNBUFFERED=1 \
      "${command[@]}" >"$EXTRACTOR_SERVER_LOG" 2>&1 &
    EXTRACTOR_SERVER_OWNS_PROCESS_GROUP=1
  else
    env \
      HF_HOME="$model_cache" HF_HUB_CACHE="$model_cache/hub" \
      HUGGINGFACE_HUB_CACHE="$model_cache/hub" TRANSFORMERS_CACHE="$model_cache/hub" \
      HF_XET_CACHE="$model_cache/xet" HF_TOKEN="${HF_TOKEN:-}" \
      HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
      HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES="$ANSWER_EXTRACTOR_GPU_IDS" \
      PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" PYTHONUNBUFFERED=1 \
      "${command[@]}" >"$EXTRACTOR_SERVER_LOG" 2>&1 &
    EXTRACTOR_SERVER_OWNS_PROCESS_GROUP=0
  fi
  EXTRACTOR_SERVER_PID=$!
  local elapsed=0
  while (( elapsed < MODEL_START_TIMEOUT_SECONDS )); do
    if ! kill -0 "$EXTRACTOR_SERVER_PID" 2>/dev/null; then
      printf 'Dedicated extractor exited during startup. Last log lines:\n' >&2
      tail -n 80 "$EXTRACTOR_SERVER_LOG" >&2 || true
      stop_extractor_server
      return 1
    fi
    if server_serves_model "$ANSWER_EXTRACTOR_MODEL" "$ANSWER_EXTRACTOR_PORT"; then
      log "Dedicated extractor is ready and serves $ANSWER_EXTRACTOR_MODEL"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed % 60 == 0 )); then
      log "Waiting for dedicated extractor startup (${elapsed}s elapsed)"
    fi
  done
  printf 'Dedicated extractor did not become ready within %s seconds.\n' \
    "$MODEL_START_TIMEOUT_SECONDS" >&2
  stop_extractor_server
  return 1
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
  local request_model adapter_path="" model_source="$model_id" source_revision="$revision" hf_overrides
  local vllm_engine_mode server_kv_cache_dtype
  local -a server_environment=()
  [[ "$loading" == "unquantized" ]] || die "Refusing unsupported weight loading mode: $loading"
  request_model="$(model_request_name "$slug" "$model_id")"
  hf_overrides="$(model_hf_overrides "$slug")"
  vllm_engine_mode="$(model_vllm_engine_mode "$slug")"
  server_kv_cache_dtype="$(model_server_kv_cache_dtype "$slug")"
  if [[ "$vllm_engine_mode" == "legacy-v0" ]]; then
    server_environment+=(VLLM_USE_V1=0)
  fi
  SERVER_LOG="$OUTPUT_ROOT/$slug/vllm.log"
  mkdir -p "$(dirname "$SERVER_LOG")" "$model_cache"
  : >"$SERVER_LOG"
  port_is_available || die "Port $PORT is already in use. Set PORT to an unused local port."

  if [[ "$slug" == "phi4-multimodal" && -n "$PHI_OFFICIAL_SNAPSHOT_PATH" ]]; then
    log "Verifying the immutable official Microsoft Phi snapshot"
    verify_phi_official_snapshot >/dev/null
    model_source="$(cd "$PHI_OFFICIAL_SNAPSHOT_PATH" && pwd -P)"
    source_revision="$PHI_OFFICIAL_SNAPSHOT_COMMIT"
    adapter_path="$model_source/vision-lora"
  fi

  if (( DATA_PARALLEL_SIZE > 1 )) && [[ "$model_source" == "$model_id" ]]; then
    if ! prefetch_model_snapshot "$model_id" "$revision" "$model_cache"; then
      printf 'Could not prefetch %s at pinned revision %s before data-parallel startup.\n' \
        "$model_id" "$revision" >&2
      return 1
    fi
  fi

  if [[ "$SERVING_REPLICA_MODE" == "independent" ]]; then
    [[ "$model_source" == "$model_id" ]] \
      || die "Independent replica serving currently supports Hugging Face model sources only."
    local replica replica_port replica_log replica_pid gpu endpoint
    local -a replica_command
    SERVER_LOG="$OUTPUT_ROOT/$slug/vllm.replica-0.log"
    SERVER_ENDPOINTS=""
    for replica in "${!GPU_ID_LIST[@]}"; do
      gpu="${GPU_ID_LIST[$replica]}"
      replica_port=$((PORT + replica))
      replica_log="$OUTPUT_ROOT/$slug/vllm.replica-$replica.log"
      port_is_available "$replica_port" \
        || die "Independent replica port $replica_port is already in use."
      : >"$replica_log"
      replica_command=(
        "$VLLM_BIN" serve "$model_source"
        --host 127.0.0.1
        --port "$replica_port"
        --served-model-name "$model_id"
        --dtype "$VLLM_DTYPE"
        --kv-cache-dtype "$server_kv_cache_dtype"
        --tensor-parallel-size 1
        --data-parallel-size 1
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --max-model-len "$model_max_len"
        --max-num-seqs "$MAX_NUM_SEQS_PER_REPLICA"
        --limit-mm-per-prompt '{"image":1}'
        --generation-config vllm
        --trust-remote-code
        --revision "$revision"
      )
      if [[ -n "$hf_overrides" ]]; then
        replica_command+=(--hf-overrides "$hf_overrides")
      fi
      setsid env \
        HF_HOME="$model_cache" HF_HUB_CACHE="$model_cache/hub" \
        HUGGINGFACE_HUB_CACHE="$model_cache/hub" TRANSFORMERS_CACHE="$model_cache/hub" \
        HF_XET_CACHE="$model_cache/xet" HF_TOKEN="${HF_TOKEN:-}" \
        HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
        HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
        HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" TOKENIZERS_PARALLELISM=false \
        CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
        PYTHONUNBUFFERED=1 \
        "${server_environment[@]}" \
        "${replica_command[@]}" >"$replica_log" 2>&1 &
      replica_pid=$!
      INDEPENDENT_SERVER_PIDS+=("$replica_pid")
      INDEPENDENT_SERVER_PORTS+=("$replica_port")
      INDEPENDENT_SERVER_LOGS+=("$replica_log")
      endpoint="http://127.0.0.1:$replica_port/v1"
      SERVER_ENDPOINTS+="${SERVER_ENDPOINTS:+,}$endpoint"
      log "Started independent replica $replica on GPU $gpu, port $replica_port, PID $replica_pid"
    done

    local elapsed=0 ready_count latest
    while (( elapsed < MODEL_START_TIMEOUT_SECONDS )); do
      ready_count=0
      for replica in "${!INDEPENDENT_SERVER_PIDS[@]}"; do
        replica_pid="${INDEPENDENT_SERVER_PIDS[$replica]}"
        replica_port="${INDEPENDENT_SERVER_PORTS[$replica]}"
        replica_log="${INDEPENDENT_SERVER_LOGS[$replica]}"
        if ! kill -0 "$replica_pid" 2>/dev/null; then
          printf 'Independent replica %s exited during startup. Last log lines:\n' "$replica" >&2
          tail -n 80 "$replica_log" >&2 || true
          stop_server
          return 1
        fi
        if curl -fsS "http://127.0.0.1:$replica_port/health" >/dev/null 2>&1 \
          && server_serves_model "$request_model" "$replica_port"; then
          ready_count=$((ready_count + 1))
        fi
      done
      if (( ready_count == DATA_PARALLEL_SIZE )); then
        log "All $ready_count independent replicas are ready and serve request model $request_model"
        return 0
      fi
      sleep 5
      elapsed=$((elapsed + 5))
      if (( elapsed % 60 == 0 )); then
        latest="$(tail -c 16384 "${INDEPENDENT_SERVER_LOGS[0]}" 2>/dev/null | tr '\r' '\n' | awk 'NF { line = $0 } END { print line }')" || true
        log "Waiting for independent replicas ($ready_count/$DATA_PARALLEL_SIZE ready, ${elapsed}s elapsed)"
        [[ -n "$latest" ]] && printf '  Latest replica-0 log: %.300s\n' "$latest"
      fi
    done
    printf 'Independent replicas did not become ready within %s seconds.\n' "$MODEL_START_TIMEOUT_SECONDS" >&2
    stop_server
    return 1
  fi

  local -a command=(
    "$VLLM_BIN" serve "$model_source"
    --host 127.0.0.1
    --port "$PORT"
    --served-model-name "$model_id"
    --dtype "$VLLM_DTYPE"
    --kv-cache-dtype "$server_kv_cache_dtype"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$model_max_len"
    --max-num-seqs "$MAX_NUM_SEQS_PER_REPLICA"
    --limit-mm-per-prompt '{"image":1}'
    --generation-config vllm
    --trust-remote-code
  )
  if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
    command+=(--disable-custom-all-reduce)
  fi
  if [[ -n "$hf_overrides" ]]; then
    command+=(--hf-overrides "$hf_overrides")
  fi
  if [[ "$model_source" == "$model_id" ]]; then
    command+=(--revision "$revision")
  fi

  if [[ "$slug" == "phi4-multimodal" ]]; then
    if [[ -z "$adapter_path" ]]; then
      log "Resolving the pinned Phi-4 vision adapter"
      if ! adapter_path="$(download_phi_vision_adapter "$model_id" "$revision" "$model_cache")"; then
        printf 'Could not download the required Phi-4 vision adapter at revision %s.\n' "$revision" >&2
        return 1
      fi
    fi
    command+=(
      --enable-lora
      --max-lora-rank 320
      --max-loras 1
      --lora-modules "vision=$adapter_path"
    )
  fi

  log "Starting $model_id from $(model_source_provider "$slug") revision ${source_revision:0:12} (unquantized $VLLM_DTYPE, TP=$TENSOR_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE, concurrency=$CONCURRENCY, max sequences/replica=$MAX_NUM_SEQS_PER_REPLICA)"
  log "Model startup log: $SERVER_LOG"
  if command -v setsid >/dev/null; then
    setsid env \
      HF_HOME="$model_cache" HF_HUB_CACHE="$model_cache/hub" \
      HUGGINGFACE_HUB_CACHE="$model_cache/hub" TRANSFORMERS_CACHE="$model_cache/hub" \
      HF_XET_CACHE="$model_cache/xet" HF_TOKEN="${HF_TOKEN:-}" \
      HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
      HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
      HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
      PYTHONUNBUFFERED=1 \
      "${server_environment[@]}" \
      "${command[@]}" >"$SERVER_LOG" 2>&1 &
    SERVER_OWNS_PROCESS_GROUP=1
  else
    env \
      HF_HOME="$model_cache" HF_HUB_CACHE="$model_cache/hub" \
      HUGGINGFACE_HUB_CACHE="$model_cache/hub" TRANSFORMERS_CACHE="$model_cache/hub" \
      HF_XET_CACHE="$model_cache/xet" HF_TOKEN="${HF_TOKEN:-}" \
      HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
      HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
      HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
      PYTHONUNBUFFERED=1 \
      "${server_environment[@]}" \
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
      if server_serves_model "$request_model"; then
        log "Model server is ready and serves request model $request_model"
        return 0
      fi
      identity_mismatches=$((identity_mismatches + 1))
      if (( identity_mismatches >= 3 )); then
        served_models="$(server_model_names 2>/dev/null || printf '<unavailable>')"
        printf 'Port %s is healthy but serves [%s], not request model %s. Another evaluation likely claimed the same PORT or a required adapter failed to register. Stop the conflicting runner, assign distinct PORT values, or use evaluation/run_visual_suite_multi_gpu.sh.\n' \
          "$PORT" "$served_models" "$request_model" >&2
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
  local prompt_mode max_tokens stop_sequence temperature top_p chat_kwargs
  local extractor_model extractor_endpoints extractor_chat_kwargs
  prompt_mode="$(track_prompt_mode "$track")"
  max_tokens="$(track_max_tokens "$slug" "$track")"
  stop_sequence="$(track_stop_sequence "$track")"
  temperature="$(track_temperature "$track")"
  top_p="$(track_top_p "$track")"
  chat_kwargs="$(track_chat_kwargs "$slug" "$track")"
  extractor_model="${ANSWER_EXTRACTOR_MODEL:-$model_id}"
  extractor_endpoints="${ANSWER_EXTRACTOR_ENDPOINTS:-${SERVER_ENDPOINTS:-http://127.0.0.1:$PORT/v1}}"
  extractor_chat_kwargs="${ANSWER_EXTRACTOR_CHAT_TEMPLATE_KWARGS:-$chat_kwargs}"
  RUNNER_ARGS=(
    --model "$model_id"
    --endpoints "${SERVER_ENDPOINTS:-http://127.0.0.1:$PORT/v1}"
    --api-key EMPTY
    --image-root "$DATASET_DIR"
    --prompt-mode "$prompt_mode"
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
    --extractor-model "$extractor_model"
    --extractor-revision "$ANSWER_EXTRACTOR_REVISION"
    --extractor-endpoints "$extractor_endpoints"
    --extractor-seed "$ANSWER_EXTRACTOR_SEED"
    --extractor-max-tokens "$ANSWER_EXTRACTOR_MAX_TOKENS"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --max-final-answer-tokens "$FINAL_ANSWER_MAX_TOKENS"
  )
  if [[ -n "$max_tokens" ]]; then
    RUNNER_ARGS+=(--max-tokens "$max_tokens")
  fi
  if [[ -n "$stop_sequence" ]]; then
    RUNNER_ARGS+=(--stop "$stop_sequence" --include-stop-str-in-output)
  fi
  if [[ "$chat_kwargs" != "{}" ]]; then
    RUNNER_ARGS+=(--chat-template-kwargs "$chat_kwargs")
  fi
  if [[ "$extractor_chat_kwargs" != "{}" ]]; then
    RUNNER_ARGS+=(--extractor-chat-template-kwargs "$extractor_chat_kwargs")
  fi
}

run_track() {
  local slug="$1" model_id="$2" track="$3"
  local module questions model_dir output diagnostics smoke_diagnostics
  module="$(track_module "$track")"
  questions="$(track_questions "$track")"
  model_dir="$OUTPUT_ROOT/$slug"
  output="$model_dir/${track}_submission.jsonl"
  diagnostics="$model_dir/${track}.diagnostics.jsonl"
  smoke_diagnostics="$model_dir/${track}.smoke.diagnostics.jsonl"
  mkdir -p "$model_dir"

  if [[ "$EXTRACT_EXISTING_ONLY" != "1" && "$FORCE" != "1" ]] \
    && validate_submission "$output" "$questions" >/dev/null 2>&1; then
    log "$slug/$track already has a complete canonical output; skipping"
    return 0
  fi
  if [[ "$FORCE" == "1" ]]; then
    rm -f -- "$output" "$diagnostics" "$smoke_diagnostics"
    rm -f -- "$model_dir/${track}.attempt-"*.diagnostics.jsonl
    rm -f -- "$model_dir/${track}.smoke.attempt-"*.diagnostics.jsonl
  elif [[ -f "$output" ]]; then
    mv -- "$output" "$output.invalid.$(date -u '+%Y%m%dT%H%M%SZ')"
  fi
  runner_base_args "$slug" "$model_id" "$track" "$BASE_SEED"
  if [[ "$EXTRACT_EXISTING_ONLY" == "1" ]]; then
    [[ -f "$diagnostics" ]] || {
      printf 'Evaluation failed: %s/%s has no existing diagnostics at %s\n' \
        "$slug" "$track" "$diagnostics" >&2
      return 1
    }
    log "Running mandatory gold-blind extraction for every existing $slug/$track response"
    "$PYTHON_BIN" -m "$module" \
      "${RUNNER_ARGS[@]}" \
      --questions "$questions" \
      --resume \
      --extract-existing-diagnostics \
      --out "$output" \
      --diagnostics "$diagnostics" \
      && validate_submission "$output" "$questions"
    return
  fi

  rm -f -- "$smoke_diagnostics"
  log "Running strict $SMOKE_SAMPLES-sample smoke test with mandatory extraction for $slug/$track"
  if ! "$PYTHON_BIN" -m "$module" \
    "${RUNNER_ARGS[@]}" \
    --questions "$questions" \
    --limit "$SMOKE_SAMPLES" \
    --strict-partial \
    --diagnostics "$smoke_diagnostics"; then
    printf 'Evaluation failed: %s/%s mandatory-extractor smoke test failed. Diagnostics remain in %s.\n' \
      "$slug" "$track" "$smoke_diagnostics" >&2
    return 1
  fi
  if [[ "$SMOKE_ONLY" == "1" ]]; then
    return 0
  fi

  log "Running $slug/$track full evaluation with mandatory extraction"
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

  printf 'Evaluation failed: %s/%s did not complete its single inference-plus-extraction pass. Diagnostics remain in %s. No submission file was finalized.\n' \
    "$slug" "$track" "$diagnostics" >&2
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
  resolve_max_num_seqs_per_replica
  mkdir -p "$OUTPUT_ROOT/$slug"
  RUN_CONFIG_PATH="$OUTPUT_ROOT/$slug/.run_config.json" \
  RUN_CONFIG_MODEL_ID="$model_id" RUN_CONFIG_REVISION="$revision" \
  RUN_CONFIG_LOADING="$loading" RUN_CONFIG_MODEL_MAX_LEN="$model_max_len" \
  RUN_CONFIG_DTYPE="$VLLM_DTYPE" RUN_CONFIG_KV_CACHE_DTYPE="$VLLM_KV_CACHE_DTYPE" \
  RUN_CONFIG_TP_SIZE="$TENSOR_PARALLEL_SIZE" RUN_CONFIG_DP_SIZE="$DATA_PARALLEL_SIZE" \
  RUN_CONFIG_CONCURRENCY="$CONCURRENCY" RUN_CONFIG_MAX_NUM_SEQS="$MAX_NUM_SEQS_PER_REPLICA" \
  RUN_CONFIG_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  RUN_CONFIG_DISABLE_CUSTOM_ALL_REDUCE="$DISABLE_CUSTOM_ALL_REDUCE" \
  RUN_CONFIG_SERVING_REPLICA_MODE="$SERVING_REPLICA_MODE" \
  RUN_CONFIG_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  RUN_CONFIG_REASONING_PROFILE="$(model_reasoning_profile "$slug")" \
  RUN_CONFIG_FINAL_ANSWER_MAX_TOKENS="$FINAL_ANSWER_MAX_TOKENS" \
  RUN_CONFIG_ANSWER_EXTRACTION_METHOD="$ANSWER_EXTRACTION_METHOD_ID" \
  RUN_CONFIG_ANSWER_EXTRACTOR_MODEL="$ANSWER_EXTRACTOR_MODEL" \
  RUN_CONFIG_ANSWER_EXTRACTOR_REVISION="$ANSWER_EXTRACTOR_REVISION" \
  RUN_CONFIG_ANSWER_EXTRACTOR_ENDPOINTS="$ANSWER_EXTRACTOR_ENDPOINTS" \
  RUN_CONFIG_ANSWER_EXTRACTOR_CHAT_KWARGS="$ANSWER_EXTRACTOR_CHAT_TEMPLATE_KWARGS" \
  RUN_CONFIG_ANSWER_EXTRACTOR_MAX_TOKENS="$ANSWER_EXTRACTOR_MAX_TOKENS" \
  RUN_CONFIG_ANSWER_EXTRACTOR_SEED="$ANSWER_EXTRACTOR_SEED" \
  RUN_CONFIG_REQUEST_MODEL="$(model_request_name "$slug" "$model_id")" \
  RUN_CONFIG_SOURCE_PROVIDER="$(model_source_provider "$slug")" \
  RUN_CONFIG_SOURCE_REPO_ID="$(model_source_repo_id "$slug" "$model_id")" \
  RUN_CONFIG_SOURCE_REVISION="$(model_source_revision "$slug" "$revision")" \
  RUN_CONFIG_SOURCE_OBJECTS="$(model_source_objects "$slug")" \
  RUN_CONFIG_ADAPTER_NAME="$(model_adapter_name "$slug")" \
  RUN_CONFIG_ADAPTER_SOURCE="$(model_adapter_source "$slug")" \
  RUN_CONFIG_ADAPTER_REVISION="$(model_source_revision "$slug" "$revision")" \
  RUN_CONFIG_PROJECT_ROOT="$PROJECT_ROOT" RUN_CONFIG_DATASET_REPO_ID="$DATASET_REPO_ID" \
  RUN_CONFIG_DATASET_REVISION="$DATASET_REVISION" RUN_CONFIG_VLLM_VERSION="$VLLM_VERSION" \
  RUN_CONFIG_VLLM_ENGINE_MODE="$(model_vllm_engine_mode "$slug")" \
  RUN_CONFIG_SERVER_KV_CACHE_DTYPE="$(model_server_kv_cache_dtype "$slug")" \
  RUN_CONFIG_COMPATIBILITY_PATCH="$(model_compatibility_patch "$slug")" \
  RUN_CONFIG_HF_OVERRIDES="$(model_hf_overrides "$slug")" \
  RUN_CONFIG_UNPARSEABLE_POLICY="$UNPARSEABLE_ANSWER_POLICY_ID" \
  RUN_CONFIG_PIPELINE_REVISION="$PIPELINE_REVISION_ID" \
  RUN_CONFIG_EXTRACT_EXISTING_ONLY="$EXTRACT_EXISTING_ONLY" \
  RUN_CONFIG_BASE_SEED="$BASE_SEED" RUN_CONFIG_FORCE="$FORCE" \
  RUN_CONFIG_TOP_K="$SAMPLING_TOP_K" RUN_CONFIG_MIN_P="$SAMPLING_MIN_P" \
  RUN_CONFIG_PRESENCE_PENALTY="$PRESENCE_PENALTY" RUN_CONFIG_FREQUENCY_PENALTY="$FREQUENCY_PENALTY" \
  RUN_CONFIG_REPETITION_PENALTY="$REPETITION_PENALTY" \
  RUN_CONFIG_DYS_PROMPT_MODE="$DYS_PROMPT_MODE" RUN_CONFIG_DYS_MAX_TOKENS="$(track_max_tokens "$slug" do_you_see_me)" \
  RUN_CONFIG_DYS_MAX_TOKENS_POLICY="$(track_max_tokens_policy "$slug" do_you_see_me)" \
  RUN_CONFIG_DYS_STOP_SEQUENCE="$(track_stop_sequence do_you_see_me)" \
  RUN_CONFIG_DYS_TEMPERATURE="$DYS_TEMPERATURE" RUN_CONFIG_DYS_TOP_P="$DYS_TOP_P" \
  RUN_CONFIG_DYS_CHAT_KWARGS="$(track_chat_kwargs "$slug" do_you_see_me)" \
  RUN_CONFIG_ME_PROMPT_MODE="$MINDS_EYE_PROMPT_MODE" RUN_CONFIG_ME_MAX_TOKENS="$(track_max_tokens "$slug" minds_eye)" \
  RUN_CONFIG_ME_MAX_TOKENS_POLICY="$(track_max_tokens_policy "$slug" minds_eye)" \
  RUN_CONFIG_ME_STOP_SEQUENCE="$(track_stop_sequence minds_eye)" \
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
  digest = hashlib.sha256()
  with file_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()

def protocol(prefix: str, track: str) -> dict:
    raw_max_tokens = os.environ[f"RUN_CONFIG_{prefix}_MAX_TOKENS"]
    max_tokens = int(raw_max_tokens) if raw_max_tokens else None
    final_answer_max_tokens = int(os.environ["RUN_CONFIG_FINAL_ANSWER_MAX_TOKENS"])
    needs_separate_answer_check = (
        max_tokens is None or max_tokens > final_answer_max_tokens
    )
    stop_sequence = os.environ[f"RUN_CONFIG_{prefix}_STOP_SEQUENCE"]
    return {
        "prompt_mode": os.environ[f"RUN_CONFIG_{prefix}_PROMPT_MODE"],
        "max_tokens": max_tokens,
        "max_tokens_policy": os.environ[f"RUN_CONFIG_{prefix}_MAX_TOKENS_POLICY"],
          "final_answer_max_tokens": final_answer_max_tokens,
          "final_answer_token_enforcement": (
            "post-extraction-served-model-tokenizer"
            if needs_separate_answer_check
            else "total-completion-cap"
          ),
        "stop_sequences": [stop_sequence] if stop_sequence else [],
        "include_stop_str_in_output": bool(stop_sequence),
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
  "schema_version": 11,
    "model_id": os.environ["RUN_CONFIG_MODEL_ID"],
    "model_revision": os.environ["RUN_CONFIG_REVISION"],
    "reasoning_profile": os.environ["RUN_CONFIG_REASONING_PROFILE"],
  "checkpoint_source": {
    "provider": os.environ["RUN_CONFIG_SOURCE_PROVIDER"],
    "repo_id": os.environ["RUN_CONFIG_SOURCE_REPO_ID"],
    "revision": os.environ["RUN_CONFIG_SOURCE_REVISION"],
    "object_sha256": json.loads(os.environ["RUN_CONFIG_SOURCE_OBJECTS"]),
  },
    "serving_engine": {
        "name": "vllm",
        "version": os.environ["RUN_CONFIG_VLLM_VERSION"],
        "engine_mode": os.environ["RUN_CONFIG_VLLM_ENGINE_MODE"],
        "request_model": os.environ["RUN_CONFIG_REQUEST_MODEL"],
      "kv_cache_cli_value": os.environ["RUN_CONFIG_SERVER_KV_CACHE_DTYPE"],
      "max_num_seqs_per_replica": int(os.environ["RUN_CONFIG_MAX_NUM_SEQS"]),
      "gpu_memory_utilization": float(
        os.environ["RUN_CONFIG_GPU_MEMORY_UTILIZATION"]
      ),
      "cuda_allocator_config": os.environ["RUN_CONFIG_CUDA_ALLOC_CONF"],
      **(
        {"disable_custom_all_reduce": True}
        if os.environ["RUN_CONFIG_DISABLE_CUSTOM_ALL_REDUCE"] == "1"
        else {}
      ),
      **(
        {"replica_mode": "independent-processes"}
        if os.environ["RUN_CONFIG_SERVING_REPLICA_MODE"] == "independent"
        else {}
      ),
      **(
        {"hf_overrides": json.loads(os.environ["RUN_CONFIG_HF_OVERRIDES"])}
        if os.environ["RUN_CONFIG_HF_OVERRIDES"]
        else {}
      ),
    },
    "weight_loading": os.environ["RUN_CONFIG_LOADING"],
    "compute_dtype": os.environ["RUN_CONFIG_DTYPE"],
    "kv_cache_dtype": os.environ["RUN_CONFIG_KV_CACHE_DTYPE"],
    "tensor_parallel_size": int(os.environ["RUN_CONFIG_TP_SIZE"]),
    "data_parallel_size": int(os.environ["RUN_CONFIG_DP_SIZE"]),
    "request_concurrency": int(os.environ["RUN_CONFIG_CONCURRENCY"]),
    "max_model_len": int(os.environ["RUN_CONFIG_MODEL_MAX_LEN"]),
    "dataset": {
        "repo_id": os.environ["RUN_CONFIG_DATASET_REPO_ID"],
        "revision": os.environ["RUN_CONFIG_DATASET_REVISION"],
    },
    "generation": {
        "do_you_see_me": protocol("DYS", "do_you_see_me"),
        "minds_eye": protocol("ME", "minds_eye"),
        "base_seed": int(os.environ["RUN_CONFIG_BASE_SEED"]),
    },
    "image_preprocessing": "original-bytes-no-runner-resize-or-recompression",
    "answer_extraction": {
      "mode": "mandatory-after-every-model-response",
      "authoritative_answer_field": "diagnostics.extracted_answer",
      "extractor": {
        "method": os.environ["RUN_CONFIG_ANSWER_EXTRACTION_METHOD"],
        "model": os.environ["RUN_CONFIG_ANSWER_EXTRACTOR_MODEL"],
        "revision": os.environ["RUN_CONFIG_ANSWER_EXTRACTOR_REVISION"],
        "endpoints": os.environ["RUN_CONFIG_ANSWER_EXTRACTOR_ENDPOINTS"].split(","),
        "input_fields": [
          "question",
          "answer_type",
          "answer_domain",
          "required_output_format",
          "response_metadata",
          "candidate_response",
        ],
        "image_supplied": False,
        "ground_truth_supplied": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": int(os.environ["RUN_CONFIG_ANSWER_EXTRACTOR_SEED"]),
        "max_tokens": int(os.environ["RUN_CONFIG_ANSWER_EXTRACTOR_MAX_TOKENS"]),
        "stop_sequences": ["</answer>"],
        "include_stop_str_in_output": True,
        "chat_template_kwargs": json.loads(
          os.environ["RUN_CONFIG_ANSWER_EXTRACTOR_CHAT_KWARGS"]
        ),
        "support_validation": "answer-must-be-stated-in-candidate-response",
        "source_provenance": [
          "diagnostics_filename",
          "candidate-response-utf8-sha256",
        ],
        "output_contract": "exactly-one-answer-block",
        "unresolved_token": "UNRESOLVED",
      },
    },
    "unparseable_answers": {
      "policy": os.environ["RUN_CONFIG_UNPARSEABLE_POLICY"],
      "standardized_token": "UNRESOLVED",
      "visual_answer_retries": 0,
      "local_parser_answer_selection": False,
      "raw_output_fallback": False,
    },
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
adapter_name = os.environ["RUN_CONFIG_ADAPTER_NAME"]
if adapter_name:
    desired["adapter"] = {
        "name": adapter_name,
        "source": os.environ["RUN_CONFIG_ADAPTER_SOURCE"],
        "revision": os.environ["RUN_CONFIG_ADAPTER_REVISION"],
    }
patch = os.environ["RUN_CONFIG_COMPATIBILITY_PATCH"]
if patch:
    desired["compatibility_patches"] = [patch]

def extraction_upgrade_invariants(config: dict) -> dict:
  invariant = json.loads(json.dumps(config))
  for key in (
    "schema_version",
    "answer_extraction",
    "unparseable_answers",
    "pipeline_revision",
    "artifact_migrations",
  ):
    invariant.pop(key, None)
  invariant.get("source_hashes", {}).pop("runner", None)
  invariant.get("generation", {}).pop("format_retry_attempts", None)
  serving_engine = invariant.get("serving_engine", {})
  serving_engine.pop("max_num_seqs_per_replica", None)
  serving_engine.pop("gpu_memory_utilization", None)
  serving_engine.pop("cuda_allocator_config", None)
  return invariant

def is_supported_extraction_upgrade(existing: dict) -> bool:
  return (
    os.environ["RUN_CONFIG_EXTRACT_EXISTING_ONLY"] == "1"
    and (
      existing.get("schema_version"), existing.get("pipeline_revision")
    ) == (10, "unquantized-bf16-smoke-and-full-text-extraction-v10")
    and desired.get("schema_version") == 11
    and desired.get("pipeline_revision")
    == "unquantized-bf16-mandatory-extraction-v11"
    and extraction_upgrade_invariants(existing)
    == extraction_upgrade_invariants(desired)
  )

def serving_resource_invariants(config: dict) -> dict:
  invariant = json.loads(json.dumps(config))
  invariant.pop("artifact_migrations", None)
  serving_engine = invariant.get("serving_engine", {})
  serving_engine.pop("max_num_seqs_per_replica", None)
  serving_engine.pop("gpu_memory_utilization", None)
  serving_engine.pop("cuda_allocator_config", None)
  return invariant

def is_supported_serving_resource_upgrade(existing: dict) -> bool:
  serving_engine = existing.get("serving_engine", {})
  return (
    existing.get("schema_version") == 10
    and existing.get("pipeline_revision")
    == "unquantized-bf16-smoke-and-full-text-extraction-v10"
    and not {
      "max_num_seqs_per_replica",
      "gpu_memory_utilization",
      "cuda_allocator_config",
    }.intersection(serving_engine)
    and serving_resource_invariants(existing)
    == serving_resource_invariants(desired)
  )

def local_parser_upgrade_invariants(config: dict) -> dict:
  invariant = json.loads(json.dumps(config))
  invariant.pop("artifact_migrations", None)
  invariant.get("answer_extraction", {}).pop("local_parser", None)
  invariant.get("source_hashes", {}).get("runner", {}).pop(
    "visual_pipeline", None
  )
  return invariant

def is_supported_local_parser_upgrade(existing: dict) -> bool:
  return (
    existing.get("schema_version") == desired.get("schema_version") == 10
    and existing.get("pipeline_revision")
    == desired.get("pipeline_revision")
    == "unquantized-bf16-smoke-and-full-text-extraction-v10"
    and existing.get("answer_extraction", {}).get("local_parser")
    in {
      "strict-local-final-answer-parser-v2-number-words-zero-through-twenty",
      "strict-local-final-answer-parser-v3-number-words-and-explicit-odd-figure",
      "strict-local-final-answer-parser-v4-innermost-glm-box",
    }
    and desired.get("answer_extraction", {}).get("local_parser")
    == "strict-local-final-answer-parser-v5-innermost-answer-and-glm-box"
    and local_parser_upgrade_invariants(existing)
    == local_parser_upgrade_invariants(desired)
  )

def raw_output_finalization_upgrade_invariants(config: dict) -> dict:
  invariant = json.loads(json.dumps(config))
  invariant.pop("artifact_migrations", None)
  invariant.pop("unparseable_answers", None)
  invariant.get("answer_extraction", {}).pop("local_parser", None)
  invariant.get("source_hashes", {}).pop("runner", None)
  return invariant

def is_supported_raw_output_finalization_upgrade(existing: dict) -> bool:
  return (
    existing.get("schema_version") == desired.get("schema_version") == 10
    and existing.get("pipeline_revision")
    == desired.get("pipeline_revision")
    == "unquantized-bf16-smoke-and-full-text-extraction-v10"
    and existing.get("unparseable_answers", {}).get("policy")
    == "deterministic-smoke-and-full-visual-retries-then-text-extraction-v4"
    and desired.get("unparseable_answers", {}).get("policy")
    == "deterministic-retries-text-extraction-then-exact-raw-output-v5"
    and existing.get("answer_extraction", {}).get("local_parser")
    in {
      "strict-local-final-answer-parser-v3-number-words-and-explicit-odd-figure",
      "strict-local-final-answer-parser-v4-innermost-glm-box",
      "strict-local-final-answer-parser-v5-innermost-answer-and-glm-box",
    }
    and desired.get("answer_extraction", {}).get("local_parser")
    == "strict-local-final-answer-parser-v5-innermost-answer-and-glm-box"
    and raw_output_finalization_upgrade_invariants(existing)
    == raw_output_finalization_upgrade_invariants(desired)
  )

def smoke_admission_upgrade_invariants(config: dict) -> dict:
  invariant = json.loads(json.dumps(config))
  invariant.pop("artifact_migrations", None)
  invariant.get("unparseable_answers", {}).pop("smoke_admission", None)
  return invariant

def is_supported_smoke_admission_upgrade(existing: dict) -> bool:
  return (
    existing.get("schema_version") == desired.get("schema_version") == 10
    and existing.get("pipeline_revision") == desired.get("pipeline_revision")
    and existing.get("model_id")
    == desired.get("model_id")
    == "moonshotai/Kimi-VL-A3B-Instruct"
    and "smoke_admission" not in existing.get("unparseable_answers", {})
    and desired.get("unparseable_answers", {}).get("smoke_admission")
    == {
      "applies_after": ["smoke_visual_retries", "same_model_text_extraction"],
      "eligible_records": "complete-nonerror-nonempty-output",
      "canonical_output_effect": "none",
    }
    and smoke_admission_upgrade_invariants(existing)
    == smoke_admission_upgrade_invariants(desired)
  )

def glm_extended_retry_invariants(config: dict) -> dict:
  invariant = json.loads(json.dumps(config))
  invariant.pop("artifact_migrations", None)
  invariant.get("generation", {}).pop("format_retry_attempts", None)
  invariant.get("answer_extraction", {}).pop("local_parser", None)
  invariant.get("source_hashes", {}).get("runner", {}).pop(
    "visual_pipeline", None
  )
  return invariant

def is_supported_glm_extended_retry(existing: dict) -> bool:
  model_dir = path.parent
  return (
    existing.get("schema_version") == desired.get("schema_version") == 10
    and existing.get("pipeline_revision")
    == desired.get("pipeline_revision")
    == "unquantized-bf16-smoke-and-full-text-extraction-v10"
    and existing.get("model_id")
    == desired.get("model_id")
    == "zai-org/GLM-4.6V-Flash"
    and (model_dir / "minds_eye.diagnostics.jsonl").is_file()
    and not list(model_dir.glob("*_submission.jsonl"))
    and existing.get("generation", {}).get("format_retry_attempts") == 3
    and desired.get("generation", {}).get("format_retry_attempts") == 6
    and existing.get("answer_extraction", {}).get("local_parser")
    in {
      "strict-local-final-answer-parser-v3-number-words-and-explicit-odd-figure",
      "strict-local-final-answer-parser-v4-innermost-glm-box",
      "strict-local-final-answer-parser-v5-innermost-answer-and-glm-box",
    }
    and desired.get("answer_extraction", {}).get("local_parser")
    == "strict-local-final-answer-parser-v5-innermost-answer-and-glm-box"
    and glm_extended_retry_invariants(existing)
    == glm_extended_retry_invariants(desired)
  )

def completion_cap_recovery_invariants(config: dict) -> dict:
  invariant = json.loads(json.dumps(config))
  invariant.pop("artifact_migrations", None)
  for key in (
    "tensor_parallel_size",
    "data_parallel_size",
    "request_concurrency",
  ):
    invariant.pop(key, None)
  serving_engine = invariant.get("serving_engine", {})
  for key in (
    "max_num_seqs_per_replica",
    "gpu_memory_utilization",
    "cuda_allocator_config",
  ):
    serving_engine.pop(key, None)
  for track in ("do_you_see_me", "minds_eye"):
    generation = invariant.get("generation", {}).get(track, {})
    for key in (
      "max_tokens",
      "max_tokens_policy",
      "final_answer_token_enforcement",
    ):
      generation.pop(key, None)
  return invariant

def glm_completion_recovery_invariants(config: dict) -> dict:
  invariant = completion_cap_recovery_invariants(config)
  invariant.get("generation", {}).pop("base_seed", None)
  return invariant

def completion_protocol_matches(
  config: dict,
  *,
  max_tokens: int,
  policy: str,
  enforcement: str,
) -> bool:
  return all(
    config.get("generation", {}).get(track, {}).get("max_tokens") == max_tokens
    and config["generation"][track].get("max_tokens_policy") == policy
    and config["generation"][track].get("final_answer_max_tokens") == 200
    and config["generation"][track].get("final_answer_token_enforcement")
    == enforcement
    for track in ("do_you_see_me", "minds_eye")
  )

def is_supported_completion_cap_recovery(existing: dict) -> bool:
  model_dir = path.parent
  return (
    existing.get("schema_version") == desired.get("schema_version") == 10
    and existing.get("pipeline_revision")
    == desired.get("pipeline_revision")
    == "unquantized-bf16-smoke-and-full-text-extraction-v10"
    and existing.get("model_id")
    == desired.get("model_id")
    == "Qwen/Qwen2.5-VL-7B-Instruct"
    and (model_dir / "minds_eye.diagnostics.jsonl").is_file()
    and not list(model_dir.glob("*_submission.jsonl"))
    and not (model_dir / "do_you_see_me.diagnostics.jsonl").exists()
    and completion_protocol_matches(
      existing,
      max_tokens=200,
      policy="explicit-total-completion-cap",
      enforcement="total-completion-cap",
    )
    and completion_protocol_matches(
      desired,
      max_tokens=8192,
      policy="explicit-model-completion-cap",
      enforcement="post-extraction-served-model-tokenizer",
    )
    and completion_cap_recovery_invariants(existing)
    == completion_cap_recovery_invariants(desired)
  )

def gemma_completion_cap_recovery_invariants(config: dict) -> dict:
  invariant = json.loads(json.dumps(config))
  invariant.pop("artifact_migrations", None)
  generation = invariant.get("generation", {}).get("minds_eye", {})
  for key in (
    "max_tokens",
    "max_tokens_policy",
    "final_answer_token_enforcement",
  ):
    generation.pop(key, None)
  return invariant

def is_supported_gemma_completion_cap_recovery(existing: dict) -> bool:
  model_dir = path.parent
  dys_diagnostics = model_dir / "do_you_see_me.diagnostics.jsonl"
  dys_submission = model_dir / "do_you_see_me_submission.jsonl"
  smoke_files = list(model_dir.glob("minds_eye.smoke*.diagnostics.jsonl"))
  return (
    existing.get("schema_version") == desired.get("schema_version") == 10
    and existing.get("pipeline_revision")
    == desired.get("pipeline_revision")
    == "unquantized-bf16-smoke-and-full-text-extraction-v10"
    and existing.get("model_id")
    == desired.get("model_id")
    == "google/gemma-3-12b-it"
    and dys_diagnostics.is_file()
    and dys_submission.is_file()
    and smoke_files
    and not (model_dir / "minds_eye.diagnostics.jsonl").exists()
    and not (model_dir / "minds_eye_submission.jsonl").exists()
    and existing.get("generation", {}).get("do_you_see_me")
    == desired.get("generation", {}).get("do_you_see_me")
    and existing.get("generation", {}).get("minds_eye", {}).get("max_tokens")
    == 200
    and existing["generation"]["minds_eye"].get("max_tokens_policy")
    == "explicit-total-completion-cap"
    and existing["generation"]["minds_eye"].get(
      "final_answer_token_enforcement"
    ) == "total-completion-cap"
    and desired.get("generation", {}).get("minds_eye", {}).get("max_tokens")
    == 8192
    and desired["generation"]["minds_eye"].get("max_tokens_policy")
    == "explicit-model-completion-cap"
    and desired["generation"]["minds_eye"].get(
      "final_answer_token_enforcement"
    ) == "post-extraction-served-model-tokenizer"
    and gemma_completion_cap_recovery_invariants(existing)
    == gemma_completion_cap_recovery_invariants(desired)
  )

def is_supported_glm_completion_recovery(existing: dict) -> bool:
  model_dir = path.parent
  return (
    existing.get("schema_version") == desired.get("schema_version") == 10
    and existing.get("pipeline_revision")
    == desired.get("pipeline_revision")
    == "unquantized-bf16-smoke-and-full-text-extraction-v10"
    and existing.get("model_id")
    == desired.get("model_id")
    == "zai-org/GLM-4.6V-Flash"
    and (model_dir / "do_you_see_me.diagnostics.jsonl").is_file()
    and not list(model_dir.glob("*_submission.jsonl"))
    and not (model_dir / "minds_eye.diagnostics.jsonl").exists()
    and existing.get("generation", {}).get("base_seed") == 0
    and desired.get("generation", {}).get("base_seed") == 3
    and completion_protocol_matches(
      existing,
      max_tokens=200,
      policy="explicit-total-completion-cap",
      enforcement="total-completion-cap",
    )
    and completion_protocol_matches(
      desired,
      max_tokens=8192,
      policy="explicit-model-completion-cap",
      enforcement="post-extraction-served-model-tokenizer",
    )
    and glm_completion_recovery_invariants(existing)
    == glm_completion_recovery_invariants(desired)
  )

def serving_topology_resume_invariants(config: dict) -> dict:
  invariant = json.loads(json.dumps(config))
  invariant.pop("artifact_migrations", None)
  invariant.pop("data_parallel_size", None)
  invariant.pop("request_concurrency", None)
  return invariant

def is_supported_serving_topology_resume(existing: dict) -> bool:
  model_dir = path.parent
  has_checkpoint = any(
    (model_dir / f"{track}.diagnostics.jsonl").is_file()
    for track in ("do_you_see_me", "minds_eye")
  )
  topology_changed = (
    existing.get("data_parallel_size") != desired.get("data_parallel_size")
    or existing.get("request_concurrency") != desired.get("request_concurrency")
  )
  return (
    existing.get("schema_version") == desired.get("schema_version") == 10
    and existing.get("pipeline_revision")
    == desired.get("pipeline_revision")
    == "unquantized-bf16-smoke-and-full-text-extraction-v10"
    and has_checkpoint
    and topology_changed
    and not list(model_dir.glob("*_submission.jsonl"))
    and serving_topology_resume_invariants(existing)
    == serving_topology_resume_invariants(desired)
  )

def serving_protocol(config: dict) -> dict:
  return {
    "tensor_parallel_size": config["tensor_parallel_size"],
    "data_parallel_size": config["data_parallel_size"],
    "request_concurrency": config["request_concurrency"],
    "max_num_seqs_per_replica": config["serving_engine"][
      "max_num_seqs_per_replica"
    ],
    "gpu_memory_utilization": config["serving_engine"][
      "gpu_memory_utilization"
    ],
    "cuda_allocator_config": config["serving_engine"][
      "cuda_allocator_config"
    ],
  }

def archive_completion_cap_recovery(existing: dict) -> dict:
  model_dir = path.parent
  diagnostics = model_dir / "minds_eye.diagnostics.jsonl"
  diagnostics_archive = (
    model_dir / "minds_eye.pre-completion-cap-recovery.diagnostics.jsonl"
  )
  diagnostics_bytes = diagnostics.read_bytes()
  diagnostics_digest = hashlib.sha256(diagnostics_bytes).hexdigest()
  if diagnostics_archive.exists():
    if sha256(diagnostics_archive) != diagnostics_digest:
      raise SystemExit(
        f"Recovery archive differs from {diagnostics.name}: "
        f"{diagnostics_archive.name}. Refusing to overwrite it."
      )
  else:
    diagnostics_archive.write_bytes(diagnostics_bytes)

  rows = [
    json.loads(line)
    for line in diagnostics_bytes.decode("utf-8").splitlines()
    if line.strip()
  ]
  archived_attempts = []
  for pattern in (
    "minds_eye.attempt-*.diagnostics.jsonl",
    "minds_eye.smoke.attempt-*.diagnostics.jsonl",
  ):
    for source in sorted(model_dir.glob(pattern)):
      archived = model_dir / source.name.replace(
        "minds_eye.", "minds_eye.pre-completion-cap-recovery.", 1
      )
      source_digest = sha256(source)
      if archived.exists():
        if sha256(archived) != source_digest:
          raise SystemExit(
            f"Recovery archive differs from {source.name}: "
            f"{archived.name}. Refusing to overwrite it."
          )
        source.unlink()
      else:
        os.replace(source, archived)
      archived_attempts.append(
        {
          "original_file": source.name,
          "archived_file": archived.name,
          "sha256": source_digest,
        }
      )

  return {
    "reason": "provenance-preserving-completion-cap-recovery",
    "scope": "resume-valid-records-and-regenerate-only-unresolved",
    "baseline_diagnostics": {
      "file": diagnostics_archive.name,
      "sha256": diagnostics_digest,
      "row_count": len(rows),
      "extractor_error_count": sum(bool(row.get("extractor_error")) for row in rows),
    },
    "archived_attempt_diagnostics": archived_attempts,
    "previous_generation": existing["generation"]["minds_eye"],
    "current_generation": desired["generation"]["minds_eye"],
    "previous_serving": serving_protocol(existing),
    "current_serving": serving_protocol(desired),
  }

def archive_gemma_completion_cap_recovery(existing: dict) -> dict:
  model_dir = path.parent
  completed_artifacts = {}
  for filename in (
    "do_you_see_me.diagnostics.jsonl",
    "do_you_see_me_submission.jsonl",
  ):
    artifact = model_dir / filename
    rows = [
      json.loads(line)
      for line in artifact.read_text(encoding="utf-8").splitlines()
      if line.strip()
    ]
    if len(rows) != 4500:
      raise SystemExit(
        f"Completed Gemma DYS artifact has {len(rows)} rows instead of 4500: "
        f"{artifact.name}"
      )
    completed_artifacts[filename] = {
      "row_count": len(rows),
      "sha256": sha256(artifact),
    }

  archived_smoke = []
  for source in sorted(model_dir.glob("minds_eye.smoke*.diagnostics.jsonl")):
    archived = model_dir / source.name.replace(
      "minds_eye.", "minds_eye.pre-completion-cap-recovery.", 1
    )
    source_digest = sha256(source)
    if archived.exists():
      if sha256(archived) != source_digest:
        raise SystemExit(
          f"Recovery archive differs from {source.name}: {archived.name}. "
          "Refusing to overwrite it."
        )
      source.unlink()
    else:
      os.replace(source, archived)
    archived_smoke.append(
      {
        "original_file": source.name,
        "archived_file": archived.name,
        "sha256": source_digest,
      }
    )

  return {
    "reason": "provenance-preserving-gemma-minds-eye-completion-cap-recovery",
    "scope": "preserve-completed-dys-and-rerun-minds-eye-smoke",
    "completed_do_you_see_me_artifacts": completed_artifacts,
    "archived_smoke_diagnostics": archived_smoke,
    "previous_generation": existing["generation"]["minds_eye"],
    "current_generation": desired["generation"]["minds_eye"],
  }

def archive_glm_completion_recovery(existing: dict) -> dict:
  model_dir = path.parent
  diagnostics = model_dir / "do_you_see_me.diagnostics.jsonl"
  diagnostics_archive = (
    model_dir / "do_you_see_me.pre-completion-cap-recovery.diagnostics.jsonl"
  )
  diagnostics_bytes = diagnostics.read_bytes()
  diagnostics_digest = hashlib.sha256(diagnostics_bytes).hexdigest()
  if diagnostics_archive.exists():
    if sha256(diagnostics_archive) != diagnostics_digest:
      raise SystemExit(
        f"Recovery archive differs from {diagnostics.name}: "
        f"{diagnostics_archive.name}. Refusing to overwrite it."
      )
  else:
    diagnostics_archive.write_bytes(diagnostics_bytes)

  rows = [
    json.loads(line)
    for line in diagnostics_bytes.decode("utf-8").splitlines()
    if line.strip()
  ]
  archived_diagnostics = []
  sources = [
    *sorted(model_dir.glob("do_you_see_me.attempt-*.diagnostics.jsonl")),
    *sorted(model_dir.glob("do_you_see_me.smoke.attempt-*.diagnostics.jsonl")),
  ]
  smoke = model_dir / "do_you_see_me.smoke.diagnostics.jsonl"
  if smoke.is_file():
    sources.append(smoke)
  for source in sources:
    archived = model_dir / source.name.replace(
      "do_you_see_me.", "do_you_see_me.pre-completion-cap-recovery.", 1
    )
    source_digest = sha256(source)
    if archived.exists():
      if sha256(archived) != source_digest:
        raise SystemExit(
          f"Recovery archive differs from {source.name}: "
          f"{archived.name}. Refusing to overwrite it."
        )
      source.unlink()
    else:
      os.replace(source, archived)
    archived_diagnostics.append(
      {
        "original_file": source.name,
        "archived_file": archived.name,
        "sha256": source_digest,
      }
    )

  return {
    "reason": "provenance-preserving-completion-and-seed-recovery",
    "scope": "resume-valid-records-and-regenerate-only-unresolved",
    "baseline_diagnostics": {
      "file": diagnostics_archive.name,
      "sha256": diagnostics_digest,
      "row_count": len(rows),
      "extractor_error_count": sum(bool(row.get("extractor_error")) for row in rows),
    },
    "archived_diagnostics": archived_diagnostics,
    "previous_generation": existing["generation"]["do_you_see_me"],
    "current_generation": desired["generation"]["do_you_see_me"],
    "previous_base_seed": existing["generation"]["base_seed"],
    "current_base_seed": desired["generation"]["base_seed"],
    "previous_serving": serving_protocol(existing),
    "current_serving": serving_protocol(desired),
  }

def archive_glm_extended_retry(existing: dict) -> dict:
  model_dir = path.parent
  diagnostics = model_dir / "minds_eye.diagnostics.jsonl"
  diagnostics_archive = (
    model_dir / "minds_eye.pre-extended-format-retry.diagnostics.jsonl"
  )
  diagnostics_bytes = diagnostics.read_bytes()
  diagnostics_digest = hashlib.sha256(diagnostics_bytes).hexdigest()
  if diagnostics_archive.exists():
    if sha256(diagnostics_archive) != diagnostics_digest:
      raise SystemExit(
        f"Retry archive differs from {diagnostics.name}: "
        f"{diagnostics_archive.name}. Refusing to overwrite it."
      )
  else:
    diagnostics_archive.write_bytes(diagnostics_bytes)

  rows = [
    json.loads(line)
    for line in diagnostics_bytes.decode("utf-8").splitlines()
    if line.strip()
  ]
  for source in sorted(model_dir.glob("minds_eye.attempt-*.diagnostics.jsonl")):
    archived = model_dir / source.name.replace(
      "minds_eye.", "minds_eye.pre-extended-format-retry.", 1
    )
    source_digest = sha256(source)
    if archived.exists():
      if sha256(archived) != source_digest:
        raise SystemExit(
          f"Retry archive differs from {source.name}: "
          f"{archived.name}. Refusing to overwrite it."
        )
      source.unlink()
    else:
      os.replace(source, archived)

  archived_attempts = []
  for archived in sorted(
    model_dir.glob(
      "minds_eye.pre-extended-format-retry.attempt-*.diagnostics.jsonl"
    )
  ):
    archived_attempts.append(
      {
        "original_file": archived.name.replace(
          "minds_eye.pre-extended-format-retry.", "minds_eye.", 1
        ),
        "archived_file": archived.name,
        "sha256": sha256(archived),
      }
    )

  smoke = model_dir / "minds_eye.smoke.diagnostics.jsonl"
  smoke_archive = (
    model_dir / "minds_eye.pre-extended-format-retry.smoke.diagnostics.jsonl"
  )
  archived_smoke = None
  if smoke.is_file():
    smoke_digest = sha256(smoke)
    if smoke_archive.exists():
      if sha256(smoke_archive) != smoke_digest:
        raise SystemExit(
          f"Retry archive differs from {smoke.name}: "
          f"{smoke_archive.name}. Refusing to overwrite it."
        )
    else:
      smoke_archive.write_bytes(smoke.read_bytes())
    archived_smoke = {
      "file": smoke_archive.name,
      "sha256": smoke_digest,
    }

  return {
    "reason": "provenance-preserving-glm-extended-format-retry",
    "scope": "resume-valid-records-and-regenerate-only-unresolved",
    "baseline_diagnostics": {
      "file": diagnostics_archive.name,
      "sha256": diagnostics_digest,
      "row_count": len(rows),
      "extractor_error_count": sum(bool(row.get("extractor_error")) for row in rows),
    },
    "archived_attempt_diagnostics": archived_attempts,
    **({"archived_smoke_diagnostics": archived_smoke} if archived_smoke else {}),
    "previous_format_retry_attempts": existing["generation"][
      "format_retry_attempts"
    ],
    "current_format_retry_attempts": desired["generation"][
      "format_retry_attempts"
    ],
    "previous_local_parser": existing["answer_extraction"]["local_parser"],
    "current_local_parser": desired["answer_extraction"]["local_parser"],
    "previous_visual_pipeline_sha256": existing["source_hashes"]["runner"][
      "visual_pipeline"
    ],
    "current_visual_pipeline_sha256": desired["source_hashes"]["runner"][
      "visual_pipeline"
    ],
  }

def archive_serving_topology_resume(existing: dict) -> dict:
  model_dir = path.parent
  baselines = {}
  archived_attempts = []
  for track in ("do_you_see_me", "minds_eye"):
    diagnostics = model_dir / f"{track}.diagnostics.jsonl"
    if diagnostics.is_file():
      diagnostics_archive = (
        model_dir / f"{track}.pre-serving-topology-resume.diagnostics.jsonl"
      )
      diagnostics_bytes = diagnostics.read_bytes()
      diagnostics_digest = hashlib.sha256(diagnostics_bytes).hexdigest()
      if diagnostics_archive.exists():
        if sha256(diagnostics_archive) != diagnostics_digest:
          raise SystemExit(
            f"Topology archive differs from {diagnostics.name}: "
            f"{diagnostics_archive.name}. Refusing to overwrite it."
          )
      else:
        diagnostics_archive.write_bytes(diagnostics_bytes)
      baselines[track] = {
        "file": diagnostics_archive.name,
        "sha256": diagnostics_digest,
        "row_count": sum(
          bool(line.strip())
          for line in diagnostics_bytes.decode("utf-8").splitlines()
        ),
      }

    for pattern in (
      f"{track}.attempt-*.diagnostics.jsonl",
      f"{track}.smoke.attempt-*.diagnostics.jsonl",
    ):
      for source in sorted(model_dir.glob(pattern)):
        archived = model_dir / source.name.replace(
          f"{track}.", f"{track}.pre-serving-topology-resume.", 1
        )
        source_digest = sha256(source)
        if archived.exists():
          if sha256(archived) != source_digest:
            raise SystemExit(
              f"Topology archive differs from {source.name}: "
              f"{archived.name}. Refusing to overwrite it."
            )
          source.unlink()
        else:
          os.replace(source, archived)
        archived_attempts.append(
          {
            "original_file": source.name,
            "archived_file": archived.name,
            "sha256": source_digest,
          }
        )

  return {
    "reason": "provenance-preserving-serving-topology-resume",
    "baseline_diagnostics": baselines,
    "archived_attempt_diagnostics": archived_attempts,
    "previous_serving": serving_protocol(existing),
    "current_serving": serving_protocol(desired),
  }

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
    if existing.get("artifact_migrations"):
        desired["artifact_migrations"] = existing["artifact_migrations"]
    if existing != desired and artifacts:
        if is_supported_extraction_upgrade(existing):
          desired["artifact_migrations"] = [
            *existing.get("artifact_migrations", []),
            {
              "reason": "provenance-preserving-answer-extraction-upgrade",
              "from_schema_version": existing["schema_version"],
              "from_pipeline_revision": existing["pipeline_revision"],
              "previous_answer_extraction": existing.get("answer_extraction"),
              "previous_unparseable_answers": existing.get("unparseable_answers"),
              "previous_runner_source_hashes": existing.get("source_hashes", {}).get(
                "runner", {}
              ),
              "to_schema_version": desired["schema_version"],
              "to_pipeline_revision": desired["pipeline_revision"],
            },
          ]
          print(
            f"Migrated {path.parent.name} run fingerprint from schema "
            f"{existing['schema_version']} to {desired['schema_version']}; "
            "existing diagnostics were preserved."
          )
        elif is_supported_serving_resource_upgrade(existing):
          desired["artifact_migrations"] = [
            *existing.get("artifact_migrations", []),
            {
              "reason": "provenance-preserving-serving-resource-upgrade",
              "schema_version": existing["schema_version"],
              "previous": {
                "max_num_seqs_per_replica": existing["request_concurrency"],
                "gpu_memory_utilization": 0.88,
                "cuda_allocator_config": "default",
              },
              "current": {
                "max_num_seqs_per_replica": desired["serving_engine"][
                  "max_num_seqs_per_replica"
                ],
                "gpu_memory_utilization": desired["serving_engine"][
                  "gpu_memory_utilization"
                ],
                "cuda_allocator_config": desired["serving_engine"][
                  "cuda_allocator_config"
                ],
              },
            },
          ]
          print(
            f"Recorded bounded serving resources for {path.parent.name}; "
            "existing diagnostics were preserved."
          )
        elif is_supported_local_parser_upgrade(existing):
          desired["artifact_migrations"] = [
            *existing.get("artifact_migrations", []),
            {
              "reason": "provenance-preserving-local-parser-upgrade",
              "schema_version": existing["schema_version"],
              "pipeline_revision": existing["pipeline_revision"],
              "previous_local_parser": existing["answer_extraction"][
                "local_parser"
              ],
              "current_local_parser": desired["answer_extraction"][
                "local_parser"
              ],
              "previous_visual_pipeline_sha256": existing["source_hashes"][
                "runner"
              ]["visual_pipeline"],
              "current_visual_pipeline_sha256": desired["source_hashes"][
                "runner"
              ]["visual_pipeline"],
            },
          ]
          print(
            f"Recorded strict local-parser upgrade for {path.parent.name}; "
            "existing diagnostics were preserved."
          )
        elif is_supported_raw_output_finalization_upgrade(existing):
          desired["artifact_migrations"] = [
            *existing.get("artifact_migrations", []),
            {
              "reason": "provenance-preserving-exact-raw-output-finalization-upgrade",
              "schema_version": existing["schema_version"],
              "previous_unparseable_answers": existing["unparseable_answers"],
              "current_unparseable_answers": desired["unparseable_answers"],
              "previous_local_parser": existing["answer_extraction"]["local_parser"],
              "current_local_parser": desired["answer_extraction"]["local_parser"],
              "previous_runner_source_hashes": existing["source_hashes"]["runner"],
              "current_runner_source_hashes": desired["source_hashes"]["runner"],
            },
          ]
          print(
            f"Recorded exact raw-output finalization policy for {path.parent.name}; "
            "existing diagnostics were preserved."
          )
        elif is_supported_smoke_admission_upgrade(existing):
          desired["artifact_migrations"] = [
            *existing.get("artifact_migrations", []),
            {
              "reason": "provenance-preserving-smoke-raw-output-admission",
              "schema_version": existing["schema_version"],
              "smoke_admission": desired["unparseable_answers"]["smoke_admission"],
            },
          ]
          print(
            f"Recorded complete raw-output smoke admission for {path.parent.name}; "
            "existing diagnostics were preserved."
          )
        elif is_supported_glm_extended_retry(existing):
          migration = archive_glm_extended_retry(existing)
          desired["artifact_migrations"] = [
            *existing.get("artifact_migrations", []),
            migration,
          ]
          print(
            f"Recorded extended GLM formatting recovery for {path.parent.name}; "
            f"{migration['baseline_diagnostics']['row_count']} diagnostics were "
            "preserved and only unresolved records will be regenerated."
          )
        elif is_supported_completion_cap_recovery(existing):
          migration = archive_completion_cap_recovery(existing)
          desired["artifact_migrations"] = [
            *existing.get("artifact_migrations", []),
            migration,
          ]
          print(
            f"Recorded Qwen2.5 completion-cap recovery for {path.parent.name}; "
            f"{migration['baseline_diagnostics']['row_count']} diagnostics were "
            "preserved and only unresolved records will be regenerated."
          )
        elif is_supported_gemma_completion_cap_recovery(existing):
          migration = archive_gemma_completion_cap_recovery(existing)
          desired["artifact_migrations"] = [
            *existing.get("artifact_migrations", []),
            migration,
          ]
          print(
            f"Recorded Gemma Mind's Eye completion-cap recovery for "
            f"{path.parent.name}; completed DYS artifacts were preserved and "
            "the obsolete smoke attempts were archived."
          )
        elif is_supported_glm_completion_recovery(existing):
          migration = archive_glm_completion_recovery(existing)
          desired["artifact_migrations"] = [
            *existing.get("artifact_migrations", []),
            migration,
          ]
          print(
            f"Recorded GLM completion and seed recovery for {path.parent.name}; "
            f"{migration['baseline_diagnostics']['row_count']} diagnostics were "
            "preserved and only unresolved records will be regenerated."
          )
        elif is_supported_serving_topology_resume(existing):
          migration = archive_serving_topology_resume(existing)
          desired["artifact_migrations"] = [
            *existing.get("artifact_migrations", []),
            migration,
          ]
          print(
            f"Recorded serving-topology resume for {path.parent.name}; "
            "existing checkpoints were preserved."
          )
        else:
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
  MANIFEST_DP_SIZE="$DATA_PARALLEL_SIZE" MANIFEST_CONCURRENCY="$CONCURRENCY" \
  MANIFEST_TRACKS="$manifest_tracks" MANIFEST_VLLM_VERSION="$VLLM_VERSION" \
  MANIFEST_OPENAI_VERSION="$OPENAI_VERSION" MANIFEST_COMPATIBILITY_PATCH="$(model_compatibility_patch "$slug")" \
  MANIFEST_COMPATIBILITY_PATCH_SOURCE="$(model_compatibility_patch_source "$slug")" \
  MANIFEST_PIPELINE_REVISION="$PIPELINE_REVISION_ID" \
  "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from evaluation.common.visual_pipeline import MISSING_ANSWER_TOKEN

path = Path(os.environ["MANIFEST_PATH"])
root = Path(os.environ["MANIFEST_PROJECT_ROOT"])
run_config = json.loads((path.parent / ".run_config.json").read_text(encoding="utf-8"))

def sha256(file_path: Path) -> str:
  digest = hashlib.sha256()
  with file_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()

tracks = {}
for track in os.environ["MANIFEST_TRACKS"].split(","):
    output = path.parent / f"{track}_submission.jsonl"
    diagnostics = path.parent / f"{track}.diagnostics.jsonl"
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    diagnostic_rows = [
      json.loads(line)
      for line in diagnostics.read_text(encoding="utf-8").splitlines()
      if line
    ]
    unresolved_answer_question_ids = [
      str(row["question_id"])
      for row in rows
      if str(row["answer"]).strip().upper() == MISSING_ANSWER_TOKEN
    ]
    attempt_files = sorted(path.parent.glob(f"{track}.attempt-*.diagnostics.jsonl"))
    smoke_attempt_files = sorted(path.parent.glob(f"{track}.smoke.attempt-*.diagnostics.jsonl"))
    tracks[track] = {
        "submission_file": output.name,
        "diagnostics_file": diagnostics.name,
        "row_count": len(rows),
        "submission_sha256": sha256(output),
        "diagnostics_sha256": sha256(diagnostics),
        "resolved_answer_count": len(rows) - len(unresolved_answer_question_ids),
        "unresolved_answer_count": len(unresolved_answer_question_ids),
        "unresolved_answer_question_ids": unresolved_answer_question_ids,
        "exact_raw_output_fallback_question_ids": [],
        "failed_attempt_diagnostics": [
            {
                "file": attempt.name,
                "seed": run_config["generation"]["base_seed"]
                + int(attempt.name.split(".attempt-", 1)[1].split(".", 1)[0])
                - 1,
                "sha256": sha256(attempt),
            }
            for attempt in attempt_files
        ],
        "failed_smoke_attempt_diagnostics": [
            {
                "file": attempt.name,
                "seed": run_config["generation"]["base_seed"]
                + int(attempt.name.split(".attempt-", 1)[1].split(".", 1)[0])
                - 1,
                "sha256": sha256(attempt),
            }
            for attempt in smoke_attempt_files
        ],
        "question_bundle_sha256": run_config["source_hashes"]["questions"][track],
        "generation": run_config["generation"][track],
    }

manifest = {
  "schema_version": 11,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_id": os.environ["MANIFEST_MODEL_ID"],
    "model_revision": os.environ["MANIFEST_REVISION"],
    "reasoning_profile": run_config["reasoning_profile"],
    "checkpoint_source": run_config["checkpoint_source"],
    "serving_engine": run_config["serving_engine"],
    "weight_loading": os.environ["MANIFEST_LOADING"],
    "compute_dtype": os.environ["MANIFEST_DTYPE"],
    "kv_cache_dtype": os.environ["MANIFEST_KV_CACHE_DTYPE"],
    "max_model_len": int(os.environ["MANIFEST_MODEL_MAX_LEN"]),
    "hardware": {
        "assigned_gpu_ids": os.environ["MANIFEST_GPU_IDS"].split(","),
        "description": os.environ["MANIFEST_GPU"],
        "tensor_parallel_size": int(os.environ["MANIFEST_TP_SIZE"]),
        "data_parallel_size": int(os.environ["MANIFEST_DP_SIZE"]),
    },
      "request_concurrency": int(os.environ["MANIFEST_CONCURRENCY"]),
    "dataset": run_config["dataset"],
    "image_preprocessing": run_config["image_preprocessing"],
    "answer_extraction": run_config["answer_extraction"],
    "unparseable_answers": run_config["unparseable_answers"],
    "pipeline_revision": os.environ["MANIFEST_PIPELINE_REVISION"],
    "dependencies": {"openai": os.environ["MANIFEST_OPENAI_VERSION"]},
    "tracks": tracks,
}
if "artifact_migrations" in run_config:
    manifest["artifact_migrations"] = run_config["artifact_migrations"]
if "adapter" in run_config:
    manifest["adapter"] = run_config["adapter"]
patch = os.environ["MANIFEST_COMPATIBILITY_PATCH"]
if patch:
    patch_source = root / os.environ["MANIFEST_COMPATIBILITY_PATCH_SOURCE"]
    manifest["compatibility_patches"] = [{
        "id": patch,
        "source_sha256": sha256(patch_source),
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
  local model_cache="$CACHE_ROOT/models/$slug" request_model track track_failed=0
  if [[ "$MAX_MODEL_LEN" != "auto" ]]; then
    model_max_len="$MAX_MODEL_LEN"
  fi

  ensure_run_config "$slug" "$model_id" "$revision" "$loading" "$model_max_len" || return 1
  if [[ "$EXTRACT_EXISTING_ONLY" != "1" && "$SMOKE_ONLY" != "1" && "$FORCE" != "1" ]] \
    && model_outputs_complete "$slug"; then
    log "$slug is already complete; skipping model startup"
    write_manifest "$slug" "$model_id" "$revision" "$loading" "$model_max_len"
    delete_model_cache "$model_cache"
    SKIPPED_MODELS+=("$slug")
    return 0
  fi

  request_model="$(model_request_name "$slug" "$model_id")"
  if [[ "$EXTRACT_EXISTING_ONLY" == "1" ]]; then
    if ! start_extractor_server; then
      return 1
    fi
    while IFS= read -r track; do
      if ! run_track "$slug" "$request_model" "$track"; then
        track_failed=1
        log "$slug/$track extraction failed; continuing with remaining selected tracks"
      fi
    done < <(selected_tracks)
    stop_extractor_server
    if [[ "$track_failed" == "1" ]]; then
      return 1
    fi
    write_manifest "$slug" "$model_id" "$revision" "$loading" "$model_max_len"
    return 0
  fi

  if ! start_server "$slug" "$model_id" "$revision" "$loading" "$model_cache" "$model_max_len"; then
    delete_model_cache "$model_cache"
    return 1
  fi
  if ! start_extractor_server; then
    stop_server
    delete_model_cache "$model_cache"
    return 1
  fi
  while IFS= read -r track; do
    if ! run_track "$slug" "$request_model" "$track"; then
      track_failed=1
      log "$slug/$track failed; continuing with remaining selected tracks"
    fi
  done < <(selected_tracks)
  stop_extractor_server
  stop_server
  if [[ "$track_failed" == "1" ]]; then
    delete_model_cache "$model_cache"
    return 1
  fi

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

  mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT/models"
  preflight_host
  setup_environment
  apply_model_compatibility_patches
  verify_vllm_cli
  if [[ "$EXTRACT_EXISTING_ONLY" == "1" ]]; then
    log "Extraction-only mode uses public question files and existing diagnostics; image dataset preparation is skipped"
  else
    prepare_dataset
  fi
  if [[ "$SETUP_ONLY" == "1" ]]; then
    log "Shared environment and pinned dataset are ready; no model worker was started"
    return 0
  fi
  write_active_run_marker

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
  trap cleanup_runner EXIT
  trap 'exit 130' INT TERM
  main "$@"
fi
