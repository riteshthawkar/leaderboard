#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

GPU_IDS="${GPU_IDS:-0,1}"
TRACK3_ROOT="${TRACK3_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/ms-vista-track3}"
TRACK3_SOURCE_ROOT="${TRACK3_SOURCE_ROOT:-$TRACK3_ROOT}"
TRACK3_RUN_ROOT="${TRACK3_RUN_ROOT:-$TRACK3_ROOT/runs/paper-aligned-v5}"
INTERNVL_REUSE_ROOT="${TRACK3_INTERNVL_REUSE_ROOT:-}"
CLIENT_PYTHON="${TRACK3_PYTHON:-$TRACK3_SOURCE_ROOT/conda-env/bin/python}"
VLLM="${VLLM_BIN:-$PROJECT_ROOT/.venv/visual-suite/bin/vllm}"
LMUDATA="${LMUDATA:-$TRACK3_SOURCE_ROOT/LMUData}"
TARGET_PORT="${TARGET_PORT:-8031}"
JUDGE_PORT="${JUDGE_PORT:-8032}"
MAX_TOKENS="${MAX_TOKENS:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-33792}"
RUN_SEED="${RUN_SEED:-0}"
TARGET_CONCURRENCY="${TARGET_CONCURRENCY:-2}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
START_AT="${START_AT:-internvl_target}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-0}"
JUDGE_STARTUP_TIMEOUT_SECONDS="${JUDGE_STARTUP_TIMEOUT_SECONDS:-3600}"
REBUILD_CONTRACT="${REBUILD_CONTRACT:-0}"
DATASETS="BLINK,CV-Bench-2D,CV-Bench-3D,MMVP,RealWorldQA,VStarBench,MMSIBench_wo_circular,3DSRBench,VSR_MCQ,SpatialBench,MindCube,OmniSpatial,SAT-Real"

JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
JUDGE_REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
JUDGE_HF_HOME="$TRACK3_RUN_ROOT/judge-hf-home"
CONTRACT_ROOT="${TRACK3_CONTRACT_ROOT:-$TRACK3_RUN_ROOT/contracts/paper-aligned-v5}"
PRIVATE_CONTRACT_ROOT="${TRACK3_PRIVATE_CONTRACT_ROOT:-$TRACK3_RUN_ROOT/private/contracts/paper-aligned-v5}"
BENCHMARK_VERSION="${TRACK3_BENCHMARK_VERSION:-paper-aligned-v5-2026-07}"

ACTIVE_SERVER_PID=""
ACTIVE_SERVER_NAME=""

mkdir -p "$TRACK3_RUN_ROOT"/{logs,results,tmp}
exec 9>"$TRACK3_RUN_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'Another Track-3 queue already holds %s/queue.lock\n' "$TRACK3_RUN_ROOT" >&2
  exit 2
fi
exec >>"$TRACK3_RUN_ROOT/logs/supervisor.log" 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

write_status() {
  local phase="$1" detail="$2"
  printf '%s\t%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$phase" "$detail" \
    >"$TRACK3_RUN_ROOT/status.tsv"
}

stop_server() {
  if [[ -n "$ACTIVE_SERVER_PID" ]]; then
    log "Stopping $ACTIVE_SERVER_NAME server process group $ACTIVE_SERVER_PID"
    kill -TERM -- "-$ACTIVE_SERVER_PID" 2>/dev/null || true
    wait "$ACTIVE_SERVER_PID" 2>/dev/null || true
    ACTIVE_SERVER_PID=""
    ACTIVE_SERVER_NAME=""
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_server
  if (( status != 0 )); then
    write_status "failed" "exit_status=$status"
  elif [[ ! -f "$TRACK3_RUN_ROOT/status.tsv" ]] \
    || [[ "$(cut -f2 "$TRACK3_RUN_ROOT/status.tsv")" != "complete" ]]; then
    write_status "stopped" "exit_status=0"
  fi
  log "Track-3 queue exiting with status $status"
  exit "$status"
}
trap cleanup EXIT INT TERM

require_gpu_capacity() {
  local gpu free ready waited=0
  IFS=',' read -r -a requested_gpus <<<"$GPU_IDS"
  if (( ${#requested_gpus[@]} != 2 )); then
    printf 'GPU_IDS must contain exactly two GPU indices; got %s\n' "$GPU_IDS" >&2
    exit 2
  fi
  while true; do
    ready=1
    for gpu in "${requested_gpus[@]}"; do
      free="$(
        nvidia-smi -i "$gpu" \
          --query-gpu=memory.free \
          --format=csv,noheader,nounits
      )"
      if (( free < 38000 )); then
        ready=0
        break
      fi
    done
    if (( ready == 1 )); then
      return
    fi
    if (( GPU_WAIT_TIMEOUT_SECONDS > 0 && waited >= GPU_WAIT_TIMEOUT_SECONDS )); then
      printf 'GPUs %s did not reach the 38000 MiB free threshold within %s seconds.\n' \
        "$GPU_IDS" "$GPU_WAIT_TIMEOUT_SECONDS" >&2
      exit 2
    fi
    write_status "waiting_for_gpus" \
      "gpus=$GPU_IDS required_free_mib=38000 waited_seconds=$waited"
    log "Waiting for GPUs $GPU_IDS to become available"
    sleep "$GPU_WAIT_SECONDS"
    waited=$((waited + GPU_WAIT_SECONDS))
  done
}

require_port_free() {
  local port="$1"
  if ss -ltn | awk '{print $4}' | grep -qE ":${port}$"; then
    printf 'Port %s is already in use.\n' "$port" >&2
    exit 2
  fi
}

launch_server() {
  local model="$1" revision="$2" hf_home="$3" port="$4" max_num_seqs="$5"
  local log_path="$6" offline="$7"
  local server_max_model_len="$MAX_MODEL_LEN"
  local -a server_args
  if [[ "$model" == "$JUDGE_MODEL" ]]; then
    server_max_model_len="$JUDGE_MAX_MODEL_LEN"
  fi
  require_gpu_capacity
  require_port_free "$port"
  mkdir -p "$hf_home/hub" "$(dirname -- "$log_path")"
  server_args=(
    serve "$model"
    --host 127.0.0.1
    --port "$port"
    --served-model-name "$model"
    --revision "$revision"
    --download-dir "$hf_home/hub"
    --dtype bfloat16
    --kv-cache-dtype bfloat16
    --tensor-parallel-size 2
    --data-parallel-size 1
    --gpu-memory-utilization 0.90
    --max-model-len "$server_max_model_len"
    --max-num-seqs "$max_num_seqs"
    --generation-config vllm
    --disable-custom-all-reduce
    --trust-remote-code
  )
  if [[ "$model" != "$JUDGE_MODEL" ]]; then
    server_args+=(--limit-mm-per-prompt '{"image":10}')
  fi
  if [[ "$model" == "Qwen/Qwen3.6-27B" ]]; then
    server_args+=(--default-chat-template-kwargs '{"enable_thinking":false}')
  fi
  log "Launching $model at revision $revision on GPUs $GPU_IDS (max_model_len=$server_max_model_len)"
  setsid env \
    -u HF_HUB_ENABLE_HF_TRANSFER \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    HF_HOME="$hf_home" \
    HF_HUB_CACHE="$hf_home/hub" \
    HUGGINGFACE_HUB_CACHE="$hf_home/hub" \
    TRANSFORMERS_CACHE="$hf_home/hub" \
    HF_HUB_OFFLINE="$offline" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    TMPDIR="$TRACK3_RUN_ROOT/tmp" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$VLLM" "${server_args[@]}" >"$log_path" 2>&1 &
  ACTIVE_SERVER_PID="$!"
  ACTIVE_SERVER_NAME="$model"
  printf '%s\n' "$ACTIVE_SERVER_PID" >"${log_path%.log}.pid"
}

wait_for_model_server() {
  local model="$1" port="$2" log_path="$3"
  local elapsed=0 response
  while true; do
    response="$(
      curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null \
        || true
    )"
    if [[ "$response" == *"$model"* ]]; then
      log "$model server is ready on port $port"
      return 0
    fi
    if [[ -z "$ACTIVE_SERVER_PID" ]] \
      || ! kill -0 "$ACTIVE_SERVER_PID" 2>/dev/null; then
      printf '%s server exited before readiness; inspect %s\n' \
        "$model" "$log_path" >&2
      return 1
    fi
    if (( elapsed >= JUDGE_STARTUP_TIMEOUT_SECONDS )); then
      printf '%s server was not ready within %s seconds; inspect %s\n' \
        "$model" "$JUDGE_STARTUP_TIMEOUT_SECONDS" "$log_path" >&2
      return 1
    fi
    write_status "judge_startup" \
      "$model port=$port elapsed_seconds=$elapsed"
    sleep 15
    elapsed=$((elapsed + 15))
  done
}

run_target() {
  local slug="$1" model="$2" revision="$3" hf_home="$4"
  local noncot_kwargs="$5" cot_kwargs="$6" reuse_root="${7:-}"
  local out="$TRACK3_RUN_ROOT/results/$slug"
  local attempt status server_metadata
  local -a reuse_args
  mkdir -p "$out"
  for attempt in 1 2; do
    write_status "target_inference" "$slug attempt=$attempt"
    launch_server \
      "$model" "$revision" "$hf_home" "$TARGET_PORT" 2 \
      "$out/vllm-attempt-$attempt.log" 1
    server_metadata="$(
      printf '{"vllm_version":"0.25.1","gpu_ids":"%s","tensor_parallel_size":2,"dtype":"bfloat16","quantization":null,"max_model_len":%s,"max_num_seqs":2,"completion_budget":"context_remainder","paper_max_new_tokens":32768,"temperature":0,"top_p":1,"limit_mm_per_prompt":{"image":10}}' \
        "$GPU_IDS" "$MAX_MODEL_LEN"
    )"
    reuse_args=()
    if [[ -n "$reuse_root" && -f "$reuse_root/run_config.json" ]]; then
      reuse_args=(--reuse-compatible-from "$reuse_root")
    fi
    set +e
    PYTHONUNBUFFERED=1 "$CLIENT_PYTHON" -m spatial_harness.run_track3_vllm \
      --model "$model" \
      --model-revision "$revision" \
      --endpoints "http://127.0.0.1:$TARGET_PORT/v1" \
      --api-key EMPTY \
      --lmudata "$LMUDATA" \
      --out "$out" \
      --datasets "$DATASETS" \
      --modes main noimage noimgpp \
      --prompt-modes noncot cot \
      --concurrency "$TARGET_CONCURRENCY" \
      --checkpoint-every 8 \
      --request-retries 2 \
      --timeout 900 \
      --endpoint-start-timeout 1800 \
      --temperature 0 \
      --top-p 1 \
      --seed "$RUN_SEED" \
      --max-tokens-noncot "$MAX_TOKENS" \
      --max-tokens-cot "$MAX_TOKENS" \
      --chat-template-kwargs-noncot "$noncot_kwargs" \
      --chat-template-kwargs-cot "$cot_kwargs" \
      --server-metadata "$server_metadata" \
      "${reuse_args[@]}"
    status=$?
    set -e
    stop_server
    if (( status == 0 )); then
      log "Inference complete for $slug"
      return
    fi
    log "Inference attempt $attempt failed for $slug with status $status"
    sleep 15
  done
  return 1
}

run_judge() {
  local slug="$1"
  local out="$TRACK3_RUN_ROOT/results/$slug"
  local attempt status log_path
  for attempt in 1 2; do
    write_status "judging" "$slug attempt=$attempt"
    log_path="$out/judge-vllm-attempt-$attempt.log"
    launch_server \
      "$JUDGE_MODEL" "$JUDGE_REVISION" "$JUDGE_HF_HOME" "$JUDGE_PORT" 16 \
      "$log_path" 0
    set +e
    wait_for_model_server "$JUDGE_MODEL" "$JUDGE_PORT" "$log_path"
    status=$?
    if (( status == 0 )); then
      PYTHONUNBUFFERED=1 "$CLIENT_PYTHON" -m spatial_harness.judge_track3 \
        --input "$out" \
        --endpoint "http://127.0.0.1:$JUDGE_PORT/v1" \
        --model "$JUDGE_MODEL" \
        --model-revision "$JUDGE_REVISION" \
        --api-key EMPTY \
        --concurrency "$JUDGE_CONCURRENCY" \
        --checkpoint-every 25 \
        --request-retries 2 \
        --timeout 900 \
        --server-max-model-len "$JUDGE_MAX_MODEL_LEN"
      status=$?
    fi
    set -e
    stop_server
    if (( status == 0 )); then
      log "Judging complete for $slug"
      return
    fi
    log "Judge attempt $attempt failed for $slug with status $status"
    sleep 15
  done
  return 1
}

package_model() {
  local slug="$1"
  local out="$TRACK3_RUN_ROOT/results/$slug"
  write_status "packaging" "$slug"
  "$CLIENT_PYTHON" -m spatial_harness.package_submission \
    --input "$out" \
    --contract "$CONTRACT_ROOT" \
    --output "$out/submission_package"
  log "Submission package complete for $slug"
}

[[ -x "$CLIENT_PYTHON" ]] || {
  printf 'Track-3 Python is not executable: %s\n' "$CLIENT_PYTHON" >&2
  exit 2
}
[[ -x "$VLLM" ]] || {
  printf 'vLLM executable is missing: %s\n' "$VLLM" >&2
  exit 2
}
if [[ -f "$PROJECT_ROOT/evaluation/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/evaluation/.env"
  set +a
fi
: "${HF_TOKEN:?HF_TOKEN must be set in evaluation/.env or the environment}"
case "$START_AT" in
  internvl_target|internvl_judge|qwen_target|qwen_judge) ;;
  *)
    printf 'Unsupported START_AT phase: %s\n' "$START_AT" >&2
    exit 2
    ;;
esac
case "$REBUILD_CONTRACT" in
  0|1) ;;
  *)
    printf 'REBUILD_CONTRACT must be 0 or 1, not %s.\n' \
      "$REBUILD_CONTRACT" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
write_status "preflight" "verifying_data"
"$CLIENT_PYTHON" -m spatial_harness.prepare_data \
  --verify-only \
  --lmudata "$LMUDATA" \
  --cache "$TRACK3_SOURCE_ROOT/cache" \
  --datasets "$DATASETS"
if [[ "$REBUILD_CONTRACT" == "1" \
    || ! -s "$CONTRACT_ROOT/manifest.json" \
    || ! -s "$CONTRACT_ROOT/questions.jsonl" \
    || ! -s "$CONTRACT_ROOT/submission_template.jsonl" \
    || ! -s "$PRIVATE_CONTRACT_ROOT/ground_truth.json" ]]; then
  "$CLIENT_PYTHON" -m spatial_harness.build_public_contract \
    --lmudata "$LMUDATA" \
    --output "$CONTRACT_ROOT" \
    --private-ground-truth "$PRIVATE_CONTRACT_ROOT/ground_truth.json" \
    --benchmark-version "$BENCHMARK_VERSION"
else
  log "Reusing the existing verified public Track-3 contract"
fi

log "Starting standardized Track-3 queue"
if [[ "$START_AT" == "internvl_target" ]]; then
  run_target \
    "OpenGVLab__InternVL3_5-8B" \
    "OpenGVLab/InternVL3_5-8B" \
    "9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79" \
    "$TRACK3_SOURCE_ROOT/internvl-hf-home" \
    '{}' \
    '{}' \
    "$INTERNVL_REUSE_ROOT"
fi
if [[ "$START_AT" == "internvl_target" || "$START_AT" == "internvl_judge" ]]; then
  run_judge "OpenGVLab__InternVL3_5-8B"
  package_model "OpenGVLab__InternVL3_5-8B"
fi

if [[ "$START_AT" != "qwen_judge" ]]; then
  run_target \
    "Qwen__Qwen3.6-27B" \
    "Qwen/Qwen3.6-27B" \
    "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9" \
    "$TRACK3_SOURCE_ROOT/model-hf-home" \
    '{"enable_thinking":false}' \
    '{"enable_thinking":true}'
fi
run_judge "Qwen__Qwen3.6-27B"
package_model "Qwen__Qwen3.6-27B"

write_status "complete" "all_models_judged"
log "Standardized Track-3 queue completed"
