#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

GPU_ID="${GPU_ID:-2}"
RESEARCH_DATA_ROOT="${MS_VISTA_RESEARCH_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/ms-vista-research}"
ABLATION_ROOT="${ABLATION_ROOT:-$RESEARCH_DATA_ROOT/qwen35-thinking-v1}"
DATASET_ROOT="${DATASET_ROOT:-$RESEARCH_DATA_ROOT/visual-intelligence-dataset}"
QUESTIONS="$DATASET_ROOT/minds_eye_fresh_v1/questions.jsonl"
PYTHON="${VISUAL_PYTHON:-$PROJECT_ROOT/.venv/visual-suite/bin/python}"
VLLM="${VLLM_BIN:-$PROJECT_ROOT/.venv/visual-suite/bin/vllm}"
PORT="${ABLATION_PORT:-8042}"
TARGET_CONCURRENCY="${TARGET_CONCURRENCY:-4}"
EXTRACTOR_CONCURRENCY="${EXTRACTOR_CONCURRENCY:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SEED="${SEED:-0}"

TARGET_MODEL="Qwen/Qwen3.5-9B"
TARGET_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TARGET_HF_HOME="${TARGET_HF_HOME:-$RESEARCH_DATA_ROOT/hf-cache}"
EXTRACTOR_MODEL="Qwen/Qwen3-8B"
EXTRACTOR_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
EXTRACTOR_HF_HOME="${EXTRACTOR_HF_HOME:-$RESEARCH_DATA_ROOT/qwen3-8b-extractor}"
EXTRACTOR_MAX_MODEL_LEN="${EXTRACTOR_MAX_MODEL_LEN:-40960}"

ACTIVE_SERVER_PID=""
ACTIVE_SERVER_NAME=""
declare -a CONDITIONS=(
  'direct_disabled|noncot|{"enable_thinking":false}'
  'direct_enabled|noncot|{"enable_thinking":true}'
  'cot_disabled|cot|{"enable_thinking":false}'
  'cot_enabled|cot|{"enable_thinking":true}'
)

mkdir -p "$ABLATION_ROOT"/{conditions,logs,status,tmp}
exec 9>"$ABLATION_ROOT/ablation.lock"
if ! flock -n 9; then
  printf 'Another Qwen3.5 ablation holds %s/ablation.lock\n' \
    "$ABLATION_ROOT" >&2
  exit 2
fi
exec >>"$ABLATION_ROOT/logs/supervisor.log" 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

write_status() {
  local phase="$1" detail="$2"
  printf '%s\t%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$phase" "$detail" \
    >"$ABLATION_ROOT/status/current.tsv"
  printf '%s\t%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$phase" "$detail" \
    >>"$ABLATION_ROOT/status/history.tsv"
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
  elif [[ ! -f "$ABLATION_ROOT/status/current.tsv" ]] \
    || [[ "$(cut -f2 "$ABLATION_ROOT/status/current.tsv")" != "complete" ]]; then
    write_status "stopped" "exit_status=0"
  fi
  log "Qwen3.5 ablation exiting with status $status"
  exit "$status"
}
trap cleanup EXIT INT TERM

require_gpu_capacity() {
  local free
  free="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free \
    --format=csv,noheader,nounits)"
  if (( free < 38000 )); then
    printf 'GPU %s has only %s MiB free; refusing to launch.\n' \
      "$GPU_ID" "$free" >&2
    exit 2
  fi
}

require_port_free() {
  if ss -ltn | awk '{print $4}' | grep -qE ":${PORT}$"; then
    printf 'Port %s is already in use.\n' "$PORT" >&2
    exit 2
  fi
}

wait_for_server() {
  local model="$1" elapsed=0 payload served
  while (( elapsed < 1800 )); do
    if payload="$(curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)"; then
      served="$(
        printf '%s' "$payload" \
          | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
      )"
      if [[ "$served" == "$model" ]]; then
        log "$model is ready"
        return
      fi
    fi
    if ! kill -0 "$ACTIVE_SERVER_PID" 2>/dev/null; then
      printf '%s server exited before becoming ready.\n' "$model" >&2
      return 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
    if (( elapsed % 60 == 0 )); then
      log "Waiting for $model startup ($elapsed seconds)"
    fi
  done
  printf '%s did not become ready within 1800 seconds.\n' "$model" >&2
  return 1
}

launch_server() {
  local model="$1" revision="$2" hf_home="$3" max_model_len="$4"
  local max_num_seqs="$5" multimodal="$6" log_path="$7"
  local -a args
  require_gpu_capacity
  require_port_free
  mkdir -p "$hf_home/hub" "$(dirname -- "$log_path")"
  args=(
    serve "$model"
    --host 127.0.0.1
    --port "$PORT"
    --served-model-name "$model"
    --revision "$revision"
    --download-dir "$hf_home/hub"
    --dtype bfloat16
    --kv-cache-dtype bfloat16
    --gpu-memory-utilization 0.90
    --max-model-len "$max_model_len"
    --max-num-seqs "$max_num_seqs"
    --generation-config vllm
    --trust-remote-code
  )
  if [[ "$multimodal" == "1" ]]; then
    args+=(--limit-mm-per-prompt '{"image":1}')
  else
    args+=(--default-chat-template-kwargs '{"enable_thinking":false}')
  fi
  log "Launching $model at revision $revision on GPU $GPU_ID"
  setsid env \
    -u HF_HUB_ENABLE_HF_TRANSFER \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    HF_HOME="$hf_home" \
    HF_HUB_CACHE="$hf_home/hub" \
    HUGGINGFACE_HUB_CACHE="$hf_home/hub" \
    TRANSFORMERS_CACHE="$hf_home/hub" \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    TMPDIR="$ABLATION_ROOT/tmp" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$VLLM" "${args[@]}" >"$log_path" 2>&1 &
  ACTIVE_SERVER_PID="$!"
  ACTIVE_SERVER_NAME="$model"
  printf '%s\n' "$ACTIVE_SERVER_PID" >"${log_path%.log}.pid"
  wait_for_server "$model"
}

run_raw_condition() {
  local condition="$1" prompt_mode="$2" chat_kwargs="$3"
  local run_dir="$ABLATION_ROOT/conditions/$condition"
  local attempt status
  mkdir -p "$run_dir"
  if [[ -s "$run_dir/submission.jsonl" && -s "$run_dir/run_manifest.json" ]]; then
    log "Skipping completed $condition"
    return
  fi
  for attempt in 1 2 3; do
    write_status "target_inference" "$condition attempt=$attempt"
    set +e
    "$PYTHON" -m evaluation.minds_eye.run_vllm \
      --model "$TARGET_MODEL" \
      --endpoints "http://127.0.0.1:$PORT/v1" \
      --questions "$QUESTIONS" \
      --image-root "$DATASET_ROOT" \
      --prompt-mode "$prompt_mode" \
      --temperature 0 \
      --top-p 1 \
      --seed "$SEED" \
      --chat-template-kwargs "$chat_kwargs" \
      --concurrency "$TARGET_CONCURRENCY" \
      --request-timeout 900 \
      --max-retries 2 \
      --checkpoint-every 10 \
      --resume \
      --defer-extraction \
      --out "$run_dir/submission.jsonl" \
      --diagnostics "$run_dir/diagnostics.jsonl"
    status=$?
    set -e
    if (( status == 0 )); then
      return
    fi
    log "Raw inference attempt $attempt failed for $condition; retrying pending rows"
    sleep 10
  done
  return 1
}

smoke_condition() {
  local condition="$1" prompt_mode="$2" chat_kwargs="$3"
  local run_dir="$ABLATION_ROOT/conditions/$condition"
  mkdir -p "$run_dir"
  rm -f -- "$run_dir/smoke.diagnostics.jsonl"
  write_status "smoke" "$condition"
  "$PYTHON" -m evaluation.minds_eye.run_vllm \
    --model "$TARGET_MODEL" \
    --endpoints "http://127.0.0.1:$PORT/v1" \
    --questions "$QUESTIONS" \
    --image-root "$DATASET_ROOT" \
    --prompt-mode "$prompt_mode" \
    --temperature 0 \
    --top-p 1 \
    --seed "$SEED" \
    --chat-template-kwargs "$chat_kwargs" \
    --concurrency "$TARGET_CONCURRENCY" \
    --request-timeout 900 \
    --max-retries 2 \
    --checkpoint-every 5 \
    --limit 5 \
    --defer-extraction \
    --diagnostics "$run_dir/smoke.diagnostics.jsonl"
}

extract_condition() {
  local condition="$1" prompt_mode="$2" chat_kwargs="$3"
  local run_dir="$ABLATION_ROOT/conditions/$condition"
  local attempt status
  for attempt in 1 2 3; do
    write_status "fixed_extraction" "$condition attempt=$attempt"
    set +e
    "$PYTHON" -m evaluation.minds_eye.run_vllm \
      --model "$TARGET_MODEL" \
      --endpoints "http://127.0.0.1:$PORT/v1" \
      --extractor-model "$EXTRACTOR_MODEL" \
      --extractor-revision "$EXTRACTOR_REVISION" \
      --extractor-endpoints "http://127.0.0.1:$PORT/v1" \
      --extractor-chat-template-kwargs '{"enable_thinking":false}' \
      --questions "$QUESTIONS" \
      --image-root "$DATASET_ROOT" \
      --prompt-mode "$prompt_mode" \
      --temperature 0 \
      --top-p 1 \
      --seed "$SEED" \
      --chat-template-kwargs "$chat_kwargs" \
      --max-final-answer-tokens 200 \
      --extractor-max-tokens 512 \
      --extractor-seed "$SEED" \
      --concurrency "$EXTRACTOR_CONCURRENCY" \
      --request-timeout 900 \
      --max-retries 2 \
      --checkpoint-every 10 \
      --resume \
      --extract-existing-diagnostics \
      --out "$run_dir/submission.jsonl" \
      --diagnostics "$run_dir/diagnostics.jsonl"
    status=$?
    set -e
    if (( status == 0 )); then
      break
    fi
    log "Extractor attempt $attempt failed for $condition; retrying failed rows"
    sleep 10
  done
  (( status == 0 )) || return 1

  "$PYTHON" analysis/research/audit_inference_quality.py \
    --questions "$QUESTIONS" \
    --diagnostics "$run_dir/diagnostics.jsonl" \
    --submission "$run_dir/submission.jsonl" \
    --output "$run_dir/quality" \
    --extractor-model "$EXTRACTOR_MODEL" \
    --extractor-revision "$EXTRACTOR_REVISION"
  "$PYTHON" evaluation/research/record_experiment_run.py \
    --output "$run_dir/run_manifest.json" \
    --experiment "qwen35_thinking_ablation" \
    --model "$TARGET_MODEL" \
    --model-revision "$TARGET_REVISION" \
    --extractor-model "$EXTRACTOR_MODEL" \
    --extractor-revision "$EXTRACTOR_REVISION" \
    --parameter "condition=$condition" \
    --parameter "track=minds_eye" \
    --parameter "prompt_mode=$prompt_mode" \
    --parameter "chat_template_kwargs=$chat_kwargs" \
    --parameter "temperature=0" \
    --parameter "top_p=1" \
    --parameter "max_model_len=$MAX_MODEL_LEN" \
    --parameter "completion_budget=context_remainder" \
    --parameter "dtype=bfloat16" \
    --parameter "seed=$SEED" \
    --input "questions=$QUESTIONS" \
    --submission "$run_dir/submission.jsonl" \
    --diagnostics "$run_dir/diagnostics.jsonl"
}

[[ -x "$PYTHON" ]] || {
  printf 'Visual-suite Python is not executable: %s\n' "$PYTHON" >&2
  exit 2
}
[[ -x "$VLLM" ]] || {
  printf 'vLLM executable is missing: %s\n' "$VLLM" >&2
  exit 2
}
[[ -f "$QUESTIONS" ]] || {
  printf 'Mind'\''s Eye questions are missing: %s\n' "$QUESTIONS" >&2
  exit 2
}
[[ -f "$TARGET_HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/$TARGET_REVISION/config.json" ]] || {
  printf 'Pinned Qwen3.5 snapshot is missing from %s\n' "$TARGET_HF_HOME" >&2
  exit 2
}
[[ -f "$EXTRACTOR_HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/$EXTRACTOR_REVISION/config.json" ]] || {
  printf 'Pinned extractor snapshot is missing from %s\n' "$EXTRACTOR_HF_HOME" >&2
  exit 2
}

cd "$PROJECT_ROOT"
write_status "preflight" "qwen35_thinking_factorial"
launch_server \
  "$TARGET_MODEL" "$TARGET_REVISION" "$TARGET_HF_HOME" "$MAX_MODEL_LEN" \
  "$TARGET_CONCURRENCY" 1 "$ABLATION_ROOT/logs/target-vllm.log"

smoke_condition "direct_disabled" "noncot" '{"enable_thinking":false}'
smoke_condition "cot_enabled" "cot" '{"enable_thinking":true}'
for specification in "${CONDITIONS[@]}"; do
  IFS='|' read -r condition prompt_mode chat_kwargs <<<"$specification"
  run_raw_condition "$condition" "$prompt_mode" "$chat_kwargs"
done
stop_server

launch_server \
  "$EXTRACTOR_MODEL" "$EXTRACTOR_REVISION" "$EXTRACTOR_HF_HOME" \
  "$EXTRACTOR_MAX_MODEL_LEN" "$EXTRACTOR_CONCURRENCY" 0 \
  "$ABLATION_ROOT/logs/extractor-vllm.log"
for specification in "${CONDITIONS[@]}"; do
  IFS='|' read -r condition prompt_mode chat_kwargs <<<"$specification"
  extract_condition "$condition" "$prompt_mode" "$chat_kwargs"
done
stop_server

printf '%s\n' \
  "All four conditions passed the remote inference and extraction quality gates." \
  "Run analysis/research/analyze_qwen35_thinking_ablation.py on the trusted host." \
  >"$ABLATION_ROOT/local_scoring_required.txt"
write_status "complete" "all_conditions_extracted"
log "Qwen3.5 thinking ablation completed"
