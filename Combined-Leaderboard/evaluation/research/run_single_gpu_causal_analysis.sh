#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

GPU_ID="${GPU_ID:-2}"
RESEARCH_DATA_ROOT="${MS_VISTA_RESEARCH_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/ms-vista-research}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$RESEARCH_DATA_ROOT/causal-analysis-v1}"
PYTHON="${VISUAL_PYTHON:-$PROJECT_ROOT/.venv/visual-suite/bin/python}"
VLLM="${VLLM_BIN:-$PROJECT_ROOT/.venv/visual-suite/bin/vllm}"
TARGET_MODEL="Qwen/Qwen3-VL-8B-Instruct"
TARGET_REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TARGET_HF_HOME="${TARGET_HF_HOME:-$RESEARCH_DATA_ROOT/hf-cache}"
EXTRACTOR_MODEL="Qwen/Qwen3-8B"
EXTRACTOR_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
EXTRACTOR_HF_HOME="${EXTRACTOR_HF_HOME:-$RESEARCH_DATA_ROOT/qwen3-8b-extractor}"
PORT="${ANALYSIS_PORT:-8041}"
BUNDLE="$ANALYSIS_ROOT/causal_transformations"
RUN="$ANALYSIS_ROOT/runs/qwen3-vl-8b/causal"
ACTIVE_SERVER_PID=""
ACTIVE_SERVER_NAME=""

mkdir -p "$ANALYSIS_ROOT"/{logs,tmp} "$RUN"
exec 9>"$ANALYSIS_ROOT/causal-analysis.lock"
if ! flock -n 9; then
  printf 'Another causal-analysis run already holds %s/causal-analysis.lock\n' \
    "$ANALYSIS_ROOT" >&2
  exit 2
fi
exec >>"$ANALYSIS_ROOT/logs/causal-supervisor.log" 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

write_status() {
  local phase="$1" detail="$2"
  printf '%s\t%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$phase" "$detail" \
    >"$ANALYSIS_ROOT/causal-status.tsv"
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
  write_status "stopped" "exit_status=$status"
  log "Causal analysis exiting with status $status"
  exit "$status"
}
trap cleanup EXIT INT TERM

require_resources() {
  local free
  free="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits)"
  if (( free < 38000 )); then
    printf 'GPU %s has only %s MiB free; refusing to launch.\n' "$GPU_ID" "$free" >&2
    exit 2
  fi
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
  done
  printf '%s did not become ready within 1800 seconds.\n' "$model" >&2
  return 1
}

launch_server() {
  local model="$1" revision="$2" hf_home="$3" max_num_seqs="$4" log_path="$5"
  local -a server_args
  require_resources
  server_args=(
    serve "$model"
    --host 127.0.0.1
    --port "$PORT"
    --served-model-name "$model"
    --revision "$revision"
    --download-dir "$hf_home/hub"
    --dtype bfloat16
    --kv-cache-dtype bfloat16
    --gpu-memory-utilization 0.90
    --max-model-len 32768
    --max-num-seqs "$max_num_seqs"
    --generation-config vllm
    --trust-remote-code
  )
  if [[ "$model" == "$TARGET_MODEL" ]]; then
    server_args+=(--limit-mm-per-prompt '{"image":1}')
  else
    server_args+=(--default-chat-template-kwargs '{"enable_thinking":false}')
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
    TMPDIR="$ANALYSIS_ROOT/tmp" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$VLLM" "${server_args[@]}" >"$log_path" 2>&1 &
  ACTIVE_SERVER_PID="$!"
  ACTIVE_SERVER_NAME="$model"
  printf '%s\n' "$ACTIVE_SERVER_PID" >"${log_path%.log}.pid"
  wait_for_server "$model"
}

run_target_inference() {
  local -a common_args
  common_args=(
    --model "$TARGET_MODEL"
    --endpoints "http://127.0.0.1:$PORT/v1"
    --extractor-model "$TARGET_MODEL"
    --extractor-revision "$TARGET_REVISION"
    --extractor-endpoints "http://127.0.0.1:$PORT/v1"
    --extractor-chat-template-kwargs '{"enable_thinking":false}'
    --questions "$BUNDLE/questions.jsonl"
    --image-root "$BUNDLE"
    --prompt-mode cot
    --temperature 0.1
    --top-p 1.0
    --max-tokens 8192
    --max-final-answer-tokens 200
    --concurrency 4
    --request-timeout 900
    --max-retries 2
    --checkpoint-every 10
    --resume
  )
  write_status "smoke" "$TARGET_MODEL"
  "$PYTHON" -m evaluation.minds_eye.run_vllm \
    "${common_args[@]}" \
    --limit 5 \
    --strict-partial \
    --out "$RUN/smoke-submission.jsonl" \
    --diagnostics "$RUN/smoke-diagnostics.jsonl"

  write_status "target_inference" "$TARGET_MODEL"
  "$PYTHON" -m evaluation.minds_eye.run_vllm \
    "${common_args[@]}" \
    --out "$RUN/submission.jsonl" \
    --diagnostics "$RUN/diagnostics.jsonl"
}

run_fixed_extraction() {
  write_status "fixed_extraction" "$EXTRACTOR_MODEL"
  "$PYTHON" -m evaluation.minds_eye.run_vllm \
    --model "$TARGET_MODEL" \
    --endpoints "http://127.0.0.1:$PORT/v1" \
    --extractor-model "$EXTRACTOR_MODEL" \
    --extractor-revision "$EXTRACTOR_REVISION" \
    --extractor-endpoints "http://127.0.0.1:$PORT/v1" \
    --extractor-chat-template-kwargs '{"enable_thinking":false}' \
    --questions "$BUNDLE/questions.jsonl" \
    --image-root "$BUNDLE" \
    --prompt-mode cot \
    --temperature 0.1 \
    --top-p 1.0 \
    --max-tokens 8192 \
    --max-final-answer-tokens 200 \
    --concurrency 16 \
    --request-timeout 900 \
    --max-retries 2 \
    --checkpoint-every 10 \
    --resume \
    --extract-existing-diagnostics \
    --out "$RUN/submission.jsonl" \
    --diagnostics "$RUN/diagnostics.jsonl"
}

[[ -x "$PYTHON" ]] || {
  printf 'Visual-suite Python is not executable: %s\n' "$PYTHON" >&2
  exit 2
}
[[ -x "$VLLM" ]] || {
  printf 'vLLM executable is missing: %s\n' "$VLLM" >&2
  exit 2
}
[[ -f "$TARGET_HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/$TARGET_REVISION/config.json" ]] || {
  printf 'Pinned target model snapshot is missing from %s\n' "$TARGET_HF_HOME" >&2
  exit 2
}
[[ -f "$EXTRACTOR_HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/$EXTRACTOR_REVISION/config.json" ]] || {
  printf 'Pinned extractor snapshot is missing from %s\n' "$EXTRACTOR_HF_HOME" >&2
  exit 2
}

cd "$PROJECT_ROOT"
write_status "preflight" "building_causal_bundle"
"$PYTHON" evaluation/research/generate_causal_transformations.py \
  --output "$BUNDLE" \
  --pairs 200 \
  --seed 20260723
"$PYTHON" evaluation/research/validate_experiment_bundle.py \
  --manifest "$BUNDLE/manifest.json"

launch_server \
  "$TARGET_MODEL" "$TARGET_REVISION" "$TARGET_HF_HOME" 4 \
  "$RUN/target-vllm.log"
run_target_inference
stop_server

launch_server \
  "$EXTRACTOR_MODEL" "$EXTRACTOR_REVISION" "$EXTRACTOR_HF_HOME" 16 \
  "$RUN/extractor-vllm.log"
run_fixed_extraction
stop_server

write_status "scoring" "causal_transformations"
"$PYTHON" analysis/research/score_causal_transformations.py \
  --ground-truth "$BUNDLE/private_ground_truth.jsonl" \
  --pairs "$BUNDLE/pairs.jsonl" \
  --submission "$RUN/submission.jsonl" \
  --output "$RUN/score"
"$PYTHON" evaluation/research/record_experiment_run.py \
  --output "$RUN/run_manifest.json" \
  --experiment causal_visual_transformations \
  --model "$TARGET_MODEL" \
  --model-revision "$TARGET_REVISION" \
  --extractor-model "$EXTRACTOR_MODEL" \
  --extractor-revision "$EXTRACTOR_REVISION" \
  --parameter prompt_mode=cot \
  --parameter temperature=0.1 \
  --parameter top_p=1.0 \
  --parameter max_tokens=8192 \
  --parameter seed=0 \
  --parameter gpu_id="$GPU_ID" \
  --input questions="$BUNDLE/questions.jsonl" \
  --input pairs="$BUNDLE/pairs.jsonl" \
  --input ground_truth="$BUNDLE/private_ground_truth.jsonl" \
  --submission "$RUN/submission.jsonl" \
  --diagnostics "$RUN/diagnostics.jsonl"

write_status "complete" "$RUN/score/summary.json"
log "Causal analysis completed"
