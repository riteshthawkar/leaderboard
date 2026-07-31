#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

GPU_ID="${GPU_ID:-2}"
RESEARCH_DATA_ROOT="${MS_VISTA_RESEARCH_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/ms-vista-research}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$RESEARCH_DATA_ROOT/analysis-v2}"
DATASET_ROOT="${DATASET_ROOT:-$RESEARCH_DATA_ROOT/visual-intelligence-dataset}"
PYTHON="${VISUAL_PYTHON:-$PROJECT_ROOT/.venv/visual-suite/bin/python}"
VLLM="${VLLM_BIN:-$PROJECT_ROOT/.venv/visual-suite/bin/vllm}"
PORT="${ANALYSIS_PORT:-8041}"
TARGET_SLUGS="${TARGET_SLUGS:-qwen3-vl-8b,internvl35-8b}"
TARGET_CONCURRENCY="${TARGET_CONCURRENCY:-4}"
EXTRACTOR_CONCURRENCY="${EXTRACTOR_CONCURRENCY:-16}"
EXTRACTOR_ATTEMPTS="${EXTRACTOR_ATTEMPTS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SEED="${SEED:-0}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-20260723}"
START_PHASE="${START_PHASE:-inference}"

EXTRACTOR_MODEL="Qwen/Qwen3-8B"
EXTRACTOR_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
EXTRACTOR_HF_HOME="${EXTRACTOR_HF_HOME:-$RESEARCH_DATA_ROOT/qwen3-8b-extractor}"
EXTRACTOR_MAX_MODEL_LEN="${EXTRACTOR_MAX_MODEL_LEN:-40960}"
QWEN_TARGET_HF_HOME="${QWEN_TARGET_HF_HOME:-$RESEARCH_DATA_ROOT/hf-cache}"
INTERNVL_TARGET_HF_HOME="${INTERNVL_TARGET_HF_HOME:-$RESEARCH_DATA_ROOT/internvl-hf-home}"

BUNDLE_ROOT="$ANALYSIS_ROOT/bundles"
RUN_ROOT="$ANALYSIS_ROOT/runs"
CAUSAL_ROOT="$BUNDLE_ROOT/causal_transformations"
FIDELITY_ROOT="$BUNDLE_ROOT/visual_fidelity"
ABSTRACTION_ROOT="$BUNDLE_ROOT/state_interventions"
ACTIVE_SERVER_PID=""
ACTIVE_SERVER_NAME=""
FIDELITY_PREP_PID=""

declare -a SPECS=(
  "causal/noncot|minds_eye|$CAUSAL_ROOT/questions.jsonl|$CAUSAL_ROOT|noncot"
  "causal/cot|minds_eye|$CAUSAL_ROOT/questions.jsonl|$CAUSAL_ROOT|cot"
  "fidelity/do_you_see_me/native/noncot|do_you_see_me|$FIDELITY_ROOT/do_you_see_me/native/questions.jsonl|$FIDELITY_ROOT/do_you_see_me/native|noncot"
  "fidelity/do_you_see_me/downsample_50/noncot|do_you_see_me|$FIDELITY_ROOT/do_you_see_me/downsample_50/questions.jsonl|$FIDELITY_ROOT/do_you_see_me/downsample_50|noncot"
  "fidelity/do_you_see_me/downsample_25/noncot|do_you_see_me|$FIDELITY_ROOT/do_you_see_me/downsample_25/questions.jsonl|$FIDELITY_ROOT/do_you_see_me/downsample_25|noncot"
  "fidelity/minds_eye/native/cot|minds_eye|$FIDELITY_ROOT/minds_eye/native/questions.jsonl|$FIDELITY_ROOT/minds_eye/native|cot"
  "fidelity/minds_eye/downsample_50/cot|minds_eye|$FIDELITY_ROOT/minds_eye/downsample_50/questions.jsonl|$FIDELITY_ROOT/minds_eye/downsample_50|cot"
  "fidelity/minds_eye/downsample_25/cot|minds_eye|$FIDELITY_ROOT/minds_eye/downsample_25/questions.jsonl|$FIDELITY_ROOT/minds_eye/downsample_25|cot"
  "abstraction/baseline/cot|minds_eye|$ABSTRACTION_ROOT/baseline/questions.jsonl|$DATASET_ROOT|cot"
  "abstraction/oracle_abstraction/cot|minds_eye|$ABSTRACTION_ROOT/oracle_abstraction/questions.jsonl|$DATASET_ROOT|cot"
  "abstraction/mismatched_abstraction/cot|minds_eye|$ABSTRACTION_ROOT/mismatched_abstraction/questions.jsonl|$DATASET_ROOT|cot"
)

mkdir -p "$ANALYSIS_ROOT"/{logs,tmp,status} "$BUNDLE_ROOT" "$RUN_ROOT"
exec 9>"$ANALYSIS_ROOT/analysis-queue.lock"
if ! flock -n 9; then
  printf 'Another analysis queue already holds %s/analysis-queue.lock\n' \
    "$ANALYSIS_ROOT" >&2
  exit 2
fi
exec >>"$ANALYSIS_ROOT/logs/supervisor.log" 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

write_status() {
  local phase="$1" detail="$2"
  printf '%s\t%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$phase" "$detail" \
    >"$ANALYSIS_ROOT/status/current.tsv"
  printf '%s\t%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$phase" "$detail" \
    >>"$ANALYSIS_ROOT/status/history.tsv"
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
  if [[ -n "$FIDELITY_PREP_PID" ]]; then
    kill -TERM "$FIDELITY_PREP_PID" 2>/dev/null || true
    wait "$FIDELITY_PREP_PID" 2>/dev/null || true
    FIDELITY_PREP_PID=""
  fi
  if (( status != 0 )); then
    write_status "failed" "exit_status=$status"
  elif [[ ! -f "$ANALYSIS_ROOT/status/current.tsv" ]] \
    || [[ "$(cut -f2 "$ANALYSIS_ROOT/status/current.tsv")" != "complete" ]]; then
    write_status "stopped" "exit_status=0"
  fi
  log "Analysis queue exiting with status $status"
  exit "$status"
}
trap cleanup EXIT INT TERM

require_gpu_capacity() {
  local free
  free="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits)"
  if (( free < 38000 )); then
    printf 'GPU %s has only %s MiB free; refusing to launch.\n' "$GPU_ID" "$free" >&2
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
  local max_num_seqs="$5" chat_defaults="$6" multimodal="$7" log_path="$8"
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
  fi
  if [[ "$chat_defaults" != "{}" ]]; then
    args+=(--default-chat-template-kwargs "$chat_defaults")
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
    "$VLLM" "${args[@]}" >"$log_path" 2>&1 &
  ACTIVE_SERVER_PID="$!"
  ACTIVE_SERVER_NAME="$model"
  printf '%s\n' "$ACTIVE_SERVER_PID" >"${log_path%.log}.pid"
  wait_for_server "$model"
}

load_target_config() {
  local slug="$1"
  case "$slug" in
    qwen3-vl-8b)
      TARGET_MODEL="Qwen/Qwen3-VL-8B-Instruct"
      TARGET_REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
      TARGET_HF_HOME="$QWEN_TARGET_HF_HOME"
      TARGET_CHAT_KWARGS='{}'
      TARGET_SNAPSHOT="$TARGET_HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/$TARGET_REVISION/config.json"
      ;;
    internvl35-8b)
      TARGET_MODEL="OpenGVLab/InternVL3_5-8B"
      TARGET_REVISION="9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79"
      TARGET_HF_HOME="$INTERNVL_TARGET_HF_HOME"
      TARGET_CHAT_KWARGS='{}'
      TARGET_SNAPSHOT="$TARGET_HF_HOME/hub/models--OpenGVLab--InternVL3_5-8B/snapshots/$TARGET_REVISION/config.json"
      ;;
    *)
      printf 'Unsupported TARGET_SLUGS entry: %s\n' "$slug" >&2
      return 2
      ;;
  esac
  [[ -f "$TARGET_SNAPSHOT" ]] || {
    printf 'Pinned model snapshot is missing: %s\n' "$TARGET_SNAPSHOT" >&2
    return 2
  }
}

run_module() {
  local track="$1"
  printf 'evaluation.%s.run_vllm\n' "$track"
}

run_raw_spec() {
  local slug="$1" spec="$2"
  local spec_id track questions image_root prompt_mode module run_dir diagnostics
  local attempt status
  IFS='|' read -r spec_id track questions image_root prompt_mode <<<"$spec"
  if [[ "$spec_id" == fidelity/* ]]; then
    wait_for_fidelity_bundle
  fi
  module="$(run_module "$track")"
  run_dir="$RUN_ROOT/$slug/$spec_id"
  diagnostics="$run_dir/diagnostics.jsonl"
  mkdir -p "$run_dir"
  if [[ -s "$run_dir/submission.jsonl" && -s "$run_dir/run_manifest.json" ]]; then
    log "Skipping completed $slug/$spec_id"
    return
  fi
  for attempt in 1 2 3; do
    write_status "target_inference" "$slug/$spec_id attempt=$attempt"
    set +e
    "$PYTHON" -m "$module" \
      --model "$TARGET_MODEL" \
      --endpoints "http://127.0.0.1:$PORT/v1" \
      --questions "$questions" \
      --image-root "$image_root" \
      --prompt-mode "$prompt_mode" \
      --temperature 0 \
      --top-p 1 \
      --seed "$SEED" \
      --chat-template-kwargs "$TARGET_CHAT_KWARGS" \
      --concurrency "$TARGET_CONCURRENCY" \
      --request-timeout 900 \
      --max-retries 2 \
      --checkpoint-every 10 \
      --resume \
      --defer-extraction \
      --out "$run_dir/submission.jsonl" \
      --diagnostics "$diagnostics"
    status=$?
    set -e
    if (( status == 0 )); then
      return
    fi
    log "Raw inference attempt $attempt failed for $slug/$spec_id; retrying pending rows"
    sleep 10
  done
  return 1
}

smoke_target() {
  local slug="$1"
  local run_dir="$RUN_ROOT/$slug/smoke"
  mkdir -p "$run_dir"
  write_status "smoke" "$slug"
  "$PYTHON" -m evaluation.minds_eye.run_vllm \
    --model "$TARGET_MODEL" \
    --endpoints "http://127.0.0.1:$PORT/v1" \
    --questions "$CAUSAL_ROOT/questions.jsonl" \
    --image-root "$CAUSAL_ROOT" \
    --prompt-mode noncot \
    --temperature 0 \
    --top-p 1 \
    --seed "$SEED" \
    --chat-template-kwargs "$TARGET_CHAT_KWARGS" \
    --concurrency "$TARGET_CONCURRENCY" \
    --request-timeout 900 \
    --max-retries 2 \
    --checkpoint-every 5 \
    --limit 5 \
    --defer-extraction \
    --out "$run_dir/submission.jsonl" \
    --diagnostics "$run_dir/diagnostics.jsonl"
}

run_target_matrix() {
  local slug="$1" spec
  load_target_config "$slug"
  launch_server \
    "$TARGET_MODEL" "$TARGET_REVISION" "$TARGET_HF_HOME" "$MAX_MODEL_LEN" \
    "$TARGET_CONCURRENCY" '{}' 1 "$RUN_ROOT/$slug/target-vllm.log"
  smoke_target "$slug"
  for spec in "${SPECS[@]}"; do
    run_raw_spec "$slug" "$spec"
  done
  stop_server
}

extract_spec() {
  local slug="$1" spec="$2"
  local spec_id track questions image_root prompt_mode module run_dir
  local attempt status quality_status
  IFS='|' read -r spec_id track questions image_root prompt_mode <<<"$spec"
  module="$(run_module "$track")"
  run_dir="$RUN_ROOT/$slug/$spec_id"
  status=2
  for ((attempt = 1; attempt <= EXTRACTOR_ATTEMPTS; attempt++)); do
    write_status "fixed_extraction" "$slug/$spec_id attempt=$attempt"
    set +e
    "$PYTHON" -m "$module" \
      --model "$TARGET_MODEL" \
      --endpoints "http://127.0.0.1:$PORT/v1" \
      --extractor-model "$EXTRACTOR_MODEL" \
      --extractor-revision "$EXTRACTOR_REVISION" \
      --extractor-endpoints "http://127.0.0.1:$PORT/v1" \
      --extractor-chat-template-kwargs '{"enable_thinking":false}' \
      --questions "$questions" \
      --image-root "$image_root" \
      --prompt-mode "$prompt_mode" \
      --temperature 0 \
      --top-p 1 \
      --seed "$SEED" \
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
    if (( status != 0 )); then
      log "Extractor request attempt $attempt failed for $slug/$spec_id"
    else
      set +e
      "$PYTHON" analysis/research/audit_inference_quality.py \
        --questions "$questions" \
        --diagnostics "$run_dir/diagnostics.jsonl" \
        --submission "$run_dir/submission.jsonl" \
        --output "$run_dir/quality" \
        --extractor-model "$EXTRACTOR_MODEL" \
        --extractor-revision "$EXTRACTOR_REVISION"
      quality_status=$?
      set -e
      if (( quality_status == 0 )); then
        status=0
        break
      fi
      status="$quality_status"
      log "Quality gate attempt $attempt failed for $slug/$spec_id"
    fi
    if (( attempt < EXTRACTOR_ATTEMPTS )); then
      log "Retrying only failed extractor rows for $slug/$spec_id"
      sleep 10
    fi
  done
  if (( status != 0 )); then
    log "Finalizing persistently unparseable extractor rows as unresolved for $slug/$spec_id"
    "$PYTHON" evaluation/research/finalize_persistent_extractor_failures.py \
      --diagnostics "$run_dir/diagnostics.jsonl" \
      --submission "$run_dir/submission.jsonl" \
      --report "$run_dir/quality/persistent_extractor_resolution.json" \
      --extractor-model "$EXTRACTOR_MODEL" \
      --extractor-revision "$EXTRACTOR_REVISION" \
      --attempts "$EXTRACTOR_ATTEMPTS"
    "$PYTHON" analysis/research/audit_inference_quality.py \
      --questions "$questions" \
      --diagnostics "$run_dir/diagnostics.jsonl" \
      --submission "$run_dir/submission.jsonl" \
      --output "$run_dir/quality" \
      --extractor-model "$EXTRACTOR_MODEL" \
      --extractor-revision "$EXTRACTOR_REVISION"
  fi
  "$PYTHON" evaluation/research/record_experiment_run.py \
    --output "$run_dir/run_manifest.json" \
    --experiment "controlled_visual_analysis" \
    --model "$TARGET_MODEL" \
    --model-revision "$TARGET_REVISION" \
    --extractor-model "$EXTRACTOR_MODEL" \
    --extractor-revision "$EXTRACTOR_REVISION" \
    --parameter "spec=$spec_id" \
    --parameter "track=$track" \
    --parameter "prompt_mode=$prompt_mode" \
    --parameter "temperature=0" \
    --parameter "top_p=1" \
    --parameter "max_model_len=$MAX_MODEL_LEN" \
    --parameter "completion_budget=context_remainder" \
    --parameter "dtype=bfloat16" \
    --parameter "seed=$SEED" \
    --parameter "persistent_unparseable_policy=persistent-unparseable-extractor-output-v1" \
    --input "questions=$questions" \
    --submission "$run_dir/submission.jsonl" \
    --diagnostics "$run_dir/diagnostics.jsonl"
}

extract_all() {
  local slug spec
  launch_server \
    "$EXTRACTOR_MODEL" "$EXTRACTOR_REVISION" "$EXTRACTOR_HF_HOME" \
    "$EXTRACTOR_MAX_MODEL_LEN" "$EXTRACTOR_CONCURRENCY" \
    '{"enable_thinking":false}' 0 "$RUN_ROOT/extractor-vllm.log"
  IFS=',' read -r -a target_slugs <<<"$TARGET_SLUGS"
  for slug in "${target_slugs[@]}"; do
    load_target_config "$slug"
    for spec in "${SPECS[@]}"; do
      extract_spec "$slug" "$spec"
    done
  done
  stop_server
}

score_model() {
  local slug="$1"
  local root="$RUN_ROOT/$slug"
  write_status "scoring" "$slug"
  for prompt_mode in noncot cot; do
    "$PYTHON" analysis/research/score_causal_transformations.py \
      --ground-truth "$CAUSAL_ROOT/private_ground_truth.jsonl" \
      --pairs "$CAUSAL_ROOT/pairs.jsonl" \
      --submission "$root/causal/$prompt_mode/submission.jsonl" \
      --output "$root/causal/$prompt_mode/score" \
      --seed "$EXPERIMENT_SEED"
  done
  "$PYTHON" analysis/research/compare_causal_runs.py \
    --baseline "direct=$root/causal/noncot/score/pair_results.csv" \
    --candidate "cot=$root/causal/cot/score/pair_results.csv" \
    --output "$root/causal/prompt_comparison" \
    --seed "$EXPERIMENT_SEED"
  printf '%s\n' \
    "The fidelity and abstraction submissions passed remote quality gates." \
    "Score them only on the trusted local host that contains the private visual ground truth." \
    >"$root/local_scoring_required.txt"
}

score_cross_model_causal() {
  local qwen="$RUN_ROOT/qwen3-vl-8b/causal"
  local internvl="$RUN_ROOT/internvl35-8b/causal"
  [[ -f "$qwen/noncot/score/pair_results.csv" ]] || return 0
  [[ -f "$internvl/noncot/score/pair_results.csv" ]] || return 0
  for prompt_mode in noncot cot; do
    "$PYTHON" analysis/research/compare_causal_runs.py \
      --baseline "qwen3_vl_8b=$qwen/$prompt_mode/score/pair_results.csv" \
      --candidate "internvl35_8b=$internvl/$prompt_mode/score/pair_results.csv" \
      --output "$ANALYSIS_ROOT/comparisons/causal/$prompt_mode" \
      --seed "$EXPERIMENT_SEED"
  done
}

prepare_initial_bundles() {
  write_status "preflight" "preparing_causal_and_abstraction_bundles"
  if [[ ! -f "$CAUSAL_ROOT/manifest.json" ]]; then
    "$PYTHON" evaluation/research/generate_causal_transformations.py \
      --output "$CAUSAL_ROOT" \
      --pairs 200 \
      --seed "$EXPERIMENT_SEED"
  fi
  if [[ ! -f "$ABSTRACTION_ROOT/manifest.json" ]]; then
    "$PYTHON" evaluation/research/prepare_state_interventions.py \
      --output "$ABSTRACTION_ROOT" \
      --seed "$EXPERIMENT_SEED"
  fi
  "$PYTHON" evaluation/research/validate_experiment_bundle.py \
    --manifest "$CAUSAL_ROOT/manifest.json" \
    --manifest "$ABSTRACTION_ROOT/manifest.json"
}

start_fidelity_prep() {
  if [[ -f "$FIDELITY_ROOT/manifest.json" ]]; then
    "$PYTHON" evaluation/research/validate_experiment_bundle.py \
      --manifest "$FIDELITY_ROOT/manifest.json"
    return
  fi
  log "Preparing the visual-fidelity bundle in parallel with causal inference"
  "$PYTHON" evaluation/research/prepare_visual_fidelity.py \
    --output "$FIDELITY_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --do-you-see-me-per-stratum 10 \
    --minds-eye-per-task 25 \
    --seed "$EXPERIMENT_SEED" \
    --workers 12 \
    >"$ANALYSIS_ROOT/logs/fidelity-preparation.log" 2>&1 &
  FIDELITY_PREP_PID="$!"
}

wait_for_fidelity_bundle() {
  if [[ -n "$FIDELITY_PREP_PID" ]]; then
    log "Waiting for visual-fidelity preparation process $FIDELITY_PREP_PID"
    wait "$FIDELITY_PREP_PID"
    FIDELITY_PREP_PID=""
    "$PYTHON" evaluation/research/validate_experiment_bundle.py \
      --manifest "$FIDELITY_ROOT/manifest.json"
    log "Visual-fidelity bundle is ready"
  fi
}

[[ -x "$PYTHON" ]] || {
  printf 'Visual-suite Python is not executable: %s\n' "$PYTHON" >&2
  exit 2
}
[[ -x "$VLLM" ]] || {
  printf 'vLLM executable is missing: %s\n' "$VLLM" >&2
  exit 2
}
[[ -d "$DATASET_ROOT" ]] || {
  printf 'Visual dataset root is missing: %s\n' "$DATASET_ROOT" >&2
  exit 2
}
[[ -f "$EXTRACTOR_HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/$EXTRACTOR_REVISION/config.json" ]] || {
  printf 'Pinned extractor snapshot is missing from %s\n' "$EXTRACTOR_HF_HOME" >&2
  exit 2
}
case "$START_PHASE" in
  inference | extraction | scoring) ;;
  *)
    printf 'START_PHASE must be inference, extraction, or scoring, not %s.\n' \
      "$START_PHASE" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
IFS=',' read -r -a target_slugs <<<"$TARGET_SLUGS"
if [[ "$START_PHASE" == "inference" ]]; then
  prepare_initial_bundles
  start_fidelity_prep
  for slug in "${target_slugs[@]}"; do
    run_target_matrix "$slug"
  done
elif [[ "$START_PHASE" == "extraction" ]]; then
  write_status "preflight" "resuming_from_fixed_extraction"
  "$PYTHON" evaluation/research/validate_experiment_bundle.py \
    --manifest "$CAUSAL_ROOT/manifest.json" \
    --manifest "$FIDELITY_ROOT/manifest.json" \
    --manifest "$ABSTRACTION_ROOT/manifest.json"
fi

if [[ "$START_PHASE" != "scoring" ]]; then
  extract_all
fi

for slug in "${target_slugs[@]}"; do
  score_model "$slug"
done
score_cross_model_causal

write_status "complete" "all_models_extracted_causal_scored"
log "Controlled analysis queue completed"
