#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${TRACK:?Set TRACK to do_you_see_me or minds_eye}"
: "${CONDITION_ROOT:?Set CONDITION_ROOT to the prepared condition bundle}"
: "${CONDITIONS:?Set comma-separated CONDITIONS}"
: "${MODEL:?Set the exact model identifier served by vLLM}"
: "${MODEL_REVISION:?Set the immutable model checkpoint revision}"
: "${EXTRACTOR_MODEL:?Set the gold-blind extractor model identifier}"
: "${EXTRACTOR_REVISION:?Set the immutable extractor checkpoint revision}"

case "${TRACK}" in
  do_you_see_me|minds_eye) ;;
  *)
    printf 'ERROR: unsupported TRACK=%s\n' "${TRACK}" >&2
    exit 2
    ;;
esac

ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1}"
EXTRACTOR_ENDPOINT="${EXTRACTOR_ENDPOINT:-${ENDPOINT}}"
if [[ "${TRACK}" == "do_you_see_me" ]]; then
  PROMPT_MODES="${PROMPT_MODES:-noncot}"
  TEMPERATURE="${TEMPERATURE:-1.0}"
  TOP_P="${TOP_P:-0.95}"
else
  PROMPT_MODES="${PROMPT_MODES:-cot}"
  TEMPERATURE="${TEMPERATURE:-0.1}"
  TOP_P="${TOP_P:-1.0}"
fi
MAX_TOKENS="${MAX_TOKENS:-0}"
MAX_FINAL_ANSWER_TOKENS="${MAX_FINAL_ANSWER_TOKENS:-200}"
CONCURRENCY="${CONCURRENCY:-32}"
SEED="${SEED:-0}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{}}"
EXTRACTOR_CHAT_TEMPLATE_KWARGS="${EXTRACTOR_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/evaluation/research/results/gpu_runs/${TRACK}}"
FORCE="${FORCE:-0}"
LIMIT="${LIMIT:-0}"

if [[ ! "${LIMIT}" =~ ^[0-9]+$ ]]; then
  printf 'ERROR: LIMIT must be a non-negative integer.\n' >&2
  exit 2
fi
if [[ ! "${MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  printf 'ERROR: MAX_TOKENS must be a non-negative integer.\n' >&2
  exit 2
fi
max_token_args=()
if (( MAX_TOKENS > 0 )); then
  max_token_args=(--max-tokens "${MAX_TOKENS}")
fi
limit_args=()
if (( LIMIT > 0 )); then
  limit_args=(--limit "${LIMIT}" --strict-partial)
fi

IFS=',' read -r -a condition_array <<< "${CONDITIONS}"
IFS=',' read -r -a prompt_array <<< "${PROMPT_MODES}"

cd "${PROJECT_ROOT}"
for condition in "${condition_array[@]}"; do
  questions="${CONDITION_ROOT}/${condition}/questions.jsonl"
  if [[ ! -f "${questions}" ]]; then
    printf 'ERROR: condition questions not found: %s\n' "${questions}" >&2
    exit 2
  fi
  for prompt_mode in "${prompt_array[@]}"; do
    case "${prompt_mode}" in
      noncot|cot) ;;
      *)
        printf 'ERROR: unsupported prompt mode: %s\n' "${prompt_mode}" >&2
        exit 2
        ;;
    esac
    run_dir="${OUTPUT_ROOT}/${condition}/${prompt_mode}"
    submission="${run_dir}/submission.jsonl"
    diagnostics="${run_dir}/diagnostics.jsonl"
    manifest="${run_dir}/run_manifest.json"
    mkdir -p "${run_dir}"
    if [[ "${FORCE}" != "1" && -s "${submission}" && -s "${manifest}" ]]; then
      printf 'Skipping completed %s/%s\n' "${condition}" "${prompt_mode}"
      continue
    fi
    printf 'Running %s/%s with %s\n' "${condition}" "${prompt_mode}" "${MODEL}"
    ".venv/visual-suite/bin/python" -m "evaluation.${TRACK}.run_vllm" \
      --model "${MODEL}" \
      --endpoints "${ENDPOINT}" \
      --extractor-model "${EXTRACTOR_MODEL}" \
      --extractor-revision "${EXTRACTOR_REVISION}" \
      --extractor-endpoints "${EXTRACTOR_ENDPOINT}" \
      --questions "${questions}" \
      --image-root "${CONDITION_ROOT}/${condition}" \
      --prompt-mode "${prompt_mode}" \
      "${max_token_args[@]}" \
      --max-final-answer-tokens "${MAX_FINAL_ANSWER_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --seed "${SEED}" \
      --chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}" \
      --extractor-chat-template-kwargs "${EXTRACTOR_CHAT_TEMPLATE_KWARGS}" \
      --concurrency "${CONCURRENCY}" \
      --request-timeout 900 \
      --max-retries 2 \
      --checkpoint-every 10 \
      --resume \
      --out "${submission}" \
      --diagnostics "${diagnostics}" \
      "${limit_args[@]}"
    if (( LIMIT > 0 )); then
      printf 'Smoke run passed for %s/%s; no submission was exported.\n' "${condition}" "${prompt_mode}"
      continue
    fi
    ".venv/visual-suite/bin/python" evaluation/research/record_experiment_run.py \
      --output "${manifest}" \
      --experiment "visual_condition_matrix" \
      --model "${MODEL}" \
      --model-revision "${MODEL_REVISION}" \
      --extractor-model "${EXTRACTOR_MODEL}" \
      --extractor-revision "${EXTRACTOR_REVISION}" \
      --parameter "track=${TRACK}" \
      --parameter "condition=${condition}" \
      --parameter "prompt_mode=${prompt_mode}" \
      --parameter "max_tokens=$([[ "${MAX_TOKENS}" == "0" ]] && printf context_remainder || printf '%s' "${MAX_TOKENS}")" \
      --parameter "temperature=${TEMPERATURE}" \
      --parameter "top_p=${TOP_P}" \
      --parameter "seed=${SEED}" \
      --input "questions=${questions}" \
      --submission "${submission}" \
      --diagnostics "${diagnostics}"
  done
done

printf 'Condition matrix complete: %s\n' "${OUTPUT_ROOT}"
