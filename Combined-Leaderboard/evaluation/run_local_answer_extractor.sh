#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_ROOT="${LOCAL_EXTRACTOR_RUNTIME_ROOT:-$PROJECT_ROOT/evaluation/.runtime/local-answer-extractor}"
MODEL_ID="mlx-community/Qwen3.5-2B-8bit"
MODEL_REVISION="562fb8ea19bbe7565a3ddf5b7d49899ea2c88d2b"
PORT="${LOCAL_EXTRACTOR_PORT:-8099}"
ENDPOINT="http://127.0.0.1:${PORT}/v1"
SERVER_LOG="${LOCAL_EXTRACTOR_SERVER_LOG:-$PROJECT_ROOT/evaluation/results/local_extractor_server.log}"
STAGING_ROOT="${LOCAL_EXTRACTOR_STAGING_ROOT:-$PROJECT_ROOT/evaluation/results/local_extractor_review}"
VALIDATION_SAMPLE="${LOCAL_EXTRACTOR_VALIDATION_SAMPLE:-256}"
SLUGS="${SLUGS:-}"
MIN_FREE_DISK_GIB="${LOCAL_EXTRACTOR_MIN_FREE_DISK_GIB:-8}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ "$(uname -s)" == "Darwin" ]] || fail "The local MLX extractor requires macOS."
[[ "$(uname -m)" == "arm64" ]] || fail "The local MLX extractor requires Apple Silicon."
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python was not found: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import resource' >/dev/null 2>&1 || fail \
  "This Python runtime is blocked by macOS code-signing policy. Use Homebrew ARM64 Python."

memory_bytes="$(sysctl -n hw.memsize)"
(( memory_bytes >= 14 * 1024 * 1024 * 1024 )) || fail \
  "At least 14 GiB of unified memory is required."
free_disk_kib="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
(( free_disk_kib >= MIN_FREE_DISK_GIB * 1024 * 1024 )) || fail \
  "At least ${MIN_FREE_DISK_GIB} GiB of free disk is required."

if [[ ! -x "$RUNTIME_ROOT/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$RUNTIME_ROOT"
fi
if ! "$RUNTIME_ROOT/bin/python" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version
assert version("mlx-lm") == "0.31.3"
assert version("openai") == "2.46.0"
assert version("huggingface-hub") == "1.24.0"
PY
then
  "$RUNTIME_ROOT/bin/python" -m pip install --upgrade pip
  "$RUNTIME_ROOT/bin/python" -m pip install \
    --requirement evaluation/requirements-local-extractor.txt
fi

printf 'Preparing pinned local extractor %s at %s\n' "$MODEL_ID" "$MODEL_REVISION"
MODEL_PATH="$(HF_HUB_DISABLE_XET=1 "$RUNTIME_ROOT/bin/python" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="$MODEL_ID",
    revision="$MODEL_REVISION",
    max_workers=1,
))
PY
)"
[[ -f "$MODEL_PATH/config.json" ]] || fail "Pinned model snapshot is incomplete: $MODEL_PATH"

if curl --silent --fail "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  fail "Port ${PORT} is already serving a model. Choose LOCAL_EXTRACTOR_PORT."
fi
mkdir -p "$(dirname "$SERVER_LOG")"
"$RUNTIME_ROOT/bin/mlx_lm.server" \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-tokens 64 \
  --decode-concurrency 1 \
  --prompt-concurrency 1 \
  --prompt-cache-size 4 \
  --chat-template-args '{"enable_thinking":false}' \
  --log-level WARNING \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if curl --silent --fail "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  kill -0 "$SERVER_PID" >/dev/null 2>&1 || fail \
    "The MLX server exited during startup. Inspect $SERVER_LOG"
  sleep 1
done
curl --silent --fail "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 || fail \
  "The MLX server did not become ready. Inspect $SERVER_LOG"

args=(
  --endpoint "$ENDPOINT"
  --api-model "$MODEL_PATH"
  --model-id "$MODEL_ID"
  --model-revision "$MODEL_REVISION"
  --quantization "8-bit MLX"
  --runtime "mlx-lm 0.31.3 / MLX 0.32.0"
  --staging-root "$STAGING_ROOT"
  --validation-sample "$VALIDATION_SAMPLE"
  --max-validation-errors 0
  --min-validation-resolution 0.80
  --concurrency 1
  --max-tokens 64
)
if [[ -n "$SLUGS" ]]; then
  args+=(--slugs "$SLUGS")
fi

PYTHONPATH="$PROJECT_ROOT" "$RUNTIME_ROOT/bin/python" \
  -m evaluation.recover_visual_answers "${args[@]}"

printf '\nReview-only extraction artifacts: %s\n' "$STAGING_ROOT"
printf 'Canonical leaderboard results were not modified.\n'
