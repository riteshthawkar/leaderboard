#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to a canonical tree containing only the variants to audit.}"
AUDIT_ROOT="${AUDIT_ROOT:?Set AUDIT_ROOT to a durable output directory.}"
EXTRACTOR_SNAPSHOT="${EXTRACTOR_SNAPSHOT:?Set EXTRACTOR_SNAPSHOT to the pinned Qwen3-8B snapshot.}"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/visual-suite/bin/python}"
VLLM_BIN="${VLLM_BIN:-$PROJECT_ROOT/.venv/visual-suite/bin/vllm}"
EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-Qwen/Qwen3-8B}"
EXTRACTOR_REVISION="${EXTRACTOR_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
EXPECTED_CONTRACT="${EXPECTED_CONTRACT:-bce3aeedda9249130e0fbb0749777ae64b0c1b5e9151fe8e51136c45d0962a66}"
GPU_IDS="${GPU_IDS:-0,1}"
PORT="${PORT:-8075}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
MIN_FREE_GPU_MEMORY_MIB="${MIN_FREE_GPU_MEMORY_MIB:-18000}"
EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES:-}"

AUDIT_PATH="$AUDIT_ROOT/audit.jsonl"
SERVER_LOG="$AUDIT_ROOT/vllm.log"
CLIENT_LOG="$AUDIT_ROOT/client.log"
CACHE_ROOT="$AUDIT_ROOT/cache"
TEMP_ROOT="$AUDIT_ROOT/tmp"
SERVER_PID=""

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null \
            || kill -TERM "$SERVER_PID" 2>/dev/null \
            || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$PROJECT_ROOT"
mkdir -p "$AUDIT_ROOT" "$CACHE_ROOT" "$TEMP_ROOT"

[[ -d "$SOURCE_ROOT" ]] || {
    log "Canonical source root does not exist: $SOURCE_ROOT"
    exit 1
}
[[ -s "$EXTRACTOR_SNAPSHOT/model.safetensors.index.json" ]] || {
    log "Pinned extractor snapshot is incomplete: $EXTRACTOR_SNAPSHOT"
    exit 1
}
[[ -x "$PYTHON_BIN" && -x "$VLLM_BIN" ]] || {
    log "The configured Python or vLLM executable is unavailable."
    exit 1
}

relevant_changes="$(
    git status --porcelain --untracked-files=all -- \
        evaluation/extract_canonical_answers.py \
        evaluation/build_production_visual_results.py \
        visual_answer_contract.py
)"
[[ -z "$relevant_changes" ]] || {
    log "Canonical extraction code has uncommitted changes:"
    printf '%s\n' "$relevant_changes"
    exit 1
}

contract="$(
    "$PYTHON_BIN" - <<'PY'
from evaluation.extract_canonical_answers import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    extractor_contract_sha256,
)

print(
    extractor_contract_sha256(
        DEFAULT_EXTRACTOR_MODEL,
        256,
        DEFAULT_EXTRACTOR_REVISION,
    )
)
PY
)"
[[ "$contract" == "$EXPECTED_CONTRACT" ]] || {
    log "Extractor contract mismatch: expected $EXPECTED_CONTRACT, found $contract"
    exit 1
}

candidate_count="$(
    SOURCE_ROOT="$SOURCE_ROOT" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from evaluation.extract_canonical_answers import load_candidates

project_root = Path.cwd()
source_root = Path(os.environ["SOURCE_ROOT"])
print(len(load_candidates(project_root, source_root, "all")))
PY
)"
[[ "$candidate_count" =~ ^[0-9]+$ && "$candidate_count" -gt 0 ]] || {
    log "No audit candidates were found."
    exit 1
}
if [[ -n "$EXPECTED_CANDIDATES" && "$candidate_count" -ne "$EXPECTED_CANDIDATES" ]]; then
    log "Expected $EXPECTED_CANDIDATES candidates, found $candidate_count"
    exit 1
fi

IFS=',' read -r -a gpu_array <<<"$GPU_IDS"
tensor_parallel_size="${#gpu_array[@]}"
for gpu in "${gpu_array[@]}"; do
    free="$(
        nvidia-smi -i "$gpu" \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
            | tr -d ' '
    )"
    [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_FREE_GPU_MEMORY_MIB" ]] || {
        log "GPU $gpu has $free MiB free; $MIN_FREE_GPU_MEMORY_MIB MiB is required."
        exit 1
    }
done

"$PYTHON_BIN" - "$PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY

log "Starting pinned v4 extractor on GPUs $GPU_IDS for $candidate_count responses"
: >"$SERVER_LOG"
setsid env \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    XDG_CACHE_HOME="$CACHE_ROOT" \
    VLLM_CACHE_ROOT="$CACHE_ROOT/vllm" \
    TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" \
    TRITON_CACHE_DIR="$CACHE_ROOT/triton" \
    TMPDIR="$TEMP_ROOT" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 \
    "$VLLM_BIN" serve "$EXTRACTOR_SNAPSHOT" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --served-model-name "$EXTRACTOR_MODEL" \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --tensor-parallel-size "$tensor_parallel_size" \
    --data-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --generation-config vllm \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --disable-custom-all-reduce \
    --trust-remote-code >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 360); do
    if curl -fsS "http://127.0.0.1:$PORT/v1/models" \
        | grep -Fq "$EXTRACTOR_MODEL"; then
        ready=1
        break
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || {
        log "Extractor server exited during startup. See $SERVER_LOG"
        exit 1
    }
    sleep 5
done
[[ "$ready" -eq 1 ]] || {
    log "Extractor server did not become ready. See $SERVER_LOG"
    exit 1
}

: >"$CLIENT_LOG"
for pass in 1 2; do
    log "Running evidence audit pass $pass"
    "$PYTHON_BIN" -m evaluation.extract_canonical_answers \
        --project-root "$PROJECT_ROOT" \
        --canonical-root "$SOURCE_ROOT" \
        --endpoint "http://127.0.0.1:$PORT/v1" \
        --model "$EXTRACTOR_MODEL" \
        --revision "$EXTRACTOR_REVISION" \
        --policy all \
        --output "$AUDIT_PATH" \
        --concurrency "$CONCURRENCY" \
        --max-tokens "$MAX_TOKENS" \
        --timeout 600 \
        --endpoint-start-timeout 60 \
        --retries 2 \
        --report-every 100 >>"$CLIENT_LOG" 2>&1
done

log "Fail-closing any persistent extractor schema failures"
"$PYTHON_BIN" -m evaluation.extract_canonical_answers \
    --project-root "$PROJECT_ROOT" \
    --canonical-root "$SOURCE_ROOT" \
    --model "$EXTRACTOR_MODEL" \
    --revision "$EXTRACTOR_REVISION" \
    --policy all \
    --output "$AUDIT_PATH" \
    --max-tokens "$MAX_TOKENS" \
    --finalize-checkpoint >>"$CLIENT_LOG" 2>&1

SOURCE_ROOT="$SOURCE_ROOT" AUDIT_PATH="$AUDIT_PATH" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from evaluation.build_production_visual_results import load_completed_audit
from evaluation.extract_canonical_answers import load_candidates

project_root = Path.cwd()
source_root = Path(os.environ["SOURCE_ROOT"])
audit_path = Path(os.environ["AUDIT_PATH"])
candidates = load_candidates(project_root, source_root, "all")
audit, contract = load_completed_audit(audit_path, candidates)
if len(audit) != len(candidates):
    raise SystemExit("Completed audit coverage changed during validation.")
print(
    {
        "candidate_count": len(candidates),
        "audit_count": len(audit),
        "contract_sha256": contract,
        "audit_path": str(audit_path),
    }
)
PY

log "Canonical v4 evidence audit completed successfully"
