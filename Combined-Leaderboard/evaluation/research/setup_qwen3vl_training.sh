#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QWEN_ROOT="${QWEN_ROOT:-${PROJECT_ROOT}/evaluation/.runtime/qwen3-vl}"
TRAIN_VENV="${TRAIN_VENV:-${PROJECT_ROOT}/.venv/qwen3vl-training}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
QWEN_REPOSITORY="${QWEN_REPOSITORY:-https://github.com/QwenLM/Qwen3-VL.git}"
QWEN_REVISION="${QWEN_REVISION:-96588727e44c78b25ba03ea03b8e12f7e64fd0da}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.1}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-80}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf 'ERROR: %s is required. Do not use Python 3.13 for this training environment.\n' "${PYTHON_BIN}" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'ERROR: nvidia-smi is required on the GPU host.\n' >&2
  exit 2
fi

if ! command -v nvcc >/dev/null 2>&1; then
  printf 'ERROR: nvcc is required to build the pinned flash-attn package. Load the CUDA toolkit module first.\n' >&2
  exit 2
fi

free_kib="$(df -Pk "${PROJECT_ROOT}" | awk 'NR == 2 {print $4}')"
free_gib=$((free_kib / 1024 / 1024))
if (( free_gib < MIN_FREE_DISK_GB )); then
  printf 'ERROR: only %s GiB is free; training setup requires at least %s GiB.\n' "${free_gib}" "${MIN_FREE_DISK_GB}" >&2
  exit 2
fi

if [[ ! -d "${QWEN_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${QWEN_ROOT}")"
  git clone "${QWEN_REPOSITORY}" "${QWEN_ROOT}"
fi

git -C "${QWEN_ROOT}" fetch --quiet origin "${QWEN_REVISION}"
git -C "${QWEN_ROOT}" checkout --quiet --detach "${QWEN_REVISION}"
actual_revision="$(git -C "${QWEN_ROOT}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${QWEN_REVISION}" ]]; then
  printf 'ERROR: Qwen source revision drifted: %s\n' "${actual_revision}" >&2
  exit 2
fi

if [[ ! -x "${TRAIN_VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${TRAIN_VENV}"
fi

"${TRAIN_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${TRAIN_VENV}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
"${TRAIN_VENV}/bin/python" -m pip install \
  "transformers==${TRANSFORMERS_VERSION}" \
  deepspeed==0.17.1 \
  triton==3.2.0 \
  accelerate==1.7.0 \
  torchcodec==0.2 \
  peft==0.17.1 \
  pillow==12.3.0 \
  packaging ninja
"${TRAIN_VENV}/bin/python" -m pip install \
  flash_attn==2.7.4.post1 --no-build-isolation

"${TRAIN_VENV}/bin/python" - <<'PY'
import torch
import transformers
import deepspeed
import flash_attn
import peft

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available in the training environment")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("ERROR: the selected GPUs do not support bfloat16")
print(
    "Training environment ready:",
    f"PyTorch {torch.__version__},",
    f"Transformers {transformers.__version__},",
    f"CUDA {torch.version.cuda},",
    f"GPUs {torch.cuda.device_count()}",
)
PY

printf 'Pinned Qwen training source: %s\n' "${QWEN_ROOT}"
printf 'Training environment: %s\n' "${TRAIN_VENV}"
