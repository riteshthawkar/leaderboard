#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s perception_curriculum|recognition_control\n' "$0" >&2
  exit 2
fi

CONDITION="$1"
case "${CONDITION}" in
  perception_curriculum)
    DATASET_NAME="ms_vista_perception_curriculum"
    ;;
  recognition_control)
    DATASET_NAME="ms_vista_recognition_control"
    ;;
  *)
    printf 'ERROR: unsupported condition: %s\n' "${CONDITION}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QWEN_ROOT="${QWEN_ROOT:-${PROJECT_ROOT}/evaluation/.runtime/qwen3-vl}"
TRAIN_VENV="${TRAIN_VENV:-${PROJECT_ROOT}/.venv/qwen3vl-training}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/evaluation/research/results/perception_curriculum}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/evaluation/research/results/perception_lora}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1}"
MASTER_PORT="${MASTER_PORT:-29531}"
EPOCHS="${EPOCHS:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
SEED="${SEED:-20260723}"
MAX_STEPS="${MAX_STEPS:--1}"

IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
if [[ "${#gpu_array[@]}" -ne 2 ]]; then
  printf 'ERROR: GPU_IDS must contain exactly two comma-separated GPU IDs.\n' >&2
  exit 2
fi

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  QWEN_ROOT="${QWEN_ROOT}" TRAIN_VENV="${TRAIN_VENV}" \
    bash "${SCRIPT_DIR}/setup_qwen3vl_training.sh"
fi

"${TRAIN_VENV}/bin/python" "${SCRIPT_DIR}/register_qwen_dataset.py" \
  --qwen-root "${QWEN_ROOT}" \
  --dataset-root "${DATASET_ROOT}"

output_dir="${OUTPUT_ROOT}/seed-${SEED}/${CONDITION}"
mkdir -p "${output_dir}"

cd "${QWEN_ROOT}/qwen-vl-finetune"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${QWEN_ROOT}/qwen-vl-finetune${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

"${TRAIN_VENV}/bin/torchrun" \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path "${MODEL_ID}" \
  --dataset_use "${DATASET_NAME}" \
  --data_flatten True \
  --tune_mm_vision False \
  --tune_mm_mlp False \
  --tune_mm_llm False \
  --bf16 \
  --lora_enable True \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --output_dir "${output_dir}" \
  --num_train_epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_pixels 262144 \
  --min_pixels 784 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 250 \
  --save_total_limit 2 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --run_name "ms-vista-${CONDITION}" \
  --report_to none

printf 'Completed %s LoRA training at %s\n' "${CONDITION}" "${output_dir}"
