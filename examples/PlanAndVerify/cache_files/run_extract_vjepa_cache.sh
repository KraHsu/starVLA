#!/usr/bin/env bash
set -euo pipefail

CONFIG_YAML="${1:-examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml}"
CKPT_PATH="${2:-}"
DATASET_NAME="${DATASET_NAME:-libero_goal}"
OUT_DIR="${3:-playground/cache/vjepa/vjepa2_1_vit_b_384/${DATASET_NAME}}"
NUM_HISTORY_FRAMES="${4:-8}"
NUM_SHARDS="${5:-1}"
SHARD_ID="${6:-0}"
BATCH_SIZE="${7:-8}"
NUM_WORKERS="${8:-4}"
DEVICE="${9:-cuda:0}"
DTYPE="${10:-bf16}"
MAX_GPU_MEMORY_GB="${11:-}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Usage: $0 [CONFIG_YAML] CKPT_PATH [OUT_DIR] [NUM_HISTORY_FRAMES] [NUM_SHARDS] [SHARD_ID] [BATCH_SIZE] [NUM_WORKERS] [DEVICE] [DTYPE] [MAX_GPU_MEMORY_GB]" >&2
  exit 1
fi

CMD=(
  .venv/bin/python examples/PlanAndVerify/cache_files/extract_vjepa_cache.py
  --config_yaml "${CONFIG_YAML}" \
  --dataset_name "${DATASET_NAME}" \
  --camera_key video.primary_image \
  --vjepa_ckpt "${CKPT_PATH}" \
  --output_dir "${OUT_DIR}" \
  --num_history_frames "${NUM_HISTORY_FRAMES}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_id "${SHARD_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"
)

if [[ -n "${MAX_GPU_MEMORY_GB}" ]]; then
  CMD+=(--max_gpu_memory_gb "${MAX_GPU_MEMORY_GB}")
fi

"${CMD[@]}"
