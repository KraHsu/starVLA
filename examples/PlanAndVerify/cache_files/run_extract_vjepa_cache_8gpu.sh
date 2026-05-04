#!/usr/bin/env bash
set -euo pipefail

CONFIG_YAML="${1:-examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml}"
CKPT_PATH="${2:-}"
OUT_DIR="${3:-playground/cache/vjepa/vjepa2_1_vit_b_384/libero_goal}"
NUM_HISTORY_FRAMES="${4:-8}"
BATCH_SIZE="${5:-8}"
NUM_WORKERS="${6:-1}"
DTYPE="${7:-bf16}"
MAX_GPU_MEMORY_GB="${8:-100}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Usage: $0 [CONFIG_YAML] CKPT_PATH [OUT_DIR] [NUM_HISTORY_FRAMES] [BATCH_SIZE] [NUM_WORKERS] [DTYPE] [MAX_GPU_MEMORY_GB]" >&2
  echo "Example:" >&2
  echo "  $0 examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml ./playground/Pretrained_models/vjepa2_vitg/vjepa2_1_vitb_dist_vitG_384.pt" >&2
  exit 1
fi

NUM_SHARDS=8
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "==> Starting V-JEPA cache extraction on 8 GPUs"
echo "    config: ${CONFIG_YAML}"
echo "    ckpt:   ${CKPT_PATH}"
echo "    out:    ${OUT_DIR}"
echo "    N:      ${NUM_HISTORY_FRAMES}"
echo "    batch:  ${BATCH_SIZE}"
echo "    workers:${NUM_WORKERS}"
echo "    dtype:  ${DTYPE}"
echo "    max GPU memory/process: ${MAX_GPU_MEMORY_GB} GiB"
echo "    logs:   ${LOG_DIR}"
echo "    thread env: OMP/MKL/OPENBLAS/NUMEXPR=1"

pids=()
for SHARD_ID in $(seq 0 7); do
  LOG_FILE="${LOG_DIR}/extract_shard_${SHARD_ID}.log"
  echo "==> Launch shard ${SHARD_ID}/${NUM_SHARDS} on CUDA_VISIBLE_DEVICES=${SHARD_ID}; log=${LOG_FILE}"
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES="${SHARD_ID}" \
    .venv/bin/python examples/PlanAndVerify/cache_files/extract_vjepa_cache.py \
      --config_yaml "${CONFIG_YAML}" \
      --dataset_name libero_goal \
      --camera_key video.primary_image \
      --vjepa_ckpt "${CKPT_PATH}" \
      --output_dir "${OUT_DIR}" \
      --num_history_frames "${NUM_HISTORY_FRAMES}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --device cuda:0 \
      --dtype "${DTYPE}" \
      --max_gpu_memory_gb "${MAX_GPU_MEMORY_GB}" \
      >"${LOG_FILE}" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if wait "${pid}"; then
    echo "==> Shard ${i} finished"
  else
    echo "[FAIL] Shard ${i} failed; see ${LOG_DIR}/extract_shard_${i}.log" >&2
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[FAIL] At least one shard failed" >&2
  exit 1
fi

echo "==> All shards finished"
echo "==> Cache index: ${OUT_DIR}/index.json"
