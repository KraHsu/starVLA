#!/bin/bash
# Drive scripts/extract_vjepa_latents.py over the 4 LIBERO suites in the
# `pav_libero` mixture. Run from repo root.
#
# Override defaults via env vars:
#   GPU_LIST="0 1 2 3"   default: "0 1 2 3 4 5 6 7"
#   OUTPUT_DIR=...       default: data/latents/pav_libero
#   BATCH_SIZE=32        default: 32
#   DTYPE=fp16           default: fp16  (use bf16 only on H100/H200)
#   NO_WRIST=1           default: unset (keeps wrist view)
#
# T-W2.2.3 — full LIBERO V-JEPA 2 latent extraction.
set -e

STARVLA_DIR=$(pwd)
STARVLA_PYTHON=${STARVLA_DIR}/.venv/bin/python

GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
OUTPUT_DIR=${OUTPUT_DIR:-data/latents/pav_libero}
BATCH_SIZE=${BATCH_SIZE:-32}
DTYPE=${DTYPE:-fp16}

GPU_ARR=($GPU_LIST)
NUM_GPUS=${#GPU_ARR[@]}
GPU_CSV=$(IFS=, ; echo "${GPU_ARR[*]}")

DATASETS=(
    "libero_object_no_noops_1.0.0_lerobot"
    "libero_goal_no_noops_1.0.0_lerobot"
    "libero_spatial_no_noops_1.0.0_lerobot"
    "libero_10_no_noops_1.0.0_lerobot"
)

EXTRA_ARGS=()
if [ -n "${NO_WRIST:-}" ]; then
    EXTRA_ARGS+=(--no-wrist)
fi

mkdir -p "$OUTPUT_DIR"

# Use a master port that's unlikely to clash with concurrent torchruns.
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}

echo "==> Extracting V-JEPA 2 latents for ${#DATASETS[@]} suites on GPUs [${GPU_CSV}] → ${OUTPUT_DIR}"
for ds in "${DATASETS[@]}"; do
    # Idempotent: skip a dataset only if every rank shard is already on disk.
    shard_dir="${OUTPUT_DIR}/${ds}"
    if [ -d "$shard_dir" ]; then
        found=$(find "$shard_dir" -maxdepth 1 -name "${ds}_rank*.h5" | wc -l)
    else
        found=0
    fi
    if [ "$found" -eq "$NUM_GPUS" ]; then
        echo "--- $ds: $found/$NUM_GPUS shards present, skipping ---"
        continue
    fi
    echo "--- $ds (have $found/$NUM_GPUS shards) ---"
    CUDA_VISIBLE_DEVICES=${GPU_CSV} ${STARVLA_PYTHON} -m torch.distributed.run \
        --nproc_per_node=${NUM_GPUS} \
        --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
        scripts/extract_vjepa_latents.py \
        --data-name "${ds}" \
        --robot-type libero_franka \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size ${BATCH_SIZE} \
        --dtype ${DTYPE} \
        "${EXTRA_ARGS[@]}"
done

echo "==> Done. Shards under ${OUTPUT_DIR}/<dataset>/<dataset>_rank{NN}.h5"
