#!/bin/bash
# Stage 3 — StarVLA-OFT + offline V-JEPA pooled features on LIBERO-Goal.

set -e

[[ -n "${NCCL_SOCKET_IFNAME:-}" ]] && export NCCL_SOCKET_IFNAME
[[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}

Framework_name=QwenOFT_VJepa
freeze_module_list="${FREEZE_MODULES:-}"
base_vlm=${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${CONFIG_YAML:-./examples/PlanAndVerify/train_files/starvla_oft_vjepa_libero_goal.yaml}
libero_data_root=${LIBERO_DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}
data_mix=${DATA_MIX:-libero_goal}
vjepa_cache_dir=${VJEPA_CACHE_DIR:-playground/cache/vjepa/vjepa2_1_vit_b_384/libero_goal}
run_root_dir=${RUN_ROOT_DIR:-./playground/Checkpoints}
RUN_ID=${RUN_ID:-stage3_oft_vjepa_libero_goal_$(date -u +%Y%m%d_%H%M%S)}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-16}
WANDB_ENTITY=${WANDB_ENTITY:-${USER:-anonymous}}
WANDB_MODE=${WANDB_MODE:-online}
NUM_PROCESSES=${NUM_PROCESSES:-8}

if [[ -n "${DEBUG_STEPS:-}" ]]; then
    MAX_TRAIN_STEPS=${DEBUG_STEPS}
    RUN_ID="${RUN_ID}_smoke${DEBUG_STEPS}"
    echo "[run_oft_vjepa_libero_goal.sh] DEBUG_STEPS=${DEBUG_STEPS}; using run_id=${RUN_ID}"
fi

export WANDB_MODE
export WANDB_ENTITY

output_dir=${run_root_dir}/${RUN_ID}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${config_yaml}" "${output_dir}/"

ACCEL_BIN=${ACCEL_BIN:-.venv/bin/accelerate}

${ACCEL_BIN} launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
    --config_yaml ${config_yaml} \
    --framework.name ${Framework_name} \
    --framework.qwenvl.base_vlm ${base_vlm} \
    --framework.vjepa.cache_dir ${vjepa_cache_dir} \
    --datasets.vla_data.data_root_dir ${libero_data_root} \
    --datasets.vla_data.data_mix ${data_mix} \
    --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BATCH} \
    --trainer.freeze_modules "${freeze_module_list}" \
    --trainer.max_train_steps ${MAX_TRAIN_STEPS} \
    --trainer.save_interval 5000 \
    --trainer.eval_interval 100 \
    --trainer.logging_frequency 50 \
    --run_root_dir ${run_root_dir} \
    --run_id ${RUN_ID} \
    --wandb_project starVLA_PAV_Stage3 \
    --wandb_entity ${WANDB_ENTITY}
