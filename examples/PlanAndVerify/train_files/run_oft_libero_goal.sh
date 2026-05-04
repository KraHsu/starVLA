#!/bin/bash
# Stage 1 baseline — StarVLA-OFT + Qwen3-VL-4B on LIBERO-Goal, 8×H20, 20k steps.
# Run from repo root. Self-copies into the run dir for reproducibility.
#
# Override defaults via env vars:
#   RUN_ID              default: stage1_oft_libero_goal_<utc-timestamp>
#   MAX_TRAIN_STEPS     default: 20000
#   PER_DEVICE_BATCH    default: 16     (effective batch = 16 * 8 = 128)
#   WANDB_ENTITY        default: $WANDB_ENTITY (env)
#   WANDB_MODE          default: online   ("disabled" to skip wandb)
#   DEBUG_STEPS         unset = real run; set to e.g. 10 for smoke dry-run
#
# NOTE on gradient accumulation: starVLA/config/deepseeds/ds_config.yaml
# hard-codes "gradient_accumulation_steps": 1, and DeepSpeed takes
# precedence over both Accelerate and our trainer.gradient_accumulation_steps
# yaml field. Changing grad-accum here would require editing ds_config.yaml
# (shared module — out of scope for Stage 1). Effective batch is therefore
# PER_DEVICE_BATCH × num_processes × 1 = 128.

set -e

# === NCCL / distributed env ===
# Single-node, 8-GPU: rely on NCCL auto-detection. Original LIBERO script
# baked in bond0/mlx5_* which don't exist on the H20 dev box (only eth0-3 +
# net0 are present); leaving them unset is the right default. Override per
# multi-node setup by `export NCCL_SOCKET_IFNAME=...` before invoking.
[[ -n "${NCCL_SOCKET_IFNAME:-}" ]] && export NCCL_SOCKET_IFNAME
[[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}

###############################################################################
# === Editable variables ===
Framework_name=QwenOFT
freeze_module_list=""
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_goal
run_root_dir=./playground/Checkpoints
RUN_ID=${RUN_ID:-stage1_oft_libero_goal_$(date -u +%Y%m%d_%H%M%S)}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-16}
WANDB_ENTITY=${WANDB_ENTITY:-${USER:-anonymous}}
WANDB_MODE=${WANDB_MODE:-online}
###############################################################################

# Smoke-test override: caller sets DEBUG_STEPS=10 to do a 10-step dry run that
# reuses the full 8-GPU launcher path but exits before training cost matters.
if [[ -n "${DEBUG_STEPS:-}" ]]; then
    MAX_TRAIN_STEPS=${DEBUG_STEPS}
    RUN_ID="${RUN_ID}_smoke${DEBUG_STEPS}"
    echo "[run_oft_libero_goal.sh] DEBUG_STEPS=${DEBUG_STEPS}; using run_id=${RUN_ID}"
fi

export WANDB_MODE
export WANDB_ENTITY

output_dir=${run_root_dir}/${RUN_ID}
mkdir -p "${output_dir}"
# Reproducibility: snapshot launcher + yaml into the run dir.
cp "$0" "${output_dir}/"
cp "${config_yaml}" "${output_dir}/"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
ACCEL_BIN=${ACCEL_BIN:-.venv/bin/accelerate}

${ACCEL_BIN} launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
    --config_yaml ${config_yaml} \
    --framework.name ${Framework_name} \
    --framework.qwenvl.base_vlm ${base_vlm} \
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
    --wandb_project starVLA_PAV_Stage1 \
    --wandb_entity ${WANDB_ENTITY}
