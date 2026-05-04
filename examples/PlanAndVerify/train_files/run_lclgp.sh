#!/bin/bash
# LCLGP training launcher (Plan-and-Verify W3.3).
# 8×H20 default; tune NUM_PROCESSES + sampler.n_tasks for other clusters.

set -e

# === NCCL / cluster env (mirror of run_libero_train.sh) ===
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Edit per environment ===
NUM_PROCESSES=${NUM_PROCESSES:-8}
CONFIG_YAML=${CONFIG_YAML:-examples/PlanAndVerify/configs/lclgp_v1.yaml}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-./playground/Checkpoints}
RUN_ID=${RUN_ID:-pav_w3_lclgp_v1}
WANDB_PROJECT=${WANDB_PROJECT:-pav-w3-lclgp}
WANDB_ENTITY=${WANDB_ENTITY:-}
# === End ===
###########################################################################################

OUTPUT_DIR=${RUN_ROOT_DIR}/${RUN_ID}
mkdir -p ${OUTPUT_DIR}
cp $0 ${OUTPUT_DIR}/

EXTRA_ARGS=()
if [ -n "${WANDB_ENTITY}" ]; then
  EXTRA_ARGS+=(--wandb_entity ${WANDB_ENTITY})
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_lclgp.py \
  --config_yaml ${CONFIG_YAML} \
  --run_root_dir ${RUN_ROOT_DIR} \
  --run_id ${RUN_ID} \
  --wandb_project ${WANDB_PROJECT} \
  "${EXTRA_ARGS[@]}" \
  "$@"
