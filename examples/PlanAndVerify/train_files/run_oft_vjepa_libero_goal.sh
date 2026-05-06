#!/bin/bash
# Stage 3 — StarVLA-OFT + offline V-JEPA pooled features on LIBERO-Goal.
# Supports deterministic low-data runs via DATA_FRACTION + SUBSET_SEED.

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
vjepa_fusion=${VJEPA_FUSION:-film_gating}
vjepa_encoder_ckpt=${VJEPA_ENCODER_CKPT:-}
run_root_dir=${RUN_ROOT_DIR:-./playground/Checkpoints}
RUN_ID=${RUN_ID:-stage3_oft_vjepa_libero_goal_$(date -u +%Y%m%d_%H%M%S)}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-16}
WANDB_ENTITY=${WANDB_ENTITY:-${USER:-anonymous}}
WANDB_MODE=${WANDB_MODE:-online}
NUM_PROCESSES=${NUM_PROCESSES:-8}
SEED=${SEED:-42}
DATA_FRACTION=${DATA_FRACTION:-1.0}
SUBSET_SEED=${SUBSET_SEED:-$SEED}
RESUME=${RESUME:-0}
RESUME_RUN_DIR=${RESUME_RUN_DIR:-}
RESUME_RUN_ID=${RESUME_RUN_ID:-}
RESUME_CKPT=${RESUME_CKPT:-}

if [[ -n "${RESUME_RUN_DIR}" || -n "${RESUME_RUN_ID}" || -n "${RESUME_CKPT}" ]]; then
    RESUME=1
fi

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
EXTRA_ARGS=()

resolve_resume_checkpoint() {
    local checkpoint_dir=$1
    local latest_ckpt
    if [[ ! -d "${checkpoint_dir}" ]]; then
        echo "[run_oft_vjepa_libero_goal.sh] checkpoint directory not found: ${checkpoint_dir}" >&2
        exit 1
    fi
    latest_ckpt=$(find "${checkpoint_dir}" -maxdepth 1 -type f \
        \( -name 'steps_*_pytorch_model.pt' -o -name 'steps_*_model.safetensors' \) \
        | sort -V | tail -n 1)
    if [[ -z "${latest_ckpt}" ]]; then
        echo "[run_oft_vjepa_libero_goal.sh] no checkpoint found under ${checkpoint_dir}" >&2
        exit 1
    fi
    printf '%s\n' "${latest_ckpt}"
}

if [[ -n "${vjepa_encoder_ckpt}" ]]; then
    echo "[run_oft_vjepa_libero_goal.sh] using online eval encoder ckpt: ${vjepa_encoder_ckpt}"
    EXTRA_ARGS+=(--framework.vjepa.encoder_ckpt_path "${vjepa_encoder_ckpt}")
fi

if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "True" || "${RESUME}" == "yes" ]]; then
    resume_source_run_dir=""
    resume_ckpt_path="${RESUME_CKPT}"

    if [[ -n "${RESUME_RUN_DIR}" ]]; then
        resume_source_run_dir="${RESUME_RUN_DIR}"
    elif [[ -n "${RESUME_RUN_ID}" ]]; then
        resume_source_run_dir="${run_root_dir}/${RESUME_RUN_ID}"
    fi

    if [[ -z "${resume_ckpt_path}" && -n "${resume_source_run_dir}" ]]; then
        resume_ckpt_path=$(resolve_resume_checkpoint "${resume_source_run_dir}/checkpoints")
    fi

    if [[ -z "${resume_ckpt_path}" ]]; then
        resume_source_run_dir="${output_dir}"
        resume_ckpt_path=$(resolve_resume_checkpoint "${output_dir}/checkpoints")
    fi

    if [[ ! -f "${resume_ckpt_path}" ]]; then
        echo "[run_oft_vjepa_libero_goal.sh] resume checkpoint not found: ${resume_ckpt_path}" >&2
        exit 1
    fi

    resume_ckpt_abs=$(realpath "${resume_ckpt_path}")
    output_dir_abs=$(realpath "${output_dir}")
    resume_dir_abs=$(dirname "${resume_ckpt_abs}")
    expected_output_ckpt_dir="${output_dir_abs}/checkpoints"

    if [[ "${resume_dir_abs}" != "${expected_output_ckpt_dir}" ]]; then
        mkdir -p "${expected_output_ckpt_dir}"
        cp -f "${resume_ckpt_abs}" "${expected_output_ckpt_dir}/"
        resume_ckpt_abs="${expected_output_ckpt_dir}/$(basename "${resume_ckpt_abs}")"
        echo "[run_oft_vjepa_libero_goal.sh] copied resume checkpoint into ${expected_output_ckpt_dir}"
    fi

    echo "[run_oft_vjepa_libero_goal.sh] resuming from ${resume_ckpt_abs}"
    EXTRA_ARGS+=(--trainer.is_resume true)
    EXTRA_ARGS+=(--trainer.pretrained_checkpoint "${resume_ckpt_abs}")
fi

${ACCEL_BIN} launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
    --config_yaml ${config_yaml} \
    --seed ${SEED} \
    --datasets.vla_data.seed ${SEED} \
    --datasets.vla_data.data_fraction ${DATA_FRACTION} \
    --datasets.vla_data.subset_seed ${SUBSET_SEED} \
    --framework.name ${Framework_name} \
    --framework.qwenvl.base_vlm ${base_vlm} \
    --framework.vjepa.cache_dir ${vjepa_cache_dir} \
    --framework.vjepa.fusion ${vjepa_fusion} \
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
    --wandb_entity ${WANDB_ENTITY} \
    "${EXTRA_ARGS[@]}"
