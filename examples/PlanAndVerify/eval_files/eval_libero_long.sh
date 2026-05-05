#!/bin/bash
# PAV-specific LIBERO-Long evaluation client. Run from repo root.
set -e

STARVLA_DIR=$(pwd)

DEFAULT_CKPT=${STARVLA_DIR}/playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt
CKPT=${CKPT:-$DEFAULT_CKPT}

export LIBERO_HOME=${STARVLA_DIR}/playground/LIBERO
export LIBERO_Python=${STARVLA_DIR}/.venv-libero/bin/python

export PYTHONPATH=${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH}

export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}

host=${HOST:-127.0.0.1}
base_port=${PORT:-6694}
unnorm_key=${UNNORM_KEY:-franka}
task_suite_name=${TASK_SUITE:-libero_10}
num_trials_per_task=${NUM_TRIALS:-30}

folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
model_root=$(echo "$CKPT" | awk -F'/checkpoints/' '{print $1}')
[[ "$model_root" == "$CKPT" ]] && model_root=$(dirname "$(dirname "$CKPT")")
video_out_path="${model_root}/results/${task_suite_name}/${folder_name}"

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
