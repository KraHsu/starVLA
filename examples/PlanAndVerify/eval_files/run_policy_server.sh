#!/bin/bash
# PAV-specific policy server. Run from repo root.
set -e

STARVLA_DIR=$(pwd)
STARVLA_PYTHON=${STARVLA_DIR}/.venv/bin/python

CKPT=${STARVLA_DIR}/playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt

export PYTHONPATH=${STARVLA_DIR}:${PYTHONPATH}
export HTTPS_PROXY=${HTTPS_PROXY:-http://127.0.0.1:7890}
export HTTP_PROXY=${HTTP_PROXY:-http://127.0.0.1:7890}

gpu_id=${GPU_ID:-0}
port=${PORT:-6694}

CUDA_VISIBLE_DEVICES=$gpu_id ${STARVLA_PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT}" \
    --port ${port} \
    --use_bf16
