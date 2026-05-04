#!/bin/bash
# Stage 1 sim eval — sharded LIBERO-Goal evaluation across all visible GPUs.
# Mirrors eval_libero_long_multi_gpu.sh but locks TASK_SUITE=libero_goal,
# defaults NUM_TRIALS=10 (per implementation_todo.md §1.2), and exposes
# CKPT as the only required input. Run from repo root.
#
# Usage:
#   CKPT=/abs/path/to/steps_20000_pytorch_model.pt \
#     bash examples/PlanAndVerify/eval_files/eval_libero_goal_sharded.sh
#
# Override defaults via env vars:
#   CKPT          REQUIRED. Absolute path to a starVLA checkpoint .pt file.
#   GPU_LIST      default: "0 1 2 3 4 5 6 7"
#   BASE_PORT     default: 6694 (each GPU gets BASE_PORT + idx)
#   NUM_TRIALS    default: 10
#   READY_TIMEOUT default: 180  (seconds for server to come up)

set -e

if [[ -z "${CKPT:-}" ]]; then
    echo "[FATAL] CKPT env var is required (absolute path to .pt file)" >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "[FATAL] CKPT does not exist: $CKPT" >&2
    exit 1
fi

STARVLA_DIR=$(pwd)
STARVLA_PYTHON=${STARVLA_DIR}/.venv/bin/python
LIBERO_PYTHON=${STARVLA_DIR}/.venv-libero/bin/python

SERVER_SCRIPT=deployment/model_server/server_policy.py
CLIENT_SCRIPT=examples/PlanAndVerify/eval_files/eval_libero_long_sharded.py

GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
BASE_PORT=${BASE_PORT:-6694}
NUM_TRIALS=${NUM_TRIALS:-10}
TASK_SUITE=libero_goal           # locked by name
NUM_TASKS=10                     # libero_goal is 10 tasks
READY_TIMEOUT=${READY_TIMEOUT:-180}

GPU_ARR=($GPU_LIST)
NUM_GPUS=${#GPU_ARR[@]}

# Round-robin shard tasks across GPUs.
declare -a SHARD_TASKS
for ((i=0; i<NUM_GPUS; i++)); do SHARD_TASKS[$i]=""; done
for ((t=0; t<NUM_TASKS; t++)); do
    idx=$((t % NUM_GPUS))
    if [ -z "${SHARD_TASKS[$idx]}" ]; then
        SHARD_TASKS[$idx]="$t"
    else
        SHARD_TASKS[$idx]="${SHARD_TASKS[$idx]},$t"
    fi
done

# Common env for both server and client subshells.
export PYTHONPATH=${STARVLA_DIR}/playground/LIBERO:${STARVLA_DIR}:${PYTHONPATH:-}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export HTTPS_PROXY=${HTTPS_PROXY:-http://127.0.0.1:7890}
export HTTP_PROXY=${HTTP_PROXY:-http://127.0.0.1:7890}

LOG_DIR=/tmp/pav-eval-libero-goal
mkdir -p "$LOG_DIR"
find "$LOG_DIR" -maxdepth 1 \( -name "server-*.log" -o -name "client-*.log" -o -name "*.pids" \) -delete 2>/dev/null || true

folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
model_root=$(echo "$CKPT" | awk -F'/checkpoints/' '{print $1}')
[[ "$model_root" == "$CKPT" ]] && model_root=$(dirname "$(dirname "$CKPT")")
VIDEO_OUT="${model_root}/results/${TASK_SUITE}/${folder_name}"
mkdir -p "$VIDEO_OUT"

echo "==> CKPT       = ${CKPT}"
echo "==> TASK_SUITE = ${TASK_SUITE}  NUM_TASKS=${NUM_TASKS}  NUM_TRIALS=${NUM_TRIALS}"
echo "==> VIDEO_OUT  = ${VIDEO_OUT}"

# ---------- Phase 1: launch all servers ----------
declare -a SERVER_PIDS
echo "==> Launching ${NUM_GPUS} policy servers"
for ((i=0; i<NUM_GPUS; i++)); do
    gpu=${GPU_ARR[$i]}
    port=$((BASE_PORT + i))
    log=$LOG_DIR/server-gpu${gpu}-port${port}.log
    CUDA_VISIBLE_DEVICES=$gpu nohup ${STARVLA_PYTHON} ${SERVER_SCRIPT} \
        --ckpt_path "${CKPT}" --port ${port} --use_bf16 \
        > "$log" 2>&1 &
    SERVER_PIDS[$i]=$!
    echo "  GPU ${gpu} server pid=${SERVER_PIDS[$i]} port=${port} tasks=[${SHARD_TASKS[$i]}] log=${log}"
done

echo "${SERVER_PIDS[@]}" > $LOG_DIR/server.pids

cleanup() {
    echo "==> Cleanup: killing servers ${SERVER_PIDS[*]}"
    for pid in "${SERVER_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ---------- Phase 2: wait until each server logs "server running" ----------
echo "==> Waiting up to ${READY_TIMEOUT}s for all servers to be ready"
for ((i=0; i<NUM_GPUS; i++)); do
    gpu=${GPU_ARR[$i]}
    port=$((BASE_PORT + i))
    log=$LOG_DIR/server-gpu${gpu}-port${port}.log
    waited=0
    while ! grep -q "server running" "$log" 2>/dev/null; do
        if ! ps -p ${SERVER_PIDS[$i]} > /dev/null; then
            echo "[FATAL] server on GPU $gpu died early. Last 30 log lines:"
            tail -30 "$log" >&2
            exit 1
        fi
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $READY_TIMEOUT ]; then
            echo "[FATAL] server on GPU $gpu did not become ready within ${READY_TIMEOUT}s"
            tail -20 "$log" >&2
            exit 1
        fi
    done
    echo "  GPU ${gpu} ready (waited ${waited}s)"
done

# ---------- Phase 3: launch clients ----------
declare -a CLIENT_PIDS
echo "==> Launching ${NUM_GPUS} eval clients"
for ((i=0; i<NUM_GPUS; i++)); do
    gpu=${GPU_ARR[$i]}
    port=$((BASE_PORT + i))
    tasks="${SHARD_TASKS[$i]}"
    log=$LOG_DIR/client-gpu${gpu}-port${port}.log
    if [ -z "$tasks" ]; then
        echo "  GPU ${gpu}: no tasks assigned (skipping client)"
        continue
    fi
    nohup ${LIBERO_PYTHON} ${CLIENT_SCRIPT} \
        --args.pretrained-path "${CKPT}" \
        --args.host 127.0.0.1 \
        --args.port ${port} \
        --args.task-suite-name "${TASK_SUITE}" \
        --args.num-trials-per-task ${NUM_TRIALS} \
        --args.task-ids "${tasks}" \
        --args.video-out-path "${VIDEO_OUT}" \
        > "$log" 2>&1 &
    CLIENT_PIDS[$i]=$!
    echo "  GPU ${gpu} client pid=${CLIENT_PIDS[$i]} tasks=[${tasks}] log=${log}"
done

# ---------- Phase 4: wait for all clients ----------
echo "==> Waiting for clients to finish"
fail=0
for ((i=0; i<NUM_GPUS; i++)); do
    pid=${CLIENT_PIDS[$i]:-}
    [ -z "$pid" ] && continue
    if wait $pid; then
        echo "  client on GPU ${GPU_ARR[$i]} OK"
    else
        echo "  [WARN] client on GPU ${GPU_ARR[$i]} exited non-zero"
        fail=1
    fi
done

echo "==> Done. video_out_path=${VIDEO_OUT}"
echo "==> Aggregate results into paper/tables/baseline_oft_libero_goal.csv:"
echo "    .venv/bin/python scripts/dump_baseline_table.py \\"
echo "        --eval_results ${VIDEO_OUT} \\"
echo "        --out paper/tables/baseline_oft_libero_goal.csv \\"
echo "        --label StarVLA-OFT-Qwen3VL"
exit $fail
