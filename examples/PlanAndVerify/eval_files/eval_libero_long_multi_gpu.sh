#!/bin/bash
# Multi-GPU LIBERO-Long eval. Spins up one policy server per GPU and one
# eval client per server, with the 10 LIBERO-Long tasks round-robin
# sharded across GPUs. Run from repo root.
#
# Override defaults via env vars:
#   GPU_LIST="0 1 2 3"   default: "0 1 2 3 4 5 6 7"
#   BASE_PORT=6694       default: 6694 (each GPU gets BASE_PORT + idx)
#   NUM_TRIALS=30        default: 30
#   TASK_SUITE=libero_10 default: libero_10
#   CKPT=/path/to/model.pt default: published StarVLA LIBERO-Long ckpt
#   READY_TIMEOUT=180    default: 180  (seconds to wait per server load)
set -e

STARVLA_DIR=$(pwd)
STARVLA_PYTHON=${STARVLA_DIR}/.venv/bin/python
LIBERO_PYTHON=${STARVLA_DIR}/.venv-libero/bin/python

DEFAULT_CKPT=${STARVLA_DIR}/playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt
CKPT=${CKPT:-$DEFAULT_CKPT}
SERVER_SCRIPT=deployment/model_server/server_policy.py
CLIENT_SCRIPT=examples/PlanAndVerify/eval_files/eval_libero_long_sharded.py

GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
BASE_PORT=${BASE_PORT:-6694}
NUM_TRIALS=${NUM_TRIALS:-30}
TASK_SUITE=${TASK_SUITE:-libero_10}
READY_TIMEOUT=${READY_TIMEOUT:-180}

case "$TASK_SUITE" in
  libero_10)      NUM_TASKS=10 ;;
  libero_90)     NUM_TASKS=90 ;;
  libero_spatial)NUM_TASKS=10 ;;
  libero_object) NUM_TASKS=10 ;;
  libero_goal)   NUM_TASKS=10 ;;
  *) echo "[FATAL] unknown TASK_SUITE: $TASK_SUITE" >&2; exit 1 ;;
esac

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
export PYTHONPATH=${STARVLA_DIR}/playground/LIBERO:${STARVLA_DIR}:${PYTHONPATH}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export HTTPS_PROXY=${HTTPS_PROXY:-http://127.0.0.1:7890}
export HTTP_PROXY=${HTTP_PROXY:-http://127.0.0.1:7890}

LOG_DIR=/tmp/pav-eval-multi
mkdir -p "$LOG_DIR"
# Clear stale per-server / per-client logs, but DO NOT rm -rf $LOG_DIR
# itself — nohup may have already opened driver.log inside it before we
# entered this script.
find "$LOG_DIR" -maxdepth 1 \( -name "server-*.log" -o -name "client-*.log" -o -name "*.pids" \) -delete 2>/dev/null || true

folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
model_root=$(echo "$CKPT" | awk -F'/checkpoints/' '{print $1}')
[[ "$model_root" == "$CKPT" ]] && model_root=$(dirname "$(dirname "$CKPT")")
VIDEO_OUT="${model_root}/results/${TASK_SUITE}/${folder_name}"
mkdir -p "$VIDEO_OUT"

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

# Save server pids so the client side or a kill-all can reach them.
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
exit $fail
