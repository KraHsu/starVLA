# Plan-and-Verify (PAV)

A "Lego" extension on top of [starVLA](https://github.com/starVLA/starVLA): adds a verifier-only world-model loop on top of an off-the-shelf π₀ flow-matching policy. See [docs/plan_and_verify.md](../../docs/plan_and_verify.md) for project overview and Gate timeline.

## Status

- W1 done: env + V-JEPA 2 wrapper + LIBERO-Long baseline. Gate G-W1 result lives in [paper/tables/baseline_table.csv](../../paper/tables/baseline_table.csv).
- Stage 1 done: `QwenOFT` baseline on `libero_goal`, final eval 98/100. See [BASELINE.md](./BASELINE.md).
- Stage 2 done: offline V-JEPA cache extractor + PCA sanity plot.
- Stage 3 in progress: `QwenOFT_VJepa`, TensorBoard logging, offline cache training path, and online LIBERO eval fallback.

## Environments

This project ships with **two** uv-managed virtual environments, mirroring starVLA's
two-env setup but using uv instead of conda:

| venv          | Python | Purpose                                         |
|---------------|--------|-------------------------------------------------|
| `.venv`        | 3.11   | starVLA training/inference + V-JEPA 2 stack     |
| `.venv-libero` | 3.10   | LIBERO simulator (mujoco / robosuite / robomimic) |

```bash
# Main training/inference env
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install --python .venv/bin/python -e third_party/vjepa2 --no-deps
uv pip install --python .venv/bin/python submitit braceexpand webdataset beartype python-box ftfy fire h5py peft pytest

# LIBERO simulator env (separate to keep mujoco/numpy/transformers pins from clashing)
uv venv --python 3.10 .venv-libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git playground/LIBERO
uv pip install --python .venv-libero/bin/python -e playground/LIBERO
uv pip install --python .venv-libero/bin/python \
    "mujoco==3.2.3" "numpy==1.24.4" "robosuite==1.4.0" "robomimic==0.2.0" \
    "bddl==1.0.1" "hydra-core==1.2.0" "easydict==1.9" "future==0.18.2" \
    "cloudpickle==2.1.0" "gym==0.25.2" "einops==0.4.1" \
    tyro matplotlib mediapy websockets msgpack msgpack-numpy \
    opencv-python pillow imageio thop wandb rich omegaconf transformers tdigest
apt install -y libosmesa6 libosmesa6-dev libgl1-mesa-dri  # for headless mujoco
```

## Required pretrained assets

| What                                    | Path                                                                                       |
|-----------------------------------------|--------------------------------------------------------------------------------------------|
| Qwen3-VL-4B-Instruct (HF)               | `playground/Pretrained_models/Qwen3-VL-4B-Instruct/`                                       |
| StarVLA-PI Qwen3-VL LIBERO ckpt          | `playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt` |
| V-JEPA 2-AC ViT-g (encoder + predictor) | `playground/Pretrained_models/vjepa2_vitg/vjepa2-ac-vitg.pt`                               |

## LIBERO-Long baseline (Gate G-W1)

```bash
# Terminal 1 — policy server (starVLA env)
bash examples/PlanAndVerify/eval_files/run_policy_server.sh

# Terminal 2 — LIBERO simulator client
bash examples/PlanAndVerify/eval_files/eval_libero_long.sh

# Aggregate trial-level results into baseline_table.csv
.venv/bin/python scripts/dump_baseline_table.py \
    --eval_results playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/results/libero_10/Qwen3-VL-PI-LIBERO-4in1_checkpoints_steps_100000_pytorch_model.pt \
    --out paper/tables/baseline_table.csv
```

Pass criterion: LIBERO-Long mean success rate ≥ 0.86.

## V-JEPA 2 unit tests

```bash
.venv/bin/python -m pytest tests/world_model/test_vjepa2.py -v
```

Four tests; the two GPU-bound ones load V-JEPA 2 ViT-g with random weights and run a
forward pass (~40 s on H20).

## Stage 2 cache extraction

Single-card smoke:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/PlanAndVerify/cache_files/extract_vjepa_cache.py \
  --config_yaml examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml \
  --dataset_name libero_goal \
  --camera_key video.primary_image \
  --vjepa_ckpt /path/to/vjepa2_1_vitb_dist_vitG_384.pt \
  --output_dir playground/cache/vjepa/vjepa2_1_vit_b_384/libero_goal \
  --num_history_frames 8 \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda:0 \
  --dtype bf16 \
  --max_episodes 1
```

PCA sanity:

```bash
.venv/bin/python examples/PlanAndVerify/cache_files/extract_vjepa_cache.py \
  --config_yaml examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml \
  --dataset_name libero_goal \
  --camera_key video.primary_image \
  --output_dir playground/cache/vjepa/vjepa2_1_vit_b_384/libero_goal \
  --run_pca_sanity \
  --pca_num_samples 100
```

## Stage 3 training and checks

Training launcher:

```bash
WANDB_MODE=disabled bash examples/PlanAndVerify/train_files/run_oft_vjepa_libero_goal.sh
```

Useful overrides:

```bash
DEBUG_STEPS=200 WANDB_MODE=disabled bash examples/PlanAndVerify/train_files/run_oft_vjepa_libero_goal.sh
VJEPA_FUSION=cross_attn WANDB_MODE=disabled bash examples/PlanAndVerify/train_files/run_oft_vjepa_libero_goal.sh
VJEPA_ENCODER_CKPT=/path/to/vjepa2_1_vitb_dist_vitG_384.pt WANDB_MODE=disabled bash examples/PlanAndVerify/train_files/run_oft_vjepa_libero_goal.sh
```

TensorBoard:

```bash
.venv/bin/tensorboard --logdir playground/Checkpoints --host 0.0.0.0 --port 6006
```

Setup checks:

```bash
.venv/bin/python examples/PlanAndVerify/train_files/check_oft_vjepa_setup.py \
  --config_yaml examples/PlanAndVerify/train_files/starvla_oft_vjepa_libero_goal.yaml
```

Real single-batch check:

```bash
.venv/bin/python examples/PlanAndVerify/train_files/check_oft_vjepa_setup.py \
  --config_yaml examples/PlanAndVerify/train_files/starvla_oft_vjepa_libero_goal.yaml \
  --run_single_batch
```

Notes:

- Training reads `episode_id/frame_id` from the LeRobot sample and pulls pooled V-JEPA features from the cache.
- LIBERO sim eval does not have those ids. `QwenOFT_VJepa.predict_action()` now falls back to an online V-JEPA encoder using a rolling history of the primary view.
- If `framework.vjepa.encoder_ckpt_path` is left null, the framework auto-discovers it from `cache_dir/index.json`.
