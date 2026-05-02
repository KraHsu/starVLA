# Plan-and-Verify — From-Zero Setup

This is the full reproducibility recipe for W1: bare server → reproducing the LIBERO-Long baseline (Gate G-W1). Every command here was actually executed during the W1 run; nothing is aspirational. If you are bringing up a fresh machine, follow it top to bottom.

The cluster used to validate this had: 8×H20 (97 GB each), CUDA 12.8 driver, glibc 2.35, Ubuntu 22.04. Adjust GPU/CUDA wheel pins if yours differ.

## 0. Network access

The cluster sits behind a proxy. Export it once at the top of every shell session that downloads anything (HF, GitHub releases, apt). All commands below assume these are set:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HF_TOKEN=<your-hf-token>   # required for StarVLA gated repos
```

## 1. Repository

```bash
# Fork the upstream starVLA repo on GitHub first; then:
git clone --recurse-submodules git@github.com:<you>/starVLA.git
cd starVLA
git remote add upstream https://github.com/starVLA/starVLA.git
git checkout -b pav-dev   # all PAV work happens on this branch

# If you forgot --recurse-submodules:
git submodule update --init --recursive
```

The `third_party/vjepa2` submodule must be present after this; `ls third_party/vjepa2/setup.py` should succeed.

## 2. System packages

```bash
apt update
apt install -y cmake                           # robomimic egl-probe needs it
apt install -y libosmesa6 libosmesa6-dev libgl1-mesa-dri  # mujoco software rendering
```

EGL on this cluster's NVIDIA driver does not expose the PLATFORM_DEVICE
extension that mujoco needs for headless rendering; we use OSMesa instead. If
your environment supports EGL (e.g., a workstation with X), you can skip the
osmesa packages and set `MUJOCO_GL=egl`.

## 3. Main env (`.venv`, Python 3.11) — starVLA + V-JEPA 2

V-JEPA 2's setup.py requires Python ≥ 3.11. starVLA is ≥ 3.10. We pick 3.11.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
```

`.venv` does not include `pip` by default; always invoke installs through
`uv pip install --python .venv/bin/python ...`.

flash-attn must match torch's ABI. starVLA is on `torch==2.6.0+cu124` with
`_GLIBCXX_USE_CXX11_ABI=False`, so use the prebuilt wheel:

```bash
uv pip install --python .venv/bin/python \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
```

V-JEPA 2 (install with `--no-deps` so it does not roll back transformers /
timm / decord to its own pins):

```bash
uv pip install --python .venv/bin/python -e third_party/vjepa2 --no-deps
uv pip install --python .venv/bin/python \
    submitit braceexpand webdataset beartype python-box ftfy fire h5py peft pytest
```

vjepa2 ships a research-style layout: its internal modules use `from src.x.y import z`,
which assumes the repo root is on `sys.path`. Our `_VJEPA2_Interface` wrapper
prepends `third_party/vjepa2` automatically; do not rely on
`import vjepa2` working at the top level.

Smoke check:

```bash
.venv/bin/python -c "
import torch, transformers, deepspeed, accelerate, flash_attn, starVLA
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      'devices', torch.cuda.device_count())
print('flash_attn', flash_attn.__version__)
"
```

Expect: torch 2.6.0+cu124, cuda True, 8 devices (or however many you have).

## 4. LIBERO simulator env (`.venv-libero`, Python 3.10)

LIBERO pins numpy 1.24 / transformers 4.21 / robosuite 1.4 — incompatible with
the main starVLA stack. It needs a separate venv.

```bash
uv venv --python 3.10 .venv-libero

# Clone LIBERO into playground/ (gitignored).
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git playground/LIBERO
uv pip install --python .venv-libero/bin/python -e playground/LIBERO

# Sim + eval-client deps. Versions match starVLA's examples/LIBERO/eval_files/install_libero.sh.
uv pip install --python .venv-libero/bin/python \
    "mujoco==3.2.3" "numpy==1.24.4" \
    "robosuite==1.4.0" "robomimic==0.2.0" "bddl==1.0.1" "hydra-core==1.2.0" \
    "easydict==1.9" "future==0.18.2" "cloudpickle==2.1.0" "gym==0.25.2" \
    "einops==0.4.1" \
    tyro matplotlib mediapy websockets msgpack msgpack-numpy \
    opencv-python pillow imageio thop wandb \
    rich omegaconf transformers tdigest

# rich/omegaconf/transformers/tdigest are needed because eval_libero.py
# imports starVLA.model.tools, which transitively imports a rich-backed logger.
```

Patch LIBERO for PyTorch 2.6+: the old init-state pickles use numpy globals
that `torch.load(weights_only=True)` (the new default) refuses to unpickle.

```bash
sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' \
    playground/LIBERO/libero/libero/benchmark/__init__.py
```

LIBERO's first import writes `~/.libero/config.yaml` interactively. Pipe an
"N" to accept defaults so it doesn't block:

```bash
PYTHONPATH=$(pwd)/playground/LIBERO MUJOCO_GL=osmesa \
    .venv-libero/bin/python -c "
import io, sys
sys.stdin = io.StringIO('N\n')
from libero.libero import benchmark
print(list(benchmark.get_benchmark_dict().keys()))
"
```

Expect `['libero_spatial', 'libero_object', 'libero_goal', 'libero_90', 'libero_10', 'libero_100']`.
LIBERO-Long is `libero_10`.

## 5. Pretrained checkpoints

All large weights live under `playground/Pretrained_models/` (gitignored).
For repeated runs, prefer symlinking from a shared cache rather than
re-downloading.

### 5.1 Qwen3-VL-4B-Instruct (VLM backbone, ~9 GB)

This is the backbone the StarVLA-PI ckpt's `config.yaml` points at via
`./playground/Pretrained_models/Qwen3-VL-4B-Instruct`. Download once:

```bash
.venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

Or, if you already have it cached elsewhere, symlink:

```bash
ln -s /path/to/cache/Qwen3-VL-4B-Instruct \
      playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

### 5.2 StarVLA-PI Qwen3-VL LIBERO ckpt (16.4 GB)

```bash
.venv/bin/hf download StarVLA/Qwen3-VL-PI-LIBERO-4in1 \
    --local-dir playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1
```

After the download, you should see:

```
playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/
├── checkpoints/steps_100000_pytorch_model.pt   # 16.36 GB — the actual weights
├── config.json
├── config.yaml
├── dataset_statistics.json
└── ...
```

### 5.3 V-JEPA 2-AC ViT-g (encoder + predictor, 11.8 GB)

Cluster cache exists at `/mnt/cpfs/zch/vjepa2_pretrain/vjepa2-ac-vitg.pt`;
prefer symlinking:

```bash
mkdir -p playground/Pretrained_models/vjepa2_vitg
ln -s /mnt/cpfs/zch/vjepa2_pretrain/vjepa2-ac-vitg.pt \
      playground/Pretrained_models/vjepa2_vitg/vjepa2-ac-vitg.pt
```

If you don't have the cpfs mount, fetch it directly:

```bash
curl -L -o playground/Pretrained_models/vjepa2_vitg/vjepa2-ac-vitg.pt \
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
```

Single file holds both the encoder (`encoder` key) and AC predictor
(`predictor` key) state dicts. The `_VJEPA2_Interface` wrapper splits them.

## 6. LIBERO-Long dataset

Only `libero_10` is needed for W1. Use the LeRobot v3 mirror that starVLA
trains against:

```bash
.venv/bin/hf download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
    --repo-type dataset \
    --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot
```

Note: LIBERO _evaluation_ does not actually need this dataset (the simulator
resets from bddl init states packaged with the LIBERO repo). It's needed for
the W1.3.4 integration test and for W2 onward (LCLGP training data).

If you already have the four LIBERO suites cached as
`playground/Datasets/LEROBOT_LIBERO_DATA/libero_{spatial,object,goal,10}_no_noops_1.0.0_lerobot/`
(this is the path starVLA's `examples/LIBERO/data_preparation.sh` produces),
nothing more is required.

## 7. Run the V-JEPA 2 wrapper test suite

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/ -v
```

Expect 5 passed in ~80 s on H20:

- `test_imports`
- `test_factory_routes_to_vjepa2`
- `test_encoder_forward_shape`              (ViT-g random init, fp16)
- `test_predictor_step_shape`               (vit_ac_predictor, fp32)
- `test_libero_batch_through_vjepa_encoder` (skipped if LIBERO data missing)

## 8. LIBERO-Long baseline (Gate G-W1)

### 8a. Single-GPU, sequential (slow ~4 h)

```bash
# Terminal 1
bash examples/PlanAndVerify/eval_files/run_policy_server.sh
# Terminal 2 (after 'server running' appears)
bash examples/PlanAndVerify/eval_files/eval_libero_long.sh
```

### 8b. Multi-GPU (recommended, ~1 h on 8×H20)

```bash
# Background-friendly: launches 8 servers + 8 sharded clients
mkdir -p /tmp/pav-eval-multi
nohup bash examples/PlanAndVerify/eval_files/eval_libero_long_multi_gpu.sh \
    > /tmp/pav-eval-multi/driver.log 2>&1 &
disown
```

Configurable via env vars: `GPU_LIST`, `BASE_PORT`, `NUM_TRIALS`,
`TASK_SUITE`, `READY_TIMEOUT` (see top of the script).

Both modes write `rollout_<task>_episode<i>_<success|failure>.mp4` files
under the same `playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/results/libero_10/<folder>/`
directory.

Live progress:

```bash
RESULTS_DIR=playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/results/libero_10
N_S=$(find $RESULTS_DIR -name "*_success.mp4" | wc -l)
N_F=$(find $RESULTS_DIR -name "*_failure.mp4" | wc -l)
echo "$((N_S+N_F))/300 done; $N_S successful"
```

### 8c. Lock the result table

```bash
.venv/bin/python scripts/dump_baseline_table.py \
    --eval_results playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/results/libero_10/Qwen3-VL-PI-LIBERO-4in1_checkpoints_steps_100000_pytorch_model.pt \
    --out paper/tables/baseline_table.csv
```

Pass criterion: mean success rate ≥ 0.86. Reference number from the upstream
README: 88.4 %.

## 9. Cleanup helpers

```bash
# Kill all running PAV eval processes
pkill -f deployment/model_server/server_policy.py
pkill -f eval_libero_long_sharded.py
pkill -f eval_libero.py

# Clear partial results before re-running
rm -rf playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/results/
```

## 10. Known fragility

1. **flash-attn ABI mismatch**. If you upgrade torch later, flash-attn 2.7.4.post1
   will silently break with `undefined symbol: _ZN3c10...`. Re-pull a wheel
   that matches the new torch's `_GLIBCXX_USE_CXX11_ABI` flag.
2. **EGL on cluster**. If `MUJOCO_GL=egl` ever stops working (cluster driver
   change), the script falls back to `osmesa` automatically; no rebuild
   needed because `MUJOCO_GL` is read at runtime.
3. **HF rate limits / interrupted downloads**. `hf download` is idempotent —
   re-run on the same `--local-dir` to resume. Check file size against the
   "size" column on the HF web UI before trusting a download.
4. **LIBERO `~/.libero/config.yaml` per-user**. If you switch user, the
   interactive init prompt reappears. Re-run the `StringIO('N\n')` stanza
   from §4 once per user.
5. **`task.html`, `*.code-workspace`** — IDE artifacts from VS Code. They
   appear as untracked but should never be committed.
