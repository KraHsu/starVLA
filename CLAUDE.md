# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

StarVLA is a "Lego-like" research codebase for Vision-Language-Action (VLA) models. The design goal: each axis (VLM/world-model backbone, action head, dataset, trainer, deployment) is decoupled, so a new variant typically reduces to swapping one component while reusing the rest. Several published frameworks (FAST, OFT, π/flow-matching, GR00T-style dual-system) are re-implementations on this shared substrate.

## Common Commands

### Environment

```bash
conda create -n starVLA python=3.10 -y && conda activate starVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation        # must match local CUDA + torch
pip install -e .
```
Verified: `flash-attn==2.7.4.post1` works with nvcc 12.0/12.4. The `requirements.txt` block for Qwen3.5 (torch 2.6 / triton 3.2 / transformers 5.3) is intentionally commented out — uncomment only if you target the Qwen3.5 backbone.

### Lint / Format

```bash
make check         # black --check + ruff check (whole repo — currently has backlog, expected to fail)
make autoformat    # apply black + ruff --fix
```
Black/Ruff are configured for **line-length 121, target py3.10**. Because the repo has a historical lint backlog, when contributing only check the files your change touches:
```bash
FILES=$(git diff --name-only --diff-filter=ACMR origin/starVLA_dev | grep -E '\.py$')
black $FILES && python -m ruff check --fix $FILES
```

### Smoke-test a single module (no full training run)

Every framework file and dataloader file is runnable standalone — this is the primary debugging entry point:
```bash
python starVLA/model/framework/VLM4A/QwenOFT.py --config_yaml starvla_cotrain_oxe.yaml
python starVLA/model/framework/VLM4A/QwenGR00T.py        # forward pass on fake data
python starVLA/dataloader/lerobot_datasets.py --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

### Training

Always launched via `accelerate` + DeepSpeed (ZeRO-2 or ZeRO-3):
```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct
```
Three entry points by training paradigm — pick by file, not flag:
- `starVLA/training/train_starvla.py` — VLA SFT
- `starVLA/training/train_starvla_cotrain.py` — VLA + VLM multimodal co-training
- `starVLA/training/train_starvlm.py` — VLM-only

Wrapper scripts under `examples/<bench>/train_files/run_*.sh` set NCCL/cluster env, expose the editable variables in a labeled block at top, and `cp` themselves into the run dir for reproducibility.

### Evaluation (client/server, separate conda envs)

```bash
# Terminal 1 — starVLA env
bash examples/LIBERO/eval_files/run_policy_server.sh   # serves on port 6694

# Terminal 2 — benchmark env (e.g. `libero`)
bash examples/LIBERO/eval_files/eval_libero.sh
```
The simulator client lives in its own conda env (e.g. `libero`) because mujoco/sim deps conflict with the training stack. Communication is websocket via `deployment/model_server/server_policy.py`.

## High-Level Architecture

```
starVLA/
├── model/
│   ├── framework/             # Top-level VLA assemblies (single API surface per file)
│   │   ├── base_framework.py  # `baseframework(PreTrainedModel)` + `build_framework(cfg)` entry
│   │   ├── VLM4A/             # VLM-as-backbone: QwenOFT, QwenPI, QwenGR00T, QwenFast, ...
│   │   └── WM4A/              # World-model-as-backbone: Cosmos*, Wan* (see docs/WM4A.md)
│   └── modules/               # Reusable parts: vlm/, action_model/, dino_model/, projector/, world_model/
├── dataloader/                # LeRobot + VLM datasets; returns RAW model-agnostic dicts
├── training/                  # Train loops, trainer_utils (config tracker, freeze/lr helpers)
└── config/
    ├── training/              # Default training YAMLs
    └── deepseeds/             # accelerate/DeepSpeed launcher configs
examples/<benchmark>/          # All per-benchmark code: train_files/, eval_files/, data_registry/
deployment/model_server/       # Inference policy server (websocket)
```

### Framework registry (the routing core)

`build_framework(cfg)` in `starVLA/model/framework/base_framework.py` is the single entry point. It:
1. Walks `framework/` and `framework/VLM4A/`, `framework/WM4A/` and imports every module — registration happens as a side-effect of import.
2. Looks up `cfg.framework.name` in `FRAMEWORK_REGISTRY` (`starVLA/model/tools.py`).
3. Returns the registered class instantiated with `cfg`.

Every framework file ends with `@FRAMEWORK_REGISTRY.register("Name")` above the class. **To add a new framework:** drop a new file under `starVLA/model/framework/VLM4A/` (or `WM4A/`), register a unique name, and select it via `--framework.name <Name>`. No central wiring.

Currently registered: `QwenOFT`, `QwenFast`, `QwenPI`/`QwenFM`, `QwenPI_v3`, `QwenGR00T`, `QwenDual`, `QwenAdapter`, `CosmosGR00T`, `Gemma4PI`, `ABot_M0`, `LangForce`, `WanOFT`, `WanPI`, `WanGR00T`, `CosmoPredict2OFT`, `CosmoPredict2PI`, `CosmoPredict2GR00T`.

### Module boundaries (read this before "fixing" preprocessing logic)

- **Dataloaders return raw dicts only.** A sample is `{image: list[PIL.Image]|np.ndarray, lang: str, action: np.ndarray[T,action_dim], state?: np.ndarray}`. No tokenization, no image encoding inside the dataloader.
- `framework.forward(examples)` and `framework.predict_action(examples)` consume those raw dicts directly. All model-specific preprocessing lives inside the framework. **Do not push tokenizer/image-processor calls back into the dataloader.**

### Configuration system

A single global config object built from YAML via OmegaConf, then wrapped by `AccessTrackedConfig` (`training/trainer_utils/config_tracker.py`) — unread keys can be detected. Every value can be overridden via dotted CLI args:

```bash
--framework.qwenvl.base_vlm Qwen/Qwen2.5-VL-7B-Instruct
--framework.action_model.action_dim 7
--datasets.vla_data.per_device_batch_size 16
--trainer.freeze_modules "qwen_vl_interface.model.model.visual,dino_encoder"
```
Adding `--framework.action_model.foo bar` does **not** wire `foo` into anything; it only puts `foo` on the config object. The framework class is responsible for reading it.

Per-module learning rates use a name-prefixed dict (see `trainer_tools.build_param_lr_groups`):
```yaml
trainer:
  learning_rate:
    base: 1e-05
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
```
Freezing uses a comma-separated regex/name list (`TrainerUtils.freeze_backbones`). Resume reloads weights only — **optimizer state is intentionally not checkpointed**:
```yaml
trainer:
  pretrained_checkpoint: path/to/steps_10000.pt
  reload_modules: "action_model"   # empty = full load
```

### Datasets

LeRobot-format only. Data mixtures (which subsets, with which sampling weights, mapped to which `robot_type` config) are defined per-benchmark in `examples/<bench>/train_files/data_registry/data_config.py` as `DATASET_NAMED_MIXTURES`, then selected at the CLI by `--datasets.vla_data.data_mix <key>`. Each subset directory must contain `meta/modality.json` (the `data_preparation.sh` scripts copy it in).

`playground/Datasets/` and `playground/Pretrained_models/` are conventional symlink targets — keep large data outside the repo.

## Conventions This Repo Cares About

- **File-level isolation for new contributions.** New framework → new file under `framework/VLM4A` or `WM4A`. New benchmark → new directory under `examples/<bench>/`. New dataset registry → under that benchmark's `train_files/data_registry/`. **Do not modify shared modules** (`base_framework.py`, dataloader interface, training loop) unless the PR explicitly justifies it. See `docs/branching_strategy.md` § Community PR Guidelines.
- **Branch model:** PRs target `starVLA_dev` (active) — the stable branch is `starVLA`. Squash-merge. Conventional Commits (`feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`). Branch names use prefixes `feat/`, `fix/`, `docs/`, `refactor/`, `exp/`, `hotfix/`. Details in `docs/PR_readme.md`.
- **Any `**/bar/` directory is git-ignored** by design — put scratch scripts there (e.g. `examples/LIBERO/train_files/bar/my_train.sh`) without polluting the repo.
- **Framework PRs need validation materials** (benchmark numbers + public HF checkpoint + reproducible config) — this is an actual gate, not a soft ask.
- The framework file is the "paper figure" — its top-level structure should mirror the architecture diagram. Treat `framework/.../<Name>.py` as the public API surface for that variant.

## Pointers

- New-to-the-repo walkthrough: `docs/starVLA_guideline.md` (LIBERO end-to-end).
- Bring-your-own-dataset: `docs/integrate_your_dataset.md`, plus the agent skill at `docs/agent_skills/integrate-starvla-dataset/` (drives integration via Claude/Copilot).
- World-Model-for-Action backbones (Cosmos, Wan): `docs/WM4A.md`.
- FAQ (freeze flags, LR groups, smaller VLMs, resume): `docs/faq.md`.
- Model checkpoints: `docs/model_zoo.md`.
