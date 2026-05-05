# Stage 1 baseline — StarVLA-OFT on LIBERO-Goal

> 用途：作为后续 V-JEPA 主实验（implementation_todo.md §4）所有 ΔSR 对照的零点。
> 与 W1 PAV PI-head baseline（`paper/tables/baseline_table.csv`，libero_10 SR=0.9667）互补：相同 backbone（Qwen3-VL-4B），不同 head（OFT vs PI），不同套件（libero_goal vs libero_10），自己 fine-tune 的 ckpt（不是已发布 ckpt）。
>
> 状态：**training / eval / latency all completed**。阶段 1 baseline 已完成回填，可直接作为后续 V-JEPA 主实验的零点。

## 1. 环境固定

| 字段 | 值 |
|---|---|
| Repo commit | `0fbbd1a` (branch `simple-by`) |
| Hardware | 8× NVIDIA H20 (97 GB/卡, single node) |
| Primary venv | `.venv` (uv, py3.11) |
| Eval venv | `.venv-libero` (uv, py3.10, MUJOCO_GL=osmesa) |
| torch | 2.6.0+cu124 |
| transformers | 4.57.0 |
| accelerate | 1.5.2 |
| deepspeed | 0.16.9 |
| flash-attn | 2.7.4.post1 |
| VLM backbone | `playground/Pretrained_models/Qwen3-VL-4B-Instruct` (4.5 B params total: 4.5 B trainable) |

## 2. 训练 setup

| 字段 | 值 |
|---|---|
| Framework | `QwenOFT` (`starVLA/model/framework/VLM4A/QwenOFT.py`) |
| Action head | `L1RegressionActionHead` (MLPResNet, hidden 5120, action_dim=7, action_horizon=8) |
| Data mix | `libero_goal` (10 tasks, 52042 transitions, all 10 task descriptions confirmed by `check_libero.py`) |
| Per-device batch | 16 |
| Effective batch | 128 (8 GPUs × 16 × grad_accum=1; ds_config.yaml hard-codes grad_accum=1, see launcher注释) |
| Max steps | 20000 |
| Warmup steps | 1000 (5 % of max) |
| LR (base / qwen_vl_interface / action_model) | 2.5e-5 / 1.0e-5 / 1.0e-4 |
| LR scheduler | cosine_with_min_lr (min 1.0e-6) |
| Freeze | none (full-parameter training) |
| Optimizer | AdamW betas=(0.9, 0.95), wd=1e-8 |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Save interval | 5000 (steps 5k/10k/15k/20k + final) |
| Eval interval | 100 (in-distribution MSE only — see "deviations" below) |

Launcher: `examples/PlanAndVerify/train_files/run_oft_libero_goal.sh`
Yaml: `examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml`

## 3. 训练前 profile（profile_oft_libero_goal.py，20 步、5 步 warmup）

| 指标 | 值 |
|---|---|
| Mean step time | 1265 ms |
| p50 step time | 1263 ms |
| Max step time | 1304 ms |
| Mean data wait | 10.3 ms |
| Steps/s (post-warmup) | 0.79 |
| Samples/s | 101 |
| Peak allocated GPU memory | **44.4 GB / rank** |
| Peak reserved GPU memory | 56.6 GB / rank |

源数据：`playground/Checkpoints/profile_oft_libero_goal_<ts>/profile.json`

20k 步预估墙钟：20000 / 0.79 ≈ **7.0 h**（不含 ckpt 落盘 IO 与 in-distribution eval 时间）

## 4. 训练运行

- run_id : `stage1_oft_libero_goal_20260504_185342`
- Output dir : `playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/`
- 4 个 ckpt 路径 : `stage1_oft_libero_goal_20260504_185342/checkpoints/steps_{5000,10000,15000,20000}_pytorch_model.pt`
- Final ckpt : `playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/checkpoints/steps_20000_pytorch_model.pt`
- Final exported model : `playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/final_model/pytorch_model.pt`
- Final loss (action_dit_loss) : `0.004181030672043562`
- Final in-distribution mse_score : `0.0011419133682336127`
- Final logged epoch : `49.14`
- Training complete timestamp : `2026-05-05 10:09:57 UTC`（见 `train.log` 尾部）

## 5. Sim 评测（libero_goal × 10 task × 10 episode = 100 episode）

| 任务 | SR (10 ep) |
|---|---|
| open the middle drawer of the cabinet | 1.0 (10/10) |
| open the top drawer and put the bowl inside | 1.0 (10/10) |
| push the plate to the front of the stove | 1.0 (10/10) |
| put the bowl on the plate | 1.0 (10/10) |
| put the bowl on the stove | 0.9 (9/10) |
| put the bowl on top of the cabinet | 1.0 (10/10) |
| put the cream cheese in the bowl | 1.0 (10/10) |
| put the wine bottle on the rack | 0.9 (9/10) |
| put the wine bottle on top of the cabinet | 1.0 (10/10) |
| turn on the stove | 1.0 (10/10) |
| **Suite mean SR** | **0.98 (98/100)** |

CSV：`paper/tables/baseline_oft_libero_goal.csv`（schema 与 W1 baseline_table.csv 一致：model, task_id, trial_id, success, mp4_path）
Rollout dir：`playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/results/libero_goal/stage1_oft_libero_goal_20260504_185342_checkpoints_steps_20000_pytorch_model.pt/`

仅有 2 个 failure episode：
- `put_the_bowl_on_the_stove` episode 2
- `put_the_wine_bottle_on_the_rack` episode 6

启动：
```bash
CKPT=$(pwd)/playground/Checkpoints/<run_id>/checkpoints/steps_20000_pytorch_model.pt \
    bash examples/PlanAndVerify/eval_files/eval_libero_goal_sharded.sh
```
聚合：
```bash
.venv/bin/python scripts/dump_baseline_table.py \
    --eval_results <video_out_path> \
    --out paper/tables/baseline_oft_libero_goal.csv \
    --label StarVLA-OFT-Qwen3VL
```

## 6. 推理延迟（benchmark_latency.py，单卡 H20 b=1）

| 指标 | 值 |
|---|---|
| Mean ms / action chunk | 88.92 |
| p50 ms | 76.54 |
| p95 ms | 184.40 |
| Min / Max | 69.87 / 207.41 |

源数据：`playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/checkpoints/latency.json`

启动：
```bash
.venv/bin/python examples/PlanAndVerify/eval_files/benchmark_latency.py \
    --ckpt_path <stage1 final ckpt> --warmup 5 --iters 50 --use_bf16
```

## 7. 与 implementation_todo.md §1 的偏离

1. **Backbone**：todo §1.2 字面写 Florence-2（4090 资源友好），实际用 Qwen3-VL-4B（H20 显存充裕，且本仓现成 framework + ckpt）。已在用户确认下选择。
2. **Preview sim eval**：todo §1.2 要求"每 1k 步、每任务 2 ep 的 sim eval"未实施。理由：当前 `train_starvla.py` 的 eval-interval hook 只跑 in-distribution MSE（`train_starvla.py:319-351`），加 sim eval 需在训练循环里启停 policy server / mujoco env，违反"不改主干"原则。替代：训练中只看 MSE 收敛；sim eval 只在 final ckpt 上做一次（必要时在 5k/10k/15k 中间 ckpt 各跑一轮做曲线）。
3. **Effective batch size**：todo 未明确给值；实际是 128（受 `starVLA/config/deepseeds/ds_config.yaml` 的 `gradient_accumulation_steps: 1` 限制）。在 52042 transitions × 20k 步下约 49 epochs，对 OFT 单套件足够。
4. **Path conventions**：todo §1.4 的 `docs/results/baseline.md` 与 repo 根 `BASELINE.md` 路径未采用，改用 PAV 既有约定（本文件 + `paper/tables/baseline_oft_libero_goal.csv`），与 W1 一致（commits `4ca2c33` / `867c6e2` / `0fbbd1a`）。
