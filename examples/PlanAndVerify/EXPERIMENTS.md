# Experiments

这份文档回答三个问题：

1. 原始计划一共有多少实验。
2. 现阶段真正建议跑哪些实验。
3. 每类实验怎么训练、怎么评测、怎么记结果。

## 1. 总览

### 1.1 原始计划

原始大满贯计划分三块：

| 模块 | 公式 | 训练 run 数 |
|---|---|---:|
| baseline 锚点 | Stage 1 baseline | 1 |
| 主实验网格 | 3 methods × 3 data fractions × 3 seeds | 27 |
| 消融 | 单 seed 若干变体 | 16–18 |

说明：

- 早期草案写成 `46` 个训练 run，即 `1 + 27 + 18`。
- 按当前 `implementation_todo.md` 能明确枚举出的消融，是 `16` 个而不是精确 `18` 个。
- 所以**原始计划规模可以理解为 44–46 个训练 run**。

### 1.2 现在建议执行的方案

建议分两层：

| 方案 | 内容 | 训练 run 数 | 用途 |
|---|---|---:|---|
| 最小方案 | `libero_goal 25%` A/C + `libero_10 100%` A/C，各 2 seeds | 8 | 足够写主线 |
| 标准方案 | 同上，各 3 seeds | 12 | 更稳，适合正文主表 |

这里的 A/C 指：

- A = `QwenOFT` baseline
- C = `QwenOFT_VJepa`

不建议现在优先去做 B（naive temporal），因为当前 repo 里没有现成实现。

## 2. 当前仓库支持矩阵

| 实验 | 当前是否能直接跑 | 说明 |
|---|---|---|
| A @ `libero_goal 100%` | 是 | 已完成 |
| C @ `libero_goal 100%` | 是 | 已完成 |
| A @ `libero_10 100%` | 是 | 直接改 `DATA_MIX=libero_10` |
| C @ `libero_10 100%` | 是 | 先准备 `libero_10` cache |
| A/C 多 seed | 是 | 现在可用 `SEED=` |
| A/C 多进程数 | 是 | 现在可用 `NUM_PROCESSES=` |
| `libero_goal 25% / 10%` | 否（未自动化） | 需要先准备子集数据根目录或 split-aware loader |
| B naive temporal | 否 | 无 framework / launcher |
| DINOv2 / CLIP ablation | 否 | 无接入 |
| projector + LoRA | 否 | 无训练路径 |

## 3. 已完成实验

### 3.1 Stage 1 baseline

训练：

```bash
WANDB_MODE=disabled \
RUN_ID=stage1_oft_libero_goal_20260504_185342 \
bash examples/PlanAndVerify/train_files/run_oft_libero_goal.sh
```

正式评测：

```bash
CKPT=$(pwd)/playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/checkpoints/steps_20000_pytorch_model.pt \
NUM_TRIALS=10 \
GPU_LIST="0 1 2 3 4 5 6 7" \
bash examples/PlanAndVerify/eval_files/eval_libero_goal_sharded.sh
```

结果：

- suite mean SR = `0.98`
- 结果详见 [BASELINE.md](/root/workspace/starVLA/examples/PlanAndVerify/BASELINE.md:1)

### 3.2 Stage 3 V-JEPA

训练：

```bash
WANDB_MODE=disabled \
RUN_ID=stage3_oft_vjepa_libero_goal_20260505_042345 \
bash examples/PlanAndVerify/train_files/run_oft_vjepa_libero_goal.sh
```

正式评测：

```bash
CKPT=$(pwd)/playground/Checkpoints/stage3_oft_vjepa_libero_goal_20260505_042345/final_model/pytorch_model.pt \
TASK_SUITE=libero_goal \
NUM_TRIALS=10 \
GPU_LIST="0 1 2 3 4 5 6 7" \
bash examples/PlanAndVerify/eval_files/eval_libero_long_multi_gpu.sh
```

结果：

- suite mean SR = `0.97`
- 结果目录：
  `playground/Checkpoints/stage3_oft_vjepa_libero_goal_20260505_042345/results/libero_goal/stage3_oft_vjepa_libero_goal_20260505_042345_final_model_pytorch_model.pt/`

## 4. 推荐补实验

### 4.1 最小方案：8 runs

| ID | Suite | Data | Method | Seeds |
|---|---|---:|---|---|
| G25-A | `libero_goal` | 25% | `QwenOFT` | `17,42` |
| G25-C | `libero_goal` | 25% | `QwenOFT_VJepa` | `17,42` |
| L100-A | `libero_10` | 100% | `QwenOFT` | `17,42` |
| L100-C | `libero_10` | 100% | `QwenOFT_VJepa` | `17,42` |

### 4.2 标准方案：12 runs

| ID | Suite | Data | Method | Seeds |
|---|---|---:|---|---|
| G25-A | `libero_goal` | 25% | `QwenOFT` | `17,42,1337` |
| G25-C | `libero_goal` | 25% | `QwenOFT_VJepa` | `17,42,1337` |
| L100-A | `libero_10` | 100% | `QwenOFT` | `17,42,1337` |
| L100-C | `libero_10` | 100% | `QwenOFT_VJepa` | `17,42,1337` |

## 5. 训练方法

### 5.1 Baseline：`QwenOFT`

#### `libero_goal 100%`

```bash
SEED=42 \
RUN_ID=stage1_oft_libero_goal_seed42 \
DATA_MIX=libero_goal \
WANDB_MODE=disabled \
bash examples/PlanAndVerify/train_files/run_oft_libero_goal.sh
```

#### `libero_10 100%`

```bash
SEED=42 \
RUN_ID=stage1_oft_libero_10_seed42 \
DATA_MIX=libero_10 \
WANDB_MODE=disabled \
bash examples/PlanAndVerify/train_files/run_oft_libero_goal.sh
```

#### `libero_goal 25%`

前提：你已经准备好了一个只包含 25% episode 的数据根目录，并且内部仍保持 LeRobot 原始目录结构。

```bash
SEED=42 \
RUN_ID=stage1_oft_libero_goal25_seed42 \
LIBERO_DATA_ROOT=/abs/path/to/LEROBOT_LIBERO_DATA_25 \
DATA_MIX=libero_goal \
WANDB_MODE=disabled \
bash examples/PlanAndVerify/train_files/run_oft_libero_goal.sh
```

### 5.2 V-JEPA：`QwenOFT_VJepa`

#### `libero_goal 100%`

```bash
SEED=42 \
RUN_ID=stage3_oft_vjepa_libero_goal_seed42 \
DATA_MIX=libero_goal \
VJEPA_CACHE_DIR=playground/cache/vjepa/vjepa2_1_vit_b_384/libero_goal \
WANDB_MODE=disabled \
bash examples/PlanAndVerify/train_files/run_oft_vjepa_libero_goal.sh
```

#### `libero_10 100%`

前提：`playground/cache/vjepa/vjepa2_1_vit_b_384/libero_10/` 已准备好。

```bash
SEED=42 \
RUN_ID=stage3_oft_vjepa_libero_10_seed42 \
DATA_MIX=libero_10 \
VJEPA_CACHE_DIR=playground/cache/vjepa/vjepa2_1_vit_b_384/libero_10 \
WANDB_MODE=disabled \
bash examples/PlanAndVerify/train_files/run_oft_vjepa_libero_goal.sh
```

#### `libero_goal 25%`

前提：

- 25% 子集数据根目录已准备好
- 对应 25% 数据也准备好了独立的 V-JEPA cache

```bash
SEED=42 \
RUN_ID=stage3_oft_vjepa_libero_goal25_seed42 \
LIBERO_DATA_ROOT=/abs/path/to/LEROBOT_LIBERO_DATA_25 \
DATA_MIX=libero_goal \
VJEPA_CACHE_DIR=/abs/path/to/vjepa_cache_libero_goal_25 \
WANDB_MODE=disabled \
bash examples/PlanAndVerify/train_files/run_oft_vjepa_libero_goal.sh
```

## 6. 评测方法

### 6.1 `libero_goal`

Smoke:

```bash
CKPT=/abs/path/to/pytorch_model.pt \
NUM_TRIALS=1 \
GPU_LIST="0" \
bash examples/PlanAndVerify/eval_files/eval_libero_goal_sharded.sh
```

正式：

```bash
CKPT=/abs/path/to/pytorch_model.pt \
NUM_TRIALS=10 \
GPU_LIST="0 1 2 3 4 5 6 7" \
bash examples/PlanAndVerify/eval_files/eval_libero_goal_sharded.sh
```

### 6.2 `libero_10`

Smoke:

```bash
CKPT=/abs/path/to/pytorch_model.pt \
TASK_SUITE=libero_10 \
NUM_TRIALS=1 \
GPU_LIST="0" \
bash examples/PlanAndVerify/eval_files/eval_libero_long_multi_gpu.sh
```

正式：

```bash
CKPT=/abs/path/to/pytorch_model.pt \
TASK_SUITE=libero_10 \
NUM_TRIALS=10 \
GPU_LIST="0 1 2 3 4 5 6 7" \
bash examples/PlanAndVerify/eval_files/eval_libero_long_multi_gpu.sh
```

如果某组结果很关键，再补：

```bash
NUM_TRIALS=30
```

### 6.3 结果统计

总数：

```bash
find "$RESULT_DIR" -name '*.mp4' | wc -l
find "$RESULT_DIR" -name '*_success.mp4' | wc -l
find "$RESULT_DIR" -name '*_failure.mp4' | wc -l
```

失败样例：

```bash
find "$RESULT_DIR" -name '*_failure.mp4' | sort
```

## 7. 每个实验必须记录什么

每个 run 建议至少记录下面这些字段到表格：

| 字段 | 说明 |
|---|---|
| run_id | 唯一标识 |
| method | `QwenOFT` / `QwenOFT_VJepa` |
| suite | `libero_goal` / `libero_10` |
| data_fraction | `100%` / `25%` / `10%` |
| seed | 训练 seed |
| ckpt_path | final model 或 step ckpt |
| train_loss | 最终 `action_dit_loss` |
| mse_score | 最终训练 MSE |
| suite_mean_sr | 最终 success rate |
| per_task_sr | 每个 task 的 success rate |
| result_dir | rollout 目录 |
| failure_videos | 失败 mp4 列表 |

## 8. 当前推荐执行顺序

1. 先把 `libero_10` 的 V-JEPA cache 准备好。
2. 跑 `L100-A` 和 `L100-C` 的 seed 42。
3. 如果 `libero_10` 已经出现 A/C 差距，再补 seed 17 和 1337。
4. 之后再决定是否投入精力做 `libero_goal 25%` 的子集物化。

原因很简单：

- `libero_goal 100%` 已经饱和；
- `libero_10` 更可能给你带来可写的增益；
- `25% data` 有价值，但前提是先把低数据子集数据准备流程补齐。
