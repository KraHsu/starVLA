# 毕设实验数据全集

> 这份文件只记录两类内容：
> 1. 已经真实完成并确认过的实验结果。
> 2. 后续还要跑的实验清单与当前支持状态。
>
> 不再保留早期草案里的虚拟数值、4090/Florence 假设结果或占位表格。

## 0. 当前结论

截至 `2026-05-05`，已经确认的核心结果如下：

| 实验 | 模型 | 套件 | 训练状态 | Final eval | 结论 |
|---|---|---|---|---|---|
| W1 对照锚点 | StarVLA-PI Qwen3-VL | `libero_10` | 非本次训练，使用已发布 ckpt | `0.9667` (300 ep) | 仅作早期对照锚点 |
| Stage 1 baseline | `QwenOFT` | `libero_goal` | 已完成 | `0.98` (98/100) | 当前主 baseline |
| Stage 3 V-JEPA | `QwenOFT_VJepa` | `libero_goal` | 已完成 | `0.97` (97/100) | 成功接入、无明显退化，但在饱和 setting 未见增益 |

当前最稳妥的论文表述是：

- `libero_goal + 100% data` 已接近饱和，Stage 3 与 baseline 基本持平；
- V-JEPA 的主收益不应再指望从这个 setting 里硬挤出来；
- 后续主战场应转到 `low-data` 或 `long-horizon`。

---

## 1. 已完成实验

### 1.1 W1 锚点：StarVLA-PI Qwen3-VL on LIBERO-Long

| 字段 | 值 |
|---|---|
| 模型 | `playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt` |
| 套件 | `libero_10` |
| 评测规模 | 10 tasks × 30 trials = 300 episodes |
| Suite mean SR | `0.9667` |
| 记录位置 | `paper/tables/baseline_table.csv` |
| 备注 | 这是已发布 ckpt 的 eval 锚点，不是我们自己训练的 OFT baseline |

### 1.2 Stage 1 baseline：QwenOFT on LIBERO-Goal

参考：[examples/PlanAndVerify/BASELINE.md](/root/workspace/starVLA/examples/PlanAndVerify/BASELINE.md:1)

| 字段 | 值 |
|---|---|
| Run id | `stage1_oft_libero_goal_20260504_185342` |
| Framework | `QwenOFT` |
| Base VLM | `Qwen3-VL-4B-Instruct` |
| 数据 | `libero_goal` |
| 训练步数 | 20k |
| Final ckpt | `playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/final_model/pytorch_model.pt` |
| Final train loss | `0.004181030672043562` |
| Final mse | `0.0011419133682336127` |
| Final eval | `98/100 = 0.98` |
| 结果目录 | `playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/results/libero_goal/stage1_oft_libero_goal_20260504_185342_checkpoints_steps_20000_pytorch_model.pt/` |

Per-task SR:

| Task | SR |
|---|---:|
| open the middle drawer of the cabinet | 1.0 |
| open the top drawer and put the bowl inside | 1.0 |
| push the plate to the front of the stove | 1.0 |
| put the bowl on the plate | 1.0 |
| put the bowl on the stove | 0.9 |
| put the bowl on top of the cabinet | 1.0 |
| put the cream cheese in the bowl | 1.0 |
| put the wine bottle on the rack | 0.9 |
| put the wine bottle on top of the cabinet | 1.0 |
| turn on the stove | 1.0 |

仅有 2 个 failure:

- `rollout_put_the_bowl_on_the_stove_episode2_failure.mp4`
- `rollout_put_the_wine_bottle_on_the_rack_episode6_failure.mp4`

### 1.3 Stage 3：QwenOFT_VJepa on LIBERO-Goal

| 字段 | 值 |
|---|---|
| Run id | `stage3_oft_vjepa_libero_goal_20260505_042345` |
| Framework | `QwenOFT_VJepa` |
| Base VLM | `Qwen3-VL-4B-Instruct` |
| V-JEPA cache | `playground/cache/vjepa/vjepa2_1_vit_b_384/libero_goal` |
| 训练步数 | 20k |
| Final ckpt | `playground/Checkpoints/stage3_oft_vjepa_libero_goal_20260505_042345/final_model/pytorch_model.pt` |
| Intermediate ckpts | `checkpoints/steps_{5000,10000,15000,20000}_pytorch_model.pt` |
| Final train loss | `0.0041139610` |
| Final mse | `0.0011409129` |
| Smoke eval | `10/10` success (`NUM_TRIALS=1`) |
| Final eval | `97/100 = 0.97` |
| 结果目录 | `playground/Checkpoints/stage3_oft_vjepa_libero_goal_20260505_042345/results/libero_goal/stage3_oft_vjepa_libero_goal_20260505_042345_final_model_pytorch_model.pt/` |

Per-task SR:

| Task | SR |
|---|---:|
| open the middle drawer of the cabinet | 1.0 |
| open the top drawer and put the bowl inside | 1.0 |
| push the plate to the front of the stove | 0.9 |
| put the bowl on the plate | 1.0 |
| put the bowl on the stove | 1.0 |
| put the bowl on top of the cabinet | 1.0 |
| put the cream cheese in the bowl | 0.9 |
| put the wine bottle on the rack | 0.9 |
| put the wine bottle on top of the cabinet | 1.0 |
| turn on the stove | 1.0 |

3 个 failure 视频：

- `rollout_push_the_plate_to_the_front_of_the_stove_episode7_failure.mp4`
- `rollout_put_the_cream_cheese_in_the_bowl_episode1_failure.mp4`
- `rollout_put_the_wine_bottle_on_the_rack_episode6_failure.mp4`

### 1.4 当前可引用对比

| 对比 | 数值 |
|---|---:|
| Stage 3 - Stage 1 (`libero_goal`, 100 ep) | `0.97 - 0.98 = -0.01` |
| 解释 | 在饱和 setting 上基本打平，不能宣称有提升，也不能认定明显退化 |

---

## 2. 当前真实环境与脚本

### 2.1 训练环境

| 项 | 值 |
|---|---|
| 主训练 Python | `.venv/bin/python` |
| 仿真 Python | `.venv-libero/bin/python` |
| 训练入口 | `starVLA/training/train_starvla.py` |
| Baseline launcher | `examples/PlanAndVerify/train_files/run_oft_libero_goal.sh` |
| V-JEPA launcher | `examples/PlanAndVerify/train_files/run_oft_vjepa_libero_goal.sh` |
| Goal eval launcher | `examples/PlanAndVerify/eval_files/eval_libero_goal_sharded.sh` |
| Multi-suite eval launcher | `examples/PlanAndVerify/eval_files/eval_libero_long_multi_gpu.sh` |

### 2.2 当前脚本已支持的覆盖项

`run_oft_libero_goal.sh`:

- `RUN_ID`
- `MAX_TRAIN_STEPS`
- `PER_DEVICE_BATCH`
- `WANDB_ENTITY`
- `WANDB_MODE`
- `DEBUG_STEPS`
- `FREEZE_MODULES`
- `BASE_VLM`
- `CONFIG_YAML`
- `LIBERO_DATA_ROOT`
- `DATA_MIX`
- `RUN_ROOT_DIR`
- `NUM_PROCESSES`
- `SEED`

`run_oft_vjepa_libero_goal.sh`:

- 上述大部分项
- 额外支持 `VJEPA_CACHE_DIR`
- `VJEPA_FUSION`
- `VJEPA_ENCODER_CKPT`

---

## 3. 原始计划与当前支持状态

### 3.1 原始“大满贯”计划

原始计划的目标是三层：

1. 主实验：3 methods × 3 data fractions × 3 seeds = **27 个训练 run**
2. 消融：若干单 seed run
3. baseline 锚点：**1 个训练 run**

旧版草案里写成：

- `27` 个主实验 run
- `18` 个消融 run
- `1` 个 baseline 锚点
- 合计 **46 个训练 run**

但这里有一个必须说清楚的问题：

- `implementation_todo.md` 现有 Stage 5 条目能明确数出来的是 `16` 个单-seed 消融变体：
  - `K ∈ {1,4,8}`: 3
  - `fusion ∈ {concat, film, cross_attn}`: 3
  - `V-JEPA size ∈ {B, L}`: 2
  - `history ∈ {4,8,16}`: 3
  - `freeze ∈ {full freeze, projector+LoRA}`: 2
  - `upstream rep ∈ {V-JEPA, DINOv2, CLIP}`: 3
- 所以 `18` 这个数字来自早期草案，不适合再拿来做精确排期。

### 3.2 当前仓库真正“能直接跑”的实验

| 类别 | 是否已具备脚本支持 | 备注 |
|---|---|---|
| A. `QwenOFT` full-data `libero_goal` | 是 | 已完成 |
| C. `QwenOFT_VJepa` full-data `libero_goal` | 是 | 已完成 |
| A. `QwenOFT` full-data `libero_10` | 是 | 用 `DATA_MIX=libero_10` 即可 |
| C. `QwenOFT_VJepa` full-data `libero_10` | 基本是 | 需要先准备 `libero_10` 的 V-JEPA cache |
| A/C 多 seed | 是 | 现在可用 `SEED=` |
| A/C 多 GPU 数 | 是 | 现在可用 `NUM_PROCESSES=` |
| low-data 10% / 25% | 否（未自动化） | 当前仓库没有子集 materialization / split-aware loader |
| B. naive temporal baseline | 否 | 当前 repo 没有对应 framework/launcher |
| DINOv2 / CLIP upstream replacement | 否 | 当前 repo 没有相应接入 |
| projector + LoRA freeze ablation | 否 | 需要补 LoRA 训练路径 |

结论：**当前最现实的下一轮实验，应只围绕 A/C 两组、full-data 或手工准备的 low-data 子集展开。**

---

## 4. 建议补实验方案

### 4.1 最小可写论文方案（推荐）

总计 **8 个训练 run**：

| 组别 | 套件 | 数据量 | 方法 | seeds | 训练 run 数 |
|---|---|---:|---|---|---:|
| G25-A | `libero_goal` | 25% | baseline (`QwenOFT`) | `17, 42` | 2 |
| G25-C | `libero_goal` | 25% | V-JEPA (`QwenOFT_VJepa`) | `17, 42` | 2 |
| L100-A | `libero_10` | 100% | baseline (`QwenOFT`) | `17, 42` | 2 |
| L100-C | `libero_10` | 100% | V-JEPA (`QwenOFT_VJepa`) | `17, 42` | 2 |

用途：

- `libero_goal 25%`：找样本效率收益
- `libero_10 100%`：找长时程收益

### 4.2 更稳的标准方案

总计 **12 个训练 run**：

| 组别 | 套件 | 数据量 | 方法 | seeds | 训练 run 数 |
|---|---|---:|---|---|---:|
| G25-A | `libero_goal` | 25% | baseline | `17, 42, 1337` | 3 |
| G25-C | `libero_goal` | 25% | V-JEPA | `17, 42, 1337` | 3 |
| L100-A | `libero_10` | 100% | baseline | `17, 42, 1337` | 3 |
| L100-C | `libero_10` | 100% | V-JEPA | `17, 42, 1337` | 3 |

### 4.3 现在不建议优先做的

- 再在 `libero_goal 100%` 上反复刷分
- 先做 naive temporal baseline
- 先做 DINOv2 / CLIP 表征替换
- 先做大而全 27-run 主网格

---

## 5. 后续实验前置条件

### 5.1 `libero_10` full-data

baseline 不需要额外准备，只需要数据集已在 `playground/Datasets/LEROBOT_LIBERO_DATA/`。

V-JEPA 需要先准备 cache，例如：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/PlanAndVerify/cache_files/extract_vjepa_cache.py \
  --config_yaml examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml \
  --dataset_name libero_10 \
  --camera_key video.primary_image \
  --vjepa_ckpt /path/to/vjepa2_1_vitb_dist_vitG_384.pt \
  --output_dir playground/cache/vjepa/vjepa2_1_vit_b_384/libero_10 \
  --num_history_frames 8 \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda:0 \
  --dtype bf16
```

### 5.2 `libero_goal` 25% / 10%

当前仓库**没有**自动化的 subset pipeline。要跑这类实验，必须先二选一：

1. 物化一个子集数据根目录，例如：
   `playground/Datasets/LEROBOT_LIBERO_DATA_SUBSETS/libero_goal_25_seed2024/`
   且目录内部仍保持 `libero_goal_no_noops_1.0.0_lerobot/` 这种 LeRobot 原始结构；
2. 或新增 split-aware dataloader，让训练时按 split JSON 过滤 episode。

在这两条路没有做完之前，低数据实验只能算“计划中”，不能直接按命令跑。

---

## 6. 评测统一规范

### 6.1 `libero_goal`

- 每 task `10` trials 作为正式结果
- 每 task `1` trial 只作为 smoke
- 总 episode:
  - smoke: `10`
  - final: `100`

### 6.2 `libero_10`

- 第一轮也用每 task `10` trials
- 如果结果有潜力，再补 `30` trials 形成更稳的最终数

### 6.3 结果记录要求

每个 run 至少记录：

- run id
- config/yaml
- final ckpt 路径
- final train loss
- final mse
- suite mean SR
- per-task SR
- failure mp4 路径

---

## 7. 引用规则

论文正文里当前可以安全引用的数值只有：

- W1 `libero_10` 锚点：`0.9667`
- Stage 1 baseline：`0.98`
- Stage 3 V-JEPA：`0.97`

其余任何 25% / 10% / `libero_10` OFT 训练结果，在没真正跑出来之前都不能写进正文，只能写成实验计划。
