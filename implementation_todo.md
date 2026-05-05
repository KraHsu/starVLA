# 毕设 TODO：starVLA fork × V-JEPA 融合

> 工程载体：fork 自 `starVLA/starVLA`（稳定分支 `starVLA`），当前工作分支 `simple-by`。
> V-JEPA 等通过 git submodule 引入 `third_party/`，**只新增文件、不改动 starVLA 主干**。
> 环境：`uv` 管理双 venv（`.venv` py3.11 主栈 / `.venv-libero` py3.10 LIBERO 沙箱），不再使用 conda。
> 进度对照：[examples/PlanAndVerify/SETUP.md](./examples/PlanAndVerify/SETUP.md)（环境复现）与 [paper/tables/baseline_table.csv](./paper/tables/baseline_table.csv)（额外 baseline 锚点）。
> 完成下面 7 个阶段就构成一份完整的本科毕设故事。
> [Overview](./overview.md)

---

## 阶段 0：仓库与环境（1–2 天）

**目标**：拿到一份可复现的 fork，明确 V-JEPA 走 submodule，环境双隔离。

### 0.1 Fork 与分支
- [x] Fork `starVLA/starVLA` → `KraHsu/starVLA`
- [x] 在 fork 上基于 `starVLA`（稳定分支）建分支 `simple-by`（当前工作分支）
- [ ] 在 README 顶部贴清楚：基于 starVLA commit hash、本仓库目标一句话（README 暂为 upstream 原文，需补 fork 信息）

### 0.2 添加 submodule
- [x] `git submodule add https://github.com/facebookresearch/vjepa2 third_party/vjepa2`（必选；实际路径已落 `third_party/`）
- [ ] `git submodule add https://github.com/facebookresearch/jepa-wms third_party/jepa-wms`（仅当走方案三 reranker 时再加，可后置）
- [x] 提交并验证 `git clone --recursive` 可一键拉全（W1 已沿用）

### 0.3 新建目录骨架（全部新增，不改主干）
- [x] `starVLA/model/modules/world_model/vjepa2.py`（W1 产物：`VJEPA2Encoder` + `VJEPA2ACPredictor`，已通过 `world_model/__init__.py` 的 `get_world_model` 工厂注册 `"vjepa2"` 分支 — 替代了原计划的 `model/encoder/vjepa_encoder.py`）
- [ ] `starVLA/model/modules/projector/vjepa_projector.py`（projector + gating + token 重塑；落在现有 `projector/` 子目录下与 `QFormer.py` 同级）
- [ ] `starVLA/model/framework/VLM4A/QwenOFT_VJepa.py`（复制 `QwenOFT.py` 后改 forward / predict_action）
- [ ] `starVLA/config/training/starvla_libero_vjepa.yaml`（在原 LIBERO config 上加 `vjepa_*` 字段）
- [ ] `examples/PlanAndVerify/cache_files/extract_vjepa_cache.py`（特征离线抽取；与 `examples/PlanAndVerify/eval_files/` 同级，沿用 PAV 子目录约定，追踪进 git）
- [ ] `examples/PlanAndVerify/train_files/run_vjepa_libero.sh`（实验脚本；同上）

### 0.4 双 uv 环境（项目已统一切到 uv，不再使用 conda）
- [x] `.venv`：`uv venv --python 3.11 .venv`，starVLA + V-JEPA 2 + flash-attn 主栈
- [x] `.venv-libero`：`uv venv --python 3.10 .venv-libero`，仅装 LIBERO + mujoco + robosuite（与主 venv 不可合并：LIBERO pin `numpy==1.24.4` + `transformers==4.21.1`，与 starVLA `transformers==4.57.0` 冲突）
- [x] 两个 venv 均已在 `.gitignore` 内
- [x] 验证 `.venv/bin/python` 可跑 starVLA framework smoke test（按 `examples/PlanAndVerify/SETUP.md` 步骤）
- [x] 关键版本号已写进 `examples/PlanAndVerify/SETUP.md`

### 0.5 数据盘约定
- [x] `playground/Datasets/` 放原 LeRobot 数据（实际：`playground/Datasets/LEROBOT_LIBERO_DATA/` 已缓存 4 套件 LIBERO LeRobot 数据）
- [ ] `playground/cache/vjepa/<ckpt_name>/<suite>/` 放 V-JEPA 特征（与 `playground/Datasets`、`playground/Pretrained_models` 风格统一，自动 git-ignore）
- [ ] `runs/<run_id>/` 放 tf 离线日志 + checkpoint

### 0.6 阶段 0 交付物
- [ ] fork 链接 + submodule 链接整理在 README（README 暂为 upstream 原文）
- [x] `examples/PlanAndVerify/SETUP.md` 完成（替代原计划的 `docs/SETUP.md`）
- [x] 一次成功的 baseline smoke test 日志（PAV W1：StarVLA-PI on LIBERO-Long 全套通过，见 §1.5）

---

## 阶段 1：跑通 starVLA LIBERO baseline（3–5 天）

**目标**：复现"没用 V-JEPA"的成绩作为零点。这一阶段**不做任何方法创新**，纯工程。

### 1.1 数据
- [x] LIBERO LeRobot 数据：`playground/Datasets/LEROBOT_LIBERO_DATA/`（4 套件本地全部就位，§0.5 已记录；本阶段实际仅用 `libero_goal_no_noops_1.0.0_lerobot`）
- [x] 数据 sanity 脚本：`examples/PlanAndVerify/cache_files/check_libero.py`（替代 todo 原计划的 `tools/bar/check_libero.py`，因为 `**/bar/` 在 .gitignore；实测 52042 transitions、10 unique task descriptions 全部可达，commit `0fbbd1a`）

### 1.2 跑通 baseline
- [x] **Backbone 替换**：todo 原写 "Florence-2 backbone（4090 资源友好首选）"，实际改为 **`Qwen3-VL-4B-Instruct`**——本机是 8×H20（97 GB/卡）不是 4090，且 Florence-2 framework 在本仓不存在，Qwen3-VL-4B 是 starVLA 官方 LIBERO 例子默认 backbone。
- [x] 训练 config：`examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml`（QwenOFT + libero_goal + 20k 步 + 8 卡 ZeRO-2，effective batch 128）
- [x] 训练 launcher：`examples/PlanAndVerify/train_files/run_oft_libero_goal.sh`（10-step DEBUG_STEPS dry-run 已通过，commit `0fbbd1a`）
- [x] 实际 20k 步训练运行（阶段 E）：`playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/`，8×H20 墙钟约 7.25 h；final ckpt `checkpoints/steps_20000_pytorch_model.pt`，final `action_dit_loss=0.004181030672043562`，`mse_score=0.0011419133682336127`（详见 `examples/PlanAndVerify/BASELINE.md` §4）
- [ ] ~~Preview eval 每 1k 步、每任务 2 episode~~ —— **不做**。原因：现有 `train_starvla.py:319-351` 的 eval-interval hook 只跑 in-distribution MSE（不是 sim SR），加 sim 评测需要在训练循环里启停 policy server/mujoco env，违反"不改主干"原则。替代方案：训练中看 MSE 收敛；sim eval 只在 final ckpt 上做一次（必要时再补 5k/10k/15k 中间 ckpt 做曲线）。详见 `examples/PlanAndVerify/BASELINE.md` §7。
- [x] Final eval：`libero_goal` 10 episode/task，final ckpt 评测完成，100 episode 中 98 success（结果目录 `playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/results/libero_goal/`）

### 1.3 指标记录（后续所有实验都重复这一套）
- [x] 平均成功率 SR：per-task & suite-mean 已记录，suite mean **0.98**（98/100），per-task 见 `examples/PlanAndVerify/BASELINE.md` §5；trial 级 CSV：`paper/tables/baseline_oft_libero_goal.csv`
- [x] 单卡推理延迟基准脚本 + 实测：`examples/PlanAndVerify/eval_files/benchmark_latency.py`，单卡 H20 b=1，5 warmup + 50 step，bf16；mean **88.92 ms/action chunk**，p50 76.54 ms，p95 184.40 ms（源数据 `playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/checkpoints/latency.json`）
- [x] 训练峰值显存（profile 实测）：**44.4 GB / rank**（peak allocated），56.6 GB / rank（peak reserved）—— 见 `playground/Checkpoints/profile_oft_libero_goal_*/profile.json`
- [x] 训练吞吐（profile 实测）：**0.79 steps/s**，**101 samples/s**（同上 profile.json）

### 1.4 阶段 1 交付物
- [x] `examples/PlanAndVerify/BASELINE.md`（替代 todo 原计划的 repo 根 `BASELINE.md` 与 `docs/results/baseline.md` —— 沿用 W1 已建立的 PAV 产物约定，与 commits `4ca2c33` / `867c6e2` / `0fbbd1a` 一致）
- [x] `paper/tables/baseline_oft_libero_goal.csv`（schema 与 W1 `baseline_table.csv` 同：model, task_id, trial_id, success, mp4_path；由 stage1 final eval 生成）
- [x] 训练日志 / config / ckpt 本地归档完成：`playground/Checkpoints/stage1_oft_libero_goal_20260504_185342/{train.log,config.yaml,config.full.yaml,summary.jsonl,checkpoints/,final_model/}`；本 run 使用 `WANDB_MODE=disabled`，无 WandB 离线目录需要归档

### 1.5 额外已完成的对照 baseline（StarVLA-PI-Qwen3VL on LIBERO-Long）

> 不替代 §1.2 的主路线 baseline（OFT + Florence-2 + libero_goal），仅作为长时程套件上的早期参照与 PAV Gate-W1 锚点。

- [x] 模型：StarVLA-PI-Qwen3VL（HF: `StarVLA/Qwen3-VL-PI-LIBERO-4in1`，ckpt: `steps_100000_pytorch_model.pt`）
- [x] 套件：`libero_10`（LIBERO-Long），10 任务 × 30 trial = 300 episode
- [x] 套件平均 SR = **0.9667**（≥ Gate-W1 阈值 0.86，commit `4ca2c33` / `867c6e2` 锁定）
- [x] trial 级结果：`paper/tables/baseline_table.csv`（由 `scripts/dump_baseline_table.py` 从 mp4 文件名聚合）
- [x] 多卡 sharded 评测脚本：`examples/PlanAndVerify/eval_files/{run_policy_server,eval_libero_long,eval_libero_long_sharded.py,eval_libero_long_multi_gpu}.sh`
- [ ] 单卡推理延迟 / 训练峰值显存 / 训练吞吐 — 未记录（评测的是已发布 ckpt，无训练日志；如要在毕设里报这三项，需补一次自己 fine-tune 的 run）

---

## 阶段 2：V-JEPA 特征离线缓存（2–3 天）

**目标**：把 V-JEPA forward 完全踢出训练循环，让方案一在 4090 上"白送"。

### 2.1 抽取脚本
- [ ] 在 `examples/PlanAndVerify/cache_files/extract_vjepa_cache.py` 实现：
  - [ ] 遍历 LeRobot dataset，对每个 `(episode_id, frame_id)` 回溯过去 \(N=8\) 或 \(N=16\) 帧
  - [ ] 送进冻结的 V-JEPA2.1 ViT-B/16 384（首选，80M 参数最省）
  - [ ] 输出池化向量 \(z \in \mathbb{R}^{D}\)（先做这种，最简单）
  - [ ] 可选：少量时序 token \(Z \in \mathbb{R}^{K\times D}\)，\(K=4\)（attention pooling 或 stride sample）
- [ ] 缓存格式：每个 episode 一个 `parquet` 或 `npz`，按 `frame_id` 索引
- [ ] 写 `index.json` 记录形状、模型 ckpt、\(N\)、\(K\)、归一化均值方差

### 2.2 数据范围
- [ ] 必做：缓存 `libero_goal` 全部数据
- [ ] 可选：磁盘允许时再缓存 `libero_10`（长时程套件）
- [ ] 后置：`libero_spatial` / `libero_object` 留到阶段 4 末尾

### 2.3 健全性检查（顺便给可解释性章节攒图）
- [ ] 随机抽 100 帧的特征做 PCA
- [ ] 看不同任务在低维空间是否可分簇
- [ ] 保存 PCA 散点图到 `docs/figures/vjepa_pca_sanity.png`

### 2.4 阶段 2 交付物
- [ ] 缓存目录 + `index.json`
- [ ] 一张 PCA 散点图（**直接进毕设可解释性章节**）

---

## 阶段 3：V-JEPA 融合 framework 接入（3–5 天）

**目标**：以"新增文件 + config 字段"的方式接入 V-JEPA，**不动 starVLA 主干**。

### 3.1 Projector 模块
- [ ] 在 `starVLA/model/modules/projector/vjepa_projector.py` 实现 `VJepaProjector(nn.Module)`：2 层 MLP + LayerNorm，把 \(D_\text{vjepa}\) 投到 VLA hidden size
- [ ] 三个融合算子并存（开关切换，便于消融）：
  - [ ] `concat_tokens`：把 \(K\) 个 V-JEPA token 拼到 VLM 视觉 token 序列前
  - [ ] `film_gating`：用 V-JEPA 池化向量产生 \(\gamma, \beta\)，对 VLA 视觉 token 做 \(\hat{h}=\gamma \odot h+\beta\)
  - [ ] `cross_attn`：在 action head 前加一层 cross-attn，query 来自 VLA、key/value 来自 V-JEPA token

### 3.2 Framework 文件
- [ ] 复制 `starVLA/model/framework/VLM4A/QwenOFT.py` → `starVLA/model/framework/VLM4A/QwenOFT_VJepa.py`
- [ ] 在 framework 里读缓存（按 starVLA 风格：preprocessing 放 framework 不放 dataloader）
  - [ ] 根据 `episode_id, frame_id` 从缓存 `mmap` 出 V-JEPA 特征
  - [ ] 加进 batch dict
- [ ] 在 forward / predict_action 三处插入 projector 调用
- [ ] **保持 forward / predict_action 签名不变**
- [ ] 文件底部写 `__main__` smoke test，单独跑 `python starVLA/model/framework/QwenOFT_VJepa.py --config_yaml ...` 必须能跑通一次 forward

### 3.3 Config
- [ ] 新建 `starvla_libero_vjepa.yaml`，关键字段：

```yaml
framework:
  name: QwenOFT_VJepa
  vjepa:
    cache_dir: ./playground/cache/vjepa/vjepa2_1_vit_b_384/libero_goal
    feat_dim: 768
    num_tokens: 4
    fusion: concat_tokens   # concat_tokens | film_gating | cross_attn
trainer:
  freeze_modules: "qwen_vl_interface.model,vision_tower"   # 走 starVLA 现成 API
  learning_rate:
    base: 1.0e-05
    vjepa_projector: 1.0e-04
    action_model: 1.0e-04
```

- [ ] 验证 `--trainer.freeze_modules` 真的冻住了对应参数（打印 `requires_grad`）
- [ ] 验证 `learning_rate.vjepa_projector` 真的在单独的 param group 里

### 3.4 阶段 3 交付物
- [ ] 一次成功的 single-batch forward+backward 日志
- [ ] 一张架构图 `docs/figures/architecture.png`（**直接进毕设方法章节**）

---

## 阶段 4：主实验（1.5–2 周）

**目标**：用一张主表 + 一张数据效率曲线把"V-JEPA 时序特征对 VLA 有帮助"讲清楚。

### 4.1 实验矩阵（最小集，9 个 run × 3 seed = 27 次训练）

| 维度 | 取值 |
|---|---|
| 方法 | A) Baseline，B) Naive temporal（最近 4 帧 ResNet18/CLIP 池化），C) **V-JEPA 融合** |
| 数据量 | 10% / 25% / 100% |
| 套件 | 主战场 `libero_goal`；扩展 `libero_10`（仅 100% 数据） |
| Seed | 3 个固定 seed |

### 4.2 训练
- [ ] 全部用阶段 1 同一份 starVLA-OFT 配置，只改 `framework.name` 与数据子集 sampler
- [ ] 方法 A × 3 数据量 × 3 seed
- [ ] 方法 B × 3 数据量 × 3 seed
- [ ] 方法 C × 3 数据量 × 3 seed
- [ ] 准备 `bar/run_main_grid.sh` 一键发起全部 27 个 run

### 4.3 评估
- [ ] Preview eval：训练中每 1k 步、每任务 2 episode
- [ ] Final eval（主战场）：`libero_goal` 10 episode/task
- [ ] Final eval（长时程）：最佳模型扩到 `libero_10` 10 episode/task
- [ ] Final eval（可选）：四套件 10 episode/task 共 400 episode

### 4.4 判定阈值（"make sense" 自检）
- [ ] 单套件 SR：方法 C 比 A 高 \(\geq 3\) pp
- [ ] `libero_10`：方法 C 比 A 高 \(\geq 5\) pp
- [ ] 25% 数据档：方法 C 在 25% 数据 \(\geq\) 方法 A 在 100% 数据的 95%
- [ ] 延迟：方法 C \(\leq\) 方法 A × 1.25

### 4.5 阶段 4 交付物（4 张核心图）
- [ ] 主结果柱状图（A / B / C 三组并列）
- [ ] data-efficiency 折线图（10% / 25% / 100%）
- [ ] 训练 loss/SR 曲线图
- [ ] 主结果 CSV 归档

---

## 阶段 5：消融（5–7 天）

**目标**：让答辩老师无法问出"你为啥这么设计"。每个消融在 100% 数据 `libero_goal` 上跑 1 个 seed。

| 消融 | 对照取值 | 期望结论 |
|---|---|---|
| 时序 token 数 \(K\) | 1 vs 4 vs 8 | 中间档最佳 |
| 融合算子 | concat vs FiLM vs cross-attn | 至少一个明显更好 |
| V-JEPA 模型规模 | ViT-B/16-384 vs ViT-L/16-256 | "小模型已够用"卖点 |
| 历史窗口 \(N\) | 4 vs 8 vs 16 帧 | 8 或 16 帧最佳 |
| 是否冻结 | 完全冻结 vs projector + 小 LoRA | 验证冻结主干合理 |
| 上游表征替换 | V-JEPA vs DINOv2 vs CLIP（同样池化） | **关键卖点**：是"时序"而非"自监督"在起作用 |

### 5.1 跑实验
- [ ] 时序 token 数：\(K \in \{1, 4, 8\}\)
- [ ] 融合算子：concat / FiLM / cross-attn
- [ ] V-JEPA 模型规模：ViT-B/16-384 vs ViT-L/16-256
- [ ] 历史窗口：\(N \in \{4, 8, 16\}\)
- [ ] 冻结策略：full freeze vs projector + LoRA
- [ ] 上游表征替换：V-JEPA / DINOv2 / CLIP（共享 projector + 融合算子）

### 5.2 阶段 5 交付物
- [ ] 一张消融汇总表（6 行 × 3 列：SR / latency / VRAM）
- [ ] 简短文字解释每个消融的结论

---

## 阶段 6：可解释性与失败分析（3–5 天）

**目标**：把毕设从"涨了几个点"升级成"知道为什么涨"。

### 6.1 Failure taxonomy
- [ ] 抽 baseline 的 50 条失败 episode + 方法 C 的 50 条失败 episode
- [ ] 人工归类为五类：看错目标 / 抓取失败 / 中途犹豫 / 提前松开 / 长时程漂移
- [ ] 画两组并列条形图

### 6.2 Rollout strip 对比
- [ ] 选 3–5 个有代表性任务
- [ ] 同初始状态下 baseline vs 方法 C 关键帧并列
- [ ] 保存到 `docs/figures/rollout_strip_*.png`

### 6.3 V-JEPA token 注意力 / gating 热图
- [ ] 把 `film_gating` 的 \(\gamma\) 或 cross-attn 权重沿时间画 heatmap
- [ ] 证明模型确实"用了"V-JEPA 信号

### 6.4 Latent PCA / UMAP
- [ ] 方法 C 的 V-JEPA token 在不同子目标阶段的低维分布
- [ ] baseline 视觉 token 作为对照同图叠加

### 6.5 答辩 demo 视频
- [ ] 60–90 秒，剪 3 个任务的 baseline vs 方法 C 并列 rollout
- [ ] 配字幕说明"提前预判 / 少犹豫 / 长时程稳"

### 6.6 阶段 6 交付物
- [ ] 4 张 PNG（taxonomy / rollout / heatmap / PCA）
- [ ] 1 段 MP4
- [ ] 失败案例 CSV

---

## 阶段 7：写作与答辩（1.5–2 周）

**目标**：把所有产出收口成一份能讲完整故事的毕设文档。

### 7.1 论文章节
- [ ] 引言：为什么 VLA + 时序表征值得做
- [ ] 相关工作：VLA 线 + JEPA 线（各 0.5 页）
- [ ] 方法：架构图 + 三种融合算子 + freeze/LR 设置
- [ ] 实验：主表 + data efficiency + 消融 + 失败分析 + 可视化
- [ ] 局限与展望：明确写"未做真机器人"、"未做 reranker"，留扩展接口

### 7.2 答辩 PPT 必备 5 张图
- [ ] 主结果柱状图
- [ ] data-efficiency 曲线
- [ ] 消融汇总表
- [ ] Failure taxonomy
- [ ] Rollout 对比 strip

### 7.3 仓库
- [ ] fork README 写清"如何在 8×4090 上一行命令复现 main result"
- [ ] 一键复现脚本 `bar/reproduce_main.sh`
- [ ] 公开 tf 日志归档

### 7.4 阶段 7 交付物
- [ ] 论文 PDF
- [ ] 答辩 PPT
- [ ] 演示视频
- [ ] fork 仓库 README + 复现脚本

---

## 整体故事线（一句话版本）

> 我把开源 VLA 框架 starVLA 作为底座，将 V-JEPA2.1 通过离线特征缓存 + 轻量 projector 接入 StarVLA-OFT 的视觉 token 流；在 LIBERO 上以"无 V-JEPA / 朴素时序 / V-JEPA 融合"三组对照、跨 10 / 25 / 100% 三档数据量，验证时序表征对 VLA 在长时程任务和少样本场景下的收益，并通过消融把收益归因到 V-JEPA 的"时序"成分而非"自监督"成分。

---

## 可砍的非关键工作（时间紧时直接砍）

如果四套件评测、teacher loss、`jepa-wms` reranker 中任何一个开始拖时间，直接砍。最小可毕业路径：

- [ ] 阶段 0–4
- [ ] 阶段 6 的前两张图（taxonomy + rollout strip）
- [ ] 阶段 7

也就是：**单套件 `libero_goal` + 三组对照 + 三档数据量 + Failure taxonomy + Rollout strip**，本身就是一个完整故事。其它都是加分项。
