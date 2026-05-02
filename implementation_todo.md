# Plan-and-Verify 实施 TODO 总表（基于 starVLA fork）

> **配套阅读**：[research_design_plan_and_verify.md](research_design_plan_and_verify.md)（设计原文）
> **基础仓库**：fork [starVLA/starVLA](https://github.com/starVLA/starVLA) 稳定分支 `starVLA`
> **算力**：2×H100 常驻 + 1×4090 + 偶尔借 8×H200/H20
> **交付窗口**：10 周

---

## 0. 总览

### 0.1 文档约定

- **任务 ID**：`T-W<week>.<sec>.<idx>`，例如 `T-W3.2.4`
- **状态**：`- [ ]` 待办 ｜ `- [x]` 完成（执行时手动改）
- **优先级**：未标记 = 必做 ｜ 🔁 = 推荐 ｜ ⚪ = 可选（标记跟在 `[ ]` 后）
- **来源标记**：★ = WM4A 报告引发的新增项 ｜ 🚧 = Gate 卡点
- **依赖**：未列出时默认依赖前一编号；跨周依赖必须显式列出
- **工时单位**：人/小时（`h`）或人/天（`d`），按 1 人独立工作估算

### 0.2 三个关键 Gate

| Gate | 时点 | 通过条件 | 不通过回退 |
|---|---|---|---|
| 🚧 **G-W1** | W1 末 | StarVLA-PI HF ckpt 在 LIBERO-Long 复现成功率 ≥ 86% | 排查环境/HF/数据；不进 W2 |
| 🚧 **G-W3** | W3 末 | LCLGP best-mode 替代 image goal，V-JEPA 2-AC 在 reach 任务上 ≥ 70% | 回退 K=2 + 仅 end goal，砍 §5.3.1 多模态贡献 |
| 🚧 **G-W6** | W6 中 | GPT-4V vs 人工抽检的 ETAR 标签一致性 ≥ 80% | 回退 rule-based + 200 chunks 人工 |

### 0.3 项目目录结构（最终态）

```
plan-and-verify/                                    # fork 自 starVLA
├── starVLA/                                        # 上游主干（尽量不动）
│   ├── model/
│   │   ├── framework/
│   │   │   └── PlanVerify/                         # ★ 新增 framework 类
│   │   │       ├── __init__.py
│   │   │       ├── lclgp.py                        # 创新 1
│   │   │       ├── msfv.py                         # 创新 2
│   │   │       ├── etar.py                         # 创新 3
│   │   │       └── runtime.py                      # control loop
│   │   ├── modules/
│   │   │   ├── world_model/
│   │   │   │   └── vjepa2.py                       # ★ V-JEPA 2 wrapper
│   │   │   └── action_model/flow_matching_head/    # 改造：noise seed
│   │   └── ...
│   ├── deployment/plan_and_verify/                 # ★ 评测 server
│   ├── training/train_lclgp.py                     # ★ 训练入口
│   └── training/train_etar.py                      # ★ 训练入口
├── examples/PlanAndVerify/                         # ★ 配置 + README
│   ├── README.md
│   ├── configs/
│   │   ├── lclgp_v1.yaml
│   │   ├── etar_v1.yaml
│   │   └── eval_libero_long.yaml
│   ├── train_files/bar/                            # git-ignored
│   └── eval_files/
├── third_party/
│   └── vjepa2/                                     # facebookresearch/vjepa2 submodule
├── scripts/
│   ├── extract_vjepa_latents.py                    # T-W2.2
│   ├── build_lclgp_dataset.py                      # T-W2.3
│   ├── collect_etar_rollouts.py                    # T-W6.1
│   ├── label_etar.py                               # T-W6.3-4
│   ├── eval_main.py                                # T-W8
│   ├── eval_perturb.py                             # T-W9.1
│   ├── eval_calvin.py                              # T-W10.1
│   ├── eval_openpi_supplementary.py                # T-W10.2 ★
│   └── run_diagnostics.py                          # T-W3.4
├── data/
│   ├── latents/                                    # WebDataset 分片
│   ├── lclgp_dataset/                              # T-W2.3 输出
│   ├── etar_rollouts/                              # T-W6.1 输出
│   ├── etar_labeled/                               # T-W6.4 输出
│   └── eval_logs/                                  # 评测日志
├── ckpts/
│   ├── starvla_pi_libero/                          # HF download
│   ├── vjepa2_vitg/                                # HF download
│   ├── lclgp/                                      # T-W3.3 输出
│   └── etar/                                       # T-W7.2 输出
├── docs/
│   └── plan_and_verify.md                          # ★ 项目说明
└── paper/
    ├── main.tex
    ├── figures/
    └── tables/
```

### 0.4 时间轴速览

```
W1  ──┬─ 环境/baseline ─────────────[G-W1]
W2  ──┴─ Latent 缓存 ──────────────────────┐
W3  ─── LCLGP 训练 ─────────[G-W3]──────── │
W4  ─── MSFV 验证器 ───────────────────────│
W5  ─── Runtime + PaV-Lite demo ──────────│
W6  ─── ETAR 数据 ─────────[G-W6]──────────┘
W7  ─── ETAR 训练 + CEM-Light ─── PaV-Full
W8  ─── LIBERO 主实验 ──── 主对比表
W9  ─── Perturb + 消融 ─── 第二张表
W10 ─── CALVIN + openpi + 论文 ─── 交付
```

### 0.5 关键路径（必跑顺序）

```
W1.4 baseline ──> W2.2 latent ──> W3 LCLGP ──> W4 MSFV ──> W5 Runtime
                                        ↓                       ↓
                                   W3.4 诊断             W6 ETAR data
                                                              ↓
                                                          W7 ETAR
                                                              ↓
                                                          W8 主实验
                                                          ↓
                                                  W9 Perturb + 消融
                                                          ↓
                                                  W10 CALVIN/openpi/论文
```

---

## W1 — 仓库 + 环境 + Baseline 复现（5 d）

### 1.1 Fork 与分支策略

- [ ] **T-W1.1.1** Fork `starVLA/starVLA` 到个人/实验室账号
  - **要点**：基于稳定分支 `starVLA`，**不**用 `starVLA_dev`
  - **验收**：`gh repo view <fork>` 返回 fork 标记
  - **工时**：0.2 h

- [ ] **T-W1.1.2** 创建工作分支 `pav-dev`
  - **依赖**：T-W1.1.1
  - **要点**：`git checkout -b pav-dev`；commit message 前缀统一 `[PAV]`
  - **工时**：0.1 h

- [ ] **T-W1.1.3** 把 `facebookresearch/vjepa2` 加为 submodule
  - **要点**：`git submodule add https://github.com/facebookresearch/vjepa2 third_party/vjepa2`
  - **验收**：`.gitmodules` 已提交
  - **工时**：0.1 h

### 1.2 环境安装

- [ ] **T-W1.2.1** 创建 conda env `pav`，Python 3.10
  - **要点**：与 starVLA `pyproject.toml` 对齐
  - **工时**：0.5 h

- [ ] **T-W1.2.2** 装 starVLA 主依赖
  - **要点**：`pip install -e .`；DeepSpeed Zero-2、accelerate、flash-attn==2.x、apex
  - **风险**：flash-attn 编译需要匹配 CUDA 版本；备 `ninja` 加速
  - **工时**：2-4 h（含编译失败重试）

- [ ] **T-W1.2.3** 装 V-JEPA 2 依赖
  - **依赖**：T-W1.2.2
  - **要点**：`pip install -e third_party/vjepa2`；与 starVLA 共享 PyTorch 版本（不要降级）
  - **验收**：`python -c "from vjepa2 import build_vjepa_predictor; print('ok')"`
  - **工时**：1 h

- [ ] **T-W1.2.4** 安装 LIBERO 与 Robosuite
  - **要点**：依据 starVLA `examples/LIBERO/README.md`；MuJoCo 200 license / mujoco-py
  - **验收**：`python -c "from libero.libero import benchmark; benchmark.get_benchmark_dict()"` 列出 4 suite
  - **工时**：1-2 h

- [ ] 🔁 **T-W1.2.5** 安装 CALVIN（可推迟到 W10）
  - **要点**：占位；W10 才用
  - **工时**：1 h

### 1.3 V-JEPA 2 接入 starVLA

- [ ] **T-W1.3.1** 写 `starVLA/model/modules/world_model/vjepa2.py`
  - **依赖**：T-W1.2.3
  - **要点**：仿 `starVLA/model/modules/world_model/CosmoPredict2.py` 接口；导出 `VJEPA2Encoder`、`VJEPA2ACPredictor`
  - **接口签名**：
    ```python
    class VJEPA2Encoder:
        def __init__(self, ckpt_path, resolution=256, dtype=torch.float16)
        @torch.no_grad()
        def encode(self, frames: torch.Tensor) -> torch.Tensor  # [B, 256, 1408]

    class VJEPA2ACPredictor:
        def step(self, z: torch.Tensor, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor
        def rollout(self, z0, s0, a_chunk: torch.Tensor) -> torch.Tensor  # [B, H, 256, 1408]
    ```
  - **验收**：`pytest tests/world_model/test_vjepa2.py::test_forward_shape`
  - **工时**：4 h

- [ ] **T-W1.3.2** 下载 V-JEPA 2 ViT-g 权重 + AC predictor 权重
  - **要点**：HF 路径见 vjepa2 README；放在 `ckpts/vjepa2_vitg/`
  - **验收**：encoder 加载 + 单帧 forward 通过
  - **工时**：1 h

- [ ] **T-W1.3.3** 注册 V-JEPA 2 到 starVLA `WORLD_MODEL_REGISTRY`
  - **要点**：仿 CosmoPredict2 的注册流程（grep `@WORLD_MODEL_REGISTRY.register`）
  - **验收**：`build_world_model("vjepa2_vitg")` 返回正确实例
  - **工时**：1 h

- [ ] **T-W1.3.4** Smoke test：starVLA dataloader → V-JEPA encoder
  - **依赖**：T-W1.3.3 + LIBERO 数据已下载（W2.1 提前一部分）
  - **要点**：写 `tests/integration/test_vjepa_on_libero.py`，验证一个 batch 能 forward
  - **验收**：测试通过；输出 shape `[B, 256, 1408]`
  - **工时**：2 h

### 1.4 Baseline 复现 [🚧 G-W1]

- [ ] **T-W1.4.1** 下载 `StarVLA/bench-libero` 的 QwenPI checkpoint
  - **要点**：HF `snapshot_download`；放在 `ckpts/starvla_pi_libero/`
  - **工时**：0.5 h（依赖网速）

- [ ] **T-W1.4.2** 准备 LIBERO 数据
  - **要点**：跑 starVLA `examples/LIBERO/data_preparation.sh`
  - **磁盘**：~80 GB
  - **工时**：2 h（主要等下载）

- [ ] **T-W1.4.3** 跑 starVLA 自带 LIBERO eval（QwenPI）
  - **依赖**：T-W1.4.1, T-W1.4.2
  - **要点**：`bash examples/LIBERO/eval_files/run_libero_eval.sh`
  - **配置**：每任务 30 trials × LIBERO-Long 10 任务 = 300 trials
  - **资源**：1×H100，~6 h
  - **验收**：**🚧 LIBERO-Long 平均成功率 ≥ 86%**（论文 ~88.4，留 2.4 个点容差）
  - **失败处理**：检查 ckpt 路径、PaliGemma vs Qwen3-VL backbone 是否对齐、LIBERO 任务集合是否一致

- [ ] **T-W1.4.4** 锁定 baseline 数字 → `paper/tables/baseline_table.csv`
  - **要点**：保存原始 trial 级数据，后面所有对比都以此为锚点
  - **验收**：CSV 含每任务每 trial 的成功标记 + 总耗时

---

## W2 — 数据 Pipeline（5 d）

### 2.1 数据下载

- [ ] **T-W2.1.1** 下载 LIBERO 全部 4 suite
  - **要点**：T-W1.4.2 已完成 → 跳过
  - **工时**：0

- [ ] **T-W2.1.2** 下载 Bridge-v2（取 50K demo 子集）
  - **要点**：starVLA 是否原生支持 Bridge-v2？grep `bridge_v2` 在 `datasets/`；不支持则用 OXE 格式 + LeRobot v3 转换
  - **磁盘**：~200 GB（subset）
  - **工时**：4 h（多数为下载等待）

- [ ] ⚪ **T-W2.1.3** 下载 AgiBot World 子集（可选，扩样本量用）
  - **决策点**：W2.4 数据集 ≥ 300K 样本则跳过

### 2.2 V-JEPA Latent 缓存

- [ ] **T-W2.2.1** 写 `scripts/extract_vjepa_latents.py`
  - **要点**：sweep starVLA `datasets.vla_data` LeRobot v3 链路；按 `task_id × demo_id × frame_idx` 缓存
  - **关键参数**：`--patch_grid 16`（720KB/frame）or `--patch_grid 8`（90KB/frame，磁盘紧张时用）
  - **存储格式**：WebDataset 分片，每片 ~1 GB，便于流式读
  - **路径**：`data/latents/{dataset}/shard_{idx:06d}.tar`
  - **工时**：6 h

- [ ] **T-W2.2.2** 估算磁盘占用
  - **要点**：500K samples × 3 frames（z_t, z_Δ, z_T）× 720KB ≈ **1.05 TB**
  - **决策**：磁盘 < 1.5 TB → 改用 patch_grid=8 或流式
  - **输出**：`docs/data_storage_plan.md`

- [ ] **T-W2.2.3** 在 8×H200 上跑 latent 抽取
  - **依赖**：T-W2.2.1
  - **资源**：8×H200 一晚（~10 h）
  - **验收**：500K × 3 = 1.5M latent 全部缓存；随机抽 100 个 reload 验 sha256

- [ ] **T-W2.2.4** 写 latent dataloader
  - **要点**：`starVLA/datasets/vjepa_latent_dataset.py`；支持 WebDataset；返回 `(text_emb, z_t, z_delta, z_end, task_id)`
  - **验收**：`pytest tests/data/test_latent_loader.py`
  - **工时**：3 h

### 2.3 LCLGP 训练样本构造

- [ ] **T-W2.3.1** 写 `scripts/build_lclgp_dataset.py`
  - **依赖**：T-W2.2.4
  - **要点**：对每条成功 demo 采样 10 个 (t, t+Δ, T) 三元组；调 starVLA 的 PaliGemma/Qwen 文本 embedder 缓存 `text_emb`
  - **逻辑**：filter `demo.success == True`，跳过失败 demo
  - **输出**：`data/lclgp_dataset/{dataset}_{split}.parquet`
  - **工时**：4 h

- [ ] **T-W2.3.2** 划分 train/val/test
  - **要点**：按 task_id 划，避免任务泄漏；test 占 10%
  - **验收**：每个任务在 train 至少 5 demo
  - **工时**：1 h

- [ ] **T-W2.3.3** 写 `TaskGroupedSampler`
  - **依赖**：T-W2.3.2
  - **要点**：每 batch 含 32 任务 × 8 demos（batch_size=256）；保证 D1 多模态训练
  - **验收**：`pytest tests/data/test_grouped_sampler.py`
  - **工时**：2 h

### 2.4 数据快照

- [ ] **T-W2.4.1** 数据集统计报告
  - **输出**：`data/lclgp_dataset/STATS.md`，包含每数据集样本数、任务数、平均 demo 长度
  - **工时**：1 h

---

## W3 — LCLGP 模块 [🚧 G-W3]（7 d）

### 3.1 模型实现

- [ ] **T-W3.1.1** 实现 `LCLGP` 类骨架
  - **入口**：`starVLA/model/framework/PlanVerify/lclgp.py`
  - **要点**：基于设计文档 §5.2；K=4 多模态 + 不确定性 head
  - **验收**：`forward` 输出 shape `(z_g_end, z_g_delta, log_sig_end, log_sig_delta)`
  - **工时**：6 h

- [ ] **T-W3.1.2** 实现 mode embedding + cross-attention（D2 解的结构侧）
  - **要点**：slot tokens 通过 4 层 `TransformerDecoderLayer`，z_t 作 K/V
  - **验收**：单元测试 `tests/lclgp/test_attention_routing.py` 验证不同 z_t 输入产生不同输出
  - **工时**：3 h

- [ ] **T-W3.1.3** 实现 z_t dropout（D2 解的训练侧）
  - **要点**：训练时 10% 概率把 `z_t` 置零
  - **验收**：dropout 开启时 forward 不报错；置零时输出仍是合法 shape
  - **工时**：1 h

- [ ] **T-W3.1.4** 实现 unc head
  - **要点**：mode-level pool 后投影到 1 维 log_sigma
  - **工时**：1 h

- [ ] **T-W3.1.5** 注册到 `FRAMEWORK_REGISTRY`
  - **依赖**：T-W3.1.1～4
  - **要点**：`@FRAMEWORK_REGISTRY.register("PlanVerify_LCLGP")`
  - **验收**：`build_framework("PlanVerify_LCLGP", config)` 返回正确实例
  - **工时**：1 h

### 3.2 训练损失

- [ ] **T-W3.2.1** 实现 min-of-K hindsight + 异方差
  - **要点**：设计文档 §5.3.1 公式；先做 batch 内 [B,K] 距离矩阵，再 min
  - **验收**：单元测试梯度反传到正确的 mode（手算 case）
  - **工时**：3 h

- [ ] **T-W3.2.2** 实现 mode-balancing 正则
  - **要点**：batch 内 mode 选中频率 vs Uniform(K) 的 KL
  - **验收**：mode 不均衡时 loss > 0；均衡时 ≈ 0
  - **工时**：1.5 h

- [ ] **T-W3.2.3** 实现对比损失
  - **要点**：InfoNCE on task-level summary（patch-pool + cosine + temp 0.07）
  - **验收**：单任务 batch loss 应低，跨任务 batch loss 应高
  - **工时**：2 h

- [ ] **T-W3.2.4** 实现反事实损失（D2 解的训练侧）
  - **要点**：z_t=0 vs z_t=正常 的最小距离 hinge
  - **关键**：注意是 **最大化** 差异（反事实分支用 `with torch.no_grad()` 包住）
  - **工时**：2 h

- [ ] **T-W3.2.5** 总损失组装
  - **依赖**：T-W3.2.1～4
  - **要点**：α=0.5, λ_bal=0.05, λ_ctr=0.1, λ_cf=0.05
  - **输出**：`L, log_dict` 含每分量值（用于 TensorBoard）
  - **工时**：1 h

### 3.3 训练循环

- [ ] **T-W3.3.1** 写 `starVLA/training/train_lclgp.py`
  - **要点**：复用 starVLA `train_starvlm.py` 模板；接 DeepSpeed Zero-2、bf16
  - **配置**：`examples/PlanAndVerify/configs/lclgp_v1.yaml`
  - **工时**：4 h

- [ ] **T-W3.3.2** 写 W&B / TensorBoard 监控
  - **要点**：记 `L_end`, `L_dlt`, `L_ctr`, `L_cf`, mode_balance.std, sigma.mean
  - **关键监控指标**：`mode_balance.std < 0.3`（mode 健康活跃）；`L_cf` 收敛到 hinge margin
  - **工时**：1 h

- [ ] **T-W3.3.3** 1 epoch quick run on 1×H100
  - **依赖**：T-W3.3.1
  - **资源**：1×H100，~40 min
  - **验收**：loss 下降；mode_balance.std < 0.5
  - **失败处理**：mode 全部坍缩 → 增大 λ_bal；L_cf 不下降 → 检查反事实分支梯度

- [ ] **T-W3.3.4** 30 epoch full run on 2×H100
  - **依赖**：T-W3.3.3
  - **资源**：2×H100，~20 h
  - **配置**：lr 5e-4 → 1e-5 cosine；batch 256
  - **输出**：`ckpts/lclgp/best.pt`（按 val L_end 选）
  - **工时**：等待时间为主

### 3.4 D1-D4 诊断（论文 §4.5 基础）

- [ ] **T-W3.4.1** 写 `scripts/run_diagnostics.py`
  - **依赖**：T-W3.3.4
  - **要点**：一键跑 §5.6 全部诊断
  - **工时**：4 h

- [ ] **T-W3.4.2** G-1 Min-of-K 余弦
  - **验收**：mean ≥ 0.75；< 0.6 触发回炉
  - **输出**：`paper/tables/lclgp_g1.csv`

- [ ] **T-W3.4.3** G-2 解码可视化
  - **要点**：用 V-JEPA 2 frame decoder（如可用）解 best-mode 回像素；找不到 decoder 则跳过解码，只画 latent space PCA
  - **输出**：`paper/figures/lclgp_decoded_samples/{task}_*.png`

- [ ] 🚧 **T-W3.4.4** G-3 reach 任务零样本规划【**Gate 关键测试**】
  - **要点**：用 LCLGP best-mode 替代 image goal，跑 V-JEPA 2-AC 单步 CEM 在 reach 任务（LIBERO-Spatial 中最简单的 5 个任务）
  - **🚧 Gate**：成功率 ≥ 70%（image goal baseline 通常 ~100%）
  - **失败时回退**：见 0.2 G-W3 的回退方案

- [ ] **T-W3.4.5** D1-a/b/c/d 多模态诊断
  - **D1-a**：模式两两余弦 ∈ [0.3, 0.7]
  - **D1-b**：最不活跃 mode 选中率 ≥ 10%
  - **D1-c**：σ vs 误差 Pearson ≥ 0.4
  - **D1-d**：与 K=1 baseline 对比 G-1 提升 ≥ 5 个点（额外训一个 K=1 ckpt）

- [ ] **T-W3.4.6** D2-a/b/c 状态依赖诊断
  - **D2-a**：反事实差异 ≥ hinge margin
  - **D2-c**：z_t-ablation 对比 G-1 掉 ≥ 5 个点（额外训一个 z_t-ablation ckpt）

- [ ] **T-W3.4.7** D3-a/b 时间尺度诊断
  - **D3-a**：互换 end/delta 后 chunk-scale Spearman 绝对值 < 0.3
  - **D3-b**：双 head 比单 head 在 chunk verify 准确率高 ≥ 5 个点

- [ ] **T-W3.4.8** 出诊断报告 → `docs/lclgp_diagnostics.md`
  - **要点**：表 + 图，可直接迁入论文 §4.5

### 3.5 Gate 决策

- [ ] **T-W3.5.1** 召开内部 review，决定继续 / 回退
  - **依赖**：T-W3.4.4
  - **决策树**：
    - G-3 ≥ 70% → 进 W4 全功能 PaV
    - G-3 50-70% → 训 K=2 简化版重测；通过则进 W4 简化路线
    - G-3 < 50% → 回退到仅 end goal 的 PaV-Lite（M1）

---

## W4 — MSFV（5 d）

### 4.1 V-JEPA 2-AC predictor 集成

- [ ] **T-W4.1.1** 写 `BatchedPredictor`
  - **入口**：`starVLA/model/framework/PlanVerify/msfv.py`
  - **要点**：包 `VJEPA2ACPredictor` 支持 batch 维度（CEM-Light 一次跑 16 chunk）
  - **签名**：
    ```python
    def rollout_batched(self, z_t, s_t, a_chunks: torch.Tensor) -> torch.Tensor
        # a_chunks: [N, H, 7] -> z_traj: [N, H, 256, 1408]
    ```
  - **工时**：3 h

- [ ] **T-W4.1.2** Micro-benchmark：单 chunk rollout 时延
  - **依赖**：T-W4.1.1
  - **目标**：H=50 单 chunk ≤ 100 ms（设计文档说 80 ms）
  - **失败处理**：> 150 ms 时检查 fp16 是否生效、KV cache、`torch.compile`
  - **输出**：`docs/perf/rollout_single.md`

- [ ] **T-W4.1.3** Micro-benchmark：批量 16 chunk rollout 时延
  - **目标**：N=16 × H=50 ≤ 1.5 s（CEM-Light 预算上限）
  - **关键**：`torch.cuda.Event` 测；warm-up 5 次
  - **输出**：`docs/perf/rollout_batched.md`

### 4.2 Chunk-scale verifier

- [ ] **T-W4.2.1** 实现 `chunk_verify` 函数
  - **依赖**：T-W4.1.1
  - **要点**：min-energy across K modes，用 σ 加权（设计文档 §6.2 公式）
  - **输出字段**：`E_traj, E_end, delta_E, monotonicity, mode_choice, mode_consistency`
  - **工时**：3 h

- [ ] **T-W4.2.2** 单元测试：成功 demo 上 Spearman 单调性
  - **依赖**：T-W4.2.1
  - **要点**：从 LIBERO 成功 demo 取 chunk，跑 verify
  - **🚧 检查**：成功 chunk 的 Spearman ρ ≤ -0.5（不达标说明 V-JEPA 表征对子任务不敏感）
  - **失败处理**：触发设计文档风险表 R2，需上 SigLIP cross-check

### 4.3 Episode-scale progress

- [ ] **T-W4.3.1** 实现 `episode_progress` 函数
  - **依赖**：T-W4.2.1
  - **要点**：取 best mode；返回 `(p_t, k_star)`
  - **关键**：不强制 clamp 到 [0, 1]
  - **工时**：2 h

- [ ] **T-W4.3.2** 单元测试：成功 demo 上 p_t 单调上升
  - **要点**：在 10 条成功 demo 上累计 p_t 时间序列，Spearman 与时间 ρ ≥ 0.6

### 4.4 双尺度信号融合

- [ ] **T-W4.4.1** 实现 `msfv_features` 函数
  - **依赖**：T-W4.2.1, T-W4.3.1
  - **要点**：10 维特征向量（含 D1 推理侧的 mode_consistency, episode_mode_switch_rate）
  - **工时**：2 h

- [ ] **T-W4.4.2** 单元测试：feature 维度与命名稳定
  - **要点**：snapshot test，避免后续改动 feature 顺序破坏 ETAR

### 4.5 集成测试

- [ ] **T-W4.5.1** End-to-end 单 chunk verify
  - **依赖**：T-W3.3.4, T-W4.4.1
  - **要点**：随机 obs → LCLGP → MSFV → features，全程不报错
  - **工时**：2 h

---

## W5 — Runtime + PaV-Lite（5 d）

### 5.1 PlanAndVerifyRuntime

- [ ] **T-W5.1.1** 写 `runtime.py` 主类
  - **入口**：`starVLA/model/framework/PlanVerify/runtime.py`
  - **要点**：基于设计文档 §10.2.3；包 π₀ + encoder + predictor + lclgp + msfv + etar
  - **状态机**：reset → step → ... → step → reset
  - **工时**：5 h

- [ ] **T-W5.1.2** 实现 EXECUTE-only 占位（W5 阶段 ETAR 还没训）
  - **依赖**：T-W5.1.1
  - **要点**：所有决策强制返回 EXECUTE，但 features 全部计算并 log
  - **工时**：1 h

- [ ] **T-W5.1.3** 实现 `TrajectoryLogger`
  - **要点**：记录每 chunk 的 (z_t, E_traj, p_t, decision)；存 HDF5
  - **路径**：`data/eval_logs/{run_id}/{episode_id}.h5`
  - **工时**：2 h

### 5.2 starVLA framework 注册

- [ ] **T-W5.2.1** 注册 `PlanVerify` framework class
  - **依赖**：T-W5.1.1
  - **要点**：实现 starVLA framework 接口（`forward`, `evaluate` 等抽象方法）
  - **工时**：3 h

- [ ] **T-W5.2.2** 写 starVLA 风格的 LIBERO eval 入口
  - **要点**：仿 `examples/LIBERO/eval_files/`；命令行可切 `framework_name=PlanVerify`
  - **配置**：`examples/PlanAndVerify/eval_files/eval_libero_long.sh`
  - **工时**：2 h

### 5.3 PaV-Lite 端到端

- [ ] **T-W5.3.1** 在 LIBERO-Long 单任务跑通
  - **依赖**：T-W5.2.2
  - **任务**：`KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
  - **🚧 检查**：
    - 端到端跑通 30 trials 不崩溃
    - 平均成功率 ≥ baseline ± 5%（EXECUTE-only 不应破坏性能）
    - chunk-scale features 全部有值
  - **资源**：1×4090 一晚

- [ ] **T-W5.3.2** 出 demo 视频
  - **要点**：录 5 个成功 + 5 个失败 case 的 RGB rollout，叠加 E(t)、p_t 曲线 overlay
  - **输出**：`paper/figures/demo/pav_lite_kitchen_*.mp4`
  - **工时**：3 h

- [ ] **T-W5.3.3** Verifier signal 散点图
  - **要点**：500 个 chunk 的 (E_end, success_label) 散点；ROC-AUC 报告
  - **🚧 检查**：AUC ≥ 0.7（说明 verifier 信号有判别力）
  - **输出**：`paper/figures/verifier_auc.pdf`

### 5.4 W5 里程碑

- [ ] **T-W5.4.1** 论文最小可发版本（PaV-Lite）锚定
  - **要点**：即使 W6+ 出问题，PaV-Lite 已经可成文；记录此时的 LIBERO-Long 结果作为 fallback baseline
  - **输出**：`paper/tables/pav_lite_anchor.csv`

---

## W6 — ETAR 数据收集 [🚧 G-W6]（5 d）

### 6.1 Rollout 收集

- [ ] **T-W6.1.1** 写 `scripts/collect_etar_rollouts.py`
  - **要点**：跑 StarVLA-PI baseline 在 LIBERO-Long 500 trajs；每 chunk 记录 `(obs, z_t, a_chunk, z_g_end, z_g_delta, E_traj, p_t)`
  - **格式**：HDF5，每 traj 一个 group
  - **路径**：`data/etar_rollouts/raw_500.h5`
  - **工时**：3 h

- [ ] **T-W6.1.2** 在 8×H200 上 rollout
  - **依赖**：T-W6.1.1
  - **资源**：8×H200 一晚（每卡 ~63 trajs，并行）
  - **预期**：~50% 成功 / ~50% 失败（LIBERO-Long 较难）
  - **验收**：500 trajs 全部完成；记 `success_label, failure_mode_hint`

### 6.2 自动标注

- [ ] **T-W6.2.1** 写 `scripts/label_etar.py` 自动规则部分
  - **依赖**：T-W6.1.2
  - **规则**（设计文档 §8.2）：
    - 成功 + chunk_t > 0.8T → EXECUTE
    - 成功 → EXECUTE
    - failure + stuck（位置变化 < ε）→ RESAMPLE
    - failure + diverging（||z_t - z_g||↑↑）→ STOP
    - chunk 内 spike > 阈值且回落 → WAIT
  - **工时**：4 h

- [ ] **T-W6.2.2** 输出统计
  - **要点**：4 类标签分布；理想 EXECUTE 占 60-70%，其他各 10-15%
  - **失败处理**：分布偏斜过严重 → 调阈值或重新平衡

### 6.3 GPT-4V 标注 pilot [🚧 G-W6]

- [ ] **T-W6.3.1** 写 GPT-4V 标注 prompt
  - **要点**：输入 chunk 时间窗的 5 帧 RGB + 任务指令 + ε 文本，输出 4-class 标签
  - **prompt 文件**：`scripts/prompts/etar_label.txt`
  - **工时**：2 h

- [ ] **T-W6.3.2** 抽 1000 个 chunk 跑 pilot
  - **依赖**：T-W6.2.1, T-W6.3.1
  - **API 成本**：~$30
  - **输出**：`data/etar_labeled/pilot_1k.json`

- [ ] 🚧 **T-W6.3.3** GPT-4V vs 人工抽检 200 个的一致性
  - **🚧 Gate**：Cohen κ ≥ 0.6（一致性 ≥ 80% 等价）
  - **失败时回退**：放弃 GPT-4V，全用 rule-based + 200 chunks 人工标
  - **工时**：人工标 4 h

### 6.4 全量标注

- [ ] **T-W6.4.1** 全量 GPT-4V 跑 10K chunks（如 G-W6 通过）
  - **依赖**：T-W6.3.3 通过
  - **API 成本**：~$300
  - **运行**：API 限流 → 后台跑 ~12 h

- [ ] **T-W6.4.2** 合并 rule + GPT-4V，rule 优先
  - **要点**：明显 case rule 接管，模糊 case GPT-4V 兜底
  - **输出**：`data/etar_labeled/full.h5`
  - **工时**：1 h

### 6.5 一致性 + 数据集划分

- [ ] **T-W6.5.1** train/val 划分
  - **要点**：按 traj 划分（不是 chunk），避免泄漏
  - **比例**：80/20

- [ ] **T-W6.5.2** 类别平衡分析
  - **输出**：`data/etar_labeled/STATS.md`

---

## W7 — ETAR + CEM-Light（5 d）

### 7.1 StarVLA-PI noise seed 改造

- [ ] **T-W7.1.1** 定位 flow matching 头
  - **要点**：grep `starVLA/model/modules/action_model/flow_matching_head/`
  - **输出**：`docs/notes/fm_head_location.md`
  - **工时**：1 h

- [ ] **T-W7.1.2** 在 sample 函数加 `init_noise: Optional[Tensor]` 参数
  - **依赖**：T-W7.1.1
  - **关键**：保持 `init_noise=None` 默认行为不变；显存与计算开销相同
  - **改动量**：~30 行
  - **工时**：3 h

- [ ] **T-W7.1.3** 单元测试：相同 seed → 相同 chunk
  - **依赖**：T-W7.1.2
  - **要点**：`pytest tests/action_head/test_seeded_sample.py`
  - **验收**：两次相同 seed 生成的 chunk L1 距离 < 1e-6

- [ ] **T-W7.1.4** 验证 baseline 不退化
  - **要点**：跑 LIBERO-Long 30 trials × 10 任务，对比 T-W1.4.4 的 baseline，应 ±2%
  - **失败处理**：明显退化说明 patch 引入 bug

### 7.2 ETARClassifier

- [ ] **T-W7.2.1** 实现 `ETARClassifier`
  - **入口**：`starVLA/model/framework/PlanVerify/etar.py`
  - **结构**：3 层 MLP，~12K 参数
  - **输入**：10 维 features（W4.4.1 的输出）
  - **输出**：4 logits
  - **工时**：2 h

- [ ] **T-W7.2.2** 写 `starVLA/training/train_etar.py`
  - **依赖**：T-W6.4.2, T-W7.2.1
  - **要点**：class-balanced cross-entropy；EarlyStopping by val accuracy
  - **资源**：1×4090，~1 h
  - **配置**：`examples/PlanAndVerify/configs/etar_v1.yaml`

- [ ] **T-W7.2.3** 训练 + 选 best ckpt
  - **依赖**：T-W7.2.2
  - **🚧 检查**：val accuracy ≥ 75%（4 类，random=25%）
  - **输出**：`ckpts/etar/best.pt`
  - **混淆矩阵**：`paper/tables/etar_confusion.csv`

- [ ] **T-W7.2.4** Rule-based 对照实现
  - **要点**：实现 §7.2 的阈值规则；阈值通过 train set ROC 选
  - **工时**：3 h

### 7.3 CEM-Light

- [ ] **T-W7.3.1** 实现 `cem_light` 函数
  - **依赖**：T-W7.1.2, T-W4.1.1
  - **要点**：N=16 个 noise seed → 16 chunks → 批量 verify → arg-min E_end
  - **关键**：复用 T-W4.1.1 的 batched rollout
  - **工时**：3 h

- [ ] **T-W7.3.2** Micro-benchmark：CEM-Light 整体延迟
  - **依赖**：T-W7.3.1
  - **目标**：≤ 3 s（设计文档 ~2.8 s）
  - **失败处理**：> 5 s 时降 N=8 或并行 chunk 生成

### 7.4 PaV-Full 集成

- [ ] **T-W7.4.1** Runtime 接入 ETAR 决策
  - **依赖**：T-W5.1.1, T-W7.2.3
  - **要点**：替换 T-W5.1.2 的 EXECUTE 占位
  - **工时**：2 h

- [ ] **T-W7.4.2** 接入 RESAMPLE 调 CEM-Light
  - **依赖**：T-W7.3.1, T-W7.4.1
  - **工时**：1 h

- [ ] **T-W7.4.3** STOP / WAIT 行为实现
  - **要点**：STOP → episode 终止；WAIT → 下一 chunk 用上一 chunk 的 action[k+1]
  - **工时**：1 h

### 7.5 PaV-Full 验证

- [ ] **T-W7.5.1** 误触发率验证
  - **依赖**：T-W7.4.3
  - **要点**：在 baseline 的 100 个成功 trajs 上跑 PaV-Full，统计 RESAMPLE / STOP 触发率
  - **🚧 检查**：成功 traj 上 RESAMPLE 触发率 ≤ 5%；STOP 触发率 ≤ 1%
  - **失败处理**：触发率高 → 调 ETAR 阈值或加 EMA 平滑

- [ ] **T-W7.5.2** 端到端延迟报告
  - **要点**：测正常 EXECUTE 路径 + RESAMPLE 路径的 p50/p95
  - **目标**：EXECUTE p50 ≤ 250 ms；RESAMPLE p50 ≤ 3 s
  - **输出**：`paper/tables/latency_breakdown.csv`

---

## W8 — LIBERO 主实验（5 d）

### 8.1 基线模型准备

- [ ] **T-W8.1.1** B0 StarVLA-PI baseline（已有）
  - **依赖**：T-W1.4.4
  - **数据**：复用

- [ ] **T-W8.1.2** B1 image-goal verifier oracle
  - **要点**：用 demo 末帧的真 image goal 替换 LCLGP 输出，跑 PaV-Full
  - **目的**：上界对比，证明 LCLGP 与 oracle 差距
  - **资源**：1×H100，~6 h
  - **工时**：2 h（实现 + 跑）

- [ ] **T-W8.1.3** B2 π₀ + Qwen2.5-VL HL 规划器
  - **要点**：starVLA 已有 Qwen-VL backbone；写一个把 Qwen-VL 输出文本子任务用作 episode-scale 进度判断的简化版
  - **决策点**：实现成本 ≥ 1 d → 砍掉，主表注明
  - **工时**：1 d

- [ ] **T-W8.1.4** B3 self-check baseline
  - **要点**：用 PaliGemma 自检"是否成功"作为 verifier，决策 EXECUTE/STOP
  - **工时**：4 h

- [ ] ⚪ **T-W8.1.5** B4 F1 (open weights) baseline
  - **决策点**：F1 在 starVLA 中 framework `examples/F1`？grep 确认；不在则砍
  - **工时**：1 d（如果存在）

- [ ] ⚪ **T-W8.1.6** B5 DINO-WM verifier 替代
  - **要点**：替换 V-JEPA 2 为 DINO-WM
  - **决策点**：看 W9 时间富余度
  - **工时**：1 d

- [ ] **T-W8.1.7** ★ B6 WM4A-OFT 对照
  - **要点**：直接拉 starVLA HF `StarVLA/world-model-to-vla` collection 的 CosmoPredict2-OFT ckpt 跑 LIBERO-Long
  - **目的**：与生成式 world-model VLA 拉差异化
  - **资源**：1×H100，~6 h
  - **工时**：2 h

### 8.2 PaV 模型评测

- [ ] **T-W8.2.1** M1 PaV-Lite（仅 end goal + chunk verify）
  - **依赖**：T-W5.4.1
  - **资源**：1×H100，~6 h

- [ ] **T-W8.2.2** M2 PaV-Rule（M1 + 双 target + episode prog + ETAR-rule）
  - **依赖**：T-W7.2.4
  - **资源**：1×H100，~6 h

- [ ] **T-W8.2.3** M3 PaV-Full（完整版）
  - **依赖**：T-W7.5.2
  - **资源**：1×H100，~8 h（含 RESAMPLE 开销）

### 8.3 主对比表

- [ ] **T-W8.3.1** 整合所有 model 的 LIBERO-Long 结果
  - **依赖**：T-W8.1.*, T-W8.2.*
  - **要点**：30 trials × 10 任务 × 10 model = 3000 trials；建议借 8×H200 一夜跑完
  - **输出**：`paper/tables/main_libero_long.csv`

- [ ] **T-W8.3.2** LIBERO-Spatial / Object / Goal（不掉点验证）
  - **要点**：仅 B0 + M3 跑这三 suite，证明 PaV 在中短任务不破坏性能
  - **资源**：8×H200 一夜
  - **输出**：`paper/tables/libero_full.csv`

### 8.4 显著性分析

- [ ] **T-W8.4.1** Paired bootstrap 显著性
  - **要点**：B0 vs M3 每对应 trial 配对；10K resample；输出 95% CI
  - **工具**：`scipy.stats.bootstrap`
  - **🚧 期望**：M3 - B0 ≥ +5pp 在 LIBERO-Long 上 p < 0.01
  - **工时**：2 h

- [ ] **T-W8.4.2** 主表加显著性星号
  - **依赖**：T-W8.4.1
  - **格式**：参考论文期望 §9.4

---

## W9 — Perturb + 消融（5 d）

### 9.1 LIBERO Perturb 协议

- [ ] **T-W9.1.1** 实现 P1 物体重置
  - **入口**：`starVLA/evaluation/pav_perturb/p1_object_reset.py`
  - **要点**：episode 50% 时随机把目标物体瞬移 5-15 cm
  - **工时**：3 h

- [ ] **T-W9.1.2** 实现 P2 干扰物加入
  - **要点**：episode 50% 时场景中加 3-5 个 distractor
  - **工时**：3 h

- [ ] **T-W9.1.3** 实现 P3 光照剧变
  - **要点**：调 Robosuite 光源参数
  - **工时**：2 h

- [ ] **T-W9.1.4** 实现 P4 视角偏移
  - **要点**：相机外参旋转 5°
  - **工时**：2 h

- [ ] **T-W9.1.5** 实现 P5 物体替换
  - **要点**：换同类不同实例（颜色/形状）
  - **工时**：3 h

- [ ] **T-W9.1.6** 主 model 跑 P1-P5
  - **依赖**：T-W9.1.1～5
  - **范围**：B0 + M3 + B6（WM4A-OFT 对照）
  - **trials**：5 任务 × 5 perturb × 30 trials × 3 model = 2250
  - **资源**：8×H200 一夜
  - **输出**：`paper/tables/perturb_recovery.csv`

### 9.2 标准消融

- [ ] **T-W9.2.1** w/o LCLGP-end
  - **要点**：把 z_g_end 强制设为 z_t（无效目标），保留 z_g_delta；跑 LIBERO-Long
  - **资源**：1×H100，~6 h

- [ ] **T-W9.2.2** w/o LCLGP-delta
  - **同**：保留 end，去掉 delta

- [ ] **T-W9.2.3** w/o contrastive loss
  - **要点**：重训 LCLGP（λ_ctr=0），跑 G-1 + LIBERO-Long
  - **资源**：2×H100 训练 ~20 h + 评测 ~6 h

- [ ] **T-W9.2.4** w/o episode progress
  - **要点**：MSFV 只输出 chunk-scale features

- [ ] **T-W9.2.5** w/o ETAR (always EXECUTE)
  - **要点**：替换为 T-W5.1.2 的 EXECUTE 占位

- [ ] **T-W9.2.6** rule ETAR vs learned ETAR
  - **依赖**：T-W7.2.4

- [ ] **T-W9.2.7** w/o CEM-Light（RESAMPLE → re-call π₀ once）
  - **要点**：RESAMPLE 时不做 16 candidates，直接重新调 π₀ 一次

### 9.3 ★ Cosmos verifier 对照（关键消融）

- [ ] **T-W9.3.1** 实现 `CosmoPredict2Verifier`
  - **入口**：`starVLA/model/framework/PlanVerify/baselines/cosmo_verifier.py`
  - **要点**：复用 starVLA 已载的 Cosmos-Predict2 DiT；取其 hidden state 替代 V-JEPA latent；用同样的 chunk_verify 公式
  - **目的**：直接回应 WM4A 报告的审稿人质疑 #2
  - **工时**：1 d

- [ ] **T-W9.3.2** 重训 LCLGP-Cosmo
  - **依赖**：T-W9.3.1
  - **要点**：LCLGP 输入维度变（Cosmos hidden 维度 ≠ V-JEPA 1408）；30 epoch
  - **资源**：2×H100 ~24 h

- [ ] **T-W9.3.3** Cosmo-verifier vs V-JEPA-verifier 对比
  - **要点**：在 LIBERO-Long 上跑两个 verifier，主指标：(1) 成功率（应 < V-JEPA）；(2) chunk verify Spearman；(3) 单 chunk rollout 延迟（应 ≫ V-JEPA）
  - **输出**：`paper/tables/verifier_choice_ablation.csv`

### 9.4 编码器消融

- [ ] ⚪ **T-W9.4.1** SigLIP 替代 V-JEPA 2 encoder
  - **要点**：starVLA 已有 SigLIP backbone；改 LCLGP 输入维度重训
  - **决策点**：W9 后期视时间富余决定是否做
  - **工时**：1.5 d

---

## W10 — CALVIN + openpi + 论文（7 d）

### 10.1 CALVIN 评测

- [ ] **T-W10.1.1** CALVIN 环境安装与冒烟
  - **依赖**：T-W1.2.5
  - **要点**：starVLA `examples/calvin/` 已有 baseline；先跑通他们的 eval

- [ ] **T-W10.1.2** PaV-Full 在 CALVIN ABC-D
  - **要点**：1000 task sequences
  - **资源**：8×H200 一夜
  - **关注指标**：average length（ABC-D 长程能力的核心指标）
  - **输出**：`paper/tables/calvin_abcd.csv`

- [ ] ⚪ **T-W10.1.3** CALVIN 消融（如时间允许）
  - **决策点**：W10 中段评估剩余时间

### 10.2 ★ openpi 补充表

- [ ] **T-W10.2.1** 拉起 openpi serve_policy
  - **要点**：openpi 自带 `serve_policy.py`；docker 起一个，端口 8000
  - **资源**：1×H100 跑 server
  - **工时**：3 h

- [ ] **T-W10.2.2** 写 PyTorch 端 `Pi0HTTPClient`
  - **入口**：`starVLA/clients/pi0_client.py`
  - **要点**：HTTP/grpc 调 openpi；接口对齐 StarVLA-PI
  - **工时**：4 h

- [ ] **T-W10.2.3** PaV-Full 用 openpi 原版 π₀ 在 LIBERO-Long
  - **依赖**：T-W10.2.1, T-W10.2.2
  - **要点**：5 任务 × 30 trials = 150 trials；验证趋势一致
  - **🚧 关注**：PaV 增益符号一致（即 M3 > B0），数值可有差异
  - **资源**：1×H100，~12 h（含 HTTP RTT）
  - **输出**：`paper/tables/openpi_supplementary.csv`

### 10.3 §9.6 诊断图（论文图表）

- [ ] **T-W10.3.1** 5 个失败 + 5 个 recovery 案例时间轴
  - **要点**：每 case 一张图：x 轴时间，左 y 轴 E(t)，右 y 轴 p_t；ETAR 决策标记；下方关键帧 thumbnails
  - **数据来源**：T-W5.1.3 的 logs
  - **输出**：`paper/figures/case_study_*.pdf`
  - **工时**：1 d

- [ ] **T-W10.3.2** 计算开销饼图
  - **依赖**：T-W7.5.2
  - **要点**：π₀ 主回路 / V-JEPA 编码 / predictor rollout / LCLGP / ETAR 决策的时间占比
  - **输出**：`paper/figures/latency_pie.pdf`

- [ ] **T-W10.3.3** LCLGP 解码可视化
  - **要点**：从 T-W3.4.3 选 6 个任务，每任务 3 张图（gt end-frame / decoded best-mode / diff heatmap）
  - **输出**：`paper/figures/lclgp_decode_grid.pdf`

- [ ] **T-W10.3.4** ETAR 混淆矩阵 + ROC
  - **依赖**：T-W7.2.3
  - **输出**：`paper/figures/etar_confusion.pdf`

### 10.4 论文写作

- [ ] **T-W10.4.1** Abstract + Introduction（§1）
  - **要点**：参考设计文档 §1；强调 plug-and-play + verifier vs generator
  - **工时**：1 d

- [ ] **T-W10.4.2** Related Work（§2）
  - **要点**：4 类 + ★ WM4A 单独段落对比
  - **工时**：1 d

- [ ] **T-W10.4.3** Method（§3）
  - **要点**：参考设计文档 §3-7
  - **图**：架构图（设计文档 §4.1）
  - **工时**：1.5 d

- [ ] **T-W10.4.4** Experiments（§4）
  - **要点**：主表 + Perturb + 消融 + 诊断
  - **工时**：1 d

- [ ] **T-W10.4.5** Discussion + Conclusion（§5-6）
  - **要点**：失败模式分析；future work（DPO、真机）
  - **工时**：0.5 d

### 10.5 演示视频

- [ ] **T-W10.5.1** 制作 3 分钟 demo 视频
  - **内容**：架构图动画 + 失败案例对比（B0 失败 / M3 recovery）+ 诊断图 overlay
  - **工具**：ffmpeg + manim
  - **输出**：`paper/figures/demo_video.mp4`
  - **工时**：1.5 d

### 10.6 代码归档

- [ ] **T-W10.6.1** README + 安装文档
  - **入口**：`docs/plan_and_verify.md`
  - **要点**：风格对齐 starVLA `docs/WM4A.md`
  - **工时**：4 h

- [ ] **T-W10.6.2** 上传 ckpts 到 HuggingFace
  - **要点**：建一个 collection `<your-name>/plan-and-verify`
  - **工时**：2 h

- [ ] **T-W10.6.3** 准备 PR 上游 starVLA（可选）
  - **决策点**：与导师讨论是否 upstream

---

## 附录 A — 完整代码文件清单

| 文件路径 | 创建于 | 内容 |
|---|---|---|
| `starVLA/model/modules/world_model/vjepa2.py` | T-W1.3.1 | V-JEPA 2 wrapper |
| `starVLA/model/framework/PlanVerify/__init__.py` | T-W3.1.5 | Framework 注册 |
| `starVLA/model/framework/PlanVerify/lclgp.py` | T-W3.1.1 | LCLGP 模型 |
| `starVLA/model/framework/PlanVerify/msfv.py` | T-W4.1.1 | MSFV verifier |
| `starVLA/model/framework/PlanVerify/etar.py` | T-W7.2.1 | ETAR 分类器 |
| `starVLA/model/framework/PlanVerify/runtime.py` | T-W5.1.1 | Runtime |
| `starVLA/model/framework/PlanVerify/baselines/cosmo_verifier.py` | T-W9.3.1 | Cosmos 对照 |
| `starVLA/model/modules/action_model/flow_matching_head/seeded_sample.py` | T-W7.1.2 | noise seed 改造 |
| `starVLA/training/train_lclgp.py` | T-W3.3.1 | LCLGP 训练入口 |
| `starVLA/training/train_etar.py` | T-W7.2.2 | ETAR 训练入口 |
| `starVLA/clients/pi0_client.py` | T-W10.2.2 | openpi HTTP client |
| `starVLA/datasets/vjepa_latent_dataset.py` | T-W2.2.4 | Latent dataloader |
| `starVLA/evaluation/pav_perturb/p1-5_*.py` | T-W9.1.1～5 | Perturb 协议 |
| `scripts/extract_vjepa_latents.py` | T-W2.2.1 | Latent 抽取 |
| `scripts/build_lclgp_dataset.py` | T-W2.3.1 | LCLGP 数据构造 |
| `scripts/collect_etar_rollouts.py` | T-W6.1.1 | ETAR rollout |
| `scripts/label_etar.py` | T-W6.2.1 | ETAR 标注 |
| `scripts/run_diagnostics.py` | T-W3.4.1 | LCLGP 诊断 |
| `scripts/eval_main.py` | T-W8.3.1 | 主实验入口 |
| `scripts/eval_perturb.py` | T-W9.1.6 | Perturb 评测 |
| `scripts/eval_calvin.py` | T-W10.1.2 | CALVIN 评测 |
| `scripts/eval_openpi_supplementary.py` | T-W10.2.3 | openpi 补充表 |

---

## 附录 B — 配置文件清单

| 配置 | 路径 | 关键参数 |
|---|---|---|
| LCLGP 训练 | `examples/PlanAndVerify/configs/lclgp_v1.yaml` | K=4, batch=256, demos_per_task=8, epochs=30 |
| ETAR 训练 | `examples/PlanAndVerify/configs/etar_v1.yaml` | hidden=64, n_classes=4 |
| LIBERO-Long eval | `examples/PlanAndVerify/configs/eval_libero_long.yaml` | trials=30, chunks_per_replan=1 |
| CALVIN eval | `examples/PlanAndVerify/configs/eval_calvin_abcd.yaml` | sequences=1000 |
| Perturb eval | `examples/PlanAndVerify/configs/eval_perturb_p{1-5}.yaml` | trials=30 per perturb |
| openpi 补充 | `examples/PlanAndVerify/configs/eval_openpi_supp.yaml` | trials=30 × 5 task subset |

---

## 附录 C — 论文章节 ↔ 实验/代码 对应表

| 论文章节 | 对应代码 | 对应实验 | 来源 TODO |
|---|---|---|---|
| §1 Intro | — | — | T-W10.4.1 |
| §2 Related Work | — | WM4A 对比段落 | T-W10.4.2 |
| §3 Problem | — | — | T-W10.4.3 |
| §4 Framework | 全部 framework 代码 | — | T-W10.4.3 |
| §5 LCLGP | `lclgp.py`, `train_lclgp.py` | D1-D4 诊断 | T-W3.* |
| §6 MSFV | `msfv.py` | Verifier signal AUC | T-W4.*, T-W5.3.3 |
| §7 ETAR | `etar.py`, `train_etar.py` | 混淆矩阵 + ROC | T-W7.* |
| §8 Training Recipe | 训练脚本 + configs | — | 各 train_*.py |
| §9.2 LIBERO 主表 | `eval_main.py` | B0-B6 + M1-M3 | T-W8.* |
| §9.3 Robustness | `eval_perturb.py` | P1-P5 | T-W9.1.* |
| §9.4 Ablation | `eval_main.py` 多次 | 9 项消融 | T-W9.2.*, T-W9.3 |
| §9.4 Cosmos 对照 ★ | `cosmo_verifier.py` | verifier 选型 | T-W9.3.* |
| §9.5 CALVIN | `eval_calvin.py` | ABC-D | T-W10.1.* |
| §9.6 Diagnostics | `run_diagnostics.py` | case study + 延迟 | T-W10.3.* |
| §9.7 openpi 补充 ★ | `eval_openpi_supp.py` | 趋势一致性 | T-W10.2.* |
| §10 Discussion | — | — | T-W10.4.5 |

---

## 附录 D — 风险登记表（动态更新）

| ID | 风险 | 触发条件 | 缓解 | 负责窗口 |
|---|---|---|---|---|
| R1 | LCLGP 多模态坍缩 | mode_balance.std > 0.5 | 增大 λ_bal、加 mode 间斥力损失 | W3 |
| R2 | V-JEPA 表征不敏感 | 成功 demo Spearman > -0.3 | 加 SigLIP cross-check verifier | W4 |
| R3 | GPT-4V 一致性低 | κ < 0.6 | 退到 rule + 200 chunks 人工 | W6 |
| R4 | RESAMPLE 误触率高 | 成功 traj 上 > 10% | 调 ETAR 阈值，加 EMA | W7 |
| R5 | π₀ 在 LIBERO-Long 已饱和 | B0 > 90% | 重心移到 Perturb（未饱和） | W8 |
| R6 | CALVIN 集成超时 | W10 中段未跑通 | 砍 CALVIN，主结果靠 LIBERO + Perturb | W10 |
| R7 | openpi 补充表 RTT 太大 | HTTP > 50 ms RTT | 改 unix socket 或砍此表 | W10 |
| R8 | starVLA 上游 breaking change | rebase 冲突 > 1d | 锁 commit hash，不 rebase | 全程 |
| R9 | Cosmos verifier 维度不匹配 | LCLGP retrain 不收敛 | 砍 Cosmos 对照，仅文字论证 | W9 |
| R10 | 磁盘不足 | latent 缓存 > 1.5 TB | 降 patch_grid=8 或流式 | W2 |

---

## 附录 E — 数据/Checkpoint 清单

### 数据集
| 名称 | 来源 | 大小 | 用途 |
|---|---|---|---|
| LIBERO（4 suite）| starVLA data_preparation | ~80 GB | LCLGP 训练 + 评测 |
| Bridge-v2（subset 50K）| HF / OXE | ~200 GB | LCLGP 训练扩展 |
| V-JEPA latent 缓存 | T-W2.2.3 | ~1.05 TB | LCLGP 训练加速 |
| LCLGP 三元组 | T-W2.3.1 | ~10 GB | LCLGP 训练 |
| ETAR rollout | T-W6.1.2 | ~50 GB | ETAR 训练 |
| ETAR labeled | T-W6.4.2 | ~50 MB | ETAR 训练 |

### Checkpoints
| 名称 | 来源 | 大小 | 用途 |
|---|---|---|---|
| StarVLA-PI LIBERO | HF `StarVLA/bench-libero` | ~5 GB | B0 baseline |
| V-JEPA 2 ViT-g | HF facebookresearch/vjepa2 | ~4 GB | encoder（frozen）|
| V-JEPA 2-AC predictor | HF facebookresearch/vjepa2 | ~1.2 GB | predictor（frozen）|
| WM4A CosmoPredict2-OFT | HF `StarVLA/world-model-to-vla` | ~10 GB | B6 对照 |
| LCLGP best | T-W3.3.4 | ~400 MB | 训练产出 |
| ETAR best | T-W7.2.3 | ~50 KB | 训练产出 |
| LCLGP-Cosmo（ablation）| T-W9.3.2 | ~400 MB | 训练产出 |

---

## 附录 F — 评测命令速查

```bash
# Stage 1: 数据
python scripts/extract_vjepa_latents.py --config configs/extract.yaml
python scripts/build_lclgp_dataset.py --config configs/lclgp_v1.yaml

# Stage 2: 训练
python -m starVLA.training.train_lclgp --config examples/PlanAndVerify/configs/lclgp_v1.yaml
python scripts/run_diagnostics.py --ckpt ckpts/lclgp/best.pt        # G-W3 gate

python scripts/collect_etar_rollouts.py --num_trajs 500 --policy starvla_pi_libero
python scripts/label_etar.py --input data/etar_rollouts/raw_500.h5 --use_gpt4v
python -m starVLA.training.train_etar --config examples/PlanAndVerify/configs/etar_v1.yaml

# Stage 3: 评测
python scripts/eval_main.py --runtime PaV-Full --suite libero_long
python scripts/eval_perturb.py --runtime PaV-Full --perturb p1
python scripts/eval_calvin.py --runtime PaV-Full
python scripts/eval_openpi_supplementary.py --runtime PaV-Full

# Stage 4: 论文
python scripts/run_diagnostics.py --produce_figures --output paper/figures/
```

---

## 工时总览

| 周 | 主要工作 | 预计工时 | 等待时间 |
|---|---|---|---|
| W1 | 环境 + baseline | ~25 h | 6 h（baseline eval）|
| W2 | 数据 pipeline | ~20 h | 10 h（latent 编码）|
| W3 | LCLGP 训练 + 诊断 | ~30 h | 20 h（训练）|
| W4 | MSFV | ~20 h | 0 |
| W5 | Runtime + PaV-Lite | ~22 h | 8 h（demo eval）|
| W6 | ETAR 数据 | ~25 h | 12 h（rollout + GPT-4V）|
| W7 | ETAR 训练 + CEM | ~22 h | 1 h |
| W8 | LIBERO 主实验 | ~20 h | 一夜 8×H200 |
| W9 | Perturb + 消融 | ~30 h | 一夜 8×H200 + 24 h LCLGP-Cosmo |
| W10 | CALVIN + openpi + 论文 | ~50 h | 一夜 8×H200 |
| **合计** | | **~265 h** | + 多次借卡过夜 |

**人/小时**：265 h ≈ 33 工作日（按 8h/日），10 周窗口（70 工作日）冗余约 1.1 倍——**可吸收 ~1 周的意外延误**，但任何 Gate 全 fail 都会击穿冗余。

---

**文档状态**：v1.0（W0 起步配套），将在每周 sprint 末更新依赖树与风险表。
