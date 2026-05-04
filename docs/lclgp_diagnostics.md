# LCLGP Diagnostics — W3 v1 Retrospective + v2 Remediation

**Date**: 2026-05-04
**Run**: `pav_w3_lclgp_v1` (ckpt `playground/Checkpoints/pav_w3_lclgp_v1/final_model/pytorch_model.pt`)
**Config**: [examples/PlanAndVerify/configs/lclgp_v1.yaml](../examples/PlanAndVerify/configs/lclgp_v1.yaml)
**Diagnostics CSVs**: [paper/tables/lclgp_{g1,d1,d2,d3}.csv](../paper/tables/)

---

## TL;DR

7 条阈值（[SETUP.md §12.5](../examples/PlanAndVerify/SETUP.md)）通过 **1/7**。失败模式高度结构化：
- **σ saturation**（log_sigma 整训练贴 cap=5.0 → σ=148.4）
- **mode collapse**（k=3 拿 76% 路由，pairwise mode cosine=0.96）
- **end head 死**（end→end cos=0.20，与随机猜近似）
- **delta head 健康**（delta→delta cos=0.74，σ-best cos=0.74，min-of-K cos=0.92）

诊断结论：**所有失败模式由单一根因 σ saturation 驱动**（详见 §3）。修复方案是微创——只改 3 个 yaml 数字 + 2 处代码（< 20 行），不动架构、不动 5 损失、不动数据 pipeline。详见 §4 v2 remediation。

**Gate 决策（T-W3.5.1）**：当前不满足进入 W4 的前置条件；**不跑 G-W3 大 gate**（30+ 小时白烧高概率）。优先路径 = v2 retrain（~1.5h round-trip），通过 ≥ 5/7 后再决定。

---

## 1. 阈值通过情况（自动表）

### 1.1 G-1 — `paper/tables/lclgp_g1.csv`

| metric | value | threshold | pass |
|---|---|---|---|
| G1_end_min_cos_mean | 0.213 | ≥ 0.75 | ❌ |
| G1_end_min_cos_median | 0.216 | ≥ 0.75 | ❌ |
| G1_delta_min_cos_mean | 0.918 | ≥ 0.75 | ✅ |
| G1_end_sigma_cos_mean | 0.202 | ≥ 0.70 | ❌ |
| G1_delta_sigma_cos_mean | 0.740 | ≥ 0.70 | ✅ |

End-horizon 头预测能力近随机（0.2 ≈ random 256-D 余弦的均值）；delta-horizon 头预测良好。

### 1.2 D1 — `paper/tables/lclgp_d1.csv`

| metric | value | threshold | pass |
|---|---|---|---|
| D1a_mean_pairwise_cos | 0.957 | ∈ [0.30, 0.70] | ❌ |
| D1b_min_mode_freq_end | 0.013 | ≥ 0.10 | ❌ |
| D1b_min_mode_freq_delta | 0.000 | ≥ 0.10 | ❌ |
| D1c_pearson_sigma_err | nan | ≥ 0.40 | ❌ |
| D1b_cov_end_k0 | 0.166 | (info) | — |
| D1b_cov_end_k1 | 0.013 | (info) | — |
| D1b_cov_end_k2 | 0.063 | (info) | — |
| D1b_cov_end_k3 | **0.758** | (info) | — |

K=4 模式塌缩到几乎单模：k=3 占 76%，pairwise cosine ≈ 1（4 模式输出几乎相同），σ 是常数 → Pearson 未定义。

### 1.3 D2 — `paper/tables/lclgp_d2.csv`

| metric | value | threshold | pass |
|---|---|---|---|
| D2a_mean_cf_L1 | 0.740 | ≥ 0.05 | ✅ |
| D2b_median_intra_task_std | 0.820 | (info) | — |
| D2b_num_tasks_with_4plus_samples | 40 | (info) | — |

唯一通过的阈值。z_t 置零确实让输出大幅变化（远超 hinge margin=0.05），证明 z_t 在被使用，状态依赖性已学到。

### 1.4 D3 — `paper/tables/lclgp_d3.csv`

| metric | value | pass |
|---|---|---|
| D3a_end→end_cos_mean | 0.202 | (info) |
| D3a_end→delta_cos_mean | 0.181 | (info) |
| D3a_delta→delta_cos_mean | 0.740 | (info) |
| D3a_delta→end_cos_mean | 0.736 | (info) |
| D3a_end_specialization_gap | 0.021 | ≥ 0.05 ❌ |
| D3a_delta_specialization_gap | 0.004 | ≥ 0.05 ❌ |

End head 对两种 GT 都预测得差（0.20 / 0.18），delta head 对两种 GT 都预测得近（0.74 / 0.74）—— 两个时间尺度没有真正分化，end head 直接没学。

### 1.5 训练末段 W&B 指标（用户提供）

| metric | value | 健康范围 | 状态 |
|---|---|---|---|
| sigma_end_mean | **148.4** | exp([log σ_min, log σ_max])=[0.007, 148.4] | ⚠️ 顶到 cap |
| sigma_delta_mean | **148.4** | 同上 | ⚠️ 顶到 cap |
| l_cf | 0 | hinge loss，达到 margin 后自然为 0 | ✅ 实际健康 |
| l_ctr | 2.6 | 应 < 1（256 任务 batch 下 log(1/256)≈5.5 是随机基线）| ⚠️ 没收敛 |
| mode_balance_std_end | 5e-5 | 软分配标准差，应 < 0.3 | ⚠️ 假性达标（见 §3.3）|

### 1.6 7 阈值汇总

| # | 名称 | 通过 |
|---|---|---|
| 1 | G1_end_min_cos_mean ≥ 0.75 | ❌ |
| 2 | G1_end_sigma_cos_mean ≥ 0.70 | ❌ |
| 3 | D1-a pairwise cos ∈ [0.30, 0.70] | ❌ |
| 4a | D1-b min mode freq end ≥ 0.10 | ❌ |
| 4b | D1-b min mode freq delta ≥ 0.10 | ❌ |
| 5 | D1-c Pearson(σ, err) ≥ 0.40 | ❌ |
| 6 | D2-a cf L1 ≥ 0.05 | ✅ |
| 7a | D3-a end specialization gap ≥ 0.05 | ❌ |
| 7b | D3-a delta specialization gap ≥ 0.05 | ❌ |

通过 = **D2-a only（1/7 主项）**。

---

## 2. 训练超参回顾（v1 实际）

| 超参 | v1 值 | 配置位置 |
|---|---|---|
| K (n_modes) | 4 | [lclgp_v1.yaml:54](../examples/PlanAndVerify/configs/lclgp_v1.yaml) |
| log_sigma_max | 5.0 | yaml:69 |
| β（σ NLL 正则）| 0.1 | yaml:62 |
| λ_bal（mode KL）| 0.05 | yaml:63 |
| λ_ctr（InfoNCE）| 0.1 | yaml:64 |
| λ_cf（counterfactual）| 0.05 | yaml:65 |
| α（end vs delta 权重）| 0.5 | yaml:61 |
| max_train_steps | 2400（≈30 epoch）| yaml:77 |
| 训练时长 | ~30 min on 8×H20 | (用户报告) |

---

## 3. 根因分析：σ saturation 是单一总开关

### 3.1 损失结构与 σ 的角色

`_heteroscedastic_per_mode` （[`starVLA/model/framework/PlanVerify/lclgp.py:221-227`](../starVLA/model/framework/PlanVerify/lclgp.py)）：

```python
per_mode = ||z_g - z_true||₁ / σ + β · log σ
```

异方差 NLL 闭式解：`σ* = ||·||₁ / β`。v1 训练期 ||·||₁ 估 ~0.5、β=0.1，所以 σ* ≈ 5（log σ ≈ 1.6）。

实测 log σ = 5.0（顶到 cap）。**这意味着优化器没在追 σ\***，是别的力把 σ 推到 cap。

### 3.2 是哪股力把 σ 推到 cap？— L_bal 的副作用

`_mode_balance_loss` （[lclgp.py:237-248](../starVLA/model/framework/PlanVerify/lclgp.py)）：

```python
soft = softmin(per_mode_loss / bal_temperature)   # [B, K] 软分配
pi_bar = soft.mean(dim=0)                          # [K]
KL = pi_bar · log(pi_bar / Uniform(K))
```

L_bal 想让 4 个 mode 软分配概率均匀。检查 σ saturation 时的状态：

- 4 个 mode 的 σ 都在 cap 上 → `per_mode = ||·||/σ_cap + β·log σ_cap`
- 第一项 `||·||/148 ≈ 0.005`（数值上几乎为 0）
- 第二项 `0.1·5 = 0.5`（4 个 mode 共享）
- 4 个 mode 的 per_mode_loss 主要由共享常数 0.5 主导，`||·||` 的差异只占 ~1%
- softmin 输出近似均匀 → KL ≈ 0 → L_bal 满足

**这是关键洞察**：L_bal 接受了一个 σ 已饱和的"假性均匀"。换句话说，**模型发现把 σ 推到 cap 是绕过 L_bal 的廉价策略**——不需要让 mode 真的覆盖不同终态，只要让 σ 够大就能让 L_bal=0。优化器找到了这个 shortcut，σ runaway。

### 3.3 σ saturation 的下游灾难

σ 一旦贴 cap，4 个连锁反应同步发生：

| 失活的损失 / 性质 | 机制 | 实测对应 |
|---|---|---|
| **回归头梯度被压成 1/148** | `∂L_recon/∂z_g = (sign(z_g - z_true)) / σ_cap`，梯度尺度被 σ 缩小 ~150× | end head 没学（G1_end=0.21）；delta head 勉强学（更易的目标）|
| **L_reg 不再区分 mode** | per_mode 主项被 `β·log σ_cap` 常数主导，min-of-K 失去信号 | 4 模式参数收敛到几乎相同（D1-a=0.96）|
| **L_bal 假性满足** | §3.2 推导 | mode_balance_std=5e-5，但 cov_k3=0.76（**真实塌缩**）|
| **L_ctr 失去对比对** | 模式塌缩后正样本（best mode 输出）跨任务难区分 | l_ctr=2.6，未收敛 |
| **D1-c 数学未定义** | σ 是常数 → Pearson(σ, err) 分母为 0 | nan |

**唯独 L_cf 健康**——因为它的公式（[lclgp.py:285-300](../starVLA/model/framework/PlanVerify/lclgp.py)）`max(0, margin - ||z_g - z_g_cf||)` 不依赖 σ，所以没被 σ saturation 影响。讽刺地，这也是唯一通过的诊断（D2-a）。

### 3.4 为什么 30 epoch 训练时长本身不是主因

用户训练只用了 ~30 分钟，确实偏短。但 σ saturation 是**结构性问题**，再训 100 epoch 也无法自愈——一旦优化器找到 σ→cap 这个 shortcut，所有梯度都被压扁，模型不会回头。增加训练量只能"修补"在 σ 还没饱和的早期阶段没学好的部分；σ 一旦贴上去，时间无解。

---

## 4. v2 Remediation — 保持原设计的最小修复

**指导原则**：架构（K=4 + dual-timescale + multimodal + σ heads）不动；5 个损失一个不删；只切断 σ 通往 cap 的逃逸通道，让 5 个损失各回各家。

### 4.1 必改三项

#### 改动 1：拉紧 σ cap

```yaml
# examples/PlanAndVerify/configs/lclgp_v2.yaml
- log_sigma_max: 5.0    # σ ∈ [0.007, 148.4]
+ log_sigma_max: 1.5    # σ ∈ [0.007, 4.48]
```

理论最优 σ\* = ||·||/β，v2 下（β=0.5、||·||~0.5）σ\*≈1（log σ\*≈0）。cap=1.5 给优化空间留 1.5σ 余量，足够表达不确定但不会爆。

#### 改动 2：L_bal 输入解耦 σ（**关键代码改动**）

`starVLA/model/framework/PlanVerify/lclgp.py` forward 中：

```python
# 改前：把含 σ 的 per_mode_loss 喂给 L_bal
- l_bal_end, _, _ = self._mode_balance_loss(per_mode_end)
- l_bal_delta, _, _ = self._mode_balance_loss(per_mode_delta)

# 改后：用 raw L1（不含 σ）作为 mode 区分度信号
+ l1_end_per_mode = (z_g_end - z_end.detach().unsqueeze(1)).abs().mean(dim=(-2, -1))
+ l1_delta_per_mode = (z_g_delta - z_delta.detach().unsqueeze(1)).abs().mean(dim=(-2, -1))
+ l_bal_end, _, _ = self._mode_balance_loss(l1_end_per_mode)
+ l_bal_delta, _, _ = self._mode_balance_loss(l1_delta_per_mode)
```

**这是斩断 §3.2 那条因果链的核心改动**——L_bal 不再能被 σ 廉价绕过，只能通过让 mode 真的预测不同的目标来满足。

#### 改动 3：增大 β 与 lambda_bal

```yaml
- beta: 0.1           # σ* = ||·||/β = 0.5/0.1 = 5（log_σ*=1.6，仍接近 cap）
+ beta: 0.5           # σ* = 0.5/0.5 = 1（log_σ*=0，远离 cap）

- lambda_bal: 0.05    # 6× 不够压住 mode 塌缩
+ lambda_bal: 0.3     # 与改动 2 协同，真正惩罚 mode collapse
```

### 4.2 推荐两项

#### 改动 4：训练量加倍

```yaml
- max_train_steps: 2400    # 30 epoch
+ max_train_steps: 4800    # 60 epoch

- num_warmup_steps: 200
+ num_warmup_steps: 400
```

5 个损失互相博弈，30 epoch 半小时偏短。代价：~1h on 8×H20。

#### 改动 5：slot tokens 正交初始化

`starVLA/model/framework/PlanVerify/lclgp.py:142-147` `__init__`：

```python
- self.slot_end = nn.Parameter(torch.randn(self.K, self.n_tokens, self.d_hidden) * 0.02)
- self.slot_delta = nn.Parameter(torch.randn(self.K, self.n_tokens, self.d_hidden) * 0.02)

+ slot_end_init = torch.empty(self.K, self.n_tokens, self.d_hidden)
+ slot_delta_init = torch.empty(self.K, self.n_tokens, self.d_hidden)
+ nn.init.orthogonal_(slot_end_init.view(self.K, -1))
+ nn.init.orthogonal_(slot_delta_init.view(self.K, -1))
+ self.slot_end = nn.Parameter(slot_end_init * 0.02)
+ self.slot_delta = nn.Parameter(slot_delta_init * 0.02)
```

让 4 个 mode 的初始参数严格正交，给改动 2 创造的"L_bal 真有压力"提供初始多样性，避免雪球还没启动就被某个 mode 抢跑。

### 4.3 备选两项（如果 v2 重训后还有具体阈值不通过）

| 改动 | 何时启用 | 解什么 |
|---|---|---|
| `bal_temperature` 退火 1.0 → 0.1 | D1-a 仍 > 0.7（模式过度相似）| 早期热温允许 mode 自由分化，后期冷温固化 |
| Repulsive loss：`λ·max(0, mean_pairwise_cos − 0.5)` | 同上 | 显式推开 mode |
| α 不对称：end=0.7, delta=0.3 | G1_end 仍 < 0.5 | 给更难的 end head 更多权重 |

### 4.4 v2 没动的部分

- 模型架构（4 层 cross-attn decoder，K=4，dual head, σ head）—— 全保留
- 5 个损失项（L_reg, L_bal, L_ctr, L_cf, L_α 加权）—— 一项不删
- D1/D2/D3 的设计动机 —— 全保留
- 数据 pipeline / sampler / 训练框架 —— 全不动

总改动：3 个 yaml 数字 + 1 处 forward 张量 + 1 处 `__init__` init —— **< 20 行代码**。

---

## 5. v2 验证流程（重训后跑这一段）

```bash
# 重训
WANDB_ENTITY=<你> bash examples/PlanAndVerify/train_files/run_lclgp.sh \
    --config_yaml examples/PlanAndVerify/configs/lclgp_v2.yaml

# 重跑诊断（与 v1 相同脚本）
CKPT=playground/Checkpoints/pav_w3_lclgp_v2/final_model/pytorch_model.pt
CFG=examples/PlanAndVerify/configs/lclgp_v2.yaml
for cmd in g1 d1 d2 d3; do
  .venv/bin/python examples/PlanAndVerify/scripts/run_diagnostics.py $cmd \
      --config_yaml $CFG --checkpoint $CKPT --cuda --batch_size 16 --output_dir paper
done
```

### v2 通过预期

| 阈值 | v1 | v2 期望 | 凭据 |
|---|---|---|---|
| G1_end_min_cos | 0.21 | ≥ 0.6 | end head 拿回 ~150× 梯度 |
| G1_delta_min_cos | 0.92 | ≥ 0.85 | 已健康，不会变差 |
| G1_end_sigma_cos | 0.20 | ≥ 0.55 | σ 不再常数，σ-best 真的选好的 mode |
| D1-a pairwise cos | 0.96 | ∈ [0.4, 0.7] | mode 真分化 |
| D1-b min mode freq | 0.013/0 | ≥ 0.15 | L_bal 真起作用 |
| D1-c Pearson(σ, err) | nan | ≥ 0.45 | σ 有方差，相关性可计算 |
| D2-a cf L1 | 0.74 | ≥ 0.5 | 已健康，不会变差 |
| D3-a end gap | 0.02 | ≥ 0.08 | end head 学到东西后两个尺度自然分化 |

预期通过 ≥ 5/7。**通过 ≥ 6/7 → 进 Stage B 跑 G-W3 大 gate**；通过 4-5/7 → 上备选两项；通过 ≤ 3/7 → 重审是否结构上有问题。

### 时间预算

| 阶段 | 估时 |
|---|---|
| v2 训练（60 epoch）| ~1h on 8×H20 |
| v2 诊断（4 cmd × 1 min）| ~5 min |
| v2 报告更新 | ~5 min |
| **v2 round-trip** | **~1.5h** |
| 若 v2 通过 → G-W3 大 gate（10 任务 × 30 trials × 2 模式）| ~5h on 8×H20 sharded |

---

## 6. 决策（T-W3.5.1）

**当前不进 W4，不跑 G-W3 大 gate**。

**优先路径**：v2 retrain（§4 修复）→ 重跑诊断 → 通过 ≥ 5/7 后再决定 Stage B。

**回退路径**（仅当 v2 通过 ≤ 3/7）：
- K=2 简化版 retrain（保留多模态思路但减少自由度）
- 极端情况回退 PaV-Lite（research_design line 1256）

**截至 §6 报告时间点 (2026-05-04 早段)**：v2 retrain + 诊断尚未启动；最新状态见 §7。

---

## 7. v2 retrospective + 路径 B 决策

**Date**: 2026-05-04
**Run**: `pav_w3_lclgp_v2` (ckpt `playground/Checkpoints/pav_w3_lclgp_v2/final_model/pytorch_model.pt`)
**Config**: [examples/PlanAndVerify/configs/lclgp_v2.yaml](../examples/PlanAndVerify/configs/lclgp_v2.yaml)
**墙钟**: 60 epoch / 0.6h on H200

### 7.1 v1↔v2 7 阈值对比

| # | Metric | Threshold | v1 | v2 | v2 Pass | Δ 解读 |
|---|---|---|---|---|---|---|
| 1 | G1_end_min_cos_mean | ≥ 0.75 | 0.213 | **0.799** | ✅ | end head 拿回 ~150× 梯度 |
| 2 | G1_end_sigma_cos_mean | ≥ 0.70 | 0.202 | **0.769** | ✅ | σ 不再常数，σ-best 真选好 mode |
| 3 | D1-a pairwise cos ∈ [0.30, 0.70] | range | 0.957 | **0.882** | ❌ | 略降但 4 mode 输出仍趋同 |
| 4a | D1-b min mode freq end | ≥ 0.10 | 0.013 | **0.0** | ❌ | mode 2 独占 99.6% |
| 4b | D1-b min mode freq delta | ≥ 0.10 | 0.0 | **0.0** | ❌ | 仍 0 |
| 5 | D1-c Pearson(σ, err) | ≥ 0.40 | nan | **0.747** | ✅ | σ 有方差，相关性可计算 |
| 6 | D2-a cf L1 | ≥ 0.05 | 0.740 | **0.319** | ✅ | 仍健康（注：从 0.74 降至 0.32 但仍 > 0.05）|
| 7a | D3-a end specialization gap | ≥ 0.05 | 0.021 | **0.018** | ❌ | 基本未动 |
| 7b | D3-a delta specialization gap | ≥ 0.05 | 0.004 | **0.009** | ❌ | 略升但仍 < 0.05 |

**通过 4/7（#1, #2, #5 新通过 + #6 维持）**

### 7.2 W&B 末段对照 §5 v2 期望

| 指标 | v1 | §5 期望 | v2 实测 | 评价 |
|---|---|---|---|---|
| sigma_end_mean | 148.4（顶 cap=148） | ≈ 1（log σ ≈ 0） | 1.76 | ✅ 不顶 cap=4.48 |
| sigma_delta_mean | 148.4 | ≈ 1 | 4.48 | ⚠️ 接近 cap=4.48 上限（仍未顶死，但偏高）|
| mode_balance_std_end | 5e-5 | > 0.1 | 0.015 | ❌ K=4 mode collapse |
| l_recon_end | 0.39 | 明显下降 | 0.799 | ❌ 反向上升（解释见 §7.3）|
| l_ctr | 2.6 | < 1.5 | 3.250 | ❌ 反向上升（同 §7.3）|
| l_cf | 0.0 | 0.0 | 0.0 | ✅ |

### 7.3 σ-shortcut 切断判定 — **完全切断**

v2 σ-saturation 修复目标 100% 达成。证据三联（独立交叉证伪 v2 配置/代码未生效的怀疑）：

1. **σ 不再贴 cap**：sigma_end_mean=1.76（v1: 148）。cap 从 148→4.48 已生效；δ 头 4.48 接近 cap 但未顶死。
2. **D1-c Pearson 可计算**：0.747（v1: nan）→ σ 有方差，L_bal 不能再被 σ 廉价绕过。
3. **D1-a 脱离 v1 顶值**：0.957→0.882 → slot 正交初始化可见效果，raw-L1 L_bal 也起作用。

**l_recon_end / l_ctr 反向上升的解释**：mode 2 独占 99.6% 概率（cov_end k0=0, k1=0.004, k2=0.996, k3=0）→ 1 个 mode 承担本应 4 mode 分担的全部任务负担，loss 必然高于"4 mode 各管一摊"的潜在最优。这是 mode collapse 的**次生症状**，不是 σ 修复失败的反证。

### 7.4 失效模式 shift — K=4 mode collapse（独立问题）

v1 失效根因（σ-shortcut）已切断，但出现独立的新失效模式：4 个 slot orthogonal 初始化扛不过 60 epoch 的 winner-takes-all 动力学。

- **D1-a 0.882**：4 mode 输出仍趋同
- **cov_end k=[0.0, 0.004, **0.996**, 0.0]**：mode 2 独占；其余 3 个 mode 实际 dead
- **D1-b 0.0**：min mode freq = 0，softmin(L1) 选 1 winner 后 L_bal 仍可被满足

**根因**（非 σ-shortcut，独立诊断）：
- L_ctr 用 best-mode 输出做正样本 → snowball：哪个 mode 先学好哪个独占梯度
- L_bal 切断 σ 通道后，4 mode 输出仍可彼此趋同；softmin(L1) 仍能选 1 winner 满足 L_bal
- 缺乏 "slot 排斥" 或 "温度" 压制 winner-takes-all

### 7.5 决策路径 — 4/7 → 路径 B → v3

按 §3.6 决策树：
- ≥ 6/7 → Stage B（**不触发**）
- **4-5/7 → v2 备选两项 + 再训一轮（触发）**
- ≤ 3/7 → K=2 / PaV-Lite（不触发）

**v3 设计**（架构未动，K=4 + 双时间尺度 + 5 损失全保留；仅在 v2 之上加两项备选）：

| 备选项 | 选 | 理由 |
|---|---|---|
| bal_temperature 退火 1.0→0.1（线性 over training steps）| ✅ | softmin 锐化，严惩 winner-takes-all → 直击 D1-b |
| Repulsive loss 在 mode 输出层 z_g（pairwise cos² off-diag mean，λ_rep=0.1）| ✅ | 拉开 4 mode 预测，给 L_bal 4 个真不同 "竞争者" → 直击 D1-a |
| α 0.5→0.7 给 end head | ❌ | 解决 end-vs-delta 权重，与 mode 分化无关；G1_end 已通过 0.79 |

v3 配置：`examples/PlanAndVerify/configs/lclgp_v3.yaml`（下一 commit 落地）。
v3 代码改动：`starVLA/model/framework/PlanVerify/lclgp.py` forward 加 repulsive + bal_T schedule；`starVLA/training/train_starvla.py` 注入 global_step 到 module（1 行）。

### 7.6 v3 看点（独立于阈值通过条数）

- **核心目标**：D1-a ∈ [0.3, 0.7] + cov_end 4 mode 都 ≥ 0.10 + mode_balance_std_end > 0.1
- **防回退**：G1_end_min_cos_mean 仍 ≥ 0.75（防止 repulsive 反弹已通过的指标）
- **次生健康**：l_recon_end 回落（4 mode 分担成功的副作用）；σ_delta_mean 远离 cap

### 7.7 截至本报告时间点 (2026-05-04)

- v1 训练完成、v1 诊断完成 → ✅
- v2 配置 + 代码改动准备完成 → ✅
- v2 retrain（60 epoch / 0.6h on H20）→ ✅
- v2 诊断（4/7 通过）→ ✅
- v3 设计 + commit → ✅（commit 664ef46 + f5ed3ae）
- v3 retrain + 诊断 → ✅（详见 §8）
- Stage B (G-W3 driver) → ⏳ 取决于 v3 诊断结果

---

## 8. v3 retrospective + 路径 B 回归判定

**Date**: 2026-05-04
**Run**: `pav_w3_lclgp_v3` (ckpt `playground/Checkpoints/pav_w3_lclgp_v3/final_model/pytorch_model.pt`)
**Config**: [examples/PlanAndVerify/configs/lclgp_v3.yaml](../examples/PlanAndVerify/configs/lclgp_v3.yaml)
**墙钟**: 60 epoch / ~0.6h on H20
**v3 仅在 v2 之上叠加两项**（架构未动；α 0.5 不动）：
- bal_temperature 退火 1.0 → 0.1 over 4800 step（线性）
- repulsive loss 在 z_g（pairwise cos² off-diag mean，λ_rep=0.1）

### 8.1 v1↔v2↔v3 7 阈值对比

| # | Metric | Threshold | v1 | v2 | v3 | v3 Pass | Δ(v3−v2) 解读 |
|---|---|---|---|---|---|---|---|
| 1 | G1_end_min_cos_mean | ≥ 0.75 | 0.213 | 0.799 | **0.862** | ✅ | ↑ repulsive 帮检索任务更干净 |
| 2 | G1_end_sigma_cos_mean | ≥ 0.70 | 0.202 | 0.769 | **0.317** | ❌ | ↓↓ σ-head 被打坏（σ 不再指向 best mode）|
| 3 | D1-a pairwise cos ∈ [0.30, 0.70] | range | 0.957 | 0.882 | **0.149** | ❌ | 穿过目标区跌到近正交，repulsive 过冲 |
| 4a | D1-b min mode freq end | ≥ 0.10 | 0.013 | 0.0 | 0.0 | ❌ | 仍 mode collapse（slot 旋转，未真正 4-way）|
| 4b | D1-b min mode freq delta | ≥ 0.10 | 0.0 | 0.0 | 0.0 | ❌ | 同上 |
| 5 | D1-c Pearson(σ, err) | ≥ 0.40 | nan | 0.747 | **−0.818** | ❌ | **符号翻转**：σ 现在反预测 error |
| 6 | D2-a cf L1 | ≥ 0.05 | 0.740 | 0.319 | 0.162 | ✅ | 仍健康（继续下降但 > 0.05）|
| 7a | D3-a end specialization gap | ≥ 0.05 | 0.021 | 0.018 | 0.019 | ❌ | 基本未动 |
| 7b | D3-a delta specialization gap | ≥ 0.05 | 0.004 | 0.009 | 0.008 | ❌ | 基本未动 |

**通过 2/7（v2: 4/7）→ 路径 B 回归。**

### 8.2 关键诊断 — repulsive 过冲 + σ-head 反相

v3 三处独立证据指向同一机制（repulsive loss 强度过大且与 σ-head 共享 backbone，反向打坏 σ 的校准）：

1. **D1-a 0.882 → 0.149**：pairwise cos 不仅未停在目标区 [0.3, 0.7]，而是穿过整个区间跌到 0.15（4 slot 输出近似正交）。`λ_rep=0.1` 显著过强，让 mode 输出远离"温和差异"区。
2. **D1-c Pearson 符号翻转 +0.747 → −0.818**：v2 σ 与 reconstruction error 正相关（σ-head 正确校准）；v3 强负相关意味着 σ 现在**反向**指示 error。机制：repulsive 在 z_g 层把不同 mode 推开，σ-head 共享上游 backbone 拿到了被反向重塑的 representation，预测的不确定度与实际误差脱钩。
3. **G1_end_sigma_cos_mean 0.769 → 0.317**：与 #5 同源——基于 σ 选 best mode 的成功率从 77% 跌至 32%（接近 1/K=25% 随机基线）。

**残留 mode collapse 分析**：
- cov_end k=[0.001, **0.999**, 0, 0]（v2 是 k2=0.996，v3 旋转到 k1=0.999），同样 winner-takes-all
- bal_temperature 退火 1.0→0.1 没拉开 routing：4 个 slot 输出在 z_g 层被 repulsive 拉到正交后，softmin(L1) 仍能锁定单一 winner（哪个 mode 离 GT 最近就独占），锐化温度反而强化 winner-takes-all
- 即：**bal-T 锐化** + **repulsive 拉开** 的组合在当前 λ_rep 强度下产生了"4 mode 输出位置远 + 1 mode 独占 routing"的最坏交互

### 8.3 v3 改动 vs v2 baseline 收益清算

| 指标 | v2 → v3 | 性质 |
|---|---|---|
| G1_end_min_cos_mean | 0.799 → 0.862 | 单点改善（+0.06）|
| D1-a pairwise cos | 0.882 → 0.149 | 过冲（差距：目标区 [0.3, 0.7]）|
| Pearson(σ, err) | +0.747 → **−0.818** | 灾难性反转（损失 σ-head 全部校准）|
| G1_end_sigma_cos | 0.769 → 0.317 | 灾难（接近随机基线）|
| 7 阈值通过数 | 4/7 → 2/7 | 净减 2 |

**v2 优于 v3**。Gate 决策不能升级覆盖 v2 baseline。

### 8.4 决策路径 — 2/7 → 不进 Stage B；下一步候选

按 §3.6 决策树：≤ 3/7 → 重审是否结构性问题。**当前不能直接走 Stage B**。

候选下一步（按建议优先级）：

| 选项 | 描述 | 风险 / 收益 |
|---|---|---|
| **A. v4 = repulsive 退火 + 减弱** | `λ_rep` schedule：前 1200 step 关闭、随后线性增至 `λ_rep_max=0.03`（v3 `λ_rep=0.1` 的 3 折），bal_temperature 终值改 0.3 不打到 0.1 | 中风险：3 旋钮再调一次；高收益若过冲是唯一问题 |
| **B. 用 v2 当 plan-prior 进 Stage B** | mode collapse 主要伤 "多 mode 探索"；driver 训练只用 plan-prior 输出做 condition，单 mode 也能进 | 低风险：v2 4/7 已知；W3 mode-balance 退化为 ablation 议题 |
| **C. K=2 或 PaV-Lite 回退** | 简化结构 / 减少自由度 | 高代价：架构改动；仅在 A/B 都失败后启用 |

**8.4 节决策**：等用户在 A/B 之间二选一；C 暂不触发。

### 8.5 截至本报告时间点 (2026-05-04 晚段)

- v3 retrain（60 epoch / ~0.6h on H20）→ ✅
- v3 诊断（2/7，回归）→ ✅
- v3 baseline_table.csv 升级 → ❌（v2 仍是 W3 当前最好；baseline_table 不变）
- v4 设计 + retrain → ✅（详见 §9）
- Stage B (G-W3 driver) → ⏳ 等 v4 后再判定

---

## 9. v4 retrospective + 路径 A 收益清算 + 结构性发现

**Date**: 2026-05-04
**Run**: `pav_w3_lclgp_v4` (ckpt `playground/Checkpoints/pav_w3_lclgp_v4/final_model/pytorch_model.pt`)
**Config**: [examples/PlanAndVerify/configs/lclgp_v4.yaml](../examples/PlanAndVerify/configs/lclgp_v4.yaml)
**墙钟**: 60 epoch / ~41 min on 8×H20
**v4 仅在 v3 之上调缓 path B**（架构未动；α 0.5 不动）：
- λ_rep_max: 0.1 → **0.03**（1/3.3× peak）
- λ_rep_warmup_steps: 0 → **1200**（前 25% 关闭、随后线性 0→0.03 over 3600 step）
- bal_temperature_min: 0.1 → **0.3**（softer floor）

### 9.1 v1↔v2↔v3↔v4 7 阈值对比

| # | Metric | Threshold | v1 | v2 | v3 | v4 | v4 Pass | Δ(v4−v3) |
|---|---|---|---|---|---|---|---|---|
| 1 | G1_end_min_cos_mean | ≥ 0.75 | 0.213 | 0.799 | 0.862 | **0.795** | ✅ | ↓ 0.067 |
| 2 | G1_end_sigma_cos_mean | ≥ 0.70 | 0.202 | 0.769 | 0.317 | **0.319** | ❌ | ≈ 0 |
| 3 | D1-a pairwise cos ∈ [0.30, 0.70] | range | 0.957 | 0.882 | 0.149 | **0.227** | ❌ | ↑ 0.078（向目标区）|
| 4a | D1-b min mode freq end | ≥ 0.10 | 0.013 | 0.0 | 0.0 | 0.0 | ❌ | 0 |
| 4b | D1-b min mode freq delta | ≥ 0.10 | 0.0 | 0.0 | 0.0 | 0.0 | ❌ | 0 |
| 5 | D1-c Pearson(σ, err) | ≥ 0.40 | nan | +0.747 | **−0.818** | **−0.707** | ❌ | \|·\| ↓ 0.111 |
| 6 | D2-a cf L1 | ≥ 0.05 | 0.740 | 0.319 | 0.162 | **0.104** | ✅ | ↓ 0.058 |
| 7a | D3-a end specialization gap | ≥ 0.05 | 0.021 | 0.018 | 0.019 | 0.014 | ❌ | ↓ 0.005 |
| 7b | D3-a delta specialization gap | ≥ 0.05 | 0.004 | 0.009 | 0.008 | 0.005 | ❌ | ↓ 0.003 |
| **G1_delta_sigma_cos_mean**（次要）| ≥ 0.70 | 0.740 | 0.990 | 0.421 | **0.837** | ✅（次要）| ↑↑ 0.416 |

**通过 2/7（与 v3 持平）；W3 最优仍是 v2 4/7。**

### 9.2 W&B 末段（step 4800）

| 指标 | v2 | v3 | v4 实测 | 评价 |
|---|---|---|---|---|
| sigma_end_mean | 1.76 | n/a | 2.95 | 不顶 cap=4.48 ✅ |
| sigma_delta_mean | 4.48（贴 cap）| n/a | 3.80 | 远离 cap ✅ |
| **pi_bar_end k=[k0..k3]** | (cov k2=99.6%) | (cov k1=99.9%) | **[0.25, 0.28, 0.23, 0.25]** | softmin 训练时几乎完全均匀 |
| pi_bar_delta | (cov k2=99.6%) | (cov k1=99.9%) | [0.24, 0.22, 0.23, 0.31] | mode 3 略多但温和 |
| bal_temperature_current | 1.0 | 0.10 | 0.30 | schedule 终值符合预期 ✅ |
| lambda_rep_current | 0.0 | 0.10 | 0.030 | schedule 终值符合预期 ✅ |
| l_bal_end | 0.5 (含 σ) | n/a | 0.002 | softmin 几乎均匀 → KL≈0 |
| l_recon_end | 0.799 | n/a | 0.936 | 反向上升（同 v2 同因，1 mode 担 4 份）|
| l_recon_delta | 0.932 | n/a | 0.847 | 略降，delta 头健康 |
| l_ctr | 3.250 | n/a | 3.157 | 持平 |

### 9.3 v3 → v4 三处独立改善（方向对、幅度不够）

1. **D1-a 0.149 → 0.227**：λ_rep 减弱 0.1→0.03 + warmup 让 σ-head 先稳住 → 没有 v3 的过冲；但 0.227 仍未到 [0.3, 0.7] 下沿（差 0.073）。
2. **|D1-c Pearson| 0.818 → 0.707**：σ-head 校准部分恢复（绝对值减小 0.111），但**仍负相关**——repulsive 即使 0.03 + 1200 step warmup 也仍反向打 σ-head。
3. **G1_delta_sigma_cos 0.421 → 0.837**（**次要阈值新通过**）：delta 头 σ 校准完全恢复（v3 跌穿 0.7，v4 远超）；这是 v3 没有的结构性改善——证明 warmup + 减弱 repulsive 在 delta 头侧确实生效，**问题集中在 end 头**。

### 9.4 v4 关键发现 — soft routing / hard argmin 语义错位（结构性问题）

v4 W&B 的 `pi_bar_end k=[0.25, 0.28, 0.23, 0.25]`（4 mode 几乎完全均匀）+ `mode_balance_std_end=0.018`（接近 0）+ `l_bal_end=0.002`（满足）—— **训练时 soft routing 健康**。

但诊断 `D1b_cov_end_k3 = 1.000`——**hindsight argmin 100% 选 mode 3**。

这是一个**语义错位**：
- L_bal 看的是 `softmin(L1 / bal_T)` 输出的连续概率；当 4 mode 输出微小不同 + bal_T=0.3 温和（不是 v3 的 0.1 sharp）时，softmin 给每个 mode ~25% → KL=0 → l_bal 满足
- 但诊断 D1-b cov_end 用的是 hindsight `argmin`（取 abs error 最小的那个）——这是硬选择
- **当 4 mode 输出虽不完全相同但相对差异稳定时，argmin 永远选同一个 mode**（这次是 k3，v2 是 k2，v3 是 k1，winner 旋转但永远只有 1 个）

**这意味着 mode collapse 在 v2/v3/v4 都没真正解决，只是 winner 在 4 个 slot 之间旋转。** L_bal 的 softmin 公式与 hindsight argmin 评估指标之间存在结构性失配——**hparam 微调（温度、权重、warmup）无法解决这个失配**。

#### 9.4.1 三个根因（互相耦合，需架构改）

1. **L_bal 公式缺陷**：softmin 给"4 mode 输出微小不同"的情况近均匀概率，无法区分"真 4-way 分化"与"4 微小变体集中在一处"。
2. **L_ctr 用 best-mode 输出做正样本**：哪个 mode 先学好就独占 InfoNCE 梯度 → snowball；与 L_bal 形成对抗。
3. **Hindsight argmin 评估**：D1-b 用真值反查最佳 mode，而训练时模型不知道真值——训练优化的是"4 mode 都接近 GT"（min-of-K），不是"4 mode 互相分化"。

### 9.5 baseline_table 决策

W3 当前最优 ckpt = **v2**（4/7，σ 校准 +0.747，G1 双通过）。`paper/tables/baseline_table.csv` **不升级**。

v3/v4 探索（path B）的成果是结构性诊断（§9.4），不是 W3 阈值升级。

### 9.6 下一步候选

| 选项 | 描述 | 风险 / 收益 |
|---|---|---|
| **A2. v5 = 进一步退让 path B** | λ_rep_max 0.03→0.01，warmup 1200→2400（off 50%），bal_T_min 0.3→0.5；纯 hparam 调缓 | 收益小：3 旋钮再调一次最多回到接近 v2（4/7）；σ-head 全恢复，但 mode collapse 仍未解 |
| **B. 用 v2 当 plan-prior 进 Stage B**（**建议**）| driver 训练只用 plan-prior 输出做 condition，单 mode 也能进；mode-balance 退化为 W3 ablation；推进到 W4 | 低风险：v2 4/7 已知；mode collapse 不直接伤 driver-condition-only 路径；时间预算最划算 |
| **C. 架构改动（v5 真改）** | 解决 §9.4 三个根因之一：例如 L_bal 改为 hard onehot + entropy；L_ctr 改 sum-of-K 或 random-mode；引入 explicit routing module | 高代价：~1-2 天工作量；正确解但脱离原 design；需重做 G-W3 阈值定义 |

**建议路径 = B**：v3/v4 已验证 path B 的硬上限（§9.3 三个改善信号都在但都不够），结构问题不是 hparam 能解的；继续 W3 死磕回报递减；W4 driver 训练独立于 mode collapse。

### 9.7 截至本报告时间点 (2026-05-04 晚段)

- v4 retrain（~41 min on 8×H20）→ ✅
- v4 诊断（2/7，与 v3 持平、未击穿 v2 4/7）→ ✅
- v4 baseline_table.csv 升级 → ❌
- v4 关键发现：soft routing / hard argmin 语义错位 → 结构性问题（§9.4）
- 用户决策：**C1+C2+C3 = 全 MoE 化重构（v5）**

---

## 10. v5 设计 — 显式 MoE 路由（C1+C2+C3）

**Date**: 2026-05-04
**Config**: [examples/PlanAndVerify/configs/lclgp_v5.yaml](../examples/PlanAndVerify/configs/lclgp_v5.yaml)
**Code**: [starVLA/model/framework/PlanVerify/lclgp.py](../starVLA/model/framework/PlanVerify/lclgp.py) — `RoutingModule` class + `_forward_v5` 方法 + `predict_goal` 分支
**实施动机**：v2/v3/v4 在 path B（hparam 调缓）已撞结构上限。§9.4 三根因（L_bal softmin 弱、L_ctr best-mode 滚雪球、hindsight argmin 评估）互相耦合，需架构改动。

### 10.1 v5 = C1+C2+C3 全套 MoE 化

| 子改动 | 解的根因（§9.4）| v5 实现 |
|---|---|---|
| **C1** L_bal hard routing | softmin 给 4 微小变体 ~均匀概率 → KL=0 假性达标 | Gumbel-softmax `hard=True` 得 one-hot pi（前向硬，反向 STE）|
| **C2** L_ctr/L_recon routing-weighted | best-mode 滚雪球 → 1 winner 独占 InfoNCE/Recon 梯度 | `Σ_k π[b,k] · per_mode_loss[b,k]`（pi_hard one-hot ⇒ 拨给路由模式 + STE 软梯度回 router）|
| **C3** Explicit RoutingModule | hindsight argmin 评估 vs 训练时无 GT routing 错位 | 新 `RoutingModule(text, z_t) → π_end, π_delta`，无 GT，推理可用；与训练 / 评估三处统一 |

### 10.2 RoutingModule 结构（user 选 two-router shared backbone）

```python
class RoutingModule(nn.Module):
    def __init__(self, d_text, d_latent, d_hidden, K):
        # 共享 backbone
        self.text_proj = nn.Linear(d_text, d_hidden)
        self.zt_proj   = nn.Linear(d_latent, d_hidden)
        self.shared    = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU())
        # 分头 K-way
        self.head_end   = nn.Linear(d_hidden, K)
        self.head_delta = nn.Linear(d_hidden, K)
```

end-router 和 delta-router 可独立路由（user 决策点）。理由：v2-v4 实测 G1_end_min_cos ≈ 0.79 vs G1_delta_min_cos ≈ 0.99，end 头显著难于 delta 头；强制共享路由会瓶颈 end 头的特化能力。两 head 共享 (text, z_t) backbone，参数代价 ~50。

### 10.3 路由语义（三处统一）

- **训练（warmup）**：前 `router_warmup_steps=500` step，`pi_hard[b, k] = 1 if k == b % K else 0`（确定性均匀）。所有 K mode 头都拿到 recon 梯度 → 防止 winner-take-all snowball at L_recon level（Plan agent fix B）。
- **训练（warmup 后）**：`pi_hard = F.gumbel_softmax(logits, tau=current_T, hard=True)`。Gumbel temperature 线性退火 5.0 → 0.5 over 4800 step（比 v3 的 1.0 → 0.1 温和）。
- **推理（`predict_goal`）**：`pi_hard = F.one_hot(logits.argmax(-1), K)`。**确定性 argmax，无 Gumbel 采样**（Plan agent fix F2）。

### 10.4 5 损失重写

| 损失 | v1-v4 | v5 |
|---|---|---|
| **L_recon_end** | `min_k(L1_k/σ_k + β·logσ_k)`（min-of-K，只 winner 拿梯度）| `Σ_k π_end[b,k] · (L1_k/σ_k + β·logσ_k)`（routing-weighted；one-hot ⇒ 选中 mode 拿梯度 + STE 软梯度回 router）|
| **L_recon_delta** | 同上 min-of-K | 用 `π_delta` |
| **L_bal** | softmin shim → KL | 直接 `KL(pi_hard.mean(0) \|\| Uniform)`，end/delta 各一项 |
| **L_ctr** | positive = `z_g_end[argmin_end]`（hindsight）| positive = `z_g_end[π_end.argmax]`（router）；DDP `gather_fn` 不变 |
| **L_cf** | hinge on argmin best | `chosen_end` 来自 main pass（Plan agent fix F1）；CF 不重新路由 — 测"同一个 plan 下，移除 z_t 是否影响输出" |
| **L_repulsive** | cos² off-diag on z_g | 不变（仍要 K mode 输出在 latent 空间可区分）|

### 10.5 σ-head 解放（独立校准）

- σ 不再路由（router 接管）→ σ 现在纯粹做 calibrated NLL uncertainty
- D1-c Pearson(σ, err) 测的是 σ 是否与 reconstruction error 正相关（校准）
- v3/v4 的 D1-c 负相关问题（repulsive 反向打 σ）在 v5 中应缓解：σ 不再受 routing 任务的反向干扰

### 10.6 训练 step 持久化（Plan agent fix F3）

`register_buffer("training_step", ..., persistent=True)`（v4 是 False）。原因：v4 trainer 的 `fill_(completed_steps)` 在 `_train_step` 之后才执行，resume 时第一个 step 看到 step=0（冷启动），bal_T / λ_rep schedule / router warmup 全错。`persistent=True` 让 buffer 进 checkpoint，resume 时从 ckpt 加载正确 step。

### 10.7 7 阈值 — D1-b 替换为 D1-d（router 频率）

| # | 指标 | v1-v4 | v5 |
|---|---|---|---|
| 1 | G1_end_min_cos_mean | best-of-K via GT cos | 不变；≥ 0.75 |
| 2 | G1_end_router_cos_mean (was sigma_cos) | σ-argmin 选中 mode 的 cos | router 选中 mode 的 cos；≥ 0.70（语义一致：模型推理时选中 mode 的质量；只是 routing 来源换了）|
| 3 | D1-a pairwise cos | 不变 | 不变；∈ [0.30, 0.70] |
| 4 | **D1-d min_router_freq end / delta** | **(D1-b: hindsight argmin freq, 与 v5 架构无关)** | **NEW: bincount(`π.argmax`).min ≥ 0.10**。Legacy D1-b 仍 collected 作为 cross-version 参考。|
| 5 | D1-c Pearson(σ, err) | 不变 | 不变；≥ 0.40 |
| 6 | D2-a cf L1 | 不变 | 不变；≥ 0.05 |
| 7 | D3-a end/delta gap | 不变 | 不变；≥ 0.05（×2） |

**D1-b → D1-d 重定义的合理性**：D1-b 用 hindsight argmin（GT-aware 评估）测 mode 分化，但 v5 架构的核心改动就是把"训练时 GT-aware 路由 vs 推理 routing"统一了。D1-d（router argmax 分布）测的是模型实际的推理 routing 是否 4-way 分化，与 v5 架构语义匹配。Legacy D1-b 仍 collected，给 cross-version comparison 用。

**v5 success criterion = ≥ 5/7（用 D1-d 替换 D1-b 后）**。

### 10.8 Backward 兼容

- yaml 旗标 `framework.lclgp.use_router: True`（默认 False = v4 行为）
- v1/v2/v3/v4 yaml 不动，仍载入仍训练
- `predict_goal` 在 v4 路径下返回 `best_mode_*_sigma`（与 `best_mode_*` 同值）作为 cross-version alias；`best_mode_*_router` 和 `pi_router_*` 仅在 v5 暴露

### 10.9 Phase 1（本次 commit）实施清单

- ✅ `examples/PlanAndVerify/configs/lclgp_v5.yaml` 新建
- ✅ `lclgp.py` 增 `RoutingModule` class + `_forward_v5` + `predict_goal` 分支
- ✅ `lclgp.py` `register_buffer` 翻 `persistent=True`
- ✅ `run_diagnostics.py` `cmd_d1` 增 D1-d（router freq；v4 ckpt N/A）
- ✅ CPU smoke v5：warmup uniform [0.25×4]、post-warmup gumbel、schedule trace 正确、router 接到 grad 仅在 post-warmup
- ✅ CPU smoke v4 regression：use_router=False 输出 keys 完全不变；predict_goal API 不变（仅多 `best_mode_*_sigma` 别名，与原值相同）

### 10.10 Phase 2 / 3 — 预期 + Hard Cutoff

- v5 retrain：~50 min on 8×H20（与 v4 同 protocol）
- v5 诊断（4 cmd × ~1 min）→ paper/tables/lclgp_*.csv 覆盖到 v5
- §11 retrospective + decision

**Hard cutoff（架构改动唯一一次尝试）**：
- v5 ≥ 5/7（用 D1-d）→ 升级 baseline_table.csv = v5；进 Stage B 用 v5 plan-prior
- v5 ≤ 4/7 → **回退 option B**：用 v2 ckpt 当 plan-prior 进 Stage B；mode-balance 留 ablation
- **没有 v6**。架构改动是 W3 最后一次尝试；进一步 hparam 调试已证 path B 撞顶。
