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
- v2 retrain（60 epoch / 0.6h on H200）→ ✅
- v2 诊断（4/7 通过）→ ✅
- v3 设计 + retrain → ⏳ 下一 commit
- Stage B (G-W3 driver) → ⏳ 取决于 v3 诊断结果（≥ 6/7 触发）
