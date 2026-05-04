"""LCLGP — Language-Conditioned Latent Goal Projection (PaV creation 1).

Maps a Qwen3-VL-4B text embedding sequence + the current V-JEPA 2 latent
``z_t`` into K=4 multimodal candidate goal latents at two timescales
(end-of-episode and Δ=50-step short horizon), each with a heteroscedastic
uncertainty head. The "best mode" output replaces the image-goal that
V-JEPA 2-AC's single-step CEM normally requires, allowing language-only
planning. See ``research_design_plan_and_verify.md`` §5 for the design.

Inputs (from ``LcLgpTripletDataset`` + ``collate_lclgp``)::

    text_emb   [B, L, 2560]   fp16/bf16, variable L
    text_mask  [B, L] bool    True = valid token (NOT PyTorch's PAD convention)
    z_t        [B, 256, 1408] V-JEPA latent at the anchor frame
    z_delta    [B, 256, 1408] V-JEPA latent at t + Δ
    z_end      [B, 256, 1408] V-JEPA latent at the trajectory end

Outputs from :meth:`LCLGP.forward`::

    {
        "loss":               total scalar (training objective)
        "l_recon_end/.._delta": min-of-K hetero L1 NLL components
        "l_bal_end/.._delta":   soft mode-balance KL components
        "l_ctr":              cross-task InfoNCE
        "l_cf":               counterfactual hinge (state dependence)
        "sigma_end_mean/...": diagnostic σ averages
        "mode_balance_std_end/...": std of soft mode frequencies (D1-b proxy)
        "mode_argmin_end/...": [B] argmin indices for histogram logging
    }
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _get(cfg, *keys, default=None):
    """Read a nested config attribute with a default fallback."""
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if hasattr(cur, k):
            cur = getattr(cur, k)
        elif isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur if cur is not None else default


class LCLGPDecoderLayer(nn.Module):
    """Self-attn(slots) + cross-attn(slots, text) + cross-attn(slots, z_t) + FFN.

    Standard pre-norm decoder block. Two cross-attentions in series so the
    text and visual context are absorbed into the slot tokens before the FFN
    integrates them.
    """

    def __init__(self, d_hidden: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_hidden)
        self.self_attn = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True, dropout=dropout)

        self.norm_text = nn.LayerNorm(d_hidden)
        self.text_attn = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True, dropout=dropout)

        self.norm_zt = nn.LayerNorm(d_hidden)
        self.zt_attn = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True, dropout=dropout)

        self.norm_mlp = nn.LayerNorm(d_hidden)
        hidden = int(d_hidden * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_hidden, hidden), nn.GELU(), nn.Linear(hidden, d_hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        slots: torch.Tensor,
        text_kv: torch.Tensor,
        text_pad_mask: Optional[torch.Tensor],
        zt_kv: torch.Tensor,
    ) -> torch.Tensor:
        # text_pad_mask: PyTorch key_padding_mask convention — True = ignore.
        x = slots
        q = self.norm_self(x)
        x = x + self.self_attn(q, q, q, need_weights=False)[0]

        q = self.norm_text(x)
        x = x + self.text_attn(q, text_kv, text_kv, key_padding_mask=text_pad_mask, need_weights=False)[0]

        q = self.norm_zt(x)
        x = x + self.zt_attn(q, zt_kv, zt_kv, need_weights=False)[0]

        x = x + self.dropout(self.mlp(self.norm_mlp(x)))
        return x


class RoutingModule(nn.Module):
    """W3 v5 — explicit MoE-style routing: π_end, π_delta = Pi(text, z_t).

    Two K-way heads sharing a (text, z_t) backbone — user-decided over a single
    router (see ``docs/lclgp_diagnostics.md`` §10). End and delta routings can
    legitimately differ given the empirical end-vs-delta difficulty asymmetry
    (G1_end_min_cos ≈ 0.79 vs G1_delta_min_cos ≈ 0.99 in v2-v4).

    No GT used → routing is available at inference. Outputs Gumbel-softmax
    one-hot (forward) with soft STE (backward) during training; argmax-one-hot
    at inference.
    """

    def __init__(self, d_text: int, d_latent: int, d_hidden: int, K: int):
        super().__init__()
        # Shared backbone — projects text-pool and z_t-pool into d_hidden.
        self.text_proj = nn.Linear(d_text, d_hidden)
        self.zt_proj = nn.Linear(d_latent, d_hidden)
        self.shared = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU())
        # Split K-heads for end / delta — independent routing for the two timescales.
        self.head_end = nn.Linear(d_hidden, K)
        self.head_delta = nn.Linear(d_hidden, K)

    def forward(
        self, text_emb: torch.Tensor, text_mask: torch.Tensor, z_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # V-JEPA latent / text_emb caches are fp16 on disk; model weights default
        # to fp32. Mirror ``_forward_core``'s entry cast so router Linears match.
        compute_dtype = self.text_proj.weight.dtype
        text_emb = text_emb.to(compute_dtype)
        z_t = z_t.to(compute_dtype)
        # Mask-aware mean-pool over text (PyTorch convention here is W2's:
        # text_mask True = valid token), simple mean-pool over z_t patches.
        mask_f = text_mask.to(compute_dtype).unsqueeze(-1)  # [B, L, 1]
        text_pool = (text_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        zt_pool = z_t.mean(dim=1)  # [B, d_latent]
        feat = self.shared(self.text_proj(text_pool) + self.zt_proj(zt_pool))
        return self.head_end(feat), self.head_delta(feat)  # logits [B, K] each


@FRAMEWORK_REGISTRY.register("LCLGP")
class LCLGP(baseframework):
    """Language-Conditioned Latent Goal Projection."""

    def __init__(self, config=None):
        super().__init__()

        # --- Architecture ---------------------------------------------------
        self.d_text = int(_get(config, "framework", "lclgp", "d_text", default=2560))
        self.d_latent = int(_get(config, "framework", "lclgp", "d_latent", default=1408))
        self.d_hidden = int(_get(config, "framework", "lclgp", "d_hidden", default=1024))
        self.n_heads = int(_get(config, "framework", "lclgp", "n_heads", default=16))
        self.n_layers = int(_get(config, "framework", "lclgp", "n_layers", default=4))
        self.K = int(_get(config, "framework", "lclgp", "n_modes", default=4))
        self.n_tokens = int(_get(config, "framework", "lclgp", "n_tokens", default=256))
        self.z_dropout = float(_get(config, "framework", "lclgp", "z_dropout", default=0.1))
        self.attn_dropout = float(_get(config, "framework", "lclgp", "attn_dropout", default=0.1))
        self.mlp_ratio = float(_get(config, "framework", "lclgp", "mlp_ratio", default=4.0))

        # --- Loss weights ---------------------------------------------------
        self.alpha = float(_get(config, "framework", "lclgp", "loss", "alpha", default=0.5))
        self.beta = float(_get(config, "framework", "lclgp", "loss", "beta", default=0.1))
        self.lambda_bal = float(_get(config, "framework", "lclgp", "loss", "lambda_bal", default=0.05))
        self.lambda_ctr = float(_get(config, "framework", "lclgp", "loss", "lambda_ctr", default=0.1))
        self.lambda_cf = float(_get(config, "framework", "lclgp", "loss", "lambda_cf", default=0.05))
        self.tau = float(_get(config, "framework", "lclgp", "loss", "tau", default=0.07))
        self.cf_margin = float(_get(config, "framework", "lclgp", "loss", "cf_margin", default=0.05))
        # log_sigma is clipped to keep heteroscedastic NLL well-conditioned.
        self.log_sigma_min = float(_get(config, "framework", "lclgp", "loss", "log_sigma_min", default=-5.0))
        self.log_sigma_max = float(_get(config, "framework", "lclgp", "loss", "log_sigma_max", default=5.0))
        # Soft mode-balance temperature (KL needs differentiable assignment).
        self.bal_temperature = float(_get(config, "framework", "lclgp", "loss", "bal_temperature", default=1.0))
        # bal_temperature linear anneal (W3 v3 path B): bal_T schedules from
        # ``bal_temperature`` → ``bal_temperature_min`` over training. Sharper softmin
        # punishes winner-takes-all routing → directly attacks D1-b min-mode-freq=0.
        # Default = bal_temperature (no anneal) preserves v1/v2 behavior.
        self.bal_temperature_min = float(
            _get(config, "framework", "lclgp", "loss", "bal_temperature_min", default=self.bal_temperature)
        )
        # Mode-output repulsive loss peak weight (W3 v3 path B). lambda_rep=0 → no-op.
        self.lambda_rep = float(_get(config, "framework", "lclgp", "loss", "lambda_rep", default=0.0))
        # λ_rep warmup (W3 v4): off for first ``lambda_rep_warmup_steps``, then linear
        # ramp 0 → ``lambda_rep`` over (total_steps − warmup). Default 0 = constant
        # ``lambda_rep`` for the whole run, preserving v3 behavior. v3's instant-on
        # λ_rep=0.1 decalibrated the σ-head (D1-c Pearson sign flipped, see §8);
        # delaying the turn-on lets σ stabilize before repulsion competes for backbone.
        self.lambda_rep_warmup_steps = int(
            _get(config, "framework", "lclgp", "loss", "lambda_rep_warmup_steps", default=0)
        )
        # Schedule horizon for bal_T anneal — read from trainer.max_train_steps.
        self.total_steps_for_schedule = int(_get(config, "trainer", "max_train_steps", default=4800))

        # ===== W3 v5 — explicit MoE routing (C1+C2+C3) ======================
        # When ``use_router`` is True, an explicit ``RoutingModule`` produces
        # π_end / π_delta from (text, z_t) only — no GT, available at inference.
        # All 5 losses route through π (Gumbel-softmax STE one-hot). σ-head is
        # freed from routing duty (purely calibrated NLL uncertainty).
        # See docs/lclgp_diagnostics.md §10 for design and threshold redefinition.
        # Default False preserves v1-v4 behavior (softmin-based implicit routing).
        self.use_router = bool(_get(config, "framework", "lclgp", "use_router", default=False))
        # Gumbel temperature linear anneal: high → exploratory; low → near-discrete.
        # 5.0 → 0.5 over training (less extreme than v3's 1.0 → 0.1, which decalibrated σ).
        self.gumbel_temperature_init = float(
            _get(config, "framework", "lclgp", "gumbel_temperature_init", default=5.0)
        )
        self.gumbel_temperature_min = float(
            _get(config, "framework", "lclgp", "gumbel_temperature_min", default=0.5)
        )
        # Router warmup: for first N steps, override π with deterministic uniform
        # (pi_hard[b, k] = 1 if k == b % K else 0). All K mode heads receive recon
        # gradient → prevents winner-takes-all snowball at L_recon level.
        self.router_warmup_steps = int(
            _get(config, "framework", "lclgp", "router_warmup_steps", default=500)
        )

        # Step counter updated externally by the trainer each accelerator step.
        # Persistent (W3 v5 fix): bal_T anneal, λ_rep schedule, gumbel temp anneal,
        # and router warmup all depend on training_step. ``persistent=True`` means
        # the buffer is saved/loaded with the checkpoint, so resume reads the
        # correct step before the first forward (otherwise fill_(completed_steps)
        # in the trainer happens AFTER train_step, leaving step 0 of resume cold).
        self.register_buffer("training_step", torch.zeros((), dtype=torch.long), persistent=True)

        # --- Modules --------------------------------------------------------
        self.text_proj = nn.Linear(self.d_text, self.d_hidden)
        self.latent_proj = nn.Linear(self.d_latent, self.d_hidden)

        # Slot tokens: orthogonal init across modes so the K candidates start
        # in distinct directions in parameter space. Without this, mode-balance
        # routing snowballs to whichever mode wins early — see W3 v1 retrospective
        # in docs/lclgp_diagnostics.md (cov_end_k3=0.76 mode collapse).
        slot_end_init = torch.empty(self.K, self.n_tokens, self.d_hidden)
        slot_delta_init = torch.empty(self.K, self.n_tokens, self.d_hidden)
        nn.init.orthogonal_(slot_end_init.view(self.K, -1))
        nn.init.orthogonal_(slot_delta_init.view(self.K, -1))
        self.slot_end = nn.Parameter(slot_end_init * 0.02)
        self.slot_delta = nn.Parameter(slot_delta_init * 0.02)
        self.mode_emb_end = nn.Parameter(torch.randn(self.K, 1, self.d_hidden) * 0.02)
        self.mode_emb_delta = nn.Parameter(torch.randn(self.K, 1, self.d_hidden) * 0.02)
        self.tau_end = nn.Parameter(torch.zeros(1))  # learnable timescale separator (additive)
        self.tau_delta = nn.Parameter(torch.zeros(1))

        self.layers = nn.ModuleList(
            [
                LCLGPDecoderLayer(self.d_hidden, self.n_heads, self.mlp_ratio, self.attn_dropout)
                for _ in range(self.n_layers)
            ]
        )

        self.out_end = nn.Linear(self.d_hidden, self.d_latent)
        self.out_delta = nn.Linear(self.d_hidden, self.d_latent)
        self.unc_end = nn.Linear(self.d_hidden, 1)
        self.unc_delta = nn.Linear(self.d_hidden, 1)

        # W3 v5 — explicit routing module (only built when ``use_router`` is True).
        # Kept outside the conditional below so v1-v4 yamls (no use_router key)
        # continue to construct LCLGP without an unused router.
        self.router = (
            RoutingModule(self.d_text, self.d_latent, self.d_hidden, self.K)
            if self.use_router
            else None
        )

    # ------------------------------------------------------------------
    # Core forward: text + z_t → multimodal goals
    # ------------------------------------------------------------------

    def _forward_core(
        self,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        z_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the cross-attn stack and produce dual-timescale goals + log_sigma.

        Args:
            text_emb: [B, L, d_text]
            text_mask: [B, L] bool, True = valid token (W2 collate convention).
            z_t: [B, n_tokens, d_latent]

        Returns:
            z_g_end:   [B, K, n_tokens, d_latent]
            log_s_end: [B, K]
            z_g_delta: [B, K, n_tokens, d_latent]
            log_s_delta: [B, K]
        """
        B = text_emb.shape[0]
        compute_dtype = self.text_proj.weight.dtype
        text_emb = text_emb.to(compute_dtype)
        z_t = z_t.to(compute_dtype)

        text_kv = self.text_proj(text_emb)               # [B, L, d_hidden]
        zt_kv = self.latent_proj(z_t)                    # [B, n_tokens, d_hidden]

        # PyTorch key_padding_mask: True = ignore. Flip W2's True=valid mask.
        text_pad_mask = ~text_mask if text_mask is not None else None

        # Build slot tokens: [K, N, D] → [B, K, N, D] → flatten K,N for attn.
        slots_end = (self.slot_end + self.mode_emb_end + self.tau_end).unsqueeze(0).expand(B, -1, -1, -1)
        slots_delta = (self.slot_delta + self.mode_emb_delta + self.tau_delta).unsqueeze(0).expand(B, -1, -1, -1)
        # Concatenate timescales along the slot axis so a single decoder pass handles both.
        slots = torch.cat([slots_end, slots_delta], dim=1)         # [B, 2K, N, D]
        slots = slots.reshape(B, 2 * self.K * self.n_tokens, self.d_hidden)

        for layer in self.layers:
            slots = layer(slots, text_kv, text_pad_mask, zt_kv)

        slots = slots.reshape(B, 2 * self.K, self.n_tokens, self.d_hidden)
        slots_end_out, slots_delta_out = slots[:, : self.K], slots[:, self.K :]   # each [B, K, N, D]

        z_g_end = self.out_end(slots_end_out)            # [B, K, N, d_latent]
        z_g_delta = self.out_delta(slots_delta_out)

        log_s_end = self.unc_end(slots_end_out.mean(dim=2)).squeeze(-1)            # [B, K]
        log_s_delta = self.unc_delta(slots_delta_out.mean(dim=2)).squeeze(-1)
        log_s_end = log_s_end.clamp(self.log_sigma_min, self.log_sigma_max)
        log_s_delta = log_s_delta.clamp(self.log_sigma_min, self.log_sigma_max)
        return z_g_end, log_s_end, z_g_delta, log_s_delta

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _heteroscedastic_per_mode(
        self, z_g: torch.Tensor, z_true: torch.Tensor, log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """Per-(sample, mode) heteroscedastic L1 NLL → [B, K]."""
        # z_g [B,K,N,D], z_true [B,N,D]. Stop-grad on target per design §5.3.1.
        l1 = (z_g - z_true.detach().unsqueeze(1)).abs().mean(dim=(-2, -1))   # [B, K]
        return l1 / log_sigma.exp() + self.beta * log_sigma                  # [B, K]

    def _recon_loss(
        self, z_g: torch.Tensor, z_true: torch.Tensor, log_sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Min-of-K hindsight reconstruction. Returns (loss, argmin, per_mode)."""
        per_mode = self._heteroscedastic_per_mode(z_g, z_true, log_sigma)    # [B, K]
        best, argmin = per_mode.min(dim=-1)                                   # [B], [B]
        return best.mean(), argmin, per_mode

    def _mode_balance_loss(
        self, per_mode_loss: torch.Tensor, bal_T: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Soft-assignment KL(pi_bar || Uniform(K)).

        Hard ``argmin`` has no gradient, so we use a softmin over per-mode losses
        (temperature ``bal_T`` — passed by caller for v3 anneal, falls back to the
        static ``self.bal_temperature``) to get a differentiable assignment.
        Returns (loss, soft_pi_bar, mode_balance_std).
        """
        T = bal_T if bal_T is not None else self.bal_temperature
        soft = F.softmin(per_mode_loss / T, dim=-1)                           # [B, K]
        pi_bar = soft.mean(dim=0)                                              # [K]
        uniform = 1.0 / self.K
        kl = (pi_bar * ((pi_bar + 1e-8) / uniform).log()).sum()
        return kl, pi_bar.detach(), pi_bar.detach().std()

    def _mode_repulsive_loss(self, z_g: torch.Tensor) -> torch.Tensor:
        """Pairwise mode-output repulsion (W3 v3 path B).

        Mean-pool tokens → L2-normalize in fp32 → pairwise cos². Off-diagonal mean
        across the K×K mode pairs, averaged over the batch. With orthogonal modes
        the loss is 0; the further modes converge, the larger the penalty. Targets
        D1-a (modes too similar) and gives L_bal a true K-way competition. See
        docs/lclgp_diagnostics.md §7.5.
        """
        z_pool = z_g.mean(dim=-2).float()                                      # [B, K, D]
        z_norm = F.normalize(z_pool, dim=-1)                                   # [B, K, D]
        sim = z_norm @ z_norm.transpose(-1, -2)                                # [B, K, K]
        K = sim.shape[-1]
        mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)              # [K, K]
        return sim.pow(2).masked_select(mask.unsqueeze(0).expand_as(sim)).mean()

    def _info_nce_loss(
        self,
        z_g_end: torch.Tensor,
        argmin_end: torch.Tensor,
        z_T: torch.Tensor,
        gather_fn=None,
    ) -> torch.Tensor:
        """Cross-task InfoNCE on patch-mean-pooled (z_g_end_best, z_T)."""
        B = z_g_end.shape[0]
        idx = torch.arange(B, device=z_g_end.device)
        z_g_best = z_g_end[idx, argmin_end].mean(dim=-2)                      # [B, D_lat]
        z_T_pool = z_T.mean(dim=-2)                                            # [B, D_lat]

        # Optional global gather for full DDP-batch InfoNCE (CLIP/SimCLR pattern).
        # Local positives index unchanged because gather concatenates along dim 0
        # in rank order; the local rank's positive on the diagonal still aligns.
        if gather_fn is not None:
            z_g_all = gather_fn(z_g_best)
            z_T_all = gather_fn(z_T_pool)
        else:
            z_g_all = z_g_best
            z_T_all = z_T_pool

        z_g_n = F.normalize(z_g_best.float(), dim=-1)
        z_T_n = F.normalize(z_T_all.float(), dim=-1)
        sim = z_g_n @ z_T_n.t() / self.tau                                     # [B, B_global]
        # Local rank's positives sit at columns [rank*B, rank*B + B). When no gather,
        # this is just torch.arange(B). When gathered, the trainer adjusts via gather_fn.
        # Default fallback: assume local positives = first B indices (single-rank).
        labels = torch.arange(B, device=sim.device)
        if gather_fn is not None and z_T_all.shape[0] != B:
            rank = getattr(gather_fn, "rank", 0)
            labels = labels + rank * B
        return F.cross_entropy(sim, labels)

    def _counterfactual_loss(
        self,
        z_g_end: torch.Tensor,
        argmin_end: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """Hinge: enforce ||z_g_end[best] − z_g_end_cf[best]||_1 ≥ m, no grad through CF."""
        with torch.no_grad():
            z_g_end_cf, _, _, _ = self._forward_core(text_emb, text_mask, torch.zeros_like(z_t))
        z_g_end_cf = z_g_end_cf.detach()
        B = z_g_end.shape[0]
        idx = torch.arange(B, device=z_g_end.device)
        diff = (z_g_end[idx, argmin_end] - z_g_end_cf[idx, argmin_end]).abs().mean(dim=(-2, -1))   # [B]
        return torch.clamp(self.cf_margin - diff, min=0.0).mean()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, Any], gather_fn=None) -> Dict[str, torch.Tensor]:
        """Dispatch to v1-v4 (implicit softmin routing) or v5 (explicit MoE).

        See ``docs/lclgp_diagnostics.md`` §1-§9 for the v1-v4 history that
        motivated v5; §10 for the v5 architectural design.
        """
        if self.use_router:
            return self._forward_v5(batch, gather_fn=gather_fn)
        return self._forward_v4(batch, gather_fn=gather_fn)

    def _forward_v5(self, batch: Dict[str, Any], gather_fn=None) -> Dict[str, torch.Tensor]:
        """W3 v5 — explicit MoE routing (C1+C2+C3).

        Router(text, z_t) → (π_end, π_delta) ∈ Δ^K. Gumbel-softmax STE one-hot
        forward + soft backward during training; argmax-one-hot at inference.
        All 5 losses route through π. σ-head is freed from routing duty (purely
        calibrated NLL uncertainty). Router warmup (deterministic uniform) for
        first ``router_warmup_steps`` keeps all K mode heads alive while router
        is still untrained. See ``docs/lclgp_diagnostics.md`` §10.
        """
        text_emb = batch["text_emb"]
        text_mask = batch["text_mask"]
        z_t = batch["z_t"]
        z_delta = batch["z_delta"]
        z_end = batch["z_end"]

        compute_dtype = self.text_proj.weight.dtype
        z_delta = z_delta.to(compute_dtype)
        z_end = z_end.to(compute_dtype)

        # Per-batch z_t dropout (training only) — D2 regularizer (unchanged from v4).
        z_t_input = z_t
        if self.training and self.z_dropout > 0 and torch.rand((), device=z_t.device).item() < self.z_dropout:
            z_t_input = torch.zeros_like(z_t)

        # Forward core: K mode outputs + per-mode log_sigma (unchanged from v4).
        z_g_end, log_s_end, z_g_delta, log_s_delta = self._forward_core(text_emb, text_mask, z_t_input)
        B = z_g_end.shape[0]

        # ===== Routing =====================================================
        # Always run the router (gradient through L_bal even during warmup).
        logits_end, logits_delta = self.router(text_emb, text_mask, z_t_input)

        # Gumbel temperature linear anneal: T_init → T_min over total_steps.
        progress = (self.training_step.float() / max(1, self.total_steps_for_schedule)).clamp(0.0, 1.0)
        current_gumbel_T = self.gumbel_temperature_init - (
            self.gumbel_temperature_init - self.gumbel_temperature_min
        ) * progress

        # Routing decision branches by (training, in_warmup):
        # - eval: deterministic argmax (no Gumbel sampling) — Plan agent fix F2
        # - training, in_warmup: deterministic uniform π[b, k] = 1 if k == b % K
        #   (all K mode heads receive recon gradient → prevents L_recon snowball)
        # - training, post-warmup: Gumbel-softmax STE one-hot
        in_warmup = self.training and (int(self.training_step.item()) < self.router_warmup_steps)
        if not self.training:
            # Inference: deterministic argmax.
            pi_hard_end = F.one_hot(logits_end.argmax(dim=-1), num_classes=self.K).to(z_g_end.dtype)
            pi_hard_delta = F.one_hot(logits_delta.argmax(dim=-1), num_classes=self.K).to(z_g_end.dtype)
        elif in_warmup:
            idx_b = torch.arange(B, device=z_g_end.device)
            uniform_one_hot = F.one_hot(idx_b % self.K, num_classes=self.K).to(z_g_end.dtype)
            pi_hard_end = uniform_one_hot
            pi_hard_delta = uniform_one_hot
        else:
            # Gumbel-softmax requires float32 for stability; cast back to dtype after.
            tau = float(current_gumbel_T.item())
            pi_hard_end = F.gumbel_softmax(logits_end.float(), tau=tau, hard=True).to(z_g_end.dtype)
            pi_hard_delta = F.gumbel_softmax(logits_delta.float(), tau=tau, hard=True).to(z_g_end.dtype)

        chosen_end = pi_hard_end.argmax(dim=-1)        # [B]
        chosen_delta = pi_hard_delta.argmax(dim=-1)    # [B]

        # ===== L_recon (routing-weighted) =================================
        # Per-mode heteroscedastic L1 NLL (same fn as v4); the change is the reduction:
        # v4: min-of-K (only winner gets gradient → snowball)
        # v5: routing-weighted (chosen mode gets recon, with soft STE through router)
        per_mode_nll_end = self._heteroscedastic_per_mode(z_g_end, z_end, log_s_end)        # [B, K]
        per_mode_nll_delta = self._heteroscedastic_per_mode(z_g_delta, z_delta, log_s_delta)
        l_recon_end = (pi_hard_end * per_mode_nll_end).sum(dim=-1).mean()
        l_recon_delta = (pi_hard_delta * per_mode_nll_delta).sum(dim=-1).mean()

        # ===== L_bal (KL of router routing distribution || Uniform) =======
        # Direct KL on pi_hard.mean(0) — no softmin shim. With one-hot pi_hard,
        # equalizing pi_bar requires actually different samples picking different modes.
        pi_bar_end = pi_hard_end.mean(dim=0)           # [K]
        pi_bar_delta = pi_hard_delta.mean(dim=0)
        uniform = 1.0 / self.K
        l_bal_end = (pi_bar_end * ((pi_bar_end + 1e-8) / uniform).log()).sum()
        l_bal_delta = (pi_bar_delta * ((pi_bar_delta + 1e-8) / uniform).log()).sum()

        # ===== L_repulsive (mode outputs in latent space; unchanged from v4) =
        if self.lambda_rep > 0.0:
            l_rep_end = self._mode_repulsive_loss(z_g_end)
            l_rep_delta = self._mode_repulsive_loss(z_g_delta)
            if self.lambda_rep_warmup_steps > 0:
                _step_f = self.training_step.float()
                _ramp_total = max(1, self.total_steps_for_schedule - self.lambda_rep_warmup_steps)
                _ramp_progress = ((_step_f - self.lambda_rep_warmup_steps) / _ramp_total).clamp(0.0, 1.0)
                current_lambda_rep = self.lambda_rep * _ramp_progress
            else:
                current_lambda_rep = z_g_end.new_tensor(self.lambda_rep)
        else:
            l_rep_end = z_g_end.new_zeros(())
            l_rep_delta = z_g_end.new_zeros(())
            current_lambda_rep = z_g_end.new_zeros(())

        # ===== L_ctr (router-chosen positive replaces hindsight argmin) ===
        l_ctr = self._info_nce_loss(z_g_end, chosen_end, z_end, gather_fn=gather_fn)

        # ===== L_cf (CF reuses main pass's chosen_end; not re-routed) =====
        # Tests "removing z_t affects output for the SAME chosen plan", not
        # "removing z_t affects which plan is chosen" — Plan agent fix F1.
        l_cf = self._counterfactual_loss(z_g_end, chosen_end, text_emb, text_mask, z_t)

        # ===== Total =======================================================
        total = (
            self.alpha * l_recon_end
            + (1.0 - self.alpha) * l_recon_delta
            + self.lambda_bal * (l_bal_end + l_bal_delta)
            + self.lambda_ctr * l_ctr
            + self.lambda_cf * l_cf
            + current_lambda_rep * (l_rep_end + l_rep_delta) / 2.0
        )

        return {
            "loss": total,
            "l_recon_end": l_recon_end.detach(),
            "l_recon_delta": l_recon_delta.detach(),
            "l_bal_end": l_bal_end.detach(),
            "l_bal_delta": l_bal_delta.detach(),
            "l_ctr": l_ctr.detach(),
            "l_cf": l_cf.detach(),
            "l_rep_end": l_rep_end.detach(),
            "l_rep_delta": l_rep_delta.detach(),
            "gumbel_temperature_current": current_gumbel_T.detach(),
            "lambda_rep_current": current_lambda_rep.detach(),
            "router_warmup_active": z_g_end.new_tensor(1.0 if in_warmup else 0.0),
            "sigma_end_mean": log_s_end.exp().mean().detach(),
            "sigma_delta_mean": log_s_delta.exp().mean().detach(),
            "mode_balance_std_end": pi_bar_end.detach().std(),
            "mode_balance_std_delta": pi_bar_delta.detach().std(),
            "mode_argmin_end": chosen_end.detach(),       # router-chosen (same key as v4 for downstream)
            "mode_argmin_delta": chosen_delta.detach(),
            "pi_bar_end": pi_bar_end.detach(),
            "pi_bar_delta": pi_bar_delta.detach(),
            "pi_router_end_min": pi_bar_end.detach().min(),
            "pi_router_delta_min": pi_bar_delta.detach().min(),
        }

    def _forward_v4(self, batch: Dict[str, Any], gather_fn=None) -> Dict[str, torch.Tensor]:
        """W3 v1-v4 path — implicit softmin routing on hindsight per_mode_loss."""
        text_emb = batch["text_emb"]
        text_mask = batch["text_mask"]
        z_t = batch["z_t"]
        z_delta = batch["z_delta"]
        z_end = batch["z_end"]

        compute_dtype = self.text_proj.weight.dtype
        z_delta = z_delta.to(compute_dtype)
        z_end = z_end.to(compute_dtype)

        # Per-batch z_t dropout (training only) — D2 regularizer.
        z_t_input = z_t
        if self.training and self.z_dropout > 0 and torch.rand((), device=z_t.device).item() < self.z_dropout:
            z_t_input = torch.zeros_like(z_t)

        z_g_end, log_s_end, z_g_delta, log_s_delta = self._forward_core(text_emb, text_mask, z_t_input)

        # Reconstruction (min-of-K) per timescale.
        l_recon_end, argmin_end, per_mode_end = self._recon_loss(z_g_end, z_end, log_s_end)
        l_recon_delta, argmin_delta, per_mode_delta = self._recon_loss(z_g_delta, z_delta, log_s_delta)

        # Mode balance (soft KL) — feed RAW L1, not the σ-divided per_mode_loss.
        # Rationale: per_mode_loss contains 1/σ + β·log σ. With σ saturated at
        # log_sigma_max, all modes have identical per_mode_loss → softmin uniform
        # → KL=0 trivially, so mode collapse is left unchecked. Raw L1 cannot be
        # equalized this way; routing reflects actual reconstruction quality.
        # See docs/lclgp_diagnostics.md (W3 v1 root cause).
        l1_end_per_mode = (z_g_end - z_end.detach().unsqueeze(1)).abs().mean(dim=(-2, -1))
        l1_delta_per_mode = (z_g_delta - z_delta.detach().unsqueeze(1)).abs().mean(dim=(-2, -1))
        # bal_T linear anneal (W3 v3 path B): T(step) = T_init - (T_init - T_min) * progress.
        # progress=0 at step 0 → T_init; progress=1 at total_steps → T_min. With
        # bal_temperature_min == bal_temperature (default) this collapses to the v2 static T.
        progress = (self.training_step.float() / max(1, self.total_steps_for_schedule)).clamp(0.0, 1.0)
        current_bal_T = self.bal_temperature - (self.bal_temperature - self.bal_temperature_min) * progress
        l_bal_end, pi_bar_end, mb_std_end = self._mode_balance_loss(l1_end_per_mode, bal_T=current_bal_T)
        l_bal_delta, pi_bar_delta, mb_std_delta = self._mode_balance_loss(l1_delta_per_mode, bal_T=current_bal_T)

        # Mode-output repulsive loss (W3 v3 path B). lambda_rep=0 → no-op (v1/v2 behavior).
        # λ_rep warmup schedule (W3 v4): when lambda_rep_warmup_steps > 0, the effective
        # weight is 0 for step < warmup_steps, then linearly ramps to ``lambda_rep`` by
        # ``total_steps_for_schedule``. Always log the *would-be* l_rep terms so we can
        # diagnose the warmup transition in W&B even before they contribute to total.
        if self.lambda_rep > 0.0:
            l_rep_end = self._mode_repulsive_loss(z_g_end)
            l_rep_delta = self._mode_repulsive_loss(z_g_delta)
            if self.lambda_rep_warmup_steps > 0:
                _step_f = self.training_step.float()
                _ramp_total = max(1, self.total_steps_for_schedule - self.lambda_rep_warmup_steps)
                _ramp_progress = ((_step_f - self.lambda_rep_warmup_steps) / _ramp_total).clamp(0.0, 1.0)
                current_lambda_rep = self.lambda_rep * _ramp_progress
            else:
                current_lambda_rep = z_g_end.new_tensor(self.lambda_rep)
        else:
            l_rep_end = z_g_end.new_zeros(())
            l_rep_delta = z_g_end.new_zeros(())
            current_lambda_rep = z_g_end.new_zeros(())

        # InfoNCE across tasks (uses end-goal best mode vs ground-truth z_end).
        l_ctr = self._info_nce_loss(z_g_end, argmin_end, z_end, gather_fn=gather_fn)

        # Counterfactual hinge — second forward through the model with z_t=0,
        # under no_grad to keep state-dependence pressure on the main branch only.
        l_cf = self._counterfactual_loss(z_g_end, argmin_end, text_emb, text_mask, z_t)

        total = (
            self.alpha * l_recon_end
            + (1.0 - self.alpha) * l_recon_delta
            + self.lambda_bal * (l_bal_end + l_bal_delta)
            + self.lambda_ctr * l_ctr
            + self.lambda_cf * l_cf
            + current_lambda_rep * (l_rep_end + l_rep_delta) / 2.0
        )

        return {
            "loss": total,
            "l_recon_end": l_recon_end.detach(),
            "l_recon_delta": l_recon_delta.detach(),
            "l_bal_end": l_bal_end.detach(),
            "l_bal_delta": l_bal_delta.detach(),
            "l_ctr": l_ctr.detach(),
            "l_cf": l_cf.detach(),
            "l_rep_end": l_rep_end.detach(),
            "l_rep_delta": l_rep_delta.detach(),
            "bal_temperature_current": current_bal_T.detach(),
            "lambda_rep_current": current_lambda_rep.detach(),
            "sigma_end_mean": log_s_end.exp().mean().detach(),
            "sigma_delta_mean": log_s_delta.exp().mean().detach(),
            "mode_balance_std_end": mb_std_end,
            "mode_balance_std_delta": mb_std_delta,
            "mode_argmin_end": argmin_end.detach(),
            "mode_argmin_delta": argmin_delta.detach(),
            "pi_bar_end": pi_bar_end,
            "pi_bar_delta": pi_bar_delta,
        }

    @torch.no_grad()
    def predict_goal(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Inference-time best-mode goal projection. Used by run_diagnostics.py.

        v1-v4: ``best_mode_*`` is the σ-argmin (min-σ heuristic at inference).
        v5: ``best_mode_*`` is the router argmax (deterministic, no Gumbel
        sampling). σ-argmin is also returned as ``best_mode_*_sigma`` for
        cross-check / D1-c calibration test. Router probabilities returned
        as ``pi_router_*`` for D1-d (router routing distribution).
        """
        text_emb = batch["text_emb"]
        text_mask = batch["text_mask"]
        z_t = batch["z_t"]
        z_g_end, log_s_end, z_g_delta, log_s_delta = self._forward_core(text_emb, text_mask, z_t)

        # Always compute σ-argmin (legacy / cross-check).
        best_end_sigma = log_s_end.argmin(dim=-1)
        best_delta_sigma = log_s_delta.argmin(dim=-1)

        if self.use_router:
            logits_end, logits_delta = self.router(text_emb, text_mask, z_t)
            best_end_router = logits_end.argmax(dim=-1)
            best_delta_router = logits_delta.argmax(dim=-1)
            chosen_end = best_end_router
            chosen_delta = best_delta_router
            pi_router_end = F.softmax(logits_end, dim=-1)      # [B, K] for D1-d
            pi_router_delta = F.softmax(logits_delta, dim=-1)
        else:
            chosen_end = best_end_sigma
            chosen_delta = best_delta_sigma
            best_end_router = None
            best_delta_router = None
            pi_router_end = None
            pi_router_delta = None

        idx = torch.arange(z_g_end.shape[0], device=z_g_end.device)
        out: Dict[str, torch.Tensor] = {
            "z_g_end_best": z_g_end[idx, chosen_end],            # [B, N, D_lat]
            "z_g_delta_best": z_g_delta[idx, chosen_delta],
            "z_g_end_all": z_g_end,                              # [B, K, N, D_lat]
            "z_g_delta_all": z_g_delta,
            "log_sigma_end": log_s_end,                          # [B, K]
            "log_sigma_delta": log_s_delta,
            "best_mode_end": chosen_end,                         # [B] — primary chosen mode
            "best_mode_delta": chosen_delta,
            "best_mode_end_sigma": best_end_sigma,               # [B] — σ-argmin (cross-check)
            "best_mode_delta_sigma": best_delta_sigma,
        }
        if self.use_router:
            out["best_mode_end_router"] = best_end_router         # type: ignore[assignment]
            out["best_mode_delta_router"] = best_delta_router     # type: ignore[assignment]
            out["pi_router_end"] = pi_router_end                  # type: ignore[assignment]
            out["pi_router_delta"] = pi_router_delta              # type: ignore[assignment]
        return out


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------


def _fake_batch(B=4, L=24, n_tokens=256, d_text=2560, d_latent=1408, device="cpu"):
    """Synthesize a batch with the W2 collate contract, for the smoke-test."""
    text_emb = torch.randn(B, L, d_text, device=device)
    text_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    text_mask[:, L // 2 :] = False                           # half real, half padded
    return {
        "text_emb": text_emb,
        "text_mask": text_mask,
        "z_t": torch.randn(B, n_tokens, d_latent, device=device),
        "z_delta": torch.randn(B, n_tokens, d_latent, device=device),
        "z_end": torch.randn(B, n_tokens, d_latent, device=device),
        "task_id": ["t0", "t0", "t1", "t1"],
        "lang": ["a", "a", "b", "b"],
        "data_name": ["fake"] * B,
        "traj_id": list(range(B)),
    }


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test LCLGP forward")
    parser.add_argument("--config_yaml", type=str, default=None)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    # Build a tiny in-memory config — overridden by --config_yaml when provided.
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        framework=SimpleNamespace(
            name="LCLGP",
            lclgp=SimpleNamespace(
                d_text=2560,
                d_latent=1408,
                d_hidden=256,                                # smaller for CPU smoke
                n_heads=8,
                n_layers=2,
                n_modes=4,
                n_tokens=64,                                 # smaller for CPU smoke
                z_dropout=0.1,
                attn_dropout=0.1,
                mlp_ratio=4.0,
                loss=SimpleNamespace(
                    alpha=0.5, beta=0.1, lambda_bal=0.05, lambda_ctr=0.1, lambda_cf=0.05,
                    tau=0.07, cf_margin=0.05, log_sigma_min=-5.0, log_sigma_max=5.0,
                    bal_temperature=1.0,
                ),
            ),
        ),
    )

    if args.config_yaml is not None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config_yaml)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    model = LCLGP(cfg).to(device).train()
    n_tokens = model.n_tokens
    batch = _fake_batch(B=4, L=24, n_tokens=n_tokens, d_text=model.d_text, d_latent=model.d_latent, device=device)

    out = model(batch)
    print("forward OK — keys:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<24} {tuple(v.shape)}  {v.dtype}  {('NaN!' if torch.isnan(v).any() else 'ok')}")
        else:
            print(f"  {k:<24} {type(v).__name__}")

    print("backward...")
    out["loss"].backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"params with non-zero grad: {has_grad} / {total}")

    # predict_goal sanity check.
    model.eval()
    with torch.no_grad():
        g = model.predict_goal(batch)
    for k, v in g.items():
        print(f"  predict.{k:<20} {tuple(v.shape)}  {v.dtype}")


if __name__ == "__main__":
    _main()
