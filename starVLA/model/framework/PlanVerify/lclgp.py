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

        # --- Modules --------------------------------------------------------
        self.text_proj = nn.Linear(self.d_text, self.d_hidden)
        self.latent_proj = nn.Linear(self.d_latent, self.d_hidden)

        self.slot_end = nn.Parameter(torch.randn(self.K, self.n_tokens, self.d_hidden) * 0.02)
        self.slot_delta = nn.Parameter(torch.randn(self.K, self.n_tokens, self.d_hidden) * 0.02)
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

    def _mode_balance_loss(self, per_mode_loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Soft-assignment KL(pi_bar || Uniform(K)).

        Hard ``argmin`` has no gradient, so we use a softmin over per-mode losses
        (temperature ``bal_temperature``) to get a differentiable assignment.
        Returns (loss, soft_pi_bar, mode_balance_std) with detached diagnostics.
        """
        soft = F.softmin(per_mode_loss / self.bal_temperature, dim=-1)        # [B, K]
        pi_bar = soft.mean(dim=0)                                              # [K]
        uniform = 1.0 / self.K
        kl = (pi_bar * ((pi_bar + 1e-8) / uniform).log()).sum()
        return kl, pi_bar.detach(), pi_bar.detach().std()

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

        # Mode balance (soft KL).
        l_bal_end, pi_bar_end, mb_std_end = self._mode_balance_loss(per_mode_end)
        l_bal_delta, pi_bar_delta, mb_std_delta = self._mode_balance_loss(per_mode_delta)

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
        )

        return {
            "loss": total,
            "l_recon_end": l_recon_end.detach(),
            "l_recon_delta": l_recon_delta.detach(),
            "l_bal_end": l_bal_end.detach(),
            "l_bal_delta": l_bal_delta.detach(),
            "l_ctr": l_ctr.detach(),
            "l_cf": l_cf.detach(),
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
        """Inference-time best-mode goal projection. Used by run_diagnostics.py."""
        text_emb = batch["text_emb"]
        text_mask = batch["text_mask"]
        z_t = batch["z_t"]
        z_g_end, log_s_end, z_g_delta, log_s_delta = self._forward_core(text_emb, text_mask, z_t)
        # Best mode by lowest σ — at inference there is no ground truth, so we pick the
        # most confident mode per timescale. Diagnostics may also use all K modes.
        best_end = log_s_end.argmin(dim=-1)
        best_delta = log_s_delta.argmin(dim=-1)
        idx = torch.arange(z_g_end.shape[0], device=z_g_end.device)
        return {
            "z_g_end_best": z_g_end[idx, best_end],          # [B, N, D_lat]
            "z_g_delta_best": z_g_delta[idx, best_delta],
            "z_g_end_all": z_g_end,                          # [B, K, N, D_lat]
            "z_g_delta_all": z_g_delta,
            "log_sigma_end": log_s_end,                      # [B, K]
            "log_sigma_delta": log_s_delta,
            "best_mode_end": best_end,                       # [B]
            "best_mode_delta": best_delta,
        }


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
