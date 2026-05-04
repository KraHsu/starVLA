"""LCLGP_K1 — single-mode (K=1) deterministic ablation of LCLGP.

After v1-v5 of the K=4 multimodal design (best 4/7 at v2, catastrophic 1/7 at
v5's MoE rewrite — see ``docs/lclgp_diagnostics.md`` §1-§11), the cross-version
data accumulated three observations that pointed at the multimodality
hypothesis itself:

    * G1_delta_min_cos ≈ 0.99 across all 5 versions — delta-prediction is
      unimodal-easy. K>1 was doing nothing for delta.
    * G1_end_min_cos was unstable (0.21/0.80/0.86/0.80/0.29) and v2's "good"
      relied on min-of-K hindsight retro-picking the best of 4 modes; once
      v5's routing-weighted recon removed that safety net, end collapsed.
    * LIBERO instructions ("place X on Y") are deterministic single-goal —
      multimodal goal hypothesis was never well-justified by the data domain.

K=1 is a clean test of the hypothesis: a single deterministic head per
timescale, no routing, no mode balance, no repulsive loss. If it matches or
beats v2's 4/7 with much less optimization complexity, K>1 was overengineered.

This overrides the §10.10 Hard Cutoff (which was scoped to K=4 hparam
exploration, not architectural pivots). See §12 for the cutoff override
rationale and the 6-threshold success criterion replacing the 7 thresholds.

Inputs (from ``LcLgpTripletDataset`` + ``collate_lclgp``)::

    text_emb   [B, L, 2560]   fp16/bf16, variable L
    text_mask  [B, L] bool    True = valid token
    z_t        [B, 256, 1408] V-JEPA latent at the anchor frame
    z_delta    [B, 256, 1408] V-JEPA latent at t + Δ
    z_end      [B, 256, 1408] V-JEPA latent at the trajectory end

Outputs from :meth:`LCLGPK1.forward`::

    {
        "loss":               total scalar (training objective)
        "l_recon_end/.._delta": heteroscedastic L1 NLL components (no min-of-K)
        "l_ctr":              cross-task InfoNCE
        "l_cf":               counterfactual hinge (state dependence)
        "sigma_end_mean/...": diagnostic σ averages
    }

predict_goal preserves a K=1 dim in tensor outputs (``z_g_end_all [B, 1, N, D]``,
``log_sigma_end [B, 1]``, ``best_mode_end [B]`` always 0) so the existing
diagnostic pipeline works unchanged except for D1-a pairwise cosine, which is
gated to N/A under K=1.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.PlanVerify.lclgp import LCLGPDecoderLayer, _get
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("LCLGP_K1")
class LCLGPK1(baseframework):
    """K=1 single-head LCLGP. No routing, no mode balance, no repulsive loss.

    Architecture mirrors LCLGP's ``_forward_core`` but with K=1 collapsed:
    one slot stack per timescale (end / delta), shared decoder, separate
    output heads. Heteroscedastic σ is supervised directly through the L1
    reconstruction loss (no min-of-K wrapping).
    """

    def __init__(self, config=None):
        super().__init__()

        # --- Architecture (copied from LCLGP, K dim removed) ----------------
        self.d_text = int(_get(config, "framework", "lclgp", "d_text", default=2560))
        self.d_latent = int(_get(config, "framework", "lclgp", "d_latent", default=1408))
        self.d_hidden = int(_get(config, "framework", "lclgp", "d_hidden", default=1024))
        self.n_heads = int(_get(config, "framework", "lclgp", "n_heads", default=16))
        self.n_layers = int(_get(config, "framework", "lclgp", "n_layers", default=4))
        self.n_tokens = int(_get(config, "framework", "lclgp", "n_tokens", default=256))
        self.z_dropout = float(_get(config, "framework", "lclgp", "z_dropout", default=0.1))
        self.attn_dropout = float(_get(config, "framework", "lclgp", "attn_dropout", default=0.1))
        self.mlp_ratio = float(_get(config, "framework", "lclgp", "mlp_ratio", default=4.0))

        # K is fixed; held as an attribute so diagnostic scripts that read
        # ``model.K`` continue to work and gate the K=1-specific N/A rows.
        self.K = 1

        # --- Loss weights (3-loss family: recon + ctr + cf) -----------------
        self.alpha = float(_get(config, "framework", "lclgp", "loss", "alpha", default=0.5))
        self.beta = float(_get(config, "framework", "lclgp", "loss", "beta", default=0.5))
        self.lambda_ctr = float(_get(config, "framework", "lclgp", "loss", "lambda_ctr", default=0.1))
        self.lambda_cf = float(_get(config, "framework", "lclgp", "loss", "lambda_cf", default=0.05))
        self.tau = float(_get(config, "framework", "lclgp", "loss", "tau", default=0.07))
        self.cf_margin = float(_get(config, "framework", "lclgp", "loss", "cf_margin", default=0.05))
        # σ is clipped to keep heteroscedastic NLL well-conditioned. Default
        # log_sigma_max here is 1.5 (matches v2-v5; v1's 5.0 was the σ-saturation
        # source — see §3 of lclgp_diagnostics.md).
        self.log_sigma_min = float(_get(config, "framework", "lclgp", "loss", "log_sigma_min", default=-5.0))
        self.log_sigma_max = float(_get(config, "framework", "lclgp", "loss", "log_sigma_max", default=1.5))

        # --- Modules --------------------------------------------------------
        self.text_proj = nn.Linear(self.d_text, self.d_hidden)
        self.latent_proj = nn.Linear(self.d_latent, self.d_hidden)

        # Single slot stack per timescale (was [K, N, D] in LCLGP, now [N, D]).
        # No mode_emb (no mode disambiguation needed) — slot_end / slot_delta
        # alone identify the timescale. tau_end / tau_delta retained as scalar
        # learnable timescale separator, mirroring LCLGP's design.
        self.slot_end = nn.Parameter(torch.randn(self.n_tokens, self.d_hidden) * 0.02)
        self.slot_delta = nn.Parameter(torch.randn(self.n_tokens, self.d_hidden) * 0.02)
        self.tau_end = nn.Parameter(torch.zeros(1))
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
    # Core forward: text + z_t → single (z_g_end, σ_end, z_g_delta, σ_delta)
    # ------------------------------------------------------------------

    def _forward_core(
        self,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        z_t: torch.Tensor,
    ):
        """Run the cross-attn stack and produce dual-timescale goals + log_sigma.

        Args:
            text_emb: [B, L, d_text]
            text_mask: [B, L] bool, True = valid token (W2 collate convention).
            z_t: [B, n_tokens, d_latent]

        Returns:
            z_g_end:   [B, n_tokens, d_latent]
            log_s_end: [B]
            z_g_delta: [B, n_tokens, d_latent]
            log_s_delta: [B]
        """
        B = text_emb.shape[0]
        compute_dtype = self.text_proj.weight.dtype
        text_emb = text_emb.to(compute_dtype)
        z_t = z_t.to(compute_dtype)

        text_kv = self.text_proj(text_emb)               # [B, L, d_hidden]
        zt_kv = self.latent_proj(z_t)                    # [B, n_tokens, d_hidden]

        text_pad_mask = ~text_mask if text_mask is not None else None

        # Build slot stacks: [N, D] → [B, N, D]; concat end+delta along slot
        # axis so a single decoder pass handles both timescales (mirrors LCLGP).
        slots_end = (self.slot_end + self.tau_end).unsqueeze(0).expand(B, -1, -1)        # [B, N, D]
        slots_delta = (self.slot_delta + self.tau_delta).unsqueeze(0).expand(B, -1, -1)
        slots = torch.cat([slots_end, slots_delta], dim=1)                                # [B, 2N, D]

        for layer in self.layers:
            slots = layer(slots, text_kv, text_pad_mask, zt_kv)

        slots_end_out, slots_delta_out = slots[:, : self.n_tokens], slots[:, self.n_tokens :]

        z_g_end = self.out_end(slots_end_out)             # [B, N, d_latent]
        z_g_delta = self.out_delta(slots_delta_out)

        log_s_end = self.unc_end(slots_end_out.mean(dim=1)).squeeze(-1)                  # [B]
        log_s_delta = self.unc_delta(slots_delta_out.mean(dim=1)).squeeze(-1)
        log_s_end = log_s_end.clamp(self.log_sigma_min, self.log_sigma_max)
        log_s_delta = log_s_delta.clamp(self.log_sigma_min, self.log_sigma_max)
        return z_g_end, log_s_end, z_g_delta, log_s_delta

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _heteroscedastic_recon(
        self, z_g: torch.Tensor, z_true: torch.Tensor, log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """Heteroscedastic L1 NLL on (z_g, z_true). Stop-grad on target per §5.3.1.

        z_g [B, N, D], z_true [B, N, D], log_sigma [B] → scalar mean over batch.
        """
        l1 = (z_g - z_true.detach()).abs().mean(dim=(-2, -1))                            # [B]
        return (l1 / log_sigma.exp() + self.beta * log_sigma).mean()

    def _info_nce_loss(
        self,
        z_g_end: torch.Tensor,
        z_T: torch.Tensor,
        gather_fn=None,
    ) -> torch.Tensor:
        """Cross-task InfoNCE on patch-mean-pooled (z_g_end, z_T)."""
        B = z_g_end.shape[0]
        z_g_pool = z_g_end.mean(dim=-2)                                                   # [B, D]
        z_T_pool = z_T.mean(dim=-2)                                                       # [B, D]

        # Optional global gather for full DDP-batch InfoNCE.
        if gather_fn is not None:
            z_T_all = gather_fn(z_T_pool)
        else:
            z_T_all = z_T_pool

        z_g_n = F.normalize(z_g_pool.float(), dim=-1)
        z_T_n = F.normalize(z_T_all.float(), dim=-1)
        sim = z_g_n @ z_T_n.t() / self.tau                                                # [B, B_global]
        labels = torch.arange(B, device=sim.device)
        if gather_fn is not None and z_T_all.shape[0] != B:
            rank = getattr(gather_fn, "rank", 0)
            labels = labels + rank * B
        return F.cross_entropy(sim, labels)

    def _counterfactual_loss(
        self,
        z_g_end: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """Hinge: ||z_g_end − z_g_end_cf||_1 ≥ cf_margin, no grad through CF."""
        with torch.no_grad():
            z_g_end_cf, _, _, _ = self._forward_core(text_emb, text_mask, torch.zeros_like(z_t))
        z_g_end_cf = z_g_end_cf.detach()
        diff = (z_g_end - z_g_end_cf).abs().mean(dim=(-2, -1))                           # [B]
        return torch.clamp(self.cf_margin - diff, min=0.0).mean()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, Any], gather_fn=None) -> Dict[str, torch.Tensor]:
        """3-loss training objective: recon + ctr + cf. No routing / balance / repulse."""
        text_emb = batch["text_emb"]
        text_mask = batch["text_mask"]
        z_t = batch["z_t"]
        z_delta = batch["z_delta"]
        z_end = batch["z_end"]

        compute_dtype = self.text_proj.weight.dtype
        z_delta = z_delta.to(compute_dtype)
        z_end = z_end.to(compute_dtype)

        # Per-batch z_t dropout (training only) — D2 regularizer (mirrors v4/v5).
        z_t_input = z_t
        if self.training and self.z_dropout > 0 and torch.rand((), device=z_t.device).item() < self.z_dropout:
            z_t_input = torch.zeros_like(z_t)

        z_g_end, log_s_end, z_g_delta, log_s_delta = self._forward_core(text_emb, text_mask, z_t_input)

        l_recon_end = self._heteroscedastic_recon(z_g_end, z_end, log_s_end)
        l_recon_delta = self._heteroscedastic_recon(z_g_delta, z_delta, log_s_delta)

        # InfoNCE in z_end (d_latent) space — same-task pulls together, cross-task
        # pushes apart. Mirrors LCLGP v1-v5: contrastive target is the GT goal
        # latent, not text (text already informs forward via cross-attn).
        l_ctr = self._info_nce_loss(z_g_end, z_end, gather_fn=gather_fn)

        l_cf = self._counterfactual_loss(z_g_end, text_emb, text_mask, z_t)

        total = (
            l_recon_end
            + self.alpha * l_recon_delta
            + self.lambda_ctr * l_ctr
            + self.lambda_cf * l_cf
        )

        return {
            "loss": total,
            "l_recon_end": l_recon_end.detach(),
            "l_recon_delta": l_recon_delta.detach(),
            "l_ctr": l_ctr.detach(),
            "l_cf": l_cf.detach(),
            "sigma_end_mean": log_s_end.exp().mean().detach(),
            "sigma_delta_mean": log_s_delta.exp().mean().detach(),
        }

    @torch.no_grad()
    def predict_goal(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Inference-time goal projection. Used by run_diagnostics.py.

        Returns a K=1 shim: tensor outputs preserve a singleton K dim
        (``z_g_end_all [B, 1, N, D]``, ``log_sigma_end [B, 1]``,
        ``best_mode_end [B]`` all-zero) so the existing diagnostic pipeline
        works without modification (max over K=1 = identity, bincount with
        minlength=1 → cov_k0=1.0, etc.). D1-a pairwise cosine is gated at
        the diagnostic script level since K=1 has no pairs.
        """
        text_emb = batch["text_emb"]
        text_mask = batch["text_mask"]
        z_t = batch["z_t"]
        z_g_end, log_s_end, z_g_delta, log_s_delta = self._forward_core(text_emb, text_mask, z_t)

        B = z_g_end.shape[0]
        zeros = torch.zeros(B, dtype=torch.long, device=z_g_end.device)
        return {
            "z_g_end_best": z_g_end,                                  # [B, N, d_lat]
            "z_g_delta_best": z_g_delta,
            "z_g_end_all": z_g_end.unsqueeze(1),                      # [B, 1, N, d_lat]
            "z_g_delta_all": z_g_delta.unsqueeze(1),
            "log_sigma_end": log_s_end.unsqueeze(-1),                 # [B, 1]
            "log_sigma_delta": log_s_delta.unsqueeze(-1),
            "best_mode_end": zeros,
            "best_mode_delta": zeros.clone(),
            "best_mode_end_sigma": zeros.clone(),
            "best_mode_delta_sigma": zeros.clone(),
        }


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------


def _fake_batch(B=4, L=24, n_tokens=32, d_text=2560, d_latent=1408, dtype=torch.float32, device="cpu"):
    """Synthesize a batch matching the W2 collate contract."""
    text_emb = torch.randn(B, L, d_text, dtype=dtype, device=device)
    text_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    text_mask[:, L // 2 :] = False
    return {
        "text_emb": text_emb,
        "text_mask": text_mask,
        "z_t": torch.randn(B, n_tokens, d_latent, dtype=dtype, device=device),
        "z_delta": torch.randn(B, n_tokens, d_latent, dtype=dtype, device=device),
        "z_end": torch.randn(B, n_tokens, d_latent, dtype=dtype, device=device),
        "task_id": ["t0", "t0", "t1", "t1"],
        "lang": ["a", "a", "b", "b"],
        "data_name": ["fake"] * B,
    }


def _smoke():
    """CPU smoke: forward grad on all 4 head params + predict_goal shapes; fp16 batch path."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "framework": {
            "name": "LCLGP_K1",
            "lclgp": {
                "d_text": 2560, "d_latent": 1408, "d_hidden": 256,
                "n_heads": 4, "n_layers": 2, "n_tokens": 32,
                "z_dropout": 0.0,
                "loss": {"alpha": 0.5, "beta": 0.5, "lambda_ctr": 0.1, "lambda_cf": 0.05,
                         "tau": 0.07, "cf_margin": 0.05,
                         "log_sigma_min": -5.0, "log_sigma_max": 1.5},
            },
        },
    })

    model = LCLGPK1(cfg)
    print(f"[smoke] params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"[smoke] K = {model.K} (must be 1)")
    assert model.K == 1

    # --- training path with fp32 batch ---
    model.train()
    batch = _fake_batch(B=4, L=24, n_tokens=32, dtype=torch.float32)
    out = model(batch)
    print(f"[smoke] train (fp32) loss = {out['loss'].item():.4f}")
    print(f"[smoke]   l_recon_end={out['l_recon_end'].item():.4f}  l_recon_delta={out['l_recon_delta'].item():.4f}")
    print(f"[smoke]   l_ctr={out['l_ctr'].item():.4f}  l_cf={out['l_cf'].item():.4f}")
    out["loss"].backward()

    grad_params = [(n, p.grad is not None and p.grad.abs().sum().item() > 0)
                   for n, p in model.named_parameters() if p.requires_grad]
    n_total = len(grad_params)
    n_with_grad = sum(1 for _, ok in grad_params if ok)
    print(f"[smoke] grad-receiving params: {n_with_grad}/{n_total}")
    head_names = ["out_end.weight", "out_delta.weight", "unc_end.weight", "unc_delta.weight"]
    for hn in head_names:
        ok = dict(grad_params).get(hn, False)
        assert ok, f"head param {hn} has no grad"
    print("[smoke] all 4 head params got non-zero grad ✓")

    # --- inference path ---
    model.eval()
    out_pg = model.predict_goal(batch)
    expected_shapes = {
        "z_g_end_best": (4, 32, 1408),
        "z_g_delta_best": (4, 32, 1408),
        "z_g_end_all": (4, 1, 32, 1408),
        "z_g_delta_all": (4, 1, 32, 1408),
        "log_sigma_end": (4, 1),
        "log_sigma_delta": (4, 1),
        "best_mode_end": (4,),
        "best_mode_delta": (4,),
        "best_mode_end_sigma": (4,),
        "best_mode_delta_sigma": (4,),
    }
    for k, want in expected_shapes.items():
        got = tuple(out_pg[k].shape)
        assert got == want, f"{k}: want {want}, got {got}"
    print("[smoke] predict_goal shapes ✓ (K=1 shim preserved)")
    assert (out_pg["best_mode_end"] == 0).all(), "best_mode_end must be all-zero under K=1"

    # --- fp16 batch path (V-JEPA latent / text_emb on disk are fp16) ---
    model.eval()
    batch_fp16 = _fake_batch(B=2, L=8, n_tokens=8, dtype=torch.float16)
    out_fp16 = model.predict_goal(batch_fp16)
    print(f"[smoke] fp16 batch predict_goal OK: z_g_end_best dtype = {out_fp16['z_g_end_best'].dtype}")
    print("[smoke] all checks passed.")


if __name__ == "__main__":
    _smoke()
