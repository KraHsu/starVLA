"""Project V-JEPA cache features into the VLM hidden space."""

from __future__ import annotations

import torch
import torch.nn as nn


class VJepaProjector(nn.Module):
    """Project pooled/token V-JEPA features into the VLM hidden space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        fusion: str | None = None,
        num_attention_heads: int = 8,
        use_film_gating: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or output_dim)
        self.output_dim = int(output_dim)
        if fusion is None:
            fusion = "film_gating" if use_film_gating else "residual_add"
        self.fusion = str(fusion)
        self.use_film_gating = self.fusion == "film_gating"

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.out_norm = nn.LayerNorm(output_dim)

        if self.use_film_gating:
            self.gamma = nn.Linear(output_dim, output_dim)
            self.beta = nn.Linear(output_dim, output_dim)
        else:
            self.gamma = None
            self.beta = None

        if self.fusion == "concat_tokens":
            self.concat_mixer = nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=int(num_attention_heads),
                dim_feedforward=max(hidden_dim, output_dim * 2),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        else:
            self.concat_mixer = None

        if self.fusion == "cross_attn":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=int(num_attention_heads),
                dropout=0.0,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(output_dim)
        else:
            self.cross_attn = None
            self.cross_attn_norm = None

        supported = {"film_gating", "concat_tokens", "cross_attn", "residual_add"}
        if self.fusion not in supported:
            raise ValueError(f"Unsupported V-JEPA fusion mode '{self.fusion}'. Expected one of {sorted(supported)}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.out_norm(self.mlp(features))
        return projected

    def apply_to_queries(
        self,
        projected: torch.Tensor,
        queries: torch.Tensor,
        projected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse projected V-JEPA features into action queries.

        Args:
            projected: [B, H] pooled condition or [B, K, H] token sequence
            queries: [B, T, H]
            projected_tokens: optional [B, K, H] token sequence
        Returns:
            [B, T, H]
        """
        if projected.dim() == 3:
            pooled = projected.mean(dim=1)
            source_tokens = projected if projected_tokens is None else projected_tokens
        else:
            pooled = projected
            source_tokens = projected_tokens if projected_tokens is not None else projected.unsqueeze(1)

        cond = pooled.unsqueeze(1)
        if self.fusion == "film_gating":
            gamma = torch.tanh(self.gamma(pooled)).unsqueeze(1)
            beta = self.beta(pooled).unsqueeze(1)
            return queries * (1.0 + gamma) + beta + cond
        if self.fusion == "concat_tokens":
            mixed = self.concat_mixer(torch.cat([source_tokens, queries], dim=1))
            return mixed[:, -queries.shape[1] :, :]
        if self.fusion == "cross_attn":
            attn_out, _ = self.cross_attn(queries, source_tokens, source_tokens, need_weights=False)
            return self.cross_attn_norm(queries + attn_out)
        return queries + cond
