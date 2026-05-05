"""Project V-JEPA cache features into the VLM hidden space."""

from __future__ import annotations

import torch
import torch.nn as nn


class VJepaProjector(nn.Module):
    """Two-layer projector with optional FiLM-style query modulation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        use_film_gating: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or output_dim)
        self.output_dim = int(output_dim)
        self.use_film_gating = bool(use_film_gating)

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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.out_norm(self.mlp(features))
        return projected

    def apply_to_queries(self, projected: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """Fuse projected V-JEPA pooled features into action queries.

        Args:
            projected: [B, H]
            queries: [B, T, H]
        Returns:
            [B, T, H]
        """
        cond = projected.unsqueeze(1)
        if self.use_film_gating:
            gamma = torch.tanh(self.gamma(projected)).unsqueeze(1)
            beta = self.beta(projected).unsqueeze(1)
            return queries * (1.0 + gamma) + beta + cond
        return queries + cond

