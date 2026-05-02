# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""V-JEPA 2 / V-JEPA 2-AC world model wrapper for Plan-and-Verify."""

import os
import sys
from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn

_VJEPA2_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../third_party/vjepa2")
)
if _VJEPA2_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA2_ROOT)

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

VJEPA2_HIDDEN_DIM = 1408


def _strip_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        out[k] = v
    return out


def _load_ckpt(path: Optional[str], key: Optional[str] = None):
    if path is None or not os.path.exists(path):
        return None
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if key is not None and isinstance(sd, dict) and key in sd:
        sd = sd[key]
    return _strip_prefix(sd)


class VJEPA2Encoder(nn.Module):
    """ViT-g video encoder. Accepts [B,C,T,H,W] or [B,C,H,W] frames."""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        num_frames: int = 2,
        tubelet_size: int = 2,
        img_size: int = 256,
        patch_size: int = 16,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        from src.models import vision_transformer as vit  # type: ignore

        self.encoder = vit.vit_giant_xformers(
            patch_size=patch_size,
            img_size=(img_size, img_size),
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )
        sd = _load_ckpt(ckpt_path, key="encoder")
        if sd is not None:
            missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
            logger.info(f"VJEPA2Encoder loaded {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        self.encoder = self.encoder.to(dtype=dtype)
        self.dtype = dtype
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.img_size = img_size

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 4:
            frames = frames.unsqueeze(2).expand(-1, -1, self.num_frames, -1, -1).contiguous()
        return self.encoder(frames.to(self.dtype))


class VJEPA2ACPredictor(nn.Module):
    """V-JEPA 2-AC latent action-conditioned predictor."""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        encoder_embed_dim: int = VJEPA2_HIDDEN_DIM,
        num_frames: int = 2,
        tubelet_size: int = 2,
        img_size: int = 256,
        patch_size: int = 16,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        from src.models import ac_predictor as vit_ac  # type: ignore

        self.predictor = vit_ac.vit_ac_predictor(
            img_size=(img_size, img_size),
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=encoder_embed_dim,
        )
        sd = _load_ckpt(ckpt_path, key="predictor")
        if sd is not None:
            missing, unexpected = self.predictor.load_state_dict(sd, strict=False)
            logger.info(f"VJEPA2ACPredictor loaded {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        self.predictor = self.predictor.to(dtype=dtype)
        self.dtype = dtype

    @torch.no_grad()
    def step(self, z: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # Thin pass-through to V-JEPA 2-AC predictor.
        # Shapes: z [B,N_ctxt,D]; actions [B,T,action_dim]; states [B,T,state_dim].
        return self.predictor(z.to(self.dtype), actions.to(self.dtype), states.to(self.dtype))

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, s0: torch.Tensor, a_chunk: torch.Tensor) -> torch.Tensor:
        # Full multi-step rollout API is finalized in W4 (MSFV). For now, single-shot.
        return self.step(z0, a_chunk, s0)


_VJEPA2Output = namedtuple("VJEPA2Output", ["hidden_states", "loss"])
_VJEPA2Output.__new__.__defaults__ = (None,) * len(_VJEPA2Output._fields)


class _ConfigShim:
    class config:
        hidden_size = VJEPA2_HIDDEN_DIM


class _VJEPA2_Interface(nn.Module):
    """Minimal W1 wrapper. Full build_inputs/forward come in W3 (LCLGP)."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config
        wm_cfg = config.framework.get("world_model", {}) if config is not None else {}
        encoder_ckpt = wm_cfg.get("encoder_ckpt_path", None)
        predictor_ckpt = wm_cfg.get("predictor_ckpt_path", None)
        dtype = torch.bfloat16 if wm_cfg.get("dtype", "fp16") == "bf16" else torch.float16

        self.encoder = VJEPA2Encoder(ckpt_path=encoder_ckpt, dtype=dtype)
        self.predictor = (
            VJEPA2ACPredictor(ckpt_path=predictor_ckpt, dtype=dtype)
            if predictor_ckpt is not None
            else None
        )
        self._hidden_size = VJEPA2_HIDDEN_DIM

    def build_inputs(self, images, instructions=None, **kwargs):
        return {"frames": images}

    def forward(self, frames=None, **kwargs):
        z = self.encoder.encode(frames)
        return _VJEPA2Output(hidden_states=(z,))

    def generate(self, **kwargs):
        raise NotImplementedError("V-JEPA 2 is encoder/predictor only; no pixel decoder.")

    @property
    def model(self):
        return _ConfigShim
