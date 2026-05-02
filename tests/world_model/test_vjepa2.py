"""Smoke tests for V-JEPA 2 wrapper. Most tests need a GPU."""

import pytest
import torch

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_imports():
    from starVLA.model.modules.world_model.vjepa2 import (
        VJEPA2Encoder,
        VJEPA2ACPredictor,
        _VJEPA2_Interface,
        VJEPA2_HIDDEN_DIM,
    )
    assert VJEPA2_HIDDEN_DIM == 1408


def test_factory_routes_to_vjepa2():
    from omegaconf import OmegaConf
    from starVLA.model.modules.world_model import get_world_model
    from starVLA.model.modules.world_model.vjepa2 import _VJEPA2_Interface

    cfg = OmegaConf.create(
        {"framework": {"world_model": {"base_wm": "vjepa2_vitg"}}}
    )
    obj = get_world_model(cfg)
    assert isinstance(obj, _VJEPA2_Interface)
    assert obj._hidden_size == 1408


@CUDA_REQUIRED
def test_encoder_forward_shape():
    from starVLA.model.modules.world_model.vjepa2 import VJEPA2Encoder

    enc = VJEPA2Encoder(ckpt_path=None, num_frames=2, tubelet_size=2, img_size=256, patch_size=16).cuda()
    frames = torch.randn(1, 3, 256, 256, device="cuda")
    z = enc.encode(frames)
    assert z.dim() == 3
    assert z.shape[0] == 1
    assert z.shape[-1] == 1408


@CUDA_REQUIRED
def test_predictor_step_shape():
    from starVLA.model.modules.world_model.vjepa2 import VJEPA2ACPredictor

    pred = VJEPA2ACPredictor(
        ckpt_path=None, encoder_embed_dim=1408,
        num_frames=2, tubelet_size=2, img_size=256, patch_size=16,
        dtype=torch.float32,
    ).cuda()
    B, T_ctxt, D = 1, 256, 1408
    T_pred, action_dim, state_dim = 1, 7, 7
    z0 = torch.randn(B, T_ctxt, D, device="cuda")
    actions = torch.randn(B, T_pred, action_dim, device="cuda")
    states = torch.randn(B, T_pred, state_dim, device="cuda")
    out = pred.step(z0, actions, states)
    assert out.dim() == 3 and out.shape[0] == B
