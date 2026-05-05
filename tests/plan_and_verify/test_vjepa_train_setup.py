from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


class _TinyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen_vl_interface = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 8))
        self.vision_tower = nn.Linear(8, 8)
        self.vjepa_projector = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        self.action_model = nn.Linear(8, 7)
        self.misc = nn.Linear(8, 8)


def _make_cfg():
    return OmegaConf.create(
        {
            "trainer": {
                "freeze_modules": "qwen_vl_interface,vision_tower",
                "learning_rate": {
                    "base": 1.0e-5,
                    "vjepa_projector": 1.0e-4,
                    "action_model": 5.0e-5,
                },
            }
        }
    )


def test_vjepa_freeze_modules_respected():
    model = _TinyTrainModel()
    TrainerUtils.freeze_backbones(model, freeze_modules="qwen_vl_interface,vision_tower")

    assert all(not p.requires_grad for p in model.qwen_vl_interface.parameters())
    assert all(not p.requires_grad for p in model.vision_tower.parameters())
    assert all(p.requires_grad for p in model.vjepa_projector.parameters())
    assert all(p.requires_grad for p in model.action_model.parameters())


def test_vjepa_optimizer_param_groups_include_projector_and_exclude_frozen_params():
    model = _TinyTrainModel()
    cfg = _make_cfg()
    TrainerUtils.freeze_backbones(model, freeze_modules=cfg.trainer.freeze_modules)
    param_groups = build_param_lr_groups(model, cfg)

    group_names = {group["name"] for group in param_groups}
    assert "vjepa_projector" in group_names
    assert "action_model" in group_names
    assert "base" in group_names

    frozen_param_ids = {id(p) for p in model.qwen_vl_interface.parameters()} | {
        id(p) for p in model.vision_tower.parameters()
    }
    grouped_param_ids = {id(param) for group in param_groups for param in group["params"]}

    assert frozen_param_ids.isdisjoint(grouped_param_ids)
    assert {id(p) for p in model.vjepa_projector.parameters()}.issubset(grouped_param_ids)
    assert {id(p) for p in model.action_model.parameters()}.issubset(grouped_param_ids)
