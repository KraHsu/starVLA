"""Sanity checks for the Stage 3 QwenOFT + V-JEPA training path.

This utility is intentionally lightweight:
  1. Builds the framework from YAML
  2. Applies trainer.freeze_modules and reports trainable counts
  3. Builds optimizer LR groups and prints their sizes
  4. Optionally runs one real forward/backward step on the configured dataset

Example:

    .venv/bin/python examples/PlanAndVerify/train_files/check_oft_vjepa_setup.py \
      --config_yaml examples/PlanAndVerify/train_files/starvla_oft_vjepa_libero_goal.yaml
"""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        default="examples/PlanAndVerify/train_files/starvla_oft_vjepa_libero_goal.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run_single_batch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    model = build_framework(cfg)

    freeze_modules = getattr(cfg.trainer, "freeze_modules", "")
    TrainerUtils.freeze_backbones(model, freeze_modules=freeze_modules)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_params={total_params}")
    print(f"trainable_params={trainable_params}")

    param_groups = build_param_lr_groups(model, cfg)
    for group in param_groups:
        print(f"group={group['name']} lr={group['lr']} params={sum(p.numel() for p in group['params'])}")

    if not args.run_single_batch:
        return 0

    dataset = get_vla_dataset(cfg.datasets.vla_data)
    sample = dataset[0]
    device = torch.device(args.device)
    model = model.to(device)

    output = model.forward([sample])
    loss = output["action_loss"]
    loss.backward()
    print(f"single_batch_action_loss={float(loss.detach().cpu())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
