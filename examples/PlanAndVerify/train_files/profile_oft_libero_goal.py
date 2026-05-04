"""Stage 1 / §1.3 — peak-VRAM and steps/s profile for the StarVLA-OFT
LIBERO-Goal training config, without modifying the shared training loop.

This script mirrors the bring-up path in starVLA/training/train_starvla.py
(same Accelerator, DeepSpeed plugin, framework, dataloader, optimizer) but
replaces the training loop with a fixed 50-step measurement loop that
records:

    - peak GPU memory per rank via torch.cuda.max_memory_allocated()
    - step throughput (steps/s, samples/s) over a warmup-trimmed window

Output: writes a JSON report to <run_root_dir>/<run_id>/profile.json on rank 0.

Launch identically to the real trainer:

    .venv/bin/accelerate launch \\
        --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \\
        --num_processes 8 \\
        examples/PlanAndVerify/train_files/profile_oft_libero_goal.py \\
        --config_yaml examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml \\
        --num_steps 50 --warmup 5

Why a separate script (not a flag in train_starvla.py): CLAUDE.md flags
training/train_starvla.py as a shared module that PR contributors must not
modify. This file lives in the PAV example folder and is the canonical
"how do we get peak-VRAM / throughput numbers" entry point for the thesis.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from omegaconf import OmegaConf

# Mirror train_starvla.py module-level setup so that DeepSpeed picks up the
# same accelerate launcher config.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")  # profile run never needs wandb

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

from starVLA.dataloader import build_dataloader  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402
from starVLA.model.framework.share_tools import apply_config_compat  # noqa: E402
from starVLA.training.trainer_utils.config_tracker import wrap_config  # noqa: E402
from starVLA.training.trainer_utils.trainer_tools import (  # noqa: E402
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)


def main(cfg, num_steps: int, warmup: int) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = (rank == 0)

    cfg = wrap_config(cfg)
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed + rank)

    model = build_framework(cfg)
    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    optimizer, _ = _build_optimizer_only(model, cfg)

    # Apply same freeze rules the real trainer would.
    freeze_modules = getattr(cfg.trainer, "freeze_modules", None)
    model = TrainerUtils.freeze_backbones(model, freeze_modules=freeze_modules)
    if is_main:
        TrainerUtils.print_trainable_parameters(model)

    accelerator.dataloader_config.dispatch_batches = False
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    torch.cuda.reset_peak_memory_stats()

    step_seconds: list[float] = []
    data_seconds: list[float] = []

    data_iter = iter(dataloader)
    for step in range(num_steps):
        t_data_start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        t_data_end = time.perf_counter()

        t_step_start = time.perf_counter()
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
                loss = output["action_loss"]
            accelerator.backward(loss)
            if cfg.trainer.gradient_clipping is not None:
                accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)
            optimizer.step()
        torch.cuda.synchronize()
        t_step_end = time.perf_counter()

        data_seconds.append(t_data_end - t_data_start)
        step_seconds.append(t_step_end - t_step_start)

        if is_main:
            accelerator.print(
                f"[profile] step {step + 1:3d}/{num_steps}  "
                f"data={data_seconds[-1] * 1000:6.1f} ms  "
                f"model={step_seconds[-1] * 1000:6.1f} ms  "
                f"loss={loss.item():.4f}"
            )

    timed = step_seconds[warmup:]
    timed_data = data_seconds[warmup:]
    if not timed:
        timed = step_seconds
        timed_data = data_seconds

    peak_alloc_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()

    # Gather peak memory across ranks so rank 0 can report the worst rank.
    peak_alloc_t = torch.tensor([peak_alloc_bytes], device=accelerator.device)
    peak_reserved_t = torch.tensor([peak_reserved_bytes], device=accelerator.device)
    if dist.is_initialized():
        gather_alloc = [torch.zeros_like(peak_alloc_t) for _ in range(dist.get_world_size())]
        gather_reserved = [torch.zeros_like(peak_reserved_t) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_alloc, peak_alloc_t)
        dist.all_gather(gather_reserved, peak_reserved_t)
        peak_alloc_per_rank = [int(t.item()) for t in gather_alloc]
        peak_reserved_per_rank = [int(t.item()) for t in gather_reserved]
    else:
        peak_alloc_per_rank = [peak_alloc_bytes]
        peak_reserved_per_rank = [peak_reserved_bytes]

    if is_main:
        mean_step = statistics.mean(timed)
        mean_data = statistics.mean(timed_data)
        steps_per_s = 1.0 / mean_step
        per_device_bs = cfg.datasets.vla_data.per_device_batch_size
        world_size = accelerator.num_processes
        samples_per_s = steps_per_s * per_device_bs * world_size

        report = {
            "config": {
                "config_yaml": cfg.get("config_yaml", None),
                "max_train_steps_for_profile": num_steps,
                "warmup_steps": warmup,
                "per_device_batch_size": per_device_bs,
                "world_size": world_size,
                "effective_batch_size": per_device_bs * world_size,
            },
            "timing": {
                "mean_step_ms": mean_step * 1000,
                "p50_step_ms": statistics.median(timed) * 1000,
                "max_step_ms": max(timed) * 1000,
                "mean_data_ms": mean_data * 1000,
                "steps_per_s": steps_per_s,
                "samples_per_s": samples_per_s,
            },
            "memory": {
                "peak_allocated_GB_per_rank": [b / (1024 ** 3) for b in peak_alloc_per_rank],
                "peak_reserved_GB_per_rank": [b / (1024 ** 3) for b in peak_reserved_per_rank],
                "peak_allocated_GB_max": max(peak_alloc_per_rank) / (1024 ** 3),
                "peak_reserved_GB_max": max(peak_reserved_per_rank) / (1024 ** 3),
            },
        }
        out_path = output_dir / "profile.json"
        with out_path.open("w") as f:
            json.dump(report, f, indent=2)
        accelerator.print("\n=== Stage 1 profile summary ===")
        accelerator.print(json.dumps(report, indent=2))
        accelerator.print(f"Wrote {out_path}")

    accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_optimizer_only(model, cfg):
    """Lightweight version of setup_optimizer_and_scheduler — we do not need
    a real LR scheduler for profiling."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    return optimizer, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    if "run_id" not in cfg or not cfg.run_id:
        cfg.run_id = "profile_oft_libero_goal"
    main(cfg, num_steps=args.num_steps, warmup=args.warmup)
