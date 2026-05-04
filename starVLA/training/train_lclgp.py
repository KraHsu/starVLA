"""LCLGP trainer (Plan-and-Verify W3.3).

DeepSpeed Zero-2 + bf16 training entry point for the LCLGP framework.
Distinct from train_starvla.py / train_starvlm.py because:

- The model returns a multi-component loss dict (no ``action_loss`` key);
  ``_train_step`` extracts ``output_dict["loss"]`` and copies every scalar
  from the dict into the W&B / TensorBoard log.
- The dataloader skips ``starVLA/dataloader/build_dataloader``; LCLGP
  consumes pre-extracted V-JEPA latents + text embeddings directly via
  :class:`LcLgpTripletDataset` and a :class:`DistributedTaskGroupedSampler`.
- The InfoNCE loss optionally gathers ``z_g_pool`` / ``z_T_pool`` across
  ranks (CLIP / SimCLR DDP pattern) so each rank still sees all-rank
  cross-task negatives despite the per-rank batch shrinking from 256 to 32
  on an 8-GPU cluster.

Launch::

    accelerate launch \
        --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
        --num_processes 8 \
        starVLA/training/train_lclgp.py \
        --config_yaml examples/PlanAndVerify/configs/lclgp_v1.yaml \
        --run_root_dir ./playground/Checkpoints \
        --run_id <name>
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from starVLA.datasets.distributed_task_grouped_sampler import DistributedTaskGroupedSampler
from starVLA.datasets.lclgp_triplet_dataset import LcLgpTripletDataset, collate_lclgp
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        os.makedirs(output_dir / "tb", exist_ok=True)
    return output_dir


def build_lclgp_dataloader(cfg, accelerator) -> DataLoader:
    data_cfg = cfg.data.lclgp_dataset
    sampler_cfg = cfg.data.sampler

    use_dryrun = bool(getattr(data_cfg, "use_dryrun", False))
    if use_dryrun and getattr(data_cfg, "dryrun_index", None):
        index_path = data_cfg.dryrun_index
    else:
        index_path = data_cfg.index_train

    dataset = LcLgpTripletDataset(
        index_path=index_path,
        latent_root=data_cfg.latent_root,
        text_emb_path=data_cfg.text_emb,
        view=getattr(data_cfg, "view", "primary"),
    )
    logger.info(
        f"LcLgpTripletDataset: {len(dataset)} samples, "
        f"{len(set(dataset.task_ids))} distinct tasks, text_hidden={dataset.text_hidden}"
    )

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    n_tasks_global = int(sampler_cfg.n_tasks)
    n_demos = int(sampler_cfg.n_demos)
    if n_tasks_global % world_size != 0:
        raise ValueError(
            f"sampler.n_tasks={n_tasks_global} must be divisible by world_size={world_size}"
        )
    n_tasks_per_rank = n_tasks_global // world_size

    sampler = DistributedTaskGroupedSampler(
        dataset.task_ids,
        n_tasks_per_rank=n_tasks_per_rank,
        n_demos=n_demos,
        rank=rank,
        world_size=world_size,
        seed=int(getattr(sampler_cfg, "seed", 0)),
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_lclgp,
        num_workers=int(getattr(cfg.trainer, "num_workers", 4)),
        pin_memory=True,
    )


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    if dist.is_initialized() and dist.get_rank() == 0:
        for g in optimizer.param_groups:
            logger.info(f"LR Group {g['name']}: lr={g['lr']}, num_params={len(g['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=getattr(cfg.trainer, "scheduler_specific_kwargs", None),
    )
    return optimizer, lr_scheduler


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class LcLgpTrainer(TrainerUtils):
    def __init__(self, cfg, model, dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        self.tb_writer = None
        self.best_val_loss = float("inf")
        self.total_batch_size = self._calculate_total_batch_size()

    # -- setup ----------------------------------------------------------------

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._save_initial_configs()

        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            ckpt = self.config.trainer.pretrained_checkpoint
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, ckpt, reload_modules=reload_modules)

        freeze_modules = getattr(self.config.trainer, "freeze_modules", "")
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        # DeepSpeed cannot infer batch size from a `batch_sampler`-only DataLoader,
        # so we surface it from the LCLGP sampler config before prepare().
        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is not None:
            sampler_cfg = self.config.data.sampler
            ws = max(1, self.accelerator.num_processes)
            per_gpu_bs = (int(sampler_cfg.n_tasks) // ws) * int(sampler_cfg.n_demos)
            ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = per_gpu_bs
            ds_plugin.deepspeed_config["train_batch_size"] = per_gpu_bs * ws * self.accelerator.gradient_accumulation_steps

        self.model, self.optimizer, self.dataloader = self.setup_distributed_training(
            self.accelerator, self.model, self.optimizer, self.dataloader
        )
        self._init_loggers()
        self._init_checkpointing()

    def _save_initial_configs(self):
        if not self.accelerator.is_main_process:
            return
        output_dir = Path(self.config.output_dir)
        full_cfg = self.config.unwrap() if isinstance(self.config, AccessTrackedConfig) else self.config
        OmegaConf.save(full_cfg, output_dir / "config.full.yaml", resolve=True)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)

    def _calculate_total_batch_size(self):
        sampler_cfg = self.config.data.sampler
        return int(sampler_cfg.n_tasks) * int(sampler_cfg.n_demos)

    def _init_loggers(self):
        if not self.accelerator.is_main_process:
            return
        wandb.init(
            name=self.config.run_id,
            dir=os.path.join(self.config.output_dir, "wandb"),
            project=getattr(self.config, "wandb_project", "pav-w3-lclgp"),
            entity=getattr(self.config, "wandb_entity", None),
            group="lclgp",
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.config.output_dir, "tb"))

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # -- gather helper for DDP InfoNCE ---------------------------------------

    def _build_gather_fn(self):
        """Return a callable that gathers a tensor across ranks while preserving local rank index."""
        accelerator = self.accelerator
        rank = accelerator.process_index

        def _gather(x: torch.Tensor) -> torch.Tensor:
            return accelerator.gather(x.contiguous())

        _gather.rank = rank
        return _gather

    # -- training loop --------------------------------------------------------

    def train(self):
        self._log_training_config()
        self.dataloader_iter = iter(self.dataloader)
        self.epoch_count = 0
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        gather_fn = self._build_gather_fn()

        while self.completed_steps < self.config.trainer.max_train_steps:
            batch = self._get_next_batch()
            metrics = self._train_step(batch, gather_fn=gather_fn)

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
                # LCLGP v3 path B: bal_temperature anneal reads training_step buffer.
                # No-op for frameworks without the buffer.
                _unwrapped = self.accelerator.unwrap_model(self.model)
                _ts = getattr(_unwrapped, "training_step", None)
                if isinstance(_ts, torch.Tensor):
                    _ts.fill_(self.completed_steps)

            self._log_metrics(metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _get_next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter, self.epoch_count = self._reset_dataloader(self.dataloader, self.epoch_count)
            return next(self.dataloader_iter)

    def _train_step(self, batch: Dict[str, Any], gather_fn=None) -> Dict[str, float]:
        log: Dict[str, float] = {}
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.model(batch, gather_fn=gather_fn)
                loss = output["loss"]
            self.accelerator.backward(loss)

            grad_clip = getattr(self.config.trainer, "gradient_clipping", None)
            if grad_clip is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()
            self.lr_scheduler.step()

            log["loss"] = loss.detach().float().item()
            for k, v in output.items():
                if k == "loss" or not isinstance(v, torch.Tensor):
                    continue
                if v.ndim == 0:
                    log[k] = v.detach().float().item()
            # Soft mode-balance histograms — written every step but only flushed at logging cadence.
            if "mode_argmin_end" in output:
                self._last_mode_hist_end = output["mode_argmin_end"].detach().cpu()
            if "mode_argmin_delta" in output:
                self._last_mode_hist_delta = output["mode_argmin_delta"].detach().cpu()
            if "pi_bar_end" in output:
                pi = output["pi_bar_end"].detach().float().cpu()
                for k in range(pi.shape[0]):
                    log[f"pi_bar_end/k{k}"] = float(pi[k])
            if "pi_bar_delta" in output:
                pi = output["pi_bar_delta"].detach().float().cpu()
                for k in range(pi.shape[0]):
                    log[f"pi_bar_delta/k{k}"] = float(pi[k])
        return log

    # -- logging --------------------------------------------------------------

    def _log_metrics(self, metrics: Dict[str, float]):
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return
        if not self.accelerator.is_main_process:
            return
        metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        try:
            metrics["epoch"] = round(self.completed_steps / max(1, len(self.dataloader)), 3)
        except TypeError:
            pass
        wandb.log(metrics, step=self.completed_steps)
        if self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, self.completed_steps)
            hist_every = self.config.trainer.logging_frequency * 5
            if self.completed_steps % hist_every == 0 and getattr(self, "_last_mode_hist_end", None) is not None:
                self.tb_writer.add_histogram("mode_argmin_end", self._last_mode_hist_end, self.completed_steps)
                self.tb_writer.add_histogram("mode_argmin_delta", self._last_mode_hist_delta, self.completed_steps)
                wandb.log(
                    {
                        "mode_argmin_end_hist": wandb.Histogram(self._last_mode_hist_end.numpy()),
                        "mode_argmin_delta_hist": wandb.Histogram(self._last_mode_hist_delta.numpy()),
                    },
                    step=self.completed_steps,
                )
        logger.info(f"Step {self.completed_steps}: {metrics}")

    # -- checkpoint -----------------------------------------------------------

    def _save_checkpoint(self):
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            else:
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps({"steps": self.completed_steps}) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            if isinstance(self.config, AccessTrackedConfig):
                self.config.save_accessed_config(
                    Path(self.config.output_dir) / "config.yaml", use_original_values=False
                )
        self.accelerator.wait_for_everyone()

    def _finalize_training(self):
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_dir = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_dir, "model.safetensors"))
            else:
                torch.save(state_dict, os.path.join(final_dir, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_dir}")
            if self.tb_writer is not None:
                self.tb_writer.close()
            wandb.finish()
        self.accelerator.wait_for_everyone()

    def _log_training_config(self):
        if not self.accelerator.is_main_process:
            return
        logger.info("***** LCLGP Training *****")
        logger.info(f"  max_train_steps   = {self.config.trainer.max_train_steps}")
        logger.info(f"  total_batch_size  = {self.total_batch_size}  (n_tasks×n_demos × world_size)")
        logger.info(f"  num_processes     = {self.accelerator.num_processes}")
        logger.info(f"  grad_accum_steps  = {self.accelerator.gradient_accumulation_steps}")
        logger.info(f"  logging_frequency = {self.config.trainer.logging_frequency}")
        logger.info(f"  save_interval     = {self.config.trainer.save_interval}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(cfg) -> None:
    cfg = wrap_config(cfg)
    output_dir = setup_directories(cfg)
    logger.info(f"Output dir: {output_dir}")

    model = build_framework(cfg)
    dataloader = build_lclgp_dataloader(cfg, accelerator)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg)

    trainer = LcLgpTrainer(
        cfg=cfg,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg.config_yaml = args.config_yaml

    main(cfg)
