"""Extract V-JEPA 2 ViT-g latents for every frame of a LeRobot v3 dataset.

T-W2.2.1 — see /root/.claude/plans/implementation-todo-md-w2-lazy-codd.md.

Output layout per shard (one HDF5 file per (dataset, rank) pair)::

    data/latents/<dataset>/<dataset>_rank{R:02d}.h5
      /traj_{traj_id:06d}/
          primary  [T, N_tokens, 1408]  fp16
          wrist    [T, N_tokens, 1408]  fp16   # only if dataset exposes wrist view
          lang     scalar str            (utf8)
          length   scalar int
      /index       compound dataset (traj_id:int64, length:int64)
      /attrs       data_name, robot_type, num_frames, img_size, patch_size, mean, std

Multi-GPU sharding: pass --rank/--world-size; each rank handles
trajectories where (idx % world_size == rank). Shards are independent files
that the latent dataloader (T-W2.2.4) reads transparently.

Dry-run example (one GPU, 10 trajectories of LIBERO-Long)::

    python scripts/extract_vjepa_latents.py \
        --mixture pav_libero_long --max-trajs 10 \
        --output-dir data/latents/dryrun

Full extraction (8 GPUs)::

    torchrun --nproc_per_node=8 scripts/extract_vjepa_latents.py \
        --mixture pav_libero --output-dir data/latents/pav_libero
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate_fn_map  # noqa: F401  (force import)
from tqdm.auto import tqdm

from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.modules.world_model.vjepa2 import VJEPA2Encoder

VJEPA_IMAGENET_MEAN = (0.485, 0.456, 0.406)
VJEPA_IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V-JEPA 2 latent extraction (T-W2.2.1)")
    p.add_argument("--mixture", type=str, default=None,
                   help="A DATASET_NAMED_MIXTURES key (e.g. pav_libero, pav_libero_long).")
    p.add_argument("--data-name", type=str, default=None,
                   help="Single dataset directory name (alternative to --mixture).")
    p.add_argument("--robot-type", type=str, default=None,
                   help="Required when --data-name is used.")
    p.add_argument("--data-root-dir", type=Path,
                   default=Path("playground/Datasets/LEROBOT_LIBERO_DATA"),
                   help="Parent dir holding the LeRobot dataset folders.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write per-dataset HDF5 shards.")
    p.add_argument("--vjepa-ckpt", type=Path,
                   default=Path("playground/Pretrained_models/vjepa2_vitg/vjepa2-ac-vitg.pt"))
    p.add_argument("--num-frames", type=int, default=2)
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Frames per encoder forward.")
    p.add_argument("--max-trajs", type=int, default=None,
                   help="Cap trajectories per dataset (dry-run).")
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--video-backend", type=str, default="torchvision_av")
    p.add_argument("--no-wrist", action="store_true",
                   help="Skip wrist view even if dataset has one (saves ~50% disk).")
    return p.parse_args()


def resolve_targets(args) -> list[tuple[str, str]]:
    """Return [(data_name, robot_type), ...]."""
    if args.data_name is not None:
        if args.robot_type is None:
            raise ValueError("--robot-type required when --data-name is used")
        return [(args.data_name, args.robot_type)]
    if args.mixture is None:
        raise ValueError("Pass --mixture or --data-name")
    if args.mixture not in DATASET_NAMED_MIXTURES:
        raise KeyError(f"Mixture {args.mixture!r} not registered. "
                       f"Available examples: {sorted(DATASET_NAMED_MIXTURES.keys())[:10]}...")
    return [(d, r) for (d, _, r) in DATASET_NAMED_MIXTURES[args.mixture]]


def normalize_frame(img_uint8: np.ndarray) -> torch.Tensor:
    """`(H, W, 3) uint8` → `(3, H, W) fp32` ImageNet-normalized."""
    arr = torch.from_numpy(img_uint8).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(VJEPA_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(VJEPA_IMAGENET_STD).view(3, 1, 1)
    return (arr - mean) / std


def build_encoder(args) -> VJEPA2Encoder:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    enc = VJEPA2Encoder(
        ckpt_path=str(args.vjepa_ckpt),
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        dtype=dtype,
    ).to(args.device).eval()
    return enc


def encode_traj(
    enc: VJEPA2Encoder,
    ds,
    traj_id: int,
    traj_length: int,
    *,
    batch_size: int,
    img_size: int,
    device: str,
    use_wrist: bool,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Encode every frame of one trajectory. Returns (primary[T,N,D], wrist or None, lang)."""
    primary_chunks: list[np.ndarray] = []
    wrist_chunks: list[np.ndarray] = []
    lang: str | None = None

    for start in range(0, traj_length, batch_size):
        end = min(start + batch_size, traj_length)
        prim_batch = []
        wrist_batch = []
        for f in range(start, end):
            step = ds.get_step_data(traj_id, f)
            prim_img = step["video.primary_image"][0]  # (H, W, 3) uint8
            prim_batch.append(normalize_frame(prim_img))
            if use_wrist and "video.wrist_image" in step:
                wrist_batch.append(normalize_frame(step["video.wrist_image"][0]))
            if lang is None:
                lang_field = step.get("annotation.human.action.task_description", None)
                if isinstance(lang_field, list):
                    lang = lang_field[0] if lang_field else ""
                elif isinstance(lang_field, str):
                    lang = lang_field
                else:
                    lang = ""

        prim_tensor = torch.stack(prim_batch).to(device, non_blocking=True)
        if prim_tensor.shape[-1] != img_size:
            raise RuntimeError(
                f"primary image is {prim_tensor.shape[-1]}px but encoder expects {img_size}; "
                f"add resize logic upstream."
            )
        z_primary = enc.encode(prim_tensor).float().cpu().numpy().astype(np.float16)
        primary_chunks.append(z_primary)

        if wrist_batch:
            wrist_tensor = torch.stack(wrist_batch).to(device, non_blocking=True)
            z_wrist = enc.encode(wrist_tensor).float().cpu().numpy().astype(np.float16)
            wrist_chunks.append(z_wrist)

    z_primary = np.concatenate(primary_chunks, axis=0)
    z_wrist = np.concatenate(wrist_chunks, axis=0) if wrist_chunks else None
    return z_primary, z_wrist, lang or ""


def write_shard(
    h5_path: Path,
    *,
    data_name: str,
    robot_type: str,
    encoder_meta: dict,
    traj_records: list[tuple[int, np.ndarray, np.ndarray | None, str]],
) -> None:
    """One HDF5 per (dataset, rank). All trajs written in order."""
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.attrs["data_name"] = data_name
        f.attrs["robot_type"] = robot_type
        for k, v in encoder_meta.items():
            f.attrs[k] = v

        index_dtype = np.dtype([("traj_id", "<i8"), ("length", "<i8")])
        index_arr = np.zeros(len(traj_records), dtype=index_dtype)

        for i, (traj_id, z_p, z_w, lang) in enumerate(traj_records):
            g = f.create_group(f"traj_{traj_id:06d}")
            g.create_dataset("primary", data=z_p, chunks=(1, *z_p.shape[1:]),
                             compression="gzip", compression_opts=1)
            if z_w is not None:
                g.create_dataset("wrist", data=z_w, chunks=(1, *z_w.shape[1:]),
                                 compression="gzip", compression_opts=1)
            g.attrs["lang"] = lang
            g.attrs["length"] = z_p.shape[0]
            index_arr[i]["traj_id"] = traj_id
            index_arr[i]["length"] = z_p.shape[0]

        f.create_dataset("index", data=index_arr)


def main() -> None:
    args = parse_args()
    targets = resolve_targets(args)
    if args.rank == 0:
        print(f"[rank 0] Targets ({len(targets)}): {targets}")

    if args.dtype == "bf16" and not torch.cuda.is_available():
        raise RuntimeError("bf16 requires CUDA")

    encoder = build_encoder(args)
    encoder_meta = {
        "num_frames": args.num_frames,
        "tubelet_size": args.tubelet_size,
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "dtype": args.dtype,
        "vjepa_ckpt": str(args.vjepa_ckpt),
        "imagenet_mean": json.dumps(VJEPA_IMAGENET_MEAN),
        "imagenet_std": json.dumps(VJEPA_IMAGENET_STD),
    }

    for data_name, robot_type in targets:
        if robot_type not in ROBOT_TYPE_CONFIG_MAP:
            raise KeyError(f"robot_type {robot_type!r} not in ROBOT_TYPE_CONFIG_MAP. "
                           f"Did the data_registry import fail?")

        ds = make_LeRobotSingleDataset(
            data_root_dir=args.data_root_dir,
            data_name=data_name,
            robot_type=robot_type,
            delete_pause_frame=False,
            data_cfg={"video_backend": args.video_backend},
        )
        traj_ids = ds.trajectory_ids
        traj_lengths = ds.trajectory_lengths
        if args.max_trajs is not None:
            traj_ids = traj_ids[: args.max_trajs]
            traj_lengths = traj_lengths[: args.max_trajs]

        my_indices = [i for i in range(len(traj_ids)) if i % args.world_size == args.rank]
        if not my_indices:
            print(f"[rank {args.rank}] {data_name}: nothing to do")
            continue

        # Probe wrist availability once via the first sample.
        probe = ds.get_step_data(int(traj_ids[my_indices[0]]), 0)
        has_wrist = (not args.no_wrist) and ("video.wrist_image" in probe)

        records: list[tuple[int, np.ndarray, np.ndarray | None, str]] = []
        t0 = time.time()
        for i in tqdm(my_indices, desc=f"rank{args.rank} {data_name}"):
            traj_id = int(traj_ids[i])
            traj_len = int(traj_lengths[i])
            z_p, z_w, lang = encode_traj(
                encoder, ds, traj_id, traj_len,
                batch_size=args.batch_size,
                img_size=args.img_size,
                device=args.device,
                use_wrist=has_wrist,
            )
            records.append((traj_id, z_p, z_w, lang))

        shard_path = args.output_dir / data_name / f"{data_name}_rank{args.rank:02d}.h5"
        write_shard(shard_path, data_name=data_name, robot_type=robot_type,
                    encoder_meta=encoder_meta, traj_records=records)
        elapsed = time.time() - t0
        size_mb = shard_path.stat().st_size / (1024 ** 2)
        print(f"[rank {args.rank}] {data_name}: {len(records)} trajs, "
              f"{elapsed:.1f}s, shard={shard_path} ({size_mb:.1f} MB), wrist={has_wrist}")


if __name__ == "__main__":
    main()
