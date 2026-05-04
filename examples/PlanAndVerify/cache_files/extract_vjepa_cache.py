"""Offline V-JEPA 2.1 cache extraction for Plan-and-Verify.

This script walks the underlying LeRobot single dataset episode by episode,
extracts frozen V-JEPA 2.1 ViT-B/16-384 features for the primary camera history
window, and writes one ``.npz`` file per episode.

Default target:
    - dataset mix: ``libero_goal``
    - camera key: ``video.primary_image``
    - model: ``V-JEPA 2.1 ViT-B/16 384``
    - history window: ``N=8``
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VJEPA2_ROOT = _REPO_ROOT / "third_party" / "vjepa2"
if _VJEPA2_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _VJEPA2_ROOT.as_posix())

from evals.video_classification_frozen.utils import make_transforms

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps


MODEL_NAME = "vjepa2_1_vit_base_384"
MODEL_CHECKPOINT_KEY = "ema_encoder"
MODEL_CHECKPOINT_FALLBACK_KEYS = ("encoder", "target_encoder")
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]
DEFAULT_PCA_FIGURE = _REPO_ROOT / "docs" / "figures" / "vjepa_pca_sanity.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--dataset_name", default="libero_goal")
    parser.add_argument("--camera_key", default="video.primary_image")
    parser.add_argument("--vjepa_ckpt", default=None)
    parser.add_argument("--output_dir", default="playground/cache/vjepa/vjepa2_1_vit_b_384/libero_goal")
    parser.add_argument("--num_history_frames", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument(
        "--max_gpu_memory_gb",
        type=float,
        default=None,
        help="Optional per-process CUDA allocator cap in GiB. Example: 100 limits each shard process to about 100 GiB.",
    )
    parser.add_argument("--save_tokens", action="store_true")
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--run_pca_sanity", action="store_true")
    parser.add_argument("--pca_num_samples", type=int, default=100)
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> tuple[torch.dtype, str]:
    requested = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype_name]
    if device.type == "cpu" and requested != torch.float32:
        print(
            f"[warn] device={device} does not run this script efficiently in {dtype_name}; using fp32 instead",
            file=sys.stderr,
        )
        return torch.float32, "fp32"
    return requested, dtype_name


def memory_fraction_from_gib(max_memory_gib: float, total_memory_bytes: int) -> float:
    if max_memory_gib <= 0:
        raise ValueError(f"max_gpu_memory_gb must be positive, got {max_memory_gib}")
    total_memory_gib = total_memory_bytes / (1024**3)
    return min(max_memory_gib / total_memory_gib, 1.0)


def configure_cuda_memory_cap(device: torch.device, max_memory_gib: float | None) -> None:
    if max_memory_gib is None:
        return
    if device.type != "cuda":
        print(f"[warn] --max_gpu_memory_gb is ignored for non-CUDA device {device}", file=sys.stderr)
        return

    properties = torch.cuda.get_device_properties(device)
    fraction = memory_fraction_from_gib(max_memory_gib, properties.total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    effective_gib = fraction * properties.total_memory / (1024**3)
    print(
        f"==> CUDA allocator cap for {device}: {effective_gib:.1f} GiB "
        f"({fraction:.3f} of {properties.total_memory / (1024**3):.1f} GiB)"
    )


def strip_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def patch_vjepa2_1_rope_dtype() -> None:
    """Keep RoPE-rotated q/k in the same dtype as qkv outputs.

    V-JEPA 2.1 builds RoPE positions as float32 tensors. In a bf16/fp16 model,
    the original helper promotes rotated q/k to float32 while v remains the
    model dtype, which makes PyTorch SDPA reject q/k/v. Patch only the runtime
    helper, leaving third_party source untouched.
    """
    from app.vjepa_2_1.models.utils import modules as vjepa_modules

    if getattr(vjepa_modules.rotate_queries_or_keys, "_starvla_dtype_safe", False):
        return

    original_rotate = vjepa_modules.rotate_queries_or_keys

    def dtype_safe_rotate_queries_or_keys(x, pos, n_registers, has_cls_first):
        return original_rotate(x, pos, n_registers, has_cls_first).to(dtype=x.dtype)

    dtype_safe_rotate_queries_or_keys._starvla_dtype_safe = True
    dtype_safe_rotate_queries_or_keys._starvla_original = original_rotate
    vjepa_modules.rotate_queries_or_keys = dtype_safe_rotate_queries_or_keys


def load_vjepa_encoder(
    ckpt_path: str | os.PathLike[str],
    num_frames: int,
    img_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    patch_vjepa2_1_rope_dtype()
    from app.vjepa_2_1.models import vision_transformer as vit

    encoder = vit.vit_base(
        patch_size=16,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=2,
        use_sdpa=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = None
    for key in (MODEL_CHECKPOINT_KEY, *MODEL_CHECKPOINT_FALLBACK_KEYS):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            break
    if state_dict is None:
        raise KeyError(
            f"Could not find one of checkpoint keys {(MODEL_CHECKPOINT_KEY, *MODEL_CHECKPOINT_FALLBACK_KEYS)} in {ckpt_path}"
        )

    state_dict = strip_prefix(state_dict)
    msg = encoder.load_state_dict(state_dict, strict=True)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(
            f"Unexpected V-JEPA encoder load result for {ckpt_path}: missing={msg.missing_keys}, unexpected={msg.unexpected_keys}"
        )

    encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def encoder_forward(encoder, batch: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    use_autocast = device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else nullcontext()
    with ctx:
        return encoder(batch)


def build_single_dataset(config_yaml: str, dataset_mix: str, camera_key: str, num_history_frames: int):
    cfg = OmegaConf.load(config_yaml)
    data_cfg = cfg.datasets.vla_data
    data_cfg.data_mix = dataset_mix

    if dataset_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(f"Unknown dataset mixture '{dataset_mix}'. Available keys include: {sorted(DATASET_NAMED_MIXTURES.keys())[:20]}")

    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[dataset_mix]
    unique_specs: list[tuple[str, float, str]] = []
    seen: set[tuple[str, str]] = set()
    for dataset_name, weight, robot_type in mixture_spec:
        dedup_key = (dataset_name, robot_type)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        unique_specs.append((dataset_name, weight, robot_type))

    if len(unique_specs) != 1:
        raise ValueError(
            f"This cache extractor currently expects exactly one underlying LeRobot dataset for mix '{dataset_mix}', got {len(unique_specs)}"
        )

    dataset_name, _weight, robot_type = unique_specs[0]
    dataset = make_LeRobotSingleDataset(
        Path(data_cfg.data_root_dir),
        dataset_name,
        robot_type,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )

    video_keys = dataset.modality_keys.get("video", [])
    if not video_keys:
        raise ValueError(f"Dataset '{dataset_name}' has no video modality keys")
    first_video_key = video_keys[0]
    if first_video_key != camera_key:
        raise ValueError(
            f"Extractor only uses the first video key. Expected '{camera_key}', found '{first_video_key}' in dataset '{dataset_name}'"
        )

    dataset._delta_indices[camera_key] = np.arange(-(num_history_frames - 1), 1, dtype=np.int64)
    return cfg, data_cfg, dataset_name, robot_type, dataset


def episode_output_path(output_dir: Path, trajectory_id: int) -> Path:
    return output_dir / "episodes" / f"episode_{trajectory_id:06d}.npz"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sample_tokens_uniform(tokens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    seq_len = tokens.shape[1]
    if seq_len == 0:
        raise ValueError("Encoder returned zero tokens")
    indices = torch.linspace(0, seq_len - 1, steps=num_tokens, device=tokens.device)
    indices = indices.round().long()
    return tokens.index_select(dim=1, index=indices)


def atomic_write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    tmp_path.replace(path)


def build_index(
    output_dir: Path,
    dataset_name: str,
    dataset_mix: str,
    camera_key: str,
    num_history_frames: int,
    img_size: int,
    checkpoint_path: str | None,
    dtype_name: str,
) -> dict:
    episodes_dir = output_dir / "episodes"
    files_payload = []
    feat_dim = None
    total_frames = 0
    num_tokens = 0
    token_mode = "disabled"

    for npz_path in sorted(episodes_dir.glob("episode_*.npz")):
        match = re.match(r"episode_(\d+)\.npz$", npz_path.name)
        if match is None:
            continue
        trajectory_id = int(match.group(1))
        with np.load(npz_path, allow_pickle=False) as data:
            pooled = data["pooled"]
            feat_dim = int(pooled.shape[1])
            num_frames = int(pooled.shape[0])
            if "tokens" in data.files:
                num_tokens = int(data["tokens"].shape[1])
                token_mode = "uniform_stride"
        total_frames += num_frames
        files_payload.append(
            {
                "trajectory_id": trajectory_id,
                "relative_path": npz_path.relative_to(output_dir).as_posix(),
                "num_frames": num_frames,
            }
        )

    return {
        "model_name": MODEL_NAME,
        "checkpoint_path": None if checkpoint_path is None else os.path.abspath(checkpoint_path),
        "dataset_name": dataset_name,
        "dataset_mix": dataset_mix,
        "camera_key": camera_key,
        "num_history_frames": num_history_frames,
        "img_size": img_size,
        "feat_dim": feat_dim,
        "num_tokens": num_tokens,
        "token_mode": token_mode,
        "pooling": "mean",
        "dtype": dtype_name,
        "num_episodes": len(files_payload),
        "total_frames": total_frames,
        "files": files_payload,
        "normalization": {
            "mean": NORMALIZATION_MEAN,
            "std": NORMALIZATION_STD,
        },
    }


def iter_sharded_episodes(
    trajectory_ids: np.ndarray,
    trajectory_lengths: np.ndarray,
    num_shards: int,
    shard_id: int,
    max_episodes: int | None,
) -> list[tuple[int, int]]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")

    selected = []
    for i, (trajectory_id, trajectory_length) in enumerate(zip(trajectory_ids.tolist(), trajectory_lengths.tolist())):
        if i % num_shards != shard_id:
            continue
        selected.append((int(trajectory_id), int(trajectory_length)))

    if max_episodes is not None:
        selected = selected[: max_episodes]
    return selected


def preprocess_clip(transform, clip: np.ndarray) -> torch.Tensor:
    transformed = transform(clip)
    if not isinstance(transformed, list) or len(transformed) != 1:
        raise RuntimeError(f"Unexpected transform output type: {type(transformed)}")
    return transformed[0]


def _decode_image_entry(dataset, entry) -> np.ndarray:
    from PIL import Image
    import io

    if isinstance(entry, np.ndarray):
        return entry
    if isinstance(entry, Image.Image):
        return np.array(entry)
    if isinstance(entry, dict):
        img_bytes = entry.get("bytes", None)
        img_path = entry.get("path", None)

        if img_bytes is not None:
            return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

        if img_path is not None:
            path_obj = Path(img_path)
            if not path_obj.is_absolute():
                path_obj = dataset.dataset_path / path_obj
            return np.array(Image.open(path_obj).convert("RGB"))

    raise TypeError(f"Unsupported image entry type: {type(entry)}")


def load_raw_video_clips(
    dataset,
    trajectory_id: int,
    base_indices: list[int],
    camera_key: str,
) -> list[np.ndarray]:
    """Load all raw clips for a batch with a single video reader open.

    This avoids calling dataset.get_video(...) once per base index, which would
    repeatedly open/seek the decoder and can exhaust host memory with the
    torchvision/PyAV backend under 8-way parallel extraction.
    """
    video_delta = dataset.delta_indices[camera_key]
    trajectory_index = dataset.get_trajectory_index(trajectory_id)
    max_length = int(dataset.trajectory_lengths[trajectory_index])
    step_indices = np.asarray(base_indices, dtype=np.int64)[:, None] + video_delta[None, :]
    step_indices = np.clip(step_indices, 0, max_length - 1)

    key = camera_key.replace("video.", "")
    original_key = dataset.lerobot_modality_meta.video[key].original_key
    if original_key is None:
        original_key = key

    if dataset.curr_traj_data is not None and original_key in dataset.curr_traj_data.columns:
        image_entries = dataset.curr_traj_data[original_key].tolist()
        clips = []
        for clip_indices in step_indices:
            frames = [_decode_image_entry(dataset, image_entries[int(idx)]) for idx in clip_indices]
            clips.append(np.stack(frames, axis=0))
        return clips

    video_path = dataset.get_video_path(trajectory_id, key)
    assert dataset.curr_traj_data is not None, f"No data found for {trajectory_id=}"
    assert "timestamp" in dataset.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"

    timestamps = dataset.curr_traj_data["timestamp"].to_numpy()
    flat_timestamps = timestamps[step_indices.reshape(-1)]
    if dataset._lerobot_version == "v3.0":
        episode_meta = dataset.trajectory_ids_to_metadata.get(trajectory_id, {})
        from_timestamps = episode_meta.get("videos/from_timestamps", {})
        if original_key in from_timestamps:
            flat_timestamps = flat_timestamps + float(from_timestamps[original_key])

    frames = get_frames_by_timestamps(
        video_path.as_posix(),
        flat_timestamps,
        video_backend=dataset.video_backend,
        video_backend_kwargs=dataset.video_backend_kwargs,
    )
    frames = frames.reshape(len(base_indices), len(video_delta), *frames.shape[1:])
    return [frames[i] for i in range(frames.shape[0])]


def prepare_batch_clips(
    dataset,
    trajectory_id: int,
    base_indices: list[int],
    camera_key: str,
    transform,
    num_workers: int,
) -> torch.Tensor:
    raw_clips = load_raw_video_clips(dataset, trajectory_id, base_indices, camera_key)

    def _load_one(clip: np.ndarray) -> torch.Tensor:
        return preprocess_clip(transform, clip)

    if num_workers <= 1 or len(base_indices) <= 1:
        clips = [_load_one(clip) for clip in raw_clips]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            clips = list(executor.map(_load_one, raw_clips))
    return torch.stack(clips, dim=0)


def extract_episode(
    dataset,
    trajectory_id: int,
    trajectory_length: int,
    camera_key: str,
    batch_size: int,
    num_workers: int,
    transform,
    encoder,
    device: torch.device,
    save_tokens: bool,
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    dataset.curr_traj_id = trajectory_id

    pooled_chunks = []
    token_chunks = []
    frame_ids = np.arange(trajectory_length, dtype=np.int64)
    encoder_dtype = next(encoder.parameters()).dtype

    for start in range(0, trajectory_length, batch_size):
        base_indices = list(range(start, min(start + batch_size, trajectory_length)))
        batch = prepare_batch_clips(dataset, trajectory_id, base_indices, camera_key, transform, num_workers)
        batch = batch.to(device=device, dtype=encoder_dtype, non_blocking=(device.type == "cuda"))

        with torch.inference_mode():
            tokens = encoder_forward(encoder, batch, device, encoder_dtype)
        tokens = tokens.float().cpu()

        pooled_chunks.append(tokens.mean(dim=1))
        if save_tokens:
            token_chunks.append(sample_tokens_uniform(tokens, num_tokens))

    pooled = torch.cat(pooled_chunks, dim=0).numpy().astype(np.float32, copy=False)
    sampled_tokens = None
    if save_tokens:
        sampled_tokens = torch.cat(token_chunks, dim=0).numpy().astype(np.float32, copy=False)
    return pooled, sampled_tokens, frame_ids


def save_episode_cache(path: Path, pooled: np.ndarray, tokens: np.ndarray | None, frame_ids: np.ndarray) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "pooled": pooled,
        "frame_ids": frame_ids,
    }
    if tokens is not None:
        payload["tokens"] = tokens
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
    tmp_path.replace(path)


def extract_cache(args: argparse.Namespace) -> int:
    if args.vjepa_ckpt is None:
        raise ValueError("--vjepa_ckpt is required unless --run_pca_sanity is used")
    if args.num_history_frames < 1 or args.num_history_frames % 2 != 0:
        raise ValueError("--num_history_frames must be a positive even integer for V-JEPA tubelet_size=2")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    configure_cuda_memory_cap(device, args.max_gpu_memory_gb)

    cfg, data_cfg, underlying_dataset_name, _robot_type, dataset = build_single_dataset(
        args.config_yaml,
        args.dataset_name,
        args.camera_key,
        args.num_history_frames,
    )

    model_dtype, actual_dtype_name = resolve_torch_dtype(args.dtype, device)
    encoder = load_vjepa_encoder(
        ckpt_path=args.vjepa_ckpt,
        num_frames=args.num_history_frames,
        img_size=args.img_size,
        device=device,
        dtype=model_dtype,
    )
    transform = make_transforms(training=False, crop_size=args.img_size)

    selected_episodes = iter_sharded_episodes(
        dataset.trajectory_ids,
        dataset.trajectory_lengths,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        max_episodes=args.max_episodes,
    )
    print(
        f"==> mix={args.dataset_name} dataset={underlying_dataset_name} camera={args.camera_key} "
        f"episodes_in_shard={len(selected_episodes)} history={args.num_history_frames}"
    )

    for trajectory_id, trajectory_length in tqdm(selected_episodes, desc="Extracting episodes"):
        out_path = episode_output_path(output_dir, trajectory_id)
        if out_path.exists() and not args.overwrite:
            continue

        pooled, tokens, frame_ids = extract_episode(
            dataset=dataset,
            trajectory_id=trajectory_id,
            trajectory_length=trajectory_length,
            camera_key=args.camera_key,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            transform=transform,
            encoder=encoder,
            device=device,
            save_tokens=args.save_tokens,
            num_tokens=args.num_tokens,
        )
        save_episode_cache(out_path, pooled, tokens, frame_ids)

    index_payload = build_index(
        output_dir=output_dir,
        dataset_name=underlying_dataset_name,
        dataset_mix=args.dataset_name,
        camera_key=args.camera_key,
        num_history_frames=args.num_history_frames,
        img_size=args.img_size,
        checkpoint_path=args.vjepa_ckpt,
        dtype_name=actual_dtype_name,
    )
    atomic_write_json(output_dir / "index.json", index_payload)
    print(
        f"==> wrote index.json with {index_payload['num_episodes']} episodes and {index_payload['total_frames']} frames"
    )
    return 0


def sample_pooled_features(
    output_dir: Path,
    num_samples: int,
) -> tuple[list[tuple[int, int, Path]], int]:
    index_path = output_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found at {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        index_payload = json.load(f)

    files = index_payload.get("files", [])
    if not files:
        raise ValueError(f"No cached episodes found under {output_dir}")

    cumulative = []
    running = 0
    for item in files:
        running += int(item["num_frames"])
        cumulative.append(running)
    total_frames = running
    sample_count = min(num_samples, total_frames)

    rng = np.random.default_rng(42)
    global_indices = np.sort(rng.choice(total_frames, size=sample_count, replace=False))

    samples = []
    file_idx = 0
    prev_cum = 0
    for global_index in global_indices.tolist():
        while global_index >= cumulative[file_idx]:
            prev_cum = cumulative[file_idx]
            file_idx += 1
        item = files[file_idx]
        frame_id = int(global_index - prev_cum)
        samples.append(
            (
                int(item["trajectory_id"]),
                frame_id,
                output_dir / item["relative_path"],
            )
        )
    return samples, total_frames


def run_pca_sanity(args: argparse.Namespace) -> int:
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for --run_pca_sanity. Install it with:\n"
            "  .venv/bin/python -m pip install scikit-learn"
        ) from exc

    output_dir = Path(args.output_dir)
    samples, total_frames = sample_pooled_features(output_dir, args.pca_num_samples)
    _cfg, _data_cfg, _underlying_dataset_name, _robot_type, dataset = build_single_dataset(
        args.config_yaml,
        args.dataset_name,
        args.camera_key,
        args.num_history_frames,
    )

    lang_keys = dataset.modality_keys.get("language", [])
    if not lang_keys:
        raise ValueError("Dataset has no language key; PCA sanity labels need language annotations")
    lang_key = lang_keys[0]

    grouped: dict[int, list[tuple[int, Path]]] = {}
    for trajectory_id, frame_id, npz_path in samples:
        grouped.setdefault(trajectory_id, []).append((frame_id, npz_path))

    feats = []
    labels = []
    for trajectory_id, frame_items in grouped.items():
        dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
        dataset.curr_traj_id = trajectory_id
        per_file: dict[Path, list[int]] = {}
        for frame_id, npz_path in frame_items:
            per_file.setdefault(npz_path, []).append(frame_id)

        for npz_path, frame_ids in per_file.items():
            with np.load(npz_path, allow_pickle=False) as data:
                pooled = data["pooled"]
                for frame_id in frame_ids:
                    feats.append(pooled[frame_id])
                    lang = dataset.get_language(trajectory_id, lang_key, frame_id)[0]
                    labels.append(lang)

    feats_np = np.asarray(feats, dtype=np.float32)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(feats_np)

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    for label in unique_labels:
        mask = np.array([x == label for x in labels], dtype=bool)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.8,
            label=label,
            color=cmap(label_to_idx[label]),
        )
    ax.set_title(
        f"V-JEPA pooled feature PCA ({len(feats_np)} sampled frames / {total_frames} cached)"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    if unique_labels:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()

    ensure_parent(DEFAULT_PCA_FIGURE)
    fig.savefig(DEFAULT_PCA_FIGURE)
    plt.close(fig)
    print(f"==> wrote PCA sanity plot to {DEFAULT_PCA_FIGURE}")
    return 0


def main() -> int:
    args = parse_args()
    if args.run_pca_sanity:
        return run_pca_sanity(args)
    return extract_cache(args)


if __name__ == "__main__":
    raise SystemExit(main())
