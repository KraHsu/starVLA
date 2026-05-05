import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.vjepa_cache import VJepaFeatureCache


def _write_episode(root: Path, episode_id: int, pooled: np.ndarray, tokens: np.ndarray | None = None) -> None:
    episode_path = root / "episodes" / f"episode_{episode_id:06d}.npz"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pooled": pooled.astype(np.float32),
        "frame_ids": np.arange(pooled.shape[0], dtype=np.int64),
    }
    if tokens is not None:
        payload["tokens"] = tokens.astype(np.float32)
    np.savez_compressed(episode_path, **payload)


def _write_index(root: Path, files: list[dict], num_tokens: int = 0) -> None:
    total_frames = sum(item["num_frames"] for item in files)
    payload = {
        "model_name": "vjepa2_1_vit_base_384",
        "checkpoint_path": "/tmp/vjepa.pt",
        "dataset_name": "libero_goal_no_noops_1.0.0_lerobot",
        "dataset_mix": "libero_goal",
        "camera_key": "video.primary_image",
        "num_history_frames": 8,
        "img_size": 384,
        "feat_dim": 3,
        "num_tokens": num_tokens,
        "token_mode": "uniform_stride" if num_tokens else "disabled",
        "pooling": "mean",
        "dtype": "bf16",
        "num_episodes": len(files),
        "total_frames": total_frames,
        "files": files,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    (root / "index.json").write_text(json.dumps(payload), encoding="utf-8")


def test_vjepa_feature_cache_reads_pooled_and_tokens(tmp_path: Path):
    pooled = np.arange(12, dtype=np.float32).reshape(4, 3)
    tokens = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    _write_episode(tmp_path, 7, pooled, tokens)
    _write_index(
        tmp_path,
        [{"trajectory_id": 7, "relative_path": "episodes/episode_000007.npz", "num_frames": 4}],
        num_tokens=2,
    )

    cache = VJepaFeatureCache(tmp_path, include_tokens=True)

    sample_feat = cache.get(7, 2)
    assert sample_feat["episode_id"] == 7
    assert sample_feat["frame_id"] == 2
    np.testing.assert_array_equal(sample_feat["pooled"], pooled[2])
    np.testing.assert_array_equal(sample_feat["tokens"], tokens[2])

    batch = cache.get_batch([7, 7], [0, 3], include_tokens=True)
    assert batch["pooled"].shape == (2, 3)
    assert batch["tokens"].shape == (2, 2, 3)


def test_vjepa_feature_cache_reads_from_training_sample_keys(tmp_path: Path):
    pooled = np.arange(9, dtype=np.float32).reshape(3, 3)
    _write_episode(tmp_path, 3, pooled)
    _write_index(tmp_path, [{"trajectory_id": 3, "relative_path": "episodes/episode_000003.npz", "num_frames": 3}])

    cache = VJepaFeatureCache(tmp_path)
    out = cache.get_for_sample({"episode_id": 3, "frame_id": 1})

    np.testing.assert_array_equal(out["pooled"], pooled[1])


def test_lerobot_pack_sample_includes_cache_indices():
    fake_dataset = SimpleNamespace(
        modality_keys={
            "video": ["video.primary_image"],
            "language": ["annotation.human.action.task_description"],
            "action": ["action.x", "action.y"],
        },
        tag="franka",
        data_cfg=None,
    )
    data = {
        "video.primary_image": np.zeros((1, 8, 8, 3), dtype=np.uint8),
        "annotation.human.action.task_description": ["pick up the bowl"],
        "action.x": np.zeros((8, 1), dtype=np.float32),
        "action.y": np.ones((8, 1), dtype=np.float32),
    }

    sample = LeRobotSingleDataset._pack_sample(fake_dataset, data, trajectory_id=12, frame_id=34)

    assert sample["episode_id"] == 12
    assert sample["trajectory_id"] == 12
    assert sample["frame_id"] == 34
    assert sample["action"].shape == (8, 2)
