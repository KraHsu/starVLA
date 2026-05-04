from pathlib import Path

import numpy as np
import torch

from examples.PlanAndVerify.cache_files.extract_vjepa_cache import (
    build_index,
    iter_sharded_episodes,
    memory_fraction_from_gib,
    patch_vjepa2_1_rope_dtype,
    save_episode_cache,
)


def test_save_episode_cache_and_build_index(tmp_path: Path):
    out_path = tmp_path / "episodes" / "episode_000007.npz"
    pooled = np.zeros((3, 768), dtype=np.float32)
    tokens = np.zeros((3, 4, 768), dtype=np.float32)
    frame_ids = np.arange(3, dtype=np.int64)

    save_episode_cache(out_path, pooled, tokens, frame_ids)

    with np.load(out_path, allow_pickle=False) as data:
        assert data["pooled"].shape == (3, 768)
        assert data["tokens"].shape == (3, 4, 768)
        assert data["frame_ids"].tolist() == [0, 1, 2]

    index_payload = build_index(
        output_dir=tmp_path,
        dataset_name="libero_goal_no_noops_1.0.0_lerobot",
        dataset_mix="libero_goal",
        camera_key="video.primary_image",
        num_history_frames=8,
        img_size=384,
        checkpoint_path="/tmp/vjepa.pt",
        dtype_name="bf16",
    )
    assert index_payload["num_episodes"] == 1
    assert index_payload["total_frames"] == 3
    assert index_payload["feat_dim"] == 768
    assert index_payload["num_tokens"] == 4
    assert index_payload["files"][0]["trajectory_id"] == 7
    assert index_payload["files"][0]["relative_path"] == "episodes/episode_000007.npz"


def test_iter_sharded_episodes_splits_by_episode_index():
    trajectory_ids = np.array([10, 11, 12, 13, 14])
    trajectory_lengths = np.array([3, 4, 5, 6, 7])

    shard0 = iter_sharded_episodes(trajectory_ids, trajectory_lengths, num_shards=2, shard_id=0, max_episodes=None)
    shard1 = iter_sharded_episodes(trajectory_ids, trajectory_lengths, num_shards=2, shard_id=1, max_episodes=None)

    assert shard0 == [(10, 3), (12, 5), (14, 7)]
    assert shard1 == [(11, 4), (13, 6)]
    assert iter_sharded_episodes(trajectory_ids, trajectory_lengths, 2, 0, max_episodes=2) == [(10, 3), (12, 5)]


def test_vjepa_rope_patch_preserves_input_dtype():
    patch_vjepa2_1_rope_dtype()
    from app.vjepa_2_1.models.utils import modules as vjepa_modules

    x = torch.randn(1, 2, 4, 12, dtype=torch.bfloat16)
    pos = torch.arange(4, dtype=torch.float32).reshape(1, 4)

    out = vjepa_modules.rotate_queries_or_keys(x, pos=pos, n_registers=0, has_cls_first=False)

    assert out.dtype == torch.bfloat16


def test_memory_fraction_from_gib_caps_at_one():
    assert memory_fraction_from_gib(100, 120 * 1024**3) == 100 / 120
    assert memory_fraction_from_gib(100, 80 * 1024**3) == 1.0
