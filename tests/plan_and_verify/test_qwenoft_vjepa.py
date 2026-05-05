from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.modules.projector.vjepa_projector import VJepaProjector
from starVLA.dataloader.vjepa_cache import VJepaFeatureCache


def _write_vjepa_cache(root: Path) -> None:
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "episodes" / "episode_000001.npz",
        pooled=np.arange(6, dtype=np.float32).reshape(2, 3),
        frame_ids=np.arange(2, dtype=np.int64),
    )
    (root / "index.json").write_text(
        """
{
  "model_name": "vjepa2_1_vit_base_384",
  "checkpoint_path": "/tmp/vjepa.pt",
  "dataset_name": "libero_goal_no_noops_1.0.0_lerobot",
  "dataset_mix": "libero_goal",
  "camera_key": "video.primary_image",
  "num_history_frames": 8,
  "img_size": 384,
  "feat_dim": 3,
  "num_tokens": 0,
  "token_mode": "disabled",
  "pooling": "mean",
  "dtype": "bf16",
  "num_episodes": 1,
  "total_frames": 2,
  "files": [
    {"trajectory_id": 1, "relative_path": "episodes/episode_000001.npz", "num_frames": 2}
  ],
  "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
}
""".strip(),
        encoding="utf-8",
    )


def test_vjepa_projector_shapes():
    projector = VJepaProjector(input_dim=768, output_dim=2048, hidden_dim=1024, use_film_gating=True)
    pooled = torch.randn(2, 768)
    queries = torch.randn(2, 8, 2048)

    projected = projector(pooled)
    fused = projector.apply_to_queries(projected, queries)

    assert projected.shape == (2, 2048)
    assert fused.shape == (2, 8, 2048)


def test_vjepa_cache_batch_for_framework(tmp_path: Path):
    _write_vjepa_cache(tmp_path)
    cache = VJepaFeatureCache(tmp_path)
    batch = cache.get_batch([1, 1], [0, 1])

    assert batch["pooled"].shape == (2, 3)
    np.testing.assert_array_equal(batch["pooled"][0], np.array([0, 1, 2], dtype=np.float32))


def test_vjepa_framework_config_defaults_merge():
    from starVLA.model.framework.VLM4A.QwenOFT_VJepa import QwenOFTVJepaDefaultConfig
    from starVLA.model.framework.share_tools import merge_framework_config

    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "QwenOFT_VJepa",
                "vjepa": {"cache_dir": "/tmp/cache"},
                "action_model": {"action_horizon": 8, "action_dim": 7},
            }
        }
    )
    merged = merge_framework_config(QwenOFTVJepaDefaultConfig, cfg)

    assert merged.framework.name == "QwenOFT_VJepa"
    assert merged.framework.vjepa.cache_dir == "/tmp/cache"
    assert merged.framework.vjepa.input_dim == 768
