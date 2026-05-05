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


def test_vjepa_predict_action_returns_trainer_contract():
    from starVLA.model.framework.VLM4A.QwenOFT_VJepa import QwenOFT_VJepa

    pred_actions = torch.randn(2, 8, 7)
    model = object.__new__(QwenOFT_VJepa)

    class DummyActionModel:
        def predict_action(self, _queries):
            return pred_actions

    class DummyProjector:
        def __call__(self, pooled):
            return torch.zeros(2, 4, device=pooled.device, dtype=pooled.dtype)

        def apply_to_queries(self, projected, queries):
            return queries

    model.action_model = DummyActionModel()
    model.vjepa_projector = DummyProjector()
    model.vjepa_cache = None
    model.chunk_len = 8
    model.action_token = "x"
    model.action_token_id = 1
    model.add_discretized_state_to_instruction = lambda instructions, state: instructions
    model._load_vjepa_pooled = lambda examples, device, dtype: torch.zeros(2, 3, device=device, dtype=dtype)
    model._gather_action_token_embeddings = lambda last_hidden, input_ids, action_token_id: torch.zeros(2, 8, 4)

    class DummyQwen:
        def build_qwenvl_inputs(self, images, instructions):
            return {"input_ids": torch.ones(2, 16, dtype=torch.long)}

        def __call__(self, **kwargs):
            return type("Out", (), {"hidden_states": [torch.zeros(2, 16, 4)]})()

    model.qwen_vl_interface = DummyQwen()

    out = model.predict_action(
        examples=[
            {"image": [], "lang": "a", "episode_id": 1, "frame_id": 0},
            {"image": [], "lang": "b", "episode_id": 1, "frame_id": 1},
        ]
    )

    assert isinstance(out, dict)
    assert out["normalized_actions"].shape == (2, 8, 7)
