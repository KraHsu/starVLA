"""Reader for offline V-JEPA feature caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class VJepaFeatureCache:
    """Lazy reader for episode-wise V-JEPA ``.npz`` caches.

    The cache directory must contain an ``index.json`` produced by
    ``examples/PlanAndVerify/cache_files/extract_vjepa_cache.py`` and an
    ``episodes/`` directory with one ``episode_XXXXXX.npz`` per trajectory.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        include_tokens: bool = False,
        pooled_key: str = "pooled",
        tokens_key: str = "tokens",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / "index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"V-JEPA cache index not found: {self.index_path}")

        with self.index_path.open("r", encoding="utf-8") as f:
            self.index: dict[str, Any] = json.load(f)

        self.include_tokens = include_tokens
        self.pooled_key = pooled_key
        self.tokens_key = tokens_key
        self.feat_dim = self.index.get("feat_dim")
        self.num_tokens = int(self.index.get("num_tokens", 0) or 0)
        self.files = {
            int(item["trajectory_id"]): {
                "path": self.cache_dir / item["relative_path"],
                "num_frames": int(item["num_frames"]),
            }
            for item in self.index.get("files", [])
        }
        if not self.files:
            raise ValueError(f"V-JEPA cache index contains no files: {self.index_path}")
        if self.include_tokens and self.num_tokens <= 0:
            raise ValueError(
                f"include_tokens=True but cache index says no tokens are stored: {self.index_path}"
            )

        self._episode_arrays: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return int(self.index.get("total_frames", 0) or 0)

    @property
    def episode_ids(self) -> list[int]:
        return sorted(self.files)

    def _load_episode(self, episode_id: int) -> dict[str, np.ndarray]:
        episode_id = int(episode_id)
        if episode_id in self._episode_arrays:
            return self._episode_arrays[episode_id]
        if episode_id not in self.files:
            raise KeyError(f"Episode {episode_id} is not present in V-JEPA cache {self.cache_dir}")

        path = self.files[episode_id]["path"]
        if not path.exists():
            raise FileNotFoundError(f"V-JEPA episode cache missing: {path}")

        with np.load(path, allow_pickle=False) as data:
            arrays = {
                self.pooled_key: data[self.pooled_key].astype(np.float32, copy=False),
                "frame_ids": data["frame_ids"].astype(np.int64, copy=False),
            }
            if self.tokens_key in data.files:
                arrays[self.tokens_key] = data[self.tokens_key].astype(np.float32, copy=False)

        expected_frames = self.files[episode_id]["num_frames"]
        if arrays[self.pooled_key].shape[0] != expected_frames:
            raise ValueError(
                f"Episode {episode_id} pooled length mismatch: "
                f"{arrays[self.pooled_key].shape[0]} != {expected_frames}"
            )

        self._episode_arrays[episode_id] = arrays
        return arrays

    def get_pooled(self, episode_id: int, frame_id: int) -> np.ndarray:
        arrays = self._load_episode(int(episode_id))
        frame_id = int(frame_id)
        if frame_id < 0 or frame_id >= arrays[self.pooled_key].shape[0]:
            raise IndexError(
                f"frame_id {frame_id} out of range for episode {episode_id} "
                f"with {arrays[self.pooled_key].shape[0]} frames"
            )
        return arrays[self.pooled_key][frame_id]

    def get_tokens(self, episode_id: int, frame_id: int) -> np.ndarray:
        arrays = self._load_episode(int(episode_id))
        if self.tokens_key not in arrays:
            raise KeyError(f"V-JEPA tokens were not stored for episode {episode_id}")
        frame_id = int(frame_id)
        if frame_id < 0 or frame_id >= arrays[self.tokens_key].shape[0]:
            raise IndexError(
                f"frame_id {frame_id} out of range for episode {episode_id} "
                f"with {arrays[self.tokens_key].shape[0]} token rows"
            )
        return arrays[self.tokens_key][frame_id]

    def get(self, episode_id: int, frame_id: int, include_tokens: bool | None = None) -> dict[str, np.ndarray | int]:
        use_tokens = self.include_tokens if include_tokens is None else include_tokens
        out: dict[str, np.ndarray | int] = {
            "episode_id": int(episode_id),
            "frame_id": int(frame_id),
            "pooled": self.get_pooled(episode_id, frame_id),
        }
        if use_tokens:
            out["tokens"] = self.get_tokens(episode_id, frame_id)
        return out

    def get_for_sample(self, sample: dict, include_tokens: bool | None = None) -> dict[str, np.ndarray | int]:
        episode_id = sample.get("episode_id", sample.get("trajectory_id"))
        frame_id = sample.get("frame_id")
        if episode_id is None or frame_id is None:
            raise KeyError("sample must contain episode_id/trajectory_id and frame_id")
        return self.get(int(episode_id), int(frame_id), include_tokens=include_tokens)

    def get_batch(
        self,
        episode_ids: list[int] | np.ndarray,
        frame_ids: list[int] | np.ndarray,
        include_tokens: bool | None = None,
    ) -> dict[str, np.ndarray]:
        if len(episode_ids) != len(frame_ids):
            raise ValueError(f"episode_ids and frame_ids length mismatch: {len(episode_ids)} != {len(frame_ids)}")

        use_tokens = self.include_tokens if include_tokens is None else include_tokens
        pooled = [self.get_pooled(int(ep), int(fr)) for ep, fr in zip(episode_ids, frame_ids)]
        out = {"pooled": np.stack(pooled, axis=0)}
        if use_tokens:
            tokens = [self.get_tokens(int(ep), int(fr)) for ep, fr in zip(episode_ids, frame_ids)]
            out["tokens"] = np.stack(tokens, axis=0)
        return out
