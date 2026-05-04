"""V-JEPA 2 latent shard reader for LCLGP training data construction.

T-W2.2.4. Two layers:

* :class:`VJEPALatentShardSet` — low-level, sharded, random-access store. Used
  by ``scripts/build_lclgp_dataset.py`` (T-W2.3.1) to materialize triplets
  ``(z_t, z_{t+Δ}, z_T)`` and by tests.
* :class:`VJEPALatentDataset` — thin :class:`torch.utils.data.Dataset` wrapper
  that yields per-frame samples. Sufficient for sanity tests; the LCLGP-side
  training dataset that emits ``(text_emb, z_t, z_{t+Δ}, z_T, task_id)`` is a
  separate class built on top of the W2.3 parquet output.

Shard layout (written by ``scripts/extract_vjepa_latents.py``)::

    <dir>/<dataset>_rank{R:02d}.h5
        attrs: {data_name, robot_type, num_frames, tubelet_size, img_size,
                patch_size, dtype, vjepa_ckpt, imagenet_mean, imagenet_std}
        /index    compound (traj_id:i8, length:i8)
        /traj_{id:06d}/
            primary  [T, N_tokens, 1408] fp16
            wrist    [T, N_tokens, 1408] fp16    (optional)
            attrs: {lang: str, length: int}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Attributes that must agree across shards belonging to one logical dataset.
# Mismatch usually means shards were produced by different encoder configs.
_REQUIRED_MATCHING_ATTRS = (
    "data_name",
    "robot_type",
    "num_frames",
    "tubelet_size",
    "img_size",
    "patch_size",
    "dtype",
)


@dataclass(frozen=True)
class TrajectoryRef:
    """One trajectory's location inside a shard."""

    shard_path: Path
    traj_id: int
    length: int
    has_wrist: bool


class VJEPALatentShardSet:
    """Random-access reader across multiple shards of one dataset.

    Auto-discovers shards under ``<dir>/`` matching ``*_rank*.h5``.

    Files are opened lazily on first read and kept around for the lifetime of
    the object (HDF5 file handles are cheap; closing is left to GC). Keep one
    instance per ``DataLoader`` worker — handles are NOT fork-safe.
    """

    def __init__(self, shard_dir: Path):
        self.shard_dir = Path(shard_dir)
        if not self.shard_dir.is_dir():
            raise FileNotFoundError(f"shard dir does not exist: {self.shard_dir}")

        shard_paths = sorted(self.shard_dir.glob("*_rank*.h5"))
        if not shard_paths:
            raise FileNotFoundError(
                f"no '*_rank*.h5' shards under {self.shard_dir}; did extract_vjepa_latents finish?"
            )
        self.shard_paths: List[Path] = shard_paths
        self._files: Dict[Path, h5py.File] = {}

        # Validate attrs across shards and build flat traj index.
        self._attrs: Dict[str, object] = {}
        self._trajs: List[TrajectoryRef] = []
        first = True
        for p in self.shard_paths:
            with h5py.File(p, "r") as f:
                attrs = {k: f.attrs[k] for k in f.attrs.keys()}
                if first:
                    self._attrs = attrs
                    first = False
                else:
                    for k in _REQUIRED_MATCHING_ATTRS:
                        if attrs.get(k) != self._attrs.get(k):
                            raise RuntimeError(
                                f"shard {p} attr '{k}' = {attrs.get(k)!r} "
                                f"does not match {self.shard_paths[0]} = {self._attrs.get(k)!r}"
                            )
                index = f["index"][:]
                # Probe wrist availability per shard from the first traj.
                first_group = f[f"traj_{int(index[0]['traj_id']):06d}"]
                has_wrist = "wrist" in first_group
                for row in index:
                    self._trajs.append(
                        TrajectoryRef(
                            shard_path=p,
                            traj_id=int(row["traj_id"]),
                            length=int(row["length"]),
                            has_wrist=has_wrist,
                        )
                    )

        if not self._trajs:
            raise RuntimeError(f"shards under {self.shard_dir} contain zero trajectories")

        self._traj_lookup: Dict[int, TrajectoryRef] = {t.traj_id: t for t in self._trajs}

    # -- Metadata ---------------------------------------------------------

    @property
    def data_name(self) -> str:
        return str(self._attrs["data_name"])

    @property
    def hidden_dim(self) -> int:
        # The wrapper's V-JEPA 2 ViT-g always outputs 1408. Read from a probe
        # rather than the attrs because attrs do not carry it.
        return 1408

    @property
    def num_trajectories(self) -> int:
        return len(self._trajs)

    @property
    def total_frames(self) -> int:
        return sum(t.length for t in self._trajs)

    def trajectory_refs(self) -> List[TrajectoryRef]:
        return list(self._trajs)

    def __len__(self) -> int:
        return self.num_trajectories

    # -- Random access ----------------------------------------------------

    def _get_file(self, path: Path) -> h5py.File:
        f = self._files.get(path)
        if f is None:
            f = h5py.File(path, "r", swmr=True)
            self._files[path] = f
        return f

    def _get_group(self, traj_id: int) -> Tuple[h5py.Group, TrajectoryRef]:
        ref = self._traj_lookup.get(traj_id)
        if ref is None:
            raise KeyError(f"traj_id {traj_id} not found in {self.shard_dir}")
        f = self._get_file(ref.shard_path)
        return f[f"traj_{traj_id:06d}"], ref

    def get_frame(
        self,
        traj_id: int,
        frame_idx: int,
        *,
        view: str = "primary",
    ) -> np.ndarray:
        """Return ``(N_tokens, 1408) fp16`` for one frame, one view."""
        group, ref = self._get_group(traj_id)
        if frame_idx < 0 or frame_idx >= ref.length:
            raise IndexError(
                f"frame_idx {frame_idx} out of range for traj {traj_id} "
                f"(length={ref.length})"
            )
        if view not in group:
            raise KeyError(f"view {view!r} missing in traj {traj_id}; available: {list(group.keys())}")
        return group[view][frame_idx]

    def get_trajectory(
        self,
        traj_id: int,
        *,
        view: str = "primary",
    ) -> np.ndarray:
        """Return ``(T, N_tokens, 1408) fp16`` for one whole trajectory."""
        group, _ref = self._get_group(traj_id)
        if view not in group:
            raise KeyError(f"view {view!r} missing in traj {traj_id}")
        return group[view][:]

    def get_language(self, traj_id: int) -> str:
        group, _ = self._get_group(traj_id)
        lang = group.attrs.get("lang", "")
        return str(lang)

    def iter_trajectories(self, view: str = "primary") -> Iterator[Tuple[int, np.ndarray, str]]:
        """Yield ``(traj_id, [T, N_tokens, 1408], lang)`` per trajectory."""
        for ref in self._trajs:
            yield ref.traj_id, self.get_trajectory(ref.traj_id, view=view), self.get_language(ref.traj_id)


class VJEPALatentDataset(Dataset):
    """Per-frame :class:`Dataset` over a :class:`VJEPALatentShardSet`.

    Each item is one ``(primary, wrist?, traj_id, frame_idx, lang)`` sample.
    Provided primarily for unit tests and quick sanity checks. The triplet-style
    dataset for LCLGP is built on the parquet output of T-W2.3.1.
    """

    def __init__(
        self,
        shard_dir: Path,
        *,
        return_wrist: bool = True,
    ):
        self.store = VJEPALatentShardSet(shard_dir)
        self.return_wrist = return_wrist
        # Pre-compute (traj_id, frame_idx) flat index.
        self._flat: List[Tuple[int, int]] = [
            (ref.traj_id, f)
            for ref in self.store.trajectory_refs()
            for f in range(ref.length)
        ]

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, index: int):
        traj_id, frame_idx = self._flat[index]
        primary = self.store.get_frame(traj_id, frame_idx, view="primary")
        sample: Dict[str, object] = {
            "primary": torch.from_numpy(np.asarray(primary)),
            "traj_id": traj_id,
            "frame_idx": frame_idx,
            "lang": self.store.get_language(traj_id),
        }
        if self.return_wrist:
            ref = self.store._traj_lookup[traj_id]
            if ref.has_wrist:
                sample["wrist"] = torch.from_numpy(
                    np.asarray(self.store.get_frame(traj_id, frame_idx, view="wrist"))
                )
        return sample
