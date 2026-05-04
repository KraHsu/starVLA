"""LCLGP triplet dataset that pairs the parquet index with latent + text caches.

T-W2.3 deliverable used by W3's LCLGP trainer. Compose with
:class:`starVLA.datasets.samplers.TaskGroupedSampler` for K-mode balanced
batches; see the design doc §5.4.

Item schema::

    {
        "text_emb":  Tensor [L, hidden]    # variable L; collate pads
        "z_t":       Tensor [N_tokens, 1408]
        "z_delta":   Tensor [N_tokens, 1408]
        "z_end":     Tensor [N_tokens, 1408]
        "task_id":   str                   # = lang_hash, drives TaskGroupedSampler
        "lang":      str
        "data_name": str
        "traj_id":   int
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from starVLA.datasets.vjepa_latent_dataset import VJEPALatentShardSet


class TextEmbCache:
    """Read-only access to ``text_emb.h5`` produced by ``build_lclgp_dataset.py``.

    Opens lazily on first access and keeps the handle around for the dataset's
    lifetime. Not fork-safe — instantiate one cache per dataloader worker.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"text_emb cache not found: {self.path}")
        self._file: Optional[h5py.File] = None

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def get(self, lang_hash: str) -> np.ndarray:
        f = self._open()
        if lang_hash not in f:
            raise KeyError(f"lang_hash {lang_hash!r} not in {self.path}")
        return f[lang_hash]["emb"][:]

    @property
    def hidden(self) -> int:
        f = self._open()
        first = next(iter(f.keys()))
        return int(f[first].attrs.get("hidden", f[first]["emb"].shape[1]))

    def keys(self) -> List[str]:
        return list(self._open().keys())


class LcLgpTripletDataset(Dataset):
    """Joins parquet index → V-JEPA latents + text_emb cache.

    Args:
        index_path: parquet from ``build_lclgp_dataset.py``.
        latent_root: parent dir holding ``<data_name>/<data_name>_rank*.h5``.
        text_emb_path: ``text_emb.h5`` produced alongside the parquet.
        view: ``"primary"`` (default) or ``"wrist"``.
    """

    def __init__(
        self,
        index_path: Path,
        latent_root: Path,
        text_emb_path: Path,
        *,
        view: str = "primary",
    ):
        self.index_path = Path(index_path)
        self.latent_root = Path(latent_root)
        self.view = view
        self.df: pd.DataFrame = pd.read_parquet(self.index_path).reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"empty index: {self.index_path}")

        # One shard set per data_name; opened lazily.
        self._stores: Dict[str, VJEPALatentShardSet] = {}
        for name in self.df["data_name"].unique():
            self._stores[name] = VJEPALatentShardSet(self.latent_root / name)

        self.text_cache = TextEmbCache(Path(text_emb_path))

        # Pre-extract task_ids array for TaskGroupedSampler.
        self._task_ids: List[str] = self.df["lang_hash"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    @property
    def task_ids(self) -> List[str]:
        return self._task_ids

    @property
    def text_hidden(self) -> int:
        return self.text_cache.hidden

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        store = self._stores[row.data_name]
        traj_id = int(row.traj_id)
        z_t = store.get_frame(traj_id, int(row.t), view=self.view)
        z_delta = store.get_frame(traj_id, int(row.t_delta), view=self.view)
        z_end = store.get_frame(traj_id, int(row.t_end), view=self.view)
        text_emb = self.text_cache.get(str(row.lang_hash))
        return {
            "text_emb": torch.from_numpy(np.asarray(text_emb)),
            "z_t": torch.from_numpy(np.asarray(z_t)),
            "z_delta": torch.from_numpy(np.asarray(z_delta)),
            "z_end": torch.from_numpy(np.asarray(z_end)),
            "task_id": str(row.lang_hash),
            "lang": str(row.lang),
            "data_name": str(row.data_name),
            "traj_id": traj_id,
        }


def collate_lclgp(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Right-pad ``text_emb`` to the longest L in the batch; stack everything else."""
    if not batch:
        return {}
    Ls = [item["text_emb"].shape[0] for item in batch]
    max_L = max(Ls)
    hidden = batch[0]["text_emb"].shape[1]
    text_emb = torch.zeros(len(batch), max_L, hidden, dtype=batch[0]["text_emb"].dtype)
    text_mask = torch.zeros(len(batch), max_L, dtype=torch.bool)
    for i, item in enumerate(batch):
        L = item["text_emb"].shape[0]
        text_emb[i, :L] = item["text_emb"]
        text_mask[i, :L] = True
    out = {
        "text_emb": text_emb,
        "text_mask": text_mask,
        "z_t": torch.stack([b["z_t"] for b in batch], dim=0),
        "z_delta": torch.stack([b["z_delta"] for b in batch], dim=0),
        "z_end": torch.stack([b["z_end"] for b in batch], dim=0),
        "task_id": [b["task_id"] for b in batch],
        "lang": [b["lang"] for b in batch],
        "data_name": [b["data_name"] for b in batch],
        "traj_id": [b["traj_id"] for b in batch],
    }
    return out


# ---------------------------------------------------------------------------
# Smoke entrypoint
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    from starVLA.datasets.samplers import TaskGroupedSampler

    p = argparse.ArgumentParser(description="Smoke-load LcLgpTripletDataset")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--latent-root", type=Path, required=True)
    p.add_argument("--text-emb", type=Path, required=True)
    p.add_argument("--n-tasks", type=int, default=4)
    p.add_argument("--n-demos", type=int, default=2)
    args = p.parse_args()

    ds = LcLgpTripletDataset(args.index, args.latent_root, args.text_emb)
    print(f"len(dataset)={len(ds)}  unique tasks={len(set(ds.task_ids))}  text_hidden={ds.text_hidden}")

    n_tasks = min(args.n_tasks, len(set(ds.task_ids)))
    sampler = TaskGroupedSampler(
        ds.task_ids, n_tasks=n_tasks, n_demos=args.n_demos, seed=0,
    )
    batch = next(iter(sampler))
    items = [ds[i] for i in batch]
    out = collate_lclgp(items)
    print({k: (v.shape if isinstance(v, torch.Tensor) else len(v))
           for k, v in out.items()})


if __name__ == "__main__":
    _main()
