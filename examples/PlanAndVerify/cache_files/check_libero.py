"""Stage 1 / §1.1 — data sanity for LIBERO LeRobot mixture.

Verifies the LeRobot dataset configured by the given yaml returns samples with
the keys QwenOFT.forward / .predict_action consume:

    image (list[PIL.Image | np.ndarray] of len 2 for LIBERO primary+wrist)
    action (np.ndarray, shape [chunk_len, action_dim])
    lang (str)
    state (np.ndarray, shape [1, state_dim])  -- optional, but expected for libero_franka

Run from repo root with the main starVLA venv:

    .venv/bin/python examples/PlanAndVerify/cache_files/check_libero.py \
        --config_yaml examples/PlanAndVerify/train_files/starvla_oft_libero_goal.yaml

Exit code 0 iff every check passes; 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def _summarise_value(name: str, v) -> str:
    if isinstance(v, np.ndarray):
        return f"{name}: ndarray shape={v.shape} dtype={v.dtype} min={v.min():.3f} max={v.max():.3f}"
    if isinstance(v, list):
        first = v[0] if v else None
        if hasattr(first, "size"):
            return f"{name}: list[{type(first).__name__}] len={len(v)} first.size={first.size}"
        if isinstance(first, np.ndarray):
            return f"{name}: list[ndarray] len={len(v)} first.shape={first.shape}"
        return f"{name}: list len={len(v)} first.type={type(first).__name__}"
    if isinstance(v, str):
        return f"{name}: str len={len(v)} preview={v[:80]!r}"
    return f"{name}: {type(v).__name__} value={v!r}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True, help="Path to Stage-1 yaml")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    vla_cfg = cfg.datasets.vla_data
    print(f"==> Building dataset for mix '{vla_cfg.data_mix}' from {vla_cfg.data_root_dir}")
    dataset = get_vla_dataset(data_cfg=vla_cfg)

    n_total = len(dataset)
    print(f"==> Total transitions: {n_total}")
    if n_total == 0:
        print("[FAIL] empty dataset", file=sys.stderr)
        return 1

    required_keys = {"image", "action", "lang"}
    optional_keys = {"state"}

    failures: list[str] = []
    lang_counter: Counter[str] = Counter()

    rng = np.random.default_rng(42)
    indices = rng.choice(n_total, size=min(args.num_samples, n_total), replace=False).tolist()
    for i, idx in enumerate(indices):
        idx = int(idx)
        try:
            sample = dataset[idx]
        except Exception as exc:  # noqa: BLE001
            failures.append(f"sample idx={idx} raised {exc.__class__.__name__}: {exc}")
            continue

        print(f"\n--- sample {i} (idx={idx}) keys={sorted(sample.keys())} ---")
        for k in sorted(sample.keys()):
            print("  " + _summarise_value(k, sample[k]))

        missing = required_keys - sample.keys()
        if missing:
            failures.append(f"sample idx={idx} missing required keys: {missing}")

        if "image" in sample:
            imgs = sample["image"]
            if not isinstance(imgs, list) or len(imgs) < 1:
                failures.append(f"sample idx={idx} image is not a non-empty list (got {type(imgs).__name__})")

        if "action" in sample and isinstance(sample["action"], np.ndarray):
            if sample["action"].ndim != 2:
                failures.append(f"sample idx={idx} action ndim != 2 (got {sample['action'].shape})")

        if "lang" in sample:
            lang_counter[sample["lang"]] += 1

    print("\n==> Sampled language instructions:")
    for lang, n in lang_counter.most_common():
        print(f"  [{n}x] {lang}")

    print("\n==> Scanning a wider window for unique task descriptions ...")
    wide_indices = rng.choice(n_total, size=min(500, n_total), replace=False).tolist()
    wide_lang: set[str] = set()
    for idx in wide_indices:
        try:
            wide_lang.add(dataset[int(idx)]["lang"])
        except Exception:  # noqa: BLE001
            pass
    print(f"  unique langs in {len(wide_indices)} samples: {len(wide_lang)}")
    for lang in sorted(wide_lang):
        print(f"    {lang}")

    if failures:
        print("\n[FAIL] " + "\n[FAIL] ".join(failures), file=sys.stderr)
        return 1

    print("\n[PASS] all sampled records have required keys; dataloader path looks healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
