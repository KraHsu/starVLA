"""Bridge-v2 → LeRobot v2.1 conversion driver (Plan-and-Verify W2 side task).

Three subcommands so you can stop at whichever step matches your starting
point:

  hf-download    — pull a community-converted LeRobot Bridge-v2 mirror from
                   HuggingFace and drop the matching ``meta/modality.json``
                   into place. Recommended path; the mirror is already in the
                   shape ``BridgeWidowxDataConfig`` expects.
  finalize       — given an existing LeRobot dir (e.g. produced by another
                   conversion), validate the structure, copy ``modality.json``,
                   and print a short manifest so the user can confirm the
                   ``data_config.py`` registration will work.
  from-rlds      — scaffold for converting raw RLDS-format Bridge-v2 (the
                   original release, TFDS-served) into LeRobot v2.1 chunks.
                   This path is *untested* in CI; the script aborts before
                   writing anything if any required field is missing.

Outputs land at ``--dst`` and follow LIBERO's layout exactly so
``ROBOT_TYPE_CONFIG_MAP["bridge_widowx"]`` (in
``examples/PlanAndVerify/train_files/data_registry/data_config.py``) becomes
live the moment the destination directory exists.

See ``examples/PlanAndVerify/scripts/convert_bridge_v2.md`` for end-to-end
recipes and known-good HF mirror paths.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
MODALITY_TEMPLATE_PATH = THIS_FILE.parent / "bridge_v2_modality.json"

# ---------------------------------------------------------------------------
# modality.json — matches BridgeWidowxDataConfig in
# examples/PlanAndVerify/train_files/data_registry/data_config.py.
# ---------------------------------------------------------------------------
BRIDGE_MODALITY = {
    "state": {
        "x":      {"start": 0, "end": 1},
        "y":      {"start": 1, "end": 2},
        "z":      {"start": 2, "end": 3},
        "roll":   {"start": 3, "end": 4},
        "pitch":  {"start": 4, "end": 5},
        "yaw":    {"start": 5, "end": 6},
        "gripper":{"start": 6, "end": 7},
    },
    "action": {
        "x":      {"start": 0, "end": 1},
        "y":      {"start": 1, "end": 2},
        "z":      {"start": 2, "end": 3},
        "roll":   {"start": 3, "end": 4},
        "pitch":  {"start": 4, "end": 5},
        "yaw":    {"start": 5, "end": 6},
        "gripper":{"start": 6, "end": 7},
    },
    "video": {
        "image_0": {"original_key": "observation.images.image_0"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _write_modality(dst_dir: Path) -> None:
    meta = dst_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    out = meta / "modality.json"
    out.write_text(json.dumps(BRIDGE_MODALITY, indent=2))
    print(f"📝 wrote {out}")


def _validate_lerobot_v2_layout(dst: Path) -> dict:
    """Cheap structural check; returns a manifest dict for human inspection."""
    manifest: dict = {"path": str(dst)}
    info_path = dst / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing meta/info.json under {dst}; not a LeRobot dataset")
    info = json.loads(info_path.read_text())
    manifest["codebase_version"] = info.get("codebase_version")
    manifest["total_episodes"] = info.get("total_episodes")
    manifest["total_frames"] = info.get("total_frames")
    manifest["fps"] = info.get("fps")
    manifest["features"] = sorted((info.get("features") or {}).keys())

    data_dir = dst / "data"
    video_dir = dst / "videos"
    if not data_dir.exists():
        raise FileNotFoundError(f"missing {data_dir}")
    chunks = sorted(p.name for p in data_dir.glob("chunk-*"))
    manifest["data_chunks"] = chunks[:5] + (["…"] if len(chunks) > 5 else [])
    if video_dir.exists():
        manifest["video_keys"] = sorted(p.name for p in video_dir.glob("chunk-000/*"))
    return manifest


# ---------------------------------------------------------------------------
# Subcommand: hf-download
# ---------------------------------------------------------------------------


def cmd_hf_download(args: argparse.Namespace) -> None:
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    repo = args.repo
    print(f"⬇️  hf download {repo} → {dst}")
    cmd = [
        "hf", "download", repo,
        "--repo-type", "dataset",
        "--local-dir", str(dst),
    ]
    print(f"$ {' '.join(cmd)}")
    if args.dry_run:
        print("(dry-run; would run the command above)")
        return
    subprocess.run(cmd, check=True)

    _write_modality(dst)
    manifest = _validate_lerobot_v2_layout(dst)
    print("\n✅ download complete:")
    print(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Subcommand: finalize
# ---------------------------------------------------------------------------


def cmd_finalize(args: argparse.Namespace) -> None:
    dst = Path(args.dst).resolve()
    manifest = _validate_lerobot_v2_layout(dst)
    if args.modality_only:
        _write_modality(dst)
        print("✅ modality.json updated; structure left untouched.")
        return

    _write_modality(dst)
    print("\n✅ finalize complete:")
    print(json.dumps(manifest, indent=2))

    # Symlink convenience pointer for the PaV configs.
    if args.symlink_target:
        link = Path(args.symlink_target).resolve()
        if link.exists() and not link.is_symlink():
            print(f"⚠️  {link} exists and is not a symlink; skipping link creation")
            return
        if link.is_symlink():
            link.unlink()
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(dst)
        print(f"🔗 symlinked {link} → {dst}")


# ---------------------------------------------------------------------------
# Subcommand: from-rlds
# ---------------------------------------------------------------------------


def cmd_from_rlds(args: argparse.Namespace) -> None:
    """Stub: convert RLDS-format Bridge-v2 (TFDS) → LeRobot v2.1.

    Implemented as a guarded scaffold because the Bridge-v2 raw release lives
    in TFDS and emitting LeRobot v2.1 chunks requires:
      * tensorflow / tensorflow_datasets to read RLDS shards
      * av or moviepy to encode per-episode mp4 videos
      * pandas + pyarrow for parquet step records
      * the LeRobotDataset writer (or a hand-rolled writer matching v2.1 layout)

    Without the user's actual data path + dependency stack we cannot exercise
    these paths; the function raises NotImplementedError with concrete next
    steps so the user can fill in the gaps in their environment.
    """
    raise NotImplementedError(
        "from-rlds is a scaffold. To activate it:\n"
        "  1) pip install tensorflow tensorflow-datasets pandas pyarrow av\n"
        "  2) Read the official Bridge-v2 release notes and confirm the RLDS\n"
        "     'steps' schema fields (the original release uses keys like\n"
        "     'observation.image_0', 'action', 'is_terminal', 'language_instruction').\n"
        "  3) Adapt the loop in this function to map those fields into\n"
        "     LeRobot v2.1 columns (see playground/Datasets/LEROBOT_LIBERO_DATA/\n"
        "     <suite>/meta/info.json for the exact target schema).\n"
        "  4) Write per-episode parquet under data/chunk-000/episode_XXXXXX.parquet,\n"
        "     mp4 under videos/chunk-000/observation.images.image_0/episode_XXXXXX.mp4,\n"
        "     and meta/{episodes.jsonl, tasks.jsonl, info.json, modality.json}.\n"
        "  5) Once the dataset writes successfully, re-run with the same --dst\n"
        "     plus `finalize --modality-only` to refresh meta/modality.json.\n"
        "\n"
        "The hf-download subcommand is the recommended path for now."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge-v2 → LeRobot v2.1 conversion driver")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("hf-download", help="Pull a pre-converted LeRobot Bridge-v2 mirror")
    p1.add_argument("--dst", required=True, help="Local destination dir (e.g. playground/Datasets/bridge_orig_lerobot)")
    p1.add_argument("--repo", default="IPEC-COMMUNITY/bridge_orig_1.0.0_lerobot",
                    help="HuggingFace dataset repo (default: IPEC-COMMUNITY mirror)")
    p1.add_argument("--dry-run", action="store_true", help="Print the hf command without executing")
    p1.set_defaults(func=cmd_hf_download)

    p2 = sub.add_parser("finalize", help="Validate + drop modality.json into an existing LeRobot dir")
    p2.add_argument("--dst", required=True, help="LeRobot dataset root (the dir containing data/, videos/, meta/)")
    p2.add_argument("--modality-only", action="store_true", help="Only refresh meta/modality.json")
    p2.add_argument("--symlink-target", default=None,
                    help="Optional convenience symlink, e.g. playground/Datasets/bridge_v2_lerobot")
    p2.set_defaults(func=cmd_finalize)

    p3 = sub.add_parser("from-rlds", help="(Stub) convert raw RLDS Bridge-v2 to LeRobot v2.1")
    p3.add_argument("--src", required=True, help="Path or TFDS spec for the raw RLDS Bridge-v2")
    p3.add_argument("--dst", required=True, help="Output LeRobot dir")
    p3.set_defaults(func=cmd_from_rlds)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
