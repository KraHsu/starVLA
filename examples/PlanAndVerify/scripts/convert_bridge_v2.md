# Bridge-v2 → LeRobot v2.1 conversion (PaV W2 side task)

The Plan-and-Verify project's data side currently runs on LIBERO only (W2
shipped a LIBERO-scoped V-JEPA latent + LCLGP triplet pipeline). Bridge-v2
is the second mixture component the design doc §5.4 calls for, so that the
W4 evaluation can score the LCLGP head on a more cluttered, language-rich
robot dataset. This document is the recipe for getting Bridge-v2 into the
exact LeRobot v2.1 directory shape that
`examples/PlanAndVerify/train_files/data_registry/data_config.py
::BridgeWidowxDataConfig` expects.

The conversion driver lives at
`examples/PlanAndVerify/scripts/convert_bridge_v2.py` and exposes three
subcommands. Pick the recipe that matches your starting point.

---

## Recipe A — HuggingFace mirror (recommended, ~30 min on a fast link)

If a community-maintained LeRobot mirror of Bridge-v2 is already published
on HuggingFace, this is by far the shortest path. The default repo argument
points at the IPEC-COMMUNITY namespace which hosts the LIBERO LeRobot
mirrors used by W1/W2; check whether they have published a `bridge_orig`
dataset before relying on the default.

```bash
# 1) Confirm the mirror exists and inspect its size:
hf repo info IPEC-COMMUNITY/bridge_orig_1.0.0_lerobot --repo-type dataset

# 2) Run the converter:
python examples/PlanAndVerify/scripts/convert_bridge_v2.py hf-download \
    --dst playground/Datasets/bridge_orig_1.0.0_lerobot \
    --repo IPEC-COMMUNITY/bridge_orig_1.0.0_lerobot
```

Outputs:

* The raw LeRobot dir under `--dst` (data/, videos/, meta/).
* `meta/modality.json` matching `BridgeWidowxDataConfig`.
* A printed manifest summarising total_episodes / total_frames / fps / video_keys.

Add `--dry-run` to print the `hf` command without downloading.

If the IPEC-COMMUNITY mirror does not exist, search HuggingFace for any
`*bridge*lerobot*` dataset and pass it via `--repo`. If none of them
expose the `image_0` view your `data_config.py` registration expects,
fall back to Recipe C.

---

## Recipe B — Finalize an already-converted dir (modality patch only)

If you already have a LeRobot v2.1 Bridge-v2 dataset somewhere (e.g.
produced by a teammate's converter), use `finalize` to drop the matching
`modality.json` and verify the structure:

```bash
python examples/PlanAndVerify/scripts/convert_bridge_v2.py finalize \
    --dst /path/to/bridge_orig_lerobot \
    --symlink-target playground/Datasets/bridge_v2_lerobot
```

`--modality-only` skips the manifest validation if you only want to refresh
`meta/modality.json` after a re-encode or schema bump.

The symlink `playground/Datasets/bridge_v2_lerobot` matches the conventional
location used by other `examples/<bench>/data_preparation.sh` scripts and
keeps your `data_config.py` paths short.

---

## Recipe C — From the raw RLDS / TFDS release (advanced)

The original Bridge-v2 release ships in RLDS / TFDS format
(`bridge_dataset` on TFDS). Converting it to LeRobot v2.1 requires:

1. `pip install tensorflow tensorflow-datasets pandas pyarrow av`.
2. Knowing the RLDS schema for Bridge-v2 — each step typically exposes
   `observation/image_0`, `observation/state`, `action`,
   `language_instruction`, plus episode terminators.
3. A LeRobot v2.1 writer. The repo's existing `playground/Datasets/...`
   structure follows the layout in
   `playground/Datasets/LEROBOT_LIBERO_DATA/<suite>/meta/info.json`:
   ```
   data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
   videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
   meta/info.json
   meta/episodes.jsonl
   meta/tasks.jsonl
   meta/modality.json
   ```

The converter ships a `from-rlds` subcommand as a placeholder — invoking
it raises `NotImplementedError` with a concrete checklist so you can adapt
it to your environment without re-deriving the schema. The blocker on
implementing it in this repo is twofold: the dataset is not available on
the cluster, and the dependency stack (TF, av) is not pinned in
`requirements.txt`. Once you have a working conversion in your own
sandbox, you can either:

* Push the resulting LeRobot dir to HuggingFace and switch to Recipe A.
* Or upstream the converter into this script — replace the
  `cmd_from_rlds` body with the working code and re-run on the cluster.

---

## After conversion: wiring it into PaV

When the Bridge-v2 LeRobot dir lands at, say,
`playground/Datasets/bridge_v2_lerobot`, finish the integration in
**two edits** to `examples/PlanAndVerify/train_files/data_registry/data_config.py`:

```python
ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "bridge_widowx": BridgeWidowxDataConfig(),     # ← uncomment
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka": EmbodimentTag.FRANKA,
    "bridge_widowx": EmbodimentTag.NEW_EMBODIMENT, # ← uncomment, pick the right tag
}

DATASET_NAMED_MIXTURES["pav_full"] = [
    ("libero_object_no_noops_1.0.0_lerobot",  1.0, "libero_franka"),
    ("libero_goal_no_noops_1.0.0_lerobot",    1.0, "libero_franka"),
    ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ("libero_10_no_noops_1.0.0_lerobot",      1.0, "libero_franka"),
    ("bridge_orig_1.0.0_lerobot",             1.0, "bridge_widowx"),
]
```

Then re-extract V-JEPA latents for the new mixture (T-W2.2 will pick up
Bridge automatically once the registry entry exists):

```bash
bash examples/PlanAndVerify/eval_files/extract_pav_libero.sh \
    DATA_MIX=pav_full           # if the script reads DATA_MIX env var
# OR rerun the extraction with --mixture pav_full directly
```

Finally, rebuild the LCLGP triplet dataset on the combined latents:

```bash
python scripts/build_lclgp_dataset.py \
    --mixture pav_full \
    --output-dir data/lclgp_dataset/pav_full \
    --latent-root data/latents/pav_full
```

After that, the W3 trainer's data section can simply point at
`data/lclgp_dataset/pav_full` and pick up Bridge automatically.

---

## Smoke verification

Once the dataset is on disk, this 5-line sanity check reads one episode
and prints the modality keys — a green output means the `data_config.py`
registration will work:

```bash
python -c "
from pathlib import Path
import json
root = Path('playground/Datasets/bridge_v2_lerobot')
info = json.loads((root / 'meta' / 'info.json').read_text())
modality = json.loads((root / 'meta' / 'modality.json').read_text())
print('episodes:', info['total_episodes'], 'frames:', info['total_frames'])
print('features:', sorted(info['features'].keys())[:5], '...')
print('modality keys:', sorted(modality.keys()))
"
```

Expected output (numbers will vary):

```
episodes: 60096 frames: 2125000
features: ['action', 'episode_index', 'frame_index', 'index', 'observation.images.image_0'] ...
modality keys: ['action', 'annotation', 'state', 'video']
```
