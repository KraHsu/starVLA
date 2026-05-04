"""Materialize LCLGP training triplets + cached text embeddings.

T-W2.3.1 / T-W2.3.2 / T-W2.4.1 — see implementation_todo.md and
research_design_plan_and_verify.md §5.4.

Reads the per-frame V-JEPA 2 latent shards produced by
``scripts/extract_vjepa_latents.py`` and writes:

  <output_dir>/
    index_train.parquet     (data_name, traj_id, t, t_delta, t_end, lang_hash, length, lang)
    index_val.parquet
    index_test.parquet
    text_emb.h5             /<lang_hash>/emb [L, 2560] fp16   (lang attr utf8)
    STATS.md                per-dataset / per-split / per-task tables

Triplet semantics (design doc §5.3): for each successful demo we sample
``samples_per_demo`` random ``t`` and emit ``(t, t+Δ, T-1)`` where
``Δ = delta_steps`` (default 50, one chunk).

Split is stratified per-task by *demo*, not by task — every task is in
the train split (so ``TaskGroupedSampler`` can satisfy ``n_tasks=32`` at
training time). LIBERO has ~50 demos per task, so 80/10/10 is robust.

Smoke run on the dryrun shards::

    python scripts/build_lclgp_dataset.py \\
        --latent-root data/latents/dryrun \\
        --mixture pav_libero_long \\
        --output-dir data/lclgp_dataset/dryrun \\
        --qwen-vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.datasets.vjepa_latent_dataset import VJEPALatentShardSet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LCLGP triplet + text_emb builder (T-W2.3.1)")
    p.add_argument("--latent-root", type=Path, required=True,
                   help="Dir containing one subdir per dataset of *_rank*.h5 shards.")
    p.add_argument("--mixture", type=str, default=None,
                   help="A DATASET_NAMED_MIXTURES key (e.g. pav_libero, pav_libero_long).")
    p.add_argument("--data-name", type=str, default=None,
                   help="Single dataset directory name (alternative to --mixture).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--qwen-vlm", type=Path,
                   default=Path("playground/Pretrained_models/Qwen3-VL-4B-Instruct"),
                   help="Local Qwen3-VL-4B-Instruct path used to compute text embeddings.")
    p.add_argument("--samples-per-demo", type=int, default=10)
    p.add_argument("--delta-steps", type=int, default=50)
    p.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--text-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--skip-text-emb", action="store_true",
                   help="Don't run Qwen3-VL; useful for fast structural checks. text_emb.h5 will be empty.")
    return p.parse_args()


def resolve_targets(args: argparse.Namespace) -> List[str]:
    if args.data_name is not None:
        return [args.data_name]
    if args.mixture is None:
        raise ValueError("pass --mixture or --data-name")
    if args.mixture not in DATASET_NAMED_MIXTURES:
        raise KeyError(
            f"mixture {args.mixture!r} not registered; "
            f"have e.g. {sorted(DATASET_NAMED_MIXTURES)[:8]}"
        )
    return [d for (d, _, _) in DATASET_NAMED_MIXTURES[args.mixture]]


# ---------------------------------------------------------------------------
# Triplet sampling
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TripletRow:
    data_name: str
    traj_id: int
    t: int
    t_delta: int
    t_end: int
    lang_hash: str
    length: int
    lang: str


def lang_to_hash(lang: str) -> str:
    return hashlib.sha1(lang.encode("utf-8")).hexdigest()[:16]


def sample_triplets_for_dataset(
    store: VJEPALatentShardSet,
    *,
    data_name: str,
    samples_per_demo: int,
    delta_steps: int,
    seed: int,
) -> Tuple[List[TripletRow], Dict[str, int]]:
    rows: List[TripletRow] = []
    n_skipped_short = 0
    n_skipped_no_lang = 0
    seen_lang: Dict[str, int] = collections.Counter()

    for ref in store.trajectory_refs():
        length = ref.length
        if length < delta_steps + 2:
            n_skipped_short += 1
            continue
        lang = store.get_language(ref.traj_id).strip()
        if not lang:
            n_skipped_no_lang += 1
            continue
        seen_lang[lang] += 1

        # Per-traj seeded RNG keeps regeneration deterministic.
        rng = np.random.default_rng(seed * 1_000_007 + ref.traj_id)
        max_t_inclusive = length - delta_steps - 1
        n_unique = max_t_inclusive + 1
        replace = n_unique < samples_per_demo
        ts = rng.choice(n_unique, size=samples_per_demo, replace=replace)
        h = lang_to_hash(lang)
        for t in ts:
            t_int = int(t)
            rows.append(TripletRow(
                data_name=data_name,
                traj_id=ref.traj_id,
                t=t_int,
                t_delta=t_int + delta_steps,
                t_end=length - 1,
                lang_hash=h,
                length=length,
                lang=lang,
            ))

    stats = {
        "n_trajs": store.num_trajectories,
        "n_trajs_used": store.num_trajectories - n_skipped_short - n_skipped_no_lang,
        "n_skipped_short": n_skipped_short,
        "n_skipped_no_lang": n_skipped_no_lang,
        "n_tasks": len(seen_lang),
        "n_triplets": len(rows),
        "mean_demo_length": float(np.mean([ref.length for ref in store.trajectory_refs()])),
    }
    return rows, stats


# ---------------------------------------------------------------------------
# Stratified per-task split (by demo, not by task)
# ---------------------------------------------------------------------------

def stratified_split(
    rows: Sequence[TripletRow],
    *,
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[TripletRow], List[TripletRow], List[TripletRow]]:
    """Split by (lang_hash, traj_id) demo. Every task is guaranteed present in train."""
    rng = np.random.default_rng(seed)
    by_task: Dict[str, Dict[Tuple[str, int], List[TripletRow]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for r in rows:
        by_task[r.lang_hash][(r.data_name, r.traj_id)].append(r)

    train, val, test = [], [], []
    for task_hash, demos in by_task.items():
        demo_keys = list(demos.keys())
        rng.shuffle(demo_keys)
        n = len(demo_keys)
        # Allocate with the invariant that train always gets ≥ 1 demo so the task is in train.
        n_train = max(1, int(round(n * ratios[0])))
        if n_train > n:
            n_train = n
        rem = n - n_train
        n_val = int(round(rem * ratios[1] / max(ratios[1] + ratios[2], 1e-9)))
        n_val = max(0, min(rem, n_val))
        n_test = rem - n_val
        for k in demo_keys[:n_train]:
            train.extend(demos[k])
        for k in demo_keys[n_train:n_train + n_val]:
            val.extend(demos[k])
        for k in demo_keys[n_train + n_val:]:
            test.extend(demos[k])
    return train, val, test


# ---------------------------------------------------------------------------
# Text embedding via Qwen3-VL-4B (text-only path, no vision tower)
# ---------------------------------------------------------------------------

def compute_text_embeddings(
    langs: Sequence[str],
    *,
    qwen_path: Path,
    device: str,
    dtype: str,
) -> Dict[str, np.ndarray]:
    """Run each unique instruction through Qwen3-VL's language stack (no images).

    Returns ``{lang: [L, hidden] fp16}`` un-padded per instruction. Padding is the
    dataloader's job at training time.
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    print(f"[text_emb] loading {qwen_path} dtype={dtype} device={device}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(qwen_path),
        dtype=torch_dtype,
        attn_implementation="sdpa",
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(str(qwen_path))

    # Locate the text-only language model. transformers ≥ 4.46 uses
    # `model.model.language_model` for Qwen3-VL; older builds may differ.
    inner = model.model
    text_lm = None
    for attr in ("language_model", "text_model"):
        if hasattr(inner, attr):
            text_lm = getattr(inner, attr)
            print(f"[text_emb] using submodule model.model.{attr}")
            break
    if text_lm is None:
        # Fallback: inner Qwen3VLModel.forward routes text-only when pixel_values is None.
        text_lm = inner
        print("[text_emb] using submodule model.model (no language_model attr found)")

    out: Dict[str, np.ndarray] = {}
    for lang in langs:
        # Use the raw tokenizer, not apply_chat_template — we want the
        # plain instruction tokens fed to LCLGP, not chat scaffolding.
        toks = processor.tokenizer(lang, return_tensors="pt")
        input_ids = toks.input_ids.to(device)
        attention_mask = toks.attention_mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                dtype=torch_dtype):
                lm_out = text_lm(input_ids=input_ids, attention_mask=attention_mask)
        emb = lm_out.last_hidden_state[0].float().cpu().numpy().astype(np.float16)
        out[lang] = emb
        print(f"[text_emb] {lang_to_hash(lang)} L={emb.shape[0]} d={emb.shape[1]}  '{lang[:60]}'")

    # Free the model so downstream stages don't OOM.
    del model
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    return out


def write_text_emb_h5(emb_map: Dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for lang, emb in emb_map.items():
            h = lang_to_hash(lang)
            grp = f.create_group(h)
            ds = grp.create_dataset("emb", data=emb, compression="gzip", compression_opts=1)
            grp.attrs["lang"] = lang
            grp.attrs["n_tokens"] = int(emb.shape[0])
            grp.attrs["hidden"] = int(emb.shape[1])


# ---------------------------------------------------------------------------
# Parquet + STATS writers
# ---------------------------------------------------------------------------

def rows_to_dataframe(rows: Sequence[TripletRow]) -> pd.DataFrame:
    return pd.DataFrame([dataclasses.asdict(r) for r in rows])


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_stats_md(
    *,
    output_dir: Path,
    per_dataset_stats: Dict[str, Dict[str, int]],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    samples_per_demo: int,
    delta_steps: int,
    text_hidden: int | None,
    qwen_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# LCLGP Dataset Stats")
    lines.append("")
    lines.append(f"- generator: `scripts/build_lclgp_dataset.py`")
    lines.append(f"- samples_per_demo: {samples_per_demo}")
    lines.append(f"- delta_steps: {delta_steps}")
    lines.append(f"- text encoder: `{qwen_path}` (hidden={text_hidden})")
    lines.append("")

    lines.append("## Per-dataset")
    lines.append("")
    lines.append("| dataset | trajs (used / total) | tasks | mean demo length | triplets |")
    lines.append("|---|---|---|---|---|")
    for name, st in per_dataset_stats.items():
        lines.append(
            f"| `{name}` | {st['n_trajs_used']} / {st['n_trajs']} | {st['n_tasks']} "
            f"| {st['mean_demo_length']:.1f} | {st['n_triplets']} |"
        )
    lines.append("")

    lines.append("## Per-split")
    lines.append("")
    lines.append("| split | rows | unique tasks | unique demos |")
    lines.append("|---|---|---|---|")
    for name, df in (("train", train), ("val", val), ("test", test)):
        if len(df) == 0:
            lines.append(f"| {name} | 0 | 0 | 0 |")
            continue
        n_tasks = df["lang_hash"].nunique()
        n_demos = df.drop_duplicates(["data_name", "traj_id"]).shape[0]
        lines.append(f"| {name} | {len(df)} | {n_tasks} | {n_demos} |")
    lines.append("")

    # Sanity: every task must be in train
    train_tasks = set(train["lang_hash"].unique()) if len(train) else set()
    all_tasks = set(pd.concat([train, val, test])["lang_hash"].unique()) \
        if any(len(d) for d in (train, val, test)) else set()
    missing = sorted(all_tasks - train_tasks)
    if missing:
        lines.append(f"> ⚠️ {len(missing)} tasks missing from train: {missing[:5]}…")
        lines.append("")

    lines.append("## Per-task demos (train / val / test)")
    lines.append("")
    lines.append("| lang_hash | lang | train | val | test |")
    lines.append("|---|---|---|---|---|")
    all_df = pd.concat([train.assign(_split="train"),
                        val.assign(_split="val"),
                        test.assign(_split="test")])
    if len(all_df):
        per_task = (all_df
                    .drop_duplicates(["data_name", "traj_id", "_split"])
                    .groupby(["lang_hash", "lang", "_split"])
                    .size()
                    .unstack(fill_value=0))
        for col in ("train", "val", "test"):
            if col not in per_task.columns:
                per_task[col] = 0
        per_task = per_task[["train", "val", "test"]]
        for (h, lang), row in per_task.iterrows():
            short = lang if len(lang) <= 80 else lang[:77] + "…"
            lines.append(f"| `{h}` | {short} | {int(row['train'])} | {int(row['val'])} | {int(row['test'])} |")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "STATS.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = resolve_targets(args)
    print(f"[build] targets ({len(targets)}): {targets}")

    # ------- Step 1+2: triplet sampling -------
    all_rows: List[TripletRow] = []
    per_dataset_stats: Dict[str, Dict[str, int]] = {}
    for data_name in targets:
        shard_dir = args.latent_root / data_name
        if not shard_dir.is_dir():
            raise FileNotFoundError(
                f"latent shard dir missing: {shard_dir} — did extract_vjepa_latents.py run for this dataset?"
            )
        store = VJEPALatentShardSet(shard_dir)
        rows, stats = sample_triplets_for_dataset(
            store,
            data_name=data_name,
            samples_per_demo=args.samples_per_demo,
            delta_steps=args.delta_steps,
            seed=args.seed,
        )
        all_rows.extend(rows)
        per_dataset_stats[data_name] = stats
        print(f"[build] {data_name}: {stats}")

    if not all_rows:
        raise RuntimeError("no triplets produced — check shard contents and --delta-steps")

    # ------- Step 3: stratified split -------
    train_rows, val_rows, test_rows = stratified_split(
        all_rows, ratios=tuple(args.split), seed=args.seed
    )
    train_df = rows_to_dataframe(train_rows)
    val_df = rows_to_dataframe(val_rows)
    test_df = rows_to_dataframe(test_rows)
    print(f"[build] split: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # ------- Step 4: text embeddings -------
    unique_langs = sorted({r.lang for r in all_rows})
    print(f"[build] unique tasks (by lang string): {len(unique_langs)}")
    text_hidden: int | None = None
    if args.skip_text_emb:
        write_text_emb_h5({}, args.output_dir / "text_emb.h5")
        print("[build] --skip-text-emb set; text_emb.h5 written empty")
    else:
        t0 = time.time()
        emb_map = compute_text_embeddings(
            unique_langs,
            qwen_path=args.qwen_vlm,
            device=args.device,
            dtype=args.text_dtype,
        )
        write_text_emb_h5(emb_map, args.output_dir / "text_emb.h5")
        text_hidden = next(iter(emb_map.values())).shape[1] if emb_map else None
        print(f"[build] text_emb.h5 written in {time.time() - t0:.1f}s "
              f"({len(emb_map)} entries, hidden={text_hidden})")

    # ------- Step 5: parquet + STATS -------
    write_parquet(train_df, args.output_dir / "index_train.parquet")
    write_parquet(val_df, args.output_dir / "index_val.parquet")
    write_parquet(test_df, args.output_dir / "index_test.parquet")

    write_stats_md(
        output_dir=args.output_dir,
        per_dataset_stats=per_dataset_stats,
        train=train_df, val=val_df, test=test_df,
        samples_per_demo=args.samples_per_demo,
        delta_steps=args.delta_steps,
        text_hidden=text_hidden,
        qwen_path=args.qwen_vlm,
    )

    # JSON sidecar so downstream tooling has a typed manifest.
    manifest = {
        "mixture": args.mixture,
        "data_names": targets,
        "samples_per_demo": args.samples_per_demo,
        "delta_steps": args.delta_steps,
        "split_ratios": list(args.split),
        "seed": args.seed,
        "text_encoder": str(args.qwen_vlm),
        "text_hidden": text_hidden,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_tasks": len(unique_langs),
        "per_dataset": per_dataset_stats,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[build] done → {args.output_dir}")


if __name__ == "__main__":
    main()
