"""W3.4 LCLGP diagnostics + G-W3 gate driver.

Run with::

    python examples/PlanAndVerify/scripts/run_diagnostics.py <subcmd> \\
        --config_yaml examples/PlanAndVerify/configs/lclgp_v1.yaml \\
        --checkpoint <path/to/best.pt> \\
        --output_dir paper/

Subcommands implemented end-to-end (do not require external components):
  g1         G-1 min-of-K cosine to ground truth (target ≥ 0.75 mean).
  d1         D1-a/b/c — mode pairwise cosine, coverage hist, σ-vs-error.
  d2         D2-a/b   — counterfactual L1, cross-start variation.
  d3         D3-a     — swap end↔delta Spearman.

Subcommands stubbed (need extra components — they emit a clear TODO):
  g2         G-2 decoded sample viz (needs V-JEPA 2 decoder).
  g3         G-W3 gate: best-mode → V-JEPA 2-AC single-step CEM on LIBERO-Spatial reach.
  d1d       D1-d K=4 vs K=1 ablation (needs separately trained K=1 ckpt).
  d2c       D2-c z_t-ablation training (needs no-z_t ckpt).
  d3b       D3-b dual vs end-only vs delta-only (needs ablation ckpts).

Outputs land under ``--output_dir`` as CSV / Markdown.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.datasets.lclgp_triplet_dataset import LcLgpTripletDataset, collate_lclgp
from starVLA.model.framework.base_framework import build_framework

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_model(cfg, ckpt_path: str, device: str) -> torch.nn.Module:
    model = build_framework(cfg).to(device).eval()
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"⚠️ missing keys: {len(missing)}  e.g. {list(missing)[:3]}")
    if unexpected:
        print(f"⚠️ unexpected keys: {len(unexpected)}  e.g. {list(unexpected)[:3]}")
    return model


def build_eval_loader(cfg, split: str = "val", batch_size: int = 16, num_workers: int = 2) -> DataLoader:
    data_cfg = cfg.data.lclgp_dataset
    index_path = {"train": data_cfg.index_train, "val": data_cfg.index_val, "test": data_cfg.index_test}[split]
    ds = LcLgpTripletDataset(
        index_path=index_path,
        latent_root=data_cfg.latent_root,
        text_emb_path=data_cfg.text_emb,
        view=getattr(data_cfg, "view", "primary"),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_lclgp)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"📊 wrote {path}")


def patch_mean(z: torch.Tensor) -> torch.Tensor:
    """Mean over the patch dimension (axis = -2) → [..., D]."""
    return z.mean(dim=-2)


def cosine(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=dim, eps=eps)


# ---------------------------------------------------------------------------
# G-1: Min-of-K cosine to ground truth
# ---------------------------------------------------------------------------


def cmd_g1(args, cfg, model, device) -> None:
    loader = build_eval_loader(cfg, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)
    cos_end_min, cos_delta_min = [], []
    cos_end_best_sigma, cos_delta_best_sigma = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model.predict_goal(batch)
            # All-K cosine to GT, take best per sample.
            z_g_end_pool = patch_mean(out["z_g_end_all"])         # [B, K, D]
            z_g_delta_pool = patch_mean(out["z_g_delta_all"])     # [B, K, D]
            z_end_pool = patch_mean(batch["z_end"]).unsqueeze(1).to(z_g_end_pool.dtype)
            z_delta_pool = patch_mean(batch["z_delta"]).unsqueeze(1).to(z_g_delta_pool.dtype)
            cos_end = cosine(z_g_end_pool, z_end_pool, dim=-1)    # [B, K]
            cos_delta = cosine(z_g_delta_pool, z_delta_pool, dim=-1)
            cos_end_min.append(cos_end.max(dim=-1).values.cpu())   # max over K = closest mode
            cos_delta_min.append(cos_delta.max(dim=-1).values.cpu())

            # Also report cosine of σ-best mode.
            best_e = out["best_mode_end"]
            best_d = out["best_mode_delta"]
            idx = torch.arange(z_g_end_pool.shape[0], device=device)
            cos_end_best_sigma.append(cos_end[idx, best_e].cpu())
            cos_delta_best_sigma.append(cos_delta[idx, best_d].cpu())

    end_min = torch.cat(cos_end_min)
    delta_min = torch.cat(cos_delta_min)
    end_sigma = torch.cat(cos_end_best_sigma)
    delta_sigma = torch.cat(cos_delta_best_sigma)

    rows = [
        {"metric": "G1_end_min_cos_mean",    "value": float(end_min.mean()),    "threshold": 0.75, "pass": bool(end_min.mean() >= 0.75)},
        {"metric": "G1_end_min_cos_median",  "value": float(end_min.median()),  "threshold": 0.75, "pass": bool(end_min.median() >= 0.75)},
        {"metric": "G1_delta_min_cos_mean",  "value": float(delta_min.mean()),  "threshold": 0.75, "pass": bool(delta_min.mean() >= 0.75)},
        {"metric": "G1_end_sigma_cos_mean",  "value": float(end_sigma.mean()),  "threshold": 0.70, "pass": bool(end_sigma.mean() >= 0.70)},
        {"metric": "G1_delta_sigma_cos_mean", "value": float(delta_sigma.mean()), "threshold": 0.70, "pass": bool(delta_sigma.mean() >= 0.70)},
    ]
    write_csv(Path(args.output_dir) / "tables" / "lclgp_g1.csv", rows, ["metric", "value", "threshold", "pass"])


# ---------------------------------------------------------------------------
# D1: multimodality
# ---------------------------------------------------------------------------


def cmd_d1(args, cfg, model, device) -> None:
    loader = build_eval_loader(cfg, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)

    pairwise_cos_acc: List[torch.Tensor] = []
    mode_argmin_end_acc: List[torch.Tensor] = []
    mode_argmin_delta_acc: List[torch.Tensor] = []
    sigma_acc: List[torch.Tensor] = []
    err_acc: List[torch.Tensor] = []
    # W3 v5 — D1-d router routing distribution. Populated only when
    # ``predict_goal`` returns ``pi_router_*`` (i.e., ``use_router=True``).
    # v4 ckpts: lists stay empty; D1-d row reports N/A.
    router_argmax_end_acc: List[torch.Tensor] = []
    router_argmax_delta_acc: List[torch.Tensor] = []

    K = model.K

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model.predict_goal(batch)

            pool_end = patch_mean(out["z_g_end_all"])              # [B, K, D]
            B = pool_end.shape[0]

            # D1-a: pairwise cosine across modes.
            pool_end_n = torch.nn.functional.normalize(pool_end.float(), dim=-1)
            cos_mat = pool_end_n @ pool_end_n.transpose(-2, -1)    # [B, K, K]
            tri = torch.triu(torch.ones(K, K, device=cos_mat.device, dtype=torch.bool), diagonal=1)
            pair_cos = cos_mat[:, tri].mean(dim=-1)                # [B]
            pairwise_cos_acc.append(pair_cos.cpu())

            # D1-b (legacy, hindsight argmin): per sample under heteroscedastic loss.
            # Kept for cross-version comparison. v5 success criterion uses D1-d (router).
            z_end_pool = patch_mean(batch["z_end"]).unsqueeze(1).to(pool_end.dtype)   # [B, 1, D]
            l1_end = (pool_end - z_end_pool).abs().mean(dim=-1)                       # [B, K]
            sigma_end = out["log_sigma_end"].exp()
            per_mode = l1_end / sigma_end + 0.1 * out["log_sigma_end"]                # heuristic: β=0.1
            argmin_end = per_mode.argmin(dim=-1)
            mode_argmin_end_acc.append(argmin_end.cpu())

            pool_delta = patch_mean(out["z_g_delta_all"])
            z_delta_pool = patch_mean(batch["z_delta"]).unsqueeze(1).to(pool_delta.dtype)
            l1_delta = (pool_delta - z_delta_pool).abs().mean(dim=-1)
            sigma_delta = out["log_sigma_delta"].exp()
            per_mode_delta = l1_delta / sigma_delta + 0.1 * out["log_sigma_delta"]
            mode_argmin_delta_acc.append(per_mode_delta.argmin(dim=-1).cpu())

            # D1-c: σ vs error correlation (calibration test).
            sigma_acc.append(sigma_end.flatten().cpu())
            err_acc.append(l1_end.flatten().cpu())

            # D1-d (W3 v5): router output distribution. Only collected when
            # ``predict_goal`` exposes ``pi_router_*`` (i.e., ``use_router=True``).
            if "pi_router_end" in out and out["pi_router_end"] is not None:
                router_argmax_end_acc.append(out["pi_router_end"].argmax(dim=-1).cpu())
                router_argmax_delta_acc.append(out["pi_router_delta"].argmax(dim=-1).cpu())

    pair_cos = torch.cat(pairwise_cos_acc)
    argmin_end = torch.cat(mode_argmin_end_acc)
    argmin_delta = torch.cat(mode_argmin_delta_acc)
    sig = torch.cat(sigma_acc).numpy()
    err = torch.cat(err_acc).numpy()

    cov_end = torch.bincount(argmin_end, minlength=K).float() / argmin_end.numel()
    cov_delta = torch.bincount(argmin_delta, minlength=K).float() / argmin_delta.numel()

    pearson = float(np.corrcoef(sig, err)[0, 1]) if sig.size > 1 else float("nan")

    rows = [
        {"metric": "D1a_mean_pairwise_cos", "value": float(pair_cos.mean()),
         "threshold_low": 0.30, "threshold_high": 0.70, "pass": bool(0.30 <= pair_cos.mean() <= 0.70)},
        {"metric": "D1b_min_mode_freq_end", "value": float(cov_end.min()),
         "threshold_low": 0.10, "threshold_high": None, "pass": bool(cov_end.min() >= 0.10)},
        {"metric": "D1b_min_mode_freq_delta", "value": float(cov_delta.min()),
         "threshold_low": 0.10, "threshold_high": None, "pass": bool(cov_delta.min() >= 0.10)},
        {"metric": "D1c_pearson_sigma_err",  "value": pearson,
         "threshold_low": 0.40, "threshold_high": None, "pass": bool(pearson >= 0.40)},
    ]
    for k in range(K):
        rows.append({"metric": f"D1b_cov_end_k{k}", "value": float(cov_end[k]),
                     "threshold_low": None, "threshold_high": None, "pass": ""})

    # D1-d (W3 v5 — primary success criterion): router routing distribution.
    # Threshold: min router freq ≥ 0.10. Same threshold as legacy D1-b but on
    # router output (no GT used) rather than hindsight argmin (uses GT). The v5
    # architecture explicitly addresses the L_bal/argmin mismatch (docs §9.4),
    # so the v5 success metric must measure the new architecture's behavior
    # not the v4 one. Cross-version comparison: legacy D1-b stays in the table.
    if router_argmax_end_acc:
        ra_end = torch.cat(router_argmax_end_acc)
        ra_delta = torch.cat(router_argmax_delta_acc)
        cov_router_end = torch.bincount(ra_end, minlength=K).float() / ra_end.numel()
        cov_router_delta = torch.bincount(ra_delta, minlength=K).float() / ra_delta.numel()
        rows.append({"metric": "D1d_min_router_freq_end", "value": float(cov_router_end.min()),
                     "threshold_low": 0.10, "threshold_high": None,
                     "pass": bool(cov_router_end.min() >= 0.10)})
        rows.append({"metric": "D1d_min_router_freq_delta", "value": float(cov_router_delta.min()),
                     "threshold_low": 0.10, "threshold_high": None,
                     "pass": bool(cov_router_delta.min() >= 0.10)})
        for k in range(K):
            rows.append({"metric": f"D1d_router_freq_end_k{k}", "value": float(cov_router_end[k]),
                         "threshold_low": None, "threshold_high": None, "pass": ""})
        for k in range(K):
            rows.append({"metric": f"D1d_router_freq_delta_k{k}", "value": float(cov_router_delta[k]),
                         "threshold_low": None, "threshold_high": None, "pass": ""})
    else:
        rows.append({"metric": "D1d_min_router_freq_end", "value": float("nan"),
                     "threshold_low": 0.10, "threshold_high": None, "pass": "N/A (use_router=False)"})
        rows.append({"metric": "D1d_min_router_freq_delta", "value": float("nan"),
                     "threshold_low": 0.10, "threshold_high": None, "pass": "N/A (use_router=False)"})

    write_csv(Path(args.output_dir) / "tables" / "lclgp_d1.csv", rows,
              ["metric", "value", "threshold_low", "threshold_high", "pass"])


# ---------------------------------------------------------------------------
# D2: state dependence
# ---------------------------------------------------------------------------


def cmd_d2(args, cfg, model, device) -> None:
    loader = build_eval_loader(cfg, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)
    diff_l1_acc: List[torch.Tensor] = []

    # Group by task for D2-b.
    task_to_outputs: Dict[str, List[torch.Tensor]] = {}

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out_main = model.predict_goal(batch_dev)
            zero_batch = dict(batch_dev)
            zero_batch["z_t"] = torch.zeros_like(batch_dev["z_t"])
            out_cf = model.predict_goal(zero_batch)

            # D2-a: counterfactual L1 (compare best-σ mode for each).
            z_main = patch_mean(out_main["z_g_end_best"])
            z_cf = patch_mean(out_cf["z_g_end_best"])
            diff = (z_main - z_cf).abs().mean(dim=-1)
            diff_l1_acc.append(diff.cpu())

            # D2-b: same-task variation across different starts.
            for i, tid in enumerate(batch["task_id"]):
                task_to_outputs.setdefault(tid, []).append(z_main[i].cpu())

    diff_all = torch.cat(diff_l1_acc)
    # D2-b: pick top tasks with ≥ 4 samples; compute std across those samples' z_main.
    per_task_std = []
    for tid, outs in task_to_outputs.items():
        if len(outs) < 4:
            continue
        stk = torch.stack(outs, dim=0)                # [N, D]
        per_task_std.append(stk.std(dim=0).mean().item())
    median_intra_task_std = float(np.median(per_task_std)) if per_task_std else float("nan")

    rows = [
        {"metric": "D2a_mean_cf_L1",          "value": float(diff_all.mean()),
         "threshold": 0.05, "pass": bool(diff_all.mean() >= 0.05)},
        {"metric": "D2b_median_intra_task_std", "value": median_intra_task_std,
         "threshold": None, "pass": ""},
        {"metric": "D2b_num_tasks_with_4plus_samples", "value": float(len(per_task_std)),
         "threshold": None, "pass": ""},
    ]
    write_csv(Path(args.output_dir) / "tables" / "lclgp_d2.csv", rows, ["metric", "value", "threshold", "pass"])


# ---------------------------------------------------------------------------
# D3: temporal scales
# ---------------------------------------------------------------------------


def cmd_d3(args, cfg, model, device) -> None:
    """D3-a: cosine of best end-mode to GT z_delta (and vice versa) — should be lower than direct."""
    loader = build_eval_loader(cfg, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)

    cos_end_to_end, cos_end_to_delta = [], []
    cos_delta_to_delta, cos_delta_to_end = [], []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model.predict_goal(batch_dev)
            z_e = patch_mean(out["z_g_end_best"])
            z_d = patch_mean(out["z_g_delta_best"])
            gt_e = patch_mean(batch_dev["z_end"]).to(z_e.dtype)
            gt_d = patch_mean(batch_dev["z_delta"]).to(z_d.dtype)
            cos_end_to_end.append(cosine(z_e, gt_e).cpu())
            cos_end_to_delta.append(cosine(z_e, gt_d).cpu())
            cos_delta_to_delta.append(cosine(z_d, gt_d).cpu())
            cos_delta_to_end.append(cosine(z_d, gt_e).cpu())

    rows = [
        {"metric": "D3a_end→end_cos_mean",       "value": float(torch.cat(cos_end_to_end).mean()),       "pass": ""},
        {"metric": "D3a_end→delta_cos_mean",     "value": float(torch.cat(cos_end_to_delta).mean()),     "pass": ""},
        {"metric": "D3a_delta→delta_cos_mean",   "value": float(torch.cat(cos_delta_to_delta).mean()),   "pass": ""},
        {"metric": "D3a_delta→end_cos_mean",     "value": float(torch.cat(cos_delta_to_end).mean()),     "pass": ""},
    ]
    end_diff = float(torch.cat(cos_end_to_end).mean() - torch.cat(cos_end_to_delta).mean())
    delta_diff = float(torch.cat(cos_delta_to_delta).mean() - torch.cat(cos_delta_to_end).mean())
    rows.append({"metric": "D3a_end_specialization_gap", "value": end_diff, "pass": bool(end_diff >= 0.05)})
    rows.append({"metric": "D3a_delta_specialization_gap", "value": delta_diff, "pass": bool(delta_diff >= 0.05)})
    write_csv(Path(args.output_dir) / "tables" / "lclgp_d3.csv", rows, ["metric", "value", "pass"])


# ---------------------------------------------------------------------------
# Stubs for not-yet-buildable subcommands.
# ---------------------------------------------------------------------------


def cmd_g2(args, cfg, model, device) -> None:
    print("⚠️  g2 (decoded sample viz) requires a V-JEPA 2 decoder, which is not part of the W3 deliverable.")
    print("   When the decoder is integrated, decode predict_goal()['z_g_end_best'] for 5 random tasks → save PNGs.")


def cmd_g3(args, cfg, model, device) -> None:
    print("⚠️  g3 (G-W3 reach gate) requires V-JEPA 2-AC single-step CEM, integrated separately under examples/PlanAndVerify/eval_files/.")
    print("   Pseudo-code:")
    print("     for task in LIBERO_SPATIAL_REACH_5:")
    print("         z_t = vjepa.encode(env.obs())")
    print("         z_g = lclgp.predict_goal({...})['z_g_delta_best']")
    print("         action = vjepa_ac_cem(z_t, z_g)")
    print("         env.step(action) ... rollout, count successes")
    print("   Output: paper/tables/g_w3_reach_sr.csv with per-task SR and mean.")


def cmd_d1d_d2c_d3b(args, cfg, model, device) -> None:
    print("⚠️  d1d/d2c/d3b require ablation training runs (K=1, no-z_t, end-only/delta-only).")
    print("   After main run completes and passes G-W3, retrain those variants with overrides:")
    print("     # K=1 ablation:")
    print("     bash run_lclgp.sh --framework.lclgp.n_modes 1 --run_id ${RUN_ID}_k1")
    print("     # no-z_t ablation:")
    print("     bash run_lclgp.sh --framework.lclgp.z_dropout 1.0 --run_id ${RUN_ID}_no_zt")
    print("   Then run g1/d1 separately on each ckpt and compare.")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def cmd_report(args, cfg, model, device) -> None:
    """Aggregate all CSVs under <output_dir>/tables into docs/lclgp_diagnostics.md."""
    tables_dir = Path(args.output_dir) / "tables"
    md_lines = ["# LCLGP Diagnostics Report (W3.4)\n"]
    for csv_path in sorted(tables_dir.glob("lclgp_*.csv")):
        md_lines.append(f"## {csv_path.stem}\n")
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        header, *data = rows
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in data:
            md_lines.append("| " + " | ".join(r) + " |")
        md_lines.append("")
    out = Path(args.docs_dir) / "lclgp_diagnostics.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(md_lines))
    print(f"📝 wrote {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


SUBCMD_TABLE = {
    "g1": cmd_g1, "g2": cmd_g2, "g3": cmd_g3,
    "d1": cmd_d1, "d2": cmd_d2, "d3": cmd_d3,
    "d1d": cmd_d1d_d2c_d3b, "d2c": cmd_d1d_d2c_d3b, "d3b": cmd_d1d_d2c_d3b,
    "report": cmd_report,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="LCLGP W3.4 diagnostics + G-W3 gate")
    parser.add_argument("subcmd", choices=list(SUBCMD_TABLE.keys()))
    parser.add_argument("--config_yaml", required=False, default="examples/PlanAndVerify/configs/lclgp_v1.yaml")
    parser.add_argument("--checkpoint", required=False, default=None)
    parser.add_argument("--output_dir", default="paper")
    parser.add_argument("--docs_dir", default="docs")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    needs_model = args.subcmd in {"g1", "d1", "d2", "d3", "g2", "g3", "d1d", "d2c", "d3b"}
    cfg = OmegaConf.load(args.config_yaml)

    model = None
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    if needs_model:
        if args.checkpoint is None:
            raise SystemExit(f"--checkpoint is required for subcmd {args.subcmd}")
        model = load_model(cfg, args.checkpoint, device)

    SUBCMD_TABLE[args.subcmd](args, cfg, model, device)


if __name__ == "__main__":
    main()
