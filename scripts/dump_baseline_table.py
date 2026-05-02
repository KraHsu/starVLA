#!/usr/bin/env python
"""Convert eval_libero.py mp4 outputs into a trial-level baseline_table.csv."""

import argparse
import csv
import re
from pathlib import Path

ROLLOUT_RE = re.compile(r"rollout_(?P<task>.+?)_episode(?P<trial>\d+)_(?P<status>success|failure)\.mp4")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_results", required=True, help="Directory containing rollout_*.mp4 files")
    p.add_argument("--out", default="paper/tables/baseline_table.csv")
    p.add_argument("--label", default="StarVLA-PI-Qwen3VL", help="Tag stored in CSV")
    args = p.parse_args()

    results_dir = Path(args.eval_results)
    rows = []
    for mp4 in sorted(results_dir.glob("rollout_*.mp4")):
        m = ROLLOUT_RE.match(mp4.name)
        if not m:
            continue
        rows.append({
            "model": args.label,
            "task_id": m.group("task"),
            "trial_id": int(m.group("trial")),
            "success": 1 if m.group("status") == "success" else 0,
            "mp4_path": str(mp4),
        })

    if not rows:
        raise SystemExit(f"No rollout_*.mp4 found under {results_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "task_id", "trial_id", "success", "mp4_path"])
        w.writeheader()
        w.writerows(rows)

    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r["success"])
    per_task_sr = {t: sum(s) / len(s) for t, s in by_task.items()}
    avg_sr = sum(per_task_sr.values()) / len(per_task_sr)

    print(f"Wrote {len(rows)} trials across {len(by_task)} tasks → {out}")
    print(f"Per-task success rate:")
    for t, sr in sorted(per_task_sr.items()):
        print(f"  {sr:.3f}  {t}")
    print(f"\nLIBERO-Long avg SR: {avg_sr:.4f}")
    print(f"G-W1 (≥0.86): {'PASS' if avg_sr >= 0.86 else 'FAIL'}")


if __name__ == "__main__":
    main()
