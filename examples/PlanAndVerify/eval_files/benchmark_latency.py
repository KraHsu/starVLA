"""Stage 1 / §1.3 — single-card inference latency benchmark.

Times QwenOFT.predict_action() directly (no websocket / no LIBERO sim) on
batch=1 with a synthetic LIBERO-shaped observation, so the number reflects
end-to-end model compute (vision tower + LLM forward + action head)
without sim or RPC overhead.

Usage (run from repo root):

    .venv/bin/python examples/PlanAndVerify/eval_files/benchmark_latency.py \
        --ckpt_path playground/Checkpoints/<run_id>/checkpoints/steps_20000_pytorch_model.pt \
        --warmup 5 --iters 50 --use_bf16

Reports: mean / median / p95 / max ms per predict_action call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework


def make_libero_sample(image_size: int, state_dim: int, with_state: bool) -> dict:
    img = Image.fromarray(np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8))
    sample = {
        "image": [img, img],  # primary + wrist (LIBERO returns 2 views)
        "lang": "put the bowl on the stove",
    }
    if with_state:
        sample["state"] = np.random.uniform(-1.0, 1.0, size=(1, state_dim)).astype(np.float16)
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, help="Absolute path to .pt checkpoint")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--with_state", action="store_true",
                        help="Include 'state' in the sample (LIBERO libero_franka does NOT ship state — leave off)")
    parser.add_argument("--use_bf16", action="store_true",
                        help="Cast model to bfloat16 (matches policy server default)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None,
                        help="Optional JSON output path; default: <ckpt_dir>/latency.json")
    args = parser.parse_args()

    print(f"==> Loading framework from {args.ckpt_path}")
    vla = baseframework.from_pretrained(args.ckpt_path)
    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to(args.device).eval()

    sample = make_libero_sample(args.image_size, args.state_dim, args.with_state)
    print(f"==> Sample keys: {sorted(sample.keys())}")

    # Warmup — never include in stats.
    print(f"==> Warmup {args.warmup} iters")
    for i in range(args.warmup):
        with torch.no_grad():
            out = vla.predict_action(examples=[sample])
        if i == 0:
            shape = out["normalized_actions"].shape
            print(f"  predict_action output shape={shape}")

    # Sync before each timing call so we measure full GPU work.
    torch.cuda.synchronize() if args.device.startswith("cuda") else None

    print(f"==> Timing {args.iters} iters")
    times_ms: list[float] = []
    for i in range(args.iters):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            vla.predict_action(examples=[sample])
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{args.iters}] {dt:.1f} ms")

    times_sorted = sorted(times_ms)
    n = len(times_sorted)
    p95 = times_sorted[max(int(0.95 * n) - 1, 0)]
    report = {
        "ckpt_path": args.ckpt_path,
        "device": args.device,
        "use_bf16": args.use_bf16,
        "image_size": args.image_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "mean_ms": statistics.mean(times_ms),
        "p50_ms": statistics.median(times_ms),
        "p95_ms": p95,
        "max_ms": max(times_ms),
        "min_ms": min(times_ms),
    }
    print("\n=== latency report ===")
    print(json.dumps(report, indent=2))

    out_path = Path(args.out) if args.out else Path(args.ckpt_path).parent / "latency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
