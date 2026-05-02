# Plan-and-Verify

Verifier-only world-model loop on top of an off-the-shelf π₀ flow-matching policy.
See `examples/PlanAndVerify/README.md` for an entry point and `implementation_todo.md`
for the 10-week task breakdown.

## Status (current week)

| Phase | Window | Status |
|---|---|---|
| W1 | Repo + env + V-JEPA 2 wrapper + LIBERO baseline | in progress |
| W2 | Latent caching | not started |
| W3 | LCLGP training (G-W3 gate) | not started |
| W4 | MSFV verifier | not started |
| W5 | Runtime + PaV-Lite demo | not started |
| W6 | ETAR data (G-W6 gate) | not started |
| W7 | ETAR training + CEM-Light | not started |
| W8 | LIBERO main results | not started |
| W9 | Perturbation + ablations | not started |
| W10 | CALVIN + openpi + paper writeup | not started |

## W1 deliverables

- `starVLA/model/modules/world_model/vjepa2.py` — `VJEPA2Encoder`, `VJEPA2ACPredictor`,
  and a minimal `_VJEPA2_Interface` wired into the existing `get_world_model` factory
  (`elif "vjepa2" in wm_name.lower(): ...`). Full `_VJEPA2_Interface.build_inputs/forward`
  is finalized in W3 alongside LCLGP.
- `examples/PlanAndVerify/eval_files/{run_policy_server,eval_libero_long}.sh` — PAV-local
  copies of the LIBERO eval drivers, retargeted at uv-managed venvs and the
  StarVLA-PI ckpt.
- `tests/world_model/test_vjepa2.py` — 4 unit tests (factory routing + ViT-g + AC predictor
  shape smoke).
- `paper/tables/baseline_table.csv` — locked LIBERO-Long success table from
  StarVLA-PI Qwen3-VL.

## Gates

- **G-W1** (W1 end): LIBERO-Long mean SR ≥ 0.86 with the StarVLA-PI Qwen3-VL ckpt.
- **G-W3** (W3 end): LCLGP best-mode replaces image goal; reach-task SR ≥ 70 % when
  driving V-JEPA 2-AC.
- **G-W6** (W6 mid): GPT-4V vs human ETAR-label agreement κ ≥ 0.6.

## Key design decisions

- **Two uv venvs**, not conda. `.venv` (Python 3.11) carries the starVLA training stack
  plus V-JEPA 2; `.venv-libero` (Python 3.10) carries mujoco / robosuite / LIBERO.
  Two envs are unavoidable — LIBERO's numpy 1.24 + transformers 4.21 pins clash with
  starVLA's transformers 4.57.
- **No `WORLD_MODEL_REGISTRY`** in starVLA; world models route through the
  string-matching factory `get_world_model(config)` in
  `starVLA/model/modules/world_model/__init__.py`. We added one branch
  (`"vjepa2"` substring match) and nothing else.
- **PAV scripts are copies, not edits**, of the upstream `examples/LIBERO/eval_files/*`.
  `examples/LIBERO/` stays untouched per the project's "file-level isolation" rule.
- **`playground/LIBERO/` is a third-party clone** (`Lifelong-Robot-Learning/LIBERO`).
  We applied a single one-line patch (`weights_only=False` for `torch.load` of init
  states) to make it work with PyTorch 2.6+. `playground/` is gitignored, so this
  patch lives only on the local checkout.
- **Mujoco backend = OSMesa**, not EGL — the cluster's NVIDIA EGL doesn't expose the
  PLATFORM_DEVICE extension required for headless rendering. OSMesa is software but
  fast enough for LIBERO at 256² (~50 s per trial on H20).

## Performance expectations (W1 baseline)

A LIBERO-Long pass on 1×H20 with OSMesa rendering takes ~4 h for 10 tasks × 30 trials.
Each trial averages ~50 s end-to-end (Qwen3-VL + flow-matching action sampling +
mujoco step + websocket round-trip).
