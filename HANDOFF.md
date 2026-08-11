# Graft campaign handoff — state as of 2026-08-11 evening

Point-in-time handoff for the next agent. The normative plan is
`docs/MANTIS_GRAFT_SPEC.md`; measured records and verdicts live in
`docs/ABLATIONS.md` ("Graft campaign" section). Where this file disagrees
with either, they win.

## Where things stand

Branch `mantisnet-graft` (worktree `latest-run-recovery-stack-148e60`), all
commits local and unpushed. Recent chain:

| Commit | What | Status |
|---|---|---|
| `f92f500` | run-reduced mixed class/decoder sums | landed |
| `ab3c3ae` | §5.1c cell-mediated rewrite reverted — measured negative | landed |
| `ddcd2e5` | **Step 12 BAKED**: ternary-only scope, `MODEL_REPR_VERSION` 4 | landed, all gates green |
| `5d42062` | spec: Step 15 `cell-latents` added (order …→ 3 → 15 → 6 →…) | landed |

Both 2026-08-11 owner verdicts are executed and recorded in ABLATIONS.md:
Step 12 ACCEPTED (baked at `ddcd2e5`), §5.1c cell-mediated rewrite reverted
(`ab3c3ae`; lean budgets are the sanctioned VRAM lever).

Verification at `ddcd2e5`: full pytest **397 passed, 3 skipped** and
`cargo xtask verify` all gates green; `hexo_py` rebuilt, Python sees
`MODEL_REPR_VERSION == 4`. Later commits are docs-only.

## Bake consequences worth knowing

- Binary-scope checkpoints refuse via `LEGACY_BAKED_KNOBS`
  (`mixed_windows: True`); the family registry is down to the three joint
  families and cleanly rejects binary/slot-era dicts.
- Default `MantisConfig` parameter count is now 3,866,597 (pinned in
  `test_model.py`; the +1,923,456 delta is exactly the ternary table growth).
- `incidence_plan` is three-argument; the histogram/counts path is gone and
  every class term is run-reduced (`class_row_sum`).
- `decoder.py` (binary histogram decoder), `test_decoder.py`, and
  `test_liveness.py` were deleted with the binary path.
- `lab/profile.py`'s replicated trunk was modernized to the six-stage block
  (relay + window-attention buckets, `axis_bias` signature, `window_pairs`
  kernel bucket in the fit profile). Its `profile_fit` still has the
  pre-existing KlentConfig/TrainConfig latent bug — fix with the next lab
  touch that needs it.

## Operational state

- **WSL clone `~/graft-bench` needs a reset**: it sits dirty at `2b5aec4`
  with a `_WA_NUM_WARPS` sed edit. `git checkout -- . && git fetch origin &&
  git checkout <new tip>`, then `uv sync` in `python/mantisnet` to rebuild
  `hexo_py` at repr 4. Then sanity-load an arm-B matrix checkpoint
  (`runs/lab/step12-matrix/armB/s0`) through `lab.families.load_checkpoint`
  — those checkpoints are the bake's reference shape.
- The dwm.exe VRAM leak regrows (~2.6 GiB within an hour); only an elevated
  `taskkill /f /im dwm.exe` clears it. Lean budgets are the standing
  mitigation for mixed-scope work on this card. WDDM paging signature:
  100% util at ~60 W on the 270 W 4070 Ti.
- pytest on Windows needs `--basetemp` outside the repo. `uv sync` without
  `--all-extras` drops the deck extras (fastapi) and breaks collection.

## Next

Owner-set order: 12 ✓ → **4** → 2 → 5 → 3 → 15 → 6 → 7 → 9 → 8 → 10 →
frontier (11, 13, 14). Step 4's builders landed on both sides
(`7610a7f`, `1cec048` — ternary-native 729 action classes); the model,
collation, and kernel side is not built. Do not start it without the
owner's go.
