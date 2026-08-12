# Graft campaign handoff — state as of 2026-08-11 night

Point-in-time handoff for the next agent. The normative plan is
`docs/MANTIS_GRAFT_SPEC.md`; measured records and verdicts live in
`docs/ABLATIONS.md` ("Graft campaign" section). Where this file disagrees
with either, they win.

## Where things stand

Branch `mantisnet-graft` (worktree `latest-run-recovery-stack-148e60`), all
commits local and unpushed. Recent chain:

| Commit | What | Status |
|---|---|---|
| `ddcd2e5` | **Step 12 BAKED**: ternary-only scope, `MODEL_REPR_VERSION` 4 | landed |
| `7d62232` | screen flake fix | landed |
| `adb5e8d` | Step 4 knob: ternary-native 729 action classes end to end | landed |
| `403354b` | spec: §2.2 fit bound owner-amended to 10% | landed |
| `3fec94b` | collation perf | landed |
| `22e7844` | **Step 4 BAKED**: bg-bucket decoder gone, `MODEL_REPR_VERSION` 5 | landed, all gates green |

Step 4 is ACCEPTED and baked (owner, 2026-08-11); the full record — screen
tables, the critic-bistability finding, the 4-epoch pair, §33 aliasing, and
the owner-accepted collect number — is the ABLATIONS Step 4 entry.
Verification at `22e7844`: full pytest **398 passed, 3 skipped** and
`cargo xtask verify` all gates green.

## Bake consequences worth knowing

- Default `MantisConfig` parameter count is 4,007,269, pinned in
  `test_action_rows.py` (the pin moved out of `test_model.py`).
- `LEGACY_BAKED_KNOBS` (`model.py`) now records `action_rows: True` alongside
  the seven earlier baked knobs; anything predating a baked stage refuses
  loudly at load.
  The reference state-dict shape is a knob-on Step 4 checkpoint
  (`runs/lab/step4-screen/armB/*` or `step4-epochs4/armB/s5` on the WSL
  clone). Step 12's `step12-matrix/armB` checkpoints **no longer load** —
  they predate action rows.
- The act fields (`act_class`/`act_rev`/`act_empty`) are mandatory in both
  builders and in collation; `e_bg`/`e_qbg` and the nearest-bucket path are
  gone in both languages.
- `klent/graft.py` (repr v1→v2 converter) and `test_klent_graft.py` are
  deleted; `_probe_prefixes` lives inlined in `trigraft.py`.
- `mnorm-late-v1` stays valid at repr 5: corpora store positions, and the
  act fields are collation products — the arm-B screen already trained on
  this corpus through the identical collation path. Sanity-check one load
  after the WSL sync anyway.

## The 1-epoch critic hazard

The lean screen recipe gives the critic only 95 optimizer steps, and cells
land arm-independently in an "optimist" basin (every v̂ positive) or a
discriminating one. Before reading any critic row from a 1-epoch cell,
classify its basin: optimist iff
|mean_prediction|/mean_abs_prediction > 0.97 on the moves-1–4 bucket of
`scores.json`. Step 12's matrix had 6/12 optimist cells; Step 4's screen
5/10. Four epochs escape the basin decisively.

Owner ruling (2026-08-11 night): **screens run 4-epoch cells from Step 2
onward.** The basin rule above remains only for reading historical 1-epoch
cells (Step 12 matrix, Step 4 screen).

## Operational state

- **WSL clone `~/graft-bench` sits clean at `3fec94b`** (knob era) with all
  Step 4 artifacts under `runs/lab/step4-screen` and `runs/lab/step4-epochs4`,
  bench artifacts in `~/step4-artifacts`. To sync: `git fetch` + checkout,
  then in `python/mantisnet` run `uv sync --all-extras
  --reinstall-package hexo-py` — the version number never changes, so a
  plain `uv sync` serves a STALE CACHED WHEEL of `hexo_py` and the batch
  fields silently lag the checkout (bit us at the Step 2 gate: the cached
  knob-era wheel still emitted `bg_cell`). Then sanity-load a
  `step4-screen/armB` checkpoint through `lab.families.load_checkpoint`.
- WDDM paging signature: 100% util at ~60 W on the 270 W 4070 Ti. Lean
  budgets are the standing mitigation.
- pytest on Windows needs `--basetemp` outside the repo. `uv sync` without
  `--all-extras` drops the deck extras (fastapi) and breaks collection.

## Next

Owner-set order: 12 ✓ → 4 ✓ → **2** → 5 → 3 → 15 → 6 → 7 → 9 → 8 → 10 →
frontier (11, 13, 14). Step 2 is `state-latents` (K = 4 invariant state
latents replacing the single global token); Step 5 (`tactical-scalars`)
depends on Step 4's rows and is now unblocked. **Step 2 has the owner's go
(2026-08-11 night) and is in flight** — knob first, full protocol, 4-epoch
screen seeds. The bake still needs its own owner approval.
