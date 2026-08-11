# Graft campaign handoff — state as of 2026-08-10 (evening)

Point-in-time handoff for whoever picks this up next. The normative plan is
`docs/MANTIS_GRAFT_SPEC.md`; verdicts live in `docs/ABLATIONS.md` ("Graft
campaign" section). This file summarizes state and hard-won operational
knowledge; where it disagrees with the spec, the spec wins.

## Where things stand

Branch `mantisnet-graft` (worktree `latest-run-recovery-stack-148e60`), all
commits local and unpushed:

| Commit | What | Status |
|---|---|---|
| `03f024e` | campaign spec replaces the ACT spec | landed |
| `5bab833` | Step 0: fused-Adam policy + klent-fit pinned prefetch | **baked** (owner 2026-08-10) |
| `d2bab78` | Step 1: dead softmax key biases removed | **baked** (owner 2026-08-10) |
| `b4d09cf` | §2.2 steady-window bench instrument | landed |
| `c848527` | verdict records, harness + corpus pins | landed |

Every commit passed full pytest and `cargo xtask verify` at the time it
landed.

## The campaign in one paragraph

One production model, modified in place, one step at a time. Each step:
implement behind one transient `MantisConfig` knob (kernel work in the same
change) → verify → speed gate (hard, ±2% vs pre-step baseline on the pinned
harness) → paired supervised screen (≥5 identical seeds both arms, campaign
corpus, predeclared primaries: val top-1 + critic sign accuracy, 95% CIs) →
**owner verdict** (nothing auto-decided; neutral results are presented, not
deleted) → bake (knob deleted, value into `LEGACY_BAKED_KNOBS`) or revert
(path fully deleted). One self-play head-to-head confirmation at campaign
end. All rules in spec §2.

## Execution order (owner-set, 2026-08-10)

```text
0 ✓ → 1 ✓ → 12 → 4 → 2 → 5 → 3 → 6 → 7 → 9 → 8 → 10 → frontier (11, 13, 14)
```

**Step 12 (all-nonempty / mixed windows) was elevated out of the frontier by
explicit owner priority** and runs next, BEFORE Step 4, so the counterfactual
action tables are built ternary-native (729 classes) once instead of binary
(189) then migrated. If Step 12 reverts, Step 4 proceeds binary per spec.
ACT-era ablation results are void as evidence (owner ruling) — only this
campaign's own paired measurements count.

## Overnight rulings (2026-08-10, pre-overnight session)

All committed in `de8c9aa`:

- **Screen design for Step 12**: the full 2×2 factorial — A baseline, B
  mixed windows, C mixed+wa-off, D baseline+wa-off — at **3 seeds**, with
  larger cells (~400k train samples per cell, one identical deterministic
  subset across every arm and seed, recorded in the recipe). 12 cells total.
- **Horizon tables are mandatory in every packet**: full per-distance-from-
  end-bucket paired tables for policy top-1 and critic sign accuracy / v̂
  MAE; a bucket collapse blocks promotion even when overall primaries win.
- **Failure policy overnight**: fix the issue and continue.
- **Verdict lean**: benchmark-neutral but architecturally fundamental
  candidates lean *keep* (owner still rules).
- **Night surplus**: after the Step 12 matrix, start Step 4 implementation
  (code + tests only, nothing baked overnight).

## Node-bill projection — MEASURED (594 val positions, mnorm-late-v1)

Growth factors moving window scope live-only → all-nonempty:

| stones | n | live W | mixed W | windows × | incidence × | decoder ×|
|---|---|---|---|---|---|---|
| 0–30 | 201 | 113 | 53 | 1.47 | 2.05 | 1.33 |
| 30–60 | 181 | 195 | 186 | 1.95 | 3.35 | 1.60 |
| 60–100 | 124 | 249 | 342 | 2.37 | 4.63 | 1.78 |
| 100–200 | 76 | 313 | 588 | 2.88 | 6.31 | 1.98 |
| 200+ | 12 | 420 | 1070 | 3.55 | 8.64 | 2.19 |
| **overall** | 594 | **198** | **243** | **2.22** | **4.31** | **1.69** |

Interpretation: window nodes ~2.2× on average (3.5×+ late), stone-incidence
edges ~4.3× (mixed windows hold more stones each — this is the expensive
edge family), decoder/relay edges ~1.7×. The window-attention pair set grows
super-linearly with window count, which is exactly why arm C (wa-off) is the
matrix's principal cost offset. Script: scratchpad `node_bill.py` (session
temp — rewrite from this table if needed, do not chase the file).

## Next step: Step 12 — `mixed-windows` (spec §4, Step 12)

- Builder keeps every deduplicated nonempty candidate window (own-only,
  opp-only, mixed), ternary patterns: assert 378 classes / 377 nonempty;
  joint (pattern, slot) incidence: assert 2187 / 2184. Window feature becomes
  ternary class + status; stone-incidence and decoder classes move off the
  binary 93-orbit tables; relay and window-pair tables widen.
- `MODEL_REPR_VERSION` bump (Rust builder + hexo-py + Python in one change),
  golden vectors regenerated (bump, never re-baseline).
- **Measure the node bill first**: mixed-window fraction grows with board
  density; project node/edge growth on the campaign corpus before building.
- **Owner-amended gate (2026-08-10): Step 12 runs as a matrix, not one
  arm** — A baseline, B mixed windows, C mixed + window-attention off (the
  main cost offset; pair-attention cost grows with window count), D… further
  speed-recovery arms from profiling. The hard 2% gate applies only to the
  combination proposed for bake; the owner judges the speed/strength trade
  from the whole matrix. Owner expects a slowdown and considers the idea
  worth it if the strength shows. A bake with window-attention off subsumes
  Step 3.
- Knobs: `mixed_windows: bool = False` plus resurrected
  `window_attention: bool = True` for the matrix.

Then Step 4 (action rows — the strongest strength candidate; donors
`act_encoder.rs` / `act_plans.rs` on branch `mantisnet-act` at `cc3edef`;
DEAD pre-status for spent candidate windows is specified), then latents,
tactical scalars, the window-attention removal trial, and the rest.

## Pinned measurement facts (don't re-derive these)

- **Harness**: WSL ext4 clone `~/graft-bench`, venv `~/graft-venv` (set
  `UV_PROJECT_ENVIRONMENT=$HOME/graft-venv`), `lab bench fit --corpus
  mnorm-late-v1 --split val --device cuda --compile --seed 7
  --steady-warmup 20 --steady-measure 50`, default budgets.
- **Baseline**: 205.45 samples/s median, rep spread ±0.2%, peak 10.06 GiB.
  VRAM ceiling 10.25 GiB. Campaign corpus `mnorm-late-v1`
  (SHA-256 `cd5f5d0a…`, 1M/100k/100k).
- The gate number is a **ruler, not production throughput**: `stack-939`
  reports ~1,743 pos/s because it pipelines collection with fitting over
  mostly-smaller positions; the production fit path measured ~229 sps
  windowed / ~900 sps between recompiles on today's late-heavy data. Never
  compare the gate number to run telemetry.
- Windows-native benching is **disqualified** (±40% same-arm swings under
  WDDM near the ~10.5 GiB cliff). Eager (no-compile) is ~2× slower and
  11.7 GiB — compiled mode is non-negotiable.
- fused vs foreach Adam: null at production's layout (±0.05%).

## Operational gotchas (each cost real time today)

- **pytest on this machine needs** `--basetemp=<scratch>` — the default
  `C:\Users\epicm\AppData\Local\Temp\pytest-of-epicm` has broken ACLs and
  ~136 tests error at fixture setup without it.
- **WSL processes orphan when a Windows-side `wsl.exe` wrapper is killed**
  (e.g. by a Bash timeout): the Linux child keeps running invisibly and
  holds the GPU. Always wrap long WSL commands in an inner
  `timeout <seconds>`, and check `ps aux` inside WSL when the GPU looks
  busy with no known task.
- **Never benchmark self-play collection with a fresh random model** — it
  plays to enormous positions and grinds; use a real checkpoint
  (`runs/stack-939/checkpoint_000455.pt` works; loads through the Step 1
  family stripper).
- Steady windows need enough warmup: with `--steady-warmup 5` the window is
  recompile-polluted; 20 is the pinned value. Recompiles inside the window
  show as ~15–20s p95 chunks and are honest cost, identical across arms at
  the same seed.
- The deck server (`uvicorn mantisnet.deck.app`) runs persistently in WSL
  under `/opt/venv` — it idles near 0% GPU but exists; don't mistake it for
  a stray bench.
- Codex agents on this machine could not launch `uv` (sandbox denies it):
  they can edit and self-check, but the delegating session must run the
  test suite itself before committing Codex work.

## Environments

- Windows worktree: `D:\Hexo-Shrimp-Bot\.claude\worktrees\latest-run-recovery-stack-148e60`
  (venv via `uv` in `python/mantisnet`, CUDA + Triton functional — fine for
  tests, banned for gate measurement).
- WSL bench clone: `~/graft-bench` — **sync it after every landed commit**
  (`git fetch origin mantisnet-graft && git checkout FETCH_HEAD` from inside;
  origin is the `/mnt/d` main repo, worktree commits are visible through the
  shared object store).
- ACT donor code: branch `mantisnet-act` (`cc3edef`); kernel-round history
  on `act-kernel-fusion`; the old spec text at `b735d27`.

## Resume point (exact, for the post-compaction session)

Done: rulings committed (`de8c9aa`), Step 12 task in progress, node-bill
projection measured (table above). GPU idle. Tree clean at `de8c9aa`+handoff.

Next actions, in order:
1. **Implement Step 12** in the worktree: Rust builder (`crates/models/
   mantisnet/src/encoder.rs` → `RawBatch` → `python/hexo-py` → `builder.py`)
   gains the nonempty window scope behind a builder parameter — ternary
   pattern classes (assert 378/377), joint ternary incidence/decoder classes
   (assert 2187/2184), window status field; Python builder mirrors it as the
   independent oracle; model builds binary (knob off, byte-identical
   incumbent) or ternary (knob on) tables from `mixed_windows` in
   `MantisConfig`; resurrect `window_attention: bool` from
   `LEGACY_BAKED_KNOBS` as a live field (default True) for the matrix.
   MODEL_REPR_VERSION bump. Tests first-class: class-count asserts, oracle
   parity, D6 via lab check, golden-vector regeneration (bump, never
   re-baseline). Read the engine-change skill before touching crates/.
2. **Cell recipe tooling**: `lab train` needs a recorded train-subset cap
   (~400k identical deterministic subset) — small `train_cell` extension.
3. **Run the 12-cell matrix** in the WSL harness (sync clone first),
   detached with inner timeouts (Bash cap is 600s — use nohup + artifact
   files + periodic checks). Speed-bench each arm with the steady window.
   Evaluate cells with `lab evaluate` (horizon buckets) and build the packet
   per spec §2.3/§2.4 including the mandatory horizon tables.
4. **Surplus hours**: Step 4 implementation (ternary-native if the matrix
   looks good; donors on `mantisnet-act` at `cc3edef`).
5. Morning: packet to owner; no bakes without verdicts.

## Memory

Auto-memory `mantisnet-graft-fork.md` mirrors campaign state; update it when
steps land or rulings change (it does not replace the spec/ABLATIONS as the
record — it's session bootstrap).
