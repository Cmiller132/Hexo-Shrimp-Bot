# Handoff — 2026-08-25 evening

State of the improve-and-de-bloat campaign as of ~19:30Z. The repo is the
record: measured results live in `docs/ABLATIONS.md`; this file is the
resume map only.

## Rulings in force

- **confirm-1 is cancelled**; `scripts/launch_confirm.sh` and `run/confirm-1`
  are provenance only.
- **Site attention is an open option** on `claude/site-attention` @ `0eb72e9`;
  any retry is a stability change plus a fresh screen, owner-initiated.
- **GPU stays free until the owner launches** (ruling 2026-08-25 evening):
  the v3 screen below is staged, not started.

## Today, in order

- **wattn retest killed by owner ruling** (~12:05Z, mid-s0): epoch-1
  throughput ratio 0.545, s0 epoch-3 val top-1 44.13 vs the fixture's 48.42
  final, WDDM paging crawl on the giants. Record in `docs/ABLATIONS.md`.
- **window_attention deleted** — main `2310aaf`: the knob, `window_pairs.py`
  and its four kernels, `claim_reach`, and the 48-class pair vocabulary.
  wa-era recorded configs refuse with "no longer implements"; head-count
  inference needs a cell-stage bias table or the recorded-config heads
  hint. Both fast lanes + `xtask verify` green; MODEL_SPEC/README updated.
  Wave-1 + Wave-2 pt1 merged to main earlier as `ec12f6b`.
- **Architecture pivot (owner directive)**: stop pricing add-on knobs;
  restructure the block stages. The owner asked specifically whether
  merging stones and cells is the right avenue — answer argued yes, and the
  build was authorized: **merged site-token trunk, both mixing arms,
  uncovered cells must stay reasoning-capable.**

## The merged-site trunk (Step M1) — built, tested, staged

Branch **`claude/merged-sites`** @ `40a0f81` (from main `2310aaf`).
Spec section `docs/MODEL_SPEC.md` §5M on the branch.

- Stones and legal cells are one **site** token set. Block: window ← its
  exactly-six sites (typed sum over the 2184 joint (pattern, slot) classes,
  S4 hoist extended to all slots) → window/latent cycle → site ← its ≤18
  windows (typed attention, class value rows — stones now *select* their
  windows instead of summing) + cells ← stones radius read → mixing arm →
  FFN over sites+latents. Every legal cell, covered or uncovered, carries
  state through every block and the decoder reads trunk-refined rows for
  the whole legal set (uncovered cells: radius read every block, FFN every
  block, and global state via the mixing arm — the owner's planning-ahead
  requirement; pinned by two tests).
- **Two arms, one knob**: `merged_sites=true` (full `[latents; sites]`
  self-attention) and `merged_sites=true site_self_attention=false` (linear
  latent cross-reads; no token↔token attention at all). Params 5,048,389 /
  5,315,141 vs production 5,195,909 — same band, fair screen.
- **No wire change**: MODEL_REPR_VERSION stays 9. The merged incidence and
  the site grid (`max_ts`/`site_slot`/`site_valid`) derive at collation
  (shared helper on both the Python-collate and Rust `build_batch` paths;
  parity asserted) and on-device via the existing opaque table ops. All
  kernels reused: `cell_read` (2184 classes), `aggregate_to_windows`
  (six-edge windows), `window_latents` read/broadcast (site layouts),
  `fused_attention` (site grid).
- **Tests**: `tests/test_merged_sites.py` (14: pins, config validation,
  six-edge exactness, hoist parity vs literal edges, forward/backward,
  D6 invariance both arms, uncovered-cell liveness ×2, family-registry
  roundtrips, split/merged non-cross-loading) +
  `tests/test_merged_sites_cuda.py` (CUDA vs CPU parity, kernel-path
  bit-determinism — note: embedding-gather grads are atomics-nondeterministic
  in the incumbent too; the bar is kernel-level and the test says so).
  Full CPU suite 334 green, CUDA lane 150 green, `xtask verify` green.

## Screen v3 — LAUNCHED 2026-08-26 03:20Z (owner: "you have 8 hours... make the best of it")

Queue reordered arm-first before launch (the staged fixture-first order
would have spent the night on seed noise). Running order: full s0, latent
s0, fixture s0, then s1/s2 triples, then fixture s3-s5; verdict runs
automatically when all twelve cells are scored (writes
`runs/lab/v3/verdict.json`). Interim scoreboard (val-EMA, scored):

| cell | critic_ce | policy_nll | top1 | steady sps |
| --- | --- | --- | --- | --- |
| fixture s0 | 0.6242 | 1.9581 | 46.6% | 1125 |
| msite-full s0 | 0.6257 | 2.0126 | 45.3% | ~800 |
| msite-full s1 | **0.6210** | **1.9209** | **46.9%** | ~880 |
| msite-latent s0 | 0.6256 | 2.0028 | 45.4% | 931 |

full s1 beats fixture s0 on all three metrics; the arms straddle the
fixture, so the verdict is genuinely open. No pathologies: near-terminal
critic decisive (~0.04 at 1-4 plies), no paging (VRAM ~11.9/12 GiB but
power stays ~165 W), no crashes; every cell exit 0 so far. latent s1 in
flight (slow-starting seed, monotone). Cells ~1.3-1.7 h each; full drain
~17:30Z. Monitors: persistent epoch/crash watcher in this session;
keepalive holds the WSL VM.

The original launch recipe (for relaunch-after-crash — skip-if-scored
makes it idempotent):

- `/root/run_v3_cell.sh` — per-cell train+evaluate under RECIPE, root
  `runs/lab/v3`, **no default kw** (every cell spells its config).
- `/root/queue_msite.sh` — phase 1: fixture recut at main `2310aaf`,
  6 seeds, kw `cell_latents=true cell_nodes=true cell_node_scope=all
  action_tactical=true` (the old fixtures died with the function class;
  post-deletion kw drop `window_attention`/`action_latents`). Phase 2:
  checkout `claude/merged-sites`, arms `msite-full` / `msite-latent`
  interleaved, 3 seeds each. Skip-if-scored; corpus `cn1-late-v1`;
  `/root/graft-v2` tree + venv (editable, no rebuild needed — origin is
  the mounted Windows repo, both SHAs fetched and checkout-verified).
- **Launch recipe** (when the owner says go):
  1. Keepalive first (WSL VM idle-death gotcha):
     `Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-e','bash','-lc','sleep infinity'`
  2. `wsl -e bash -lc 'setsid nohup bash /root/queue_msite.sh >/dev/null 2>&1 &'`
  3. Watch `/root/queue_msite.log` + per-cell logs
     `runs/lab/v3/<arm>/cell-s*.log`; on msite-full s0 watch nvidia-smi for
     the WDDM paging signature (~100% util at ~60W, 11.9/12 GiB).
  4. Verdict after drain:
     `python -m mantisnet.lab.screen verdict --fixture runs/lab/v3/fixture
     --arm msite-full=runs/lab/v3/msite-full --arm
     msite-latent=runs/lab/v3/msite-latent --ema`
- Cost expectation: ~12 cells. The full arm's site set is stones+all legal
  cells (~300-750 rows/position mid-game) — quadratic mixing; if it pages
  or OOMs on the late-corpus giants, that is arm evidence (the wattn
  precedent), not a bug to paper over. The latent arm is linear and should
  run at or above fixture throughput.

## Open (owner)

1. Launch call on the v3 screen (above).
2. Wave-2 remainder: `cell_adjacency` deletion = REPR 9→10 bump dropping
   `coords` + `adjacency_{src,dst,axis}` wire fields. Scoped, not started.
   Note: if the merged trunk keeps, this fold-in changes (merged never
   reads adjacency either — the bump would drop it for both).
3. trigraft deletion candidate — evidence assembled, unchanged.
4. `cell_structure` keep/delete and `uncovered` scope delete — note both
   become moot if the merged trunk keeps (no split cell stage to knob).
5. Wave-3: stronger eval anchor FIRST; lab-loader canonicalization fix is
   the known blocker. The TSS-distillation idea (solver-labeled auxiliary
   targets + eval anchor) was pitched and parked — owner said not needed
   yet.

## Environment

- GPU running the v3 screen queue since 03:20Z (launched under the 8-hour
  autonomy grant).
- Deck stack stopped; strix bridge down; nothing tonight needs them.
- WSL keepalive RUNNING (hidden `wsl.exe sleep infinity`); kill it after
  the queue drains — memory `wsl-vm-idle-death`.
- This worktree (`claude/overnight-handoff-autonomous-967f3b`) sits at the
  merged-sites tip; main is `2310aaf`.
