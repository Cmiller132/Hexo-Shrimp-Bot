# Handoff — 2026-08-25, overnight autonomous session

State of the improve-and-de-bloat campaign as of ~06:30Z. The repo is the
record: measured results live in `docs/ABLATIONS.md`; this file is the resume
map only. If the wattn screen is still running when you read this, the last
section says how to watch it.

## Rulings in force (unchanged)

- **confirm-1 is cancelled**; `scripts/launch_confirm.sh` and `run/confirm-1`
  are provenance only.
- **Site attention is an open option** on `claude/site-attention` @ `0eb72e9`;
  any retry is a stability change plus a fresh screen, owner-initiated.

## What the night executed (owner list items 1, 2-partial)

All on branch `claude/overnight-handoff-autonomous-967f3b` (pushed; **not
merged to main yet** — see Merge state).

- **Wave-1 provable cuts** — `11217f5`:
  - §4.1 stone-attention bias tables deleted (the Step-11 knock-out is the
    evidence). `fused_attention` is plain varlen attention; its backward has
    no atomics left and is bit-deterministic (pinned by test). −1,184
    parameters in every config: default 4,803,397 / production 5,195,909 /
    armB 5,213,957.
  - S9: no head reads post-trunk stone rows, so the last block computes only
    its four latent-row reads (exact — the shared TOKEN bias row cancelled
    per softmax row) and skips the stone FFN half; `trunk` returns
    `(windows, global, cells)`.
  - S4: §5.1 window class sums are one per-pattern-table gather
    (`TERN_PATTERN_CLASS_COUNTS`, reversal-orbit invariance checked at
    import); §5.2 and head class sums run-reduce once over stacked tables.
  - Quiet-machine benches (cn1-late-v1, 2M/125k budgets, compile, seed 7):
    val 947.4 → 989.3 samples/s (+4.4%), peak 3.647 → 3.781 GiB; train-split
    steady window (the sorted ply-~500 giants) 106.5 → 104.7 (−1.7%).
  - Verified: `xtask verify` all gates; CPU suite + CUDA lane green on both
    Windows and WSL.
- **Wave-2 dead-knob purge, part 1** — `69129ed`: `action_latents` and
  `line_pass` deleted with all machinery. Recorded `=False` strips as a
  legacy baked knob; `=True` refuses.
- **ABLATIONS rows** — `2d371ed`.

**Merge state:** held only for a CUDA-lane rerun at the `69129ed` tip (the
GPU has been occupied by the screen; the D+E diff is CUDA-inert by
inspection, but the lane runs before the merge per house discipline). If the
session is still live when the queue drains it will run the lane and, if
green, merge to main and update this file. Otherwise:
`pytest tests/ -m cuda_lane` from ext4, then merge `2d371ed` to main.

## wattn retest arm (owner item 2, screen running)

3 seeds, `window_attention=true` against the existing R1 fixture, tree
`98a7c88` (the fixture's own lineage — deliberately NOT the Wave-1 tree).
Cells `runs/lab/v2/wattn/s{0,1,2}`; queue log `/root/queue_wattn.log`;
per-cell logs `runs/lab/v2/wattn/cell-s*.log`.

- Launched 05:07:56Z after a first attempt died with the WSL VM (see
  Environment). Epoch-1 already prices the knob: **468.8 samples/s vs the
  fixture's 860.3 — train-throughput ratio 0.545.** Even a quality-neutral
  result fails "neutral+slower → drop"; the knob needs S ≥ 2 to live.
- ~2.5 h/cell → queue drains ≈ 13:30Z. Verdict:
  `cd /root/graft-v2/python/mantisnet && /root/graft-v2-venv/bin/python -m
  mantisnet.lab.screen verdict --fixture runs/lab/v2/fixture --arm
  wattn=runs/lab/v2/wattn`
- If the arm fails the bar, the Step-3-deferred `window_attention` deletion
  (knob + `wa_*` + the §5.1c pair machinery it alone uses) becomes
  executable — sized like tonight's purges, owner call on timing.

## Open (owner)

1. Wave-2 remainder: `cell_adjacency` deletion = `MODEL_REPR_VERSION` 9→10
   bump that also drops the now-unread `coords` and the
   `adjacency_{src,dst,axis}` wire fields. Touches
   `crates/models/mantisnet/src/encoder.rs` (struct/counts/ser/de/validate),
   hexo-py collate, `builder.py`/`collate`, `cell_nodes.py`, families,
   `lab/check.py` contract lists, `test_rust_builder`. Scoped, not started —
   it needs a clean machine window, which the screen consumed.
2. **The R1 fixture is stale once Wave-1 merges**: the function class
   changed, so any future screen arm on the new tree needs a fresh fixture
   (6 seeds). Also: screens on the new tree must drop `action_latents=false`
   from their model-kw lists (knob gone).
3. trigraft deletion candidate — evidence assembled: CLI-only module, zero
   production callers, converts a closed era; scalar checkpoints stay
   scoreable via the lab families (the scalar-joint test builds its
   transform inline). Its tests are the largest remaining suite cost
   (3×49 s). Deletion = `mantisnet/klent/trigraft.py` +
   `tests/test_klent_trigraft.py`.
4. `cell_structure` keep/delete (R1 S +1.44, policy-only) and `uncovered`
   scope delete (S −5.62, sps 1.047) — unchanged.
5. Wave-3: stronger eval anchor FIRST. Known blocker: the lab loader
   refuses production checkpoints (model_config vs infer_config
   canonicalization) — fix before Round-3 anchor evals.

## Environment

- **WSL keepalive**: the first wattn launch died because WSL2 terminates the
  VM minutes after the last client exits (the stopped deck stack had been
  the implicit keepalive all along). A hidden `wsl.exe … sleep infinity`
  process now holds it open — kill it once the queue drains. Full gotcha in
  memory `wsl-vm-idle-death`, including: `CUDA_VISIBLE_DEVICES=-1` (not
  `""`) hides the GPU on Windows torch, and never run suites/builds beside
  a bench or cell (it polluted three bench arms before quiet reruns).
- `/root/graft-v2` is checked out at `98a7c88` for the screen; the Wave-1/2
  branch is fetched there (`11217f5`/`69129ed` reachable).
- Deck stack still stopped; strix bridge still down (nothing tonight needed
  them). GPU is the screen's until the queue drains.
- This worktree has a full Windows venv (`python/mantisnet/.venv`, hexo_py
  repr 9, triton-windows).
