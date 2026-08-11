# Graft campaign handoff — state as of 2026-08-11 ~09:30 UTC

Point-in-time handoff for the next agent. The normative plan is
`docs/MANTIS_GRAFT_SPEC.md`; verdicts live in `docs/ABLATIONS.md` ("Graft
campaign" section). Where this file disagrees with the spec, the spec wins.

**A 12-cell screen matrix is RUNNING unattended in WSL right now** (see
"Matrix in flight" below). Do not kill it; it is idempotent and self-logs.

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
| `de8c9aa` | overnight rulings (factorial screen, horizon tables, keep-lean) | landed |
| `9168639` | **Step 12 implemented**: mixed-windows knob, both builders, model paths | landed, tests green |
| `0ac220a` | lab train --train-subset deterministic cap | landed |
| `5532487` | bench fit hands the supervised entry its own recipe type | landed |
| `da87926` | evaluate_cell exposes packed-inference budgets | landed |
| `7610a7f` | **Step 4 builder, Python oracle side** (action_rows knob) | landed, tests green |
| `1cec048` | **Step 4 builder, Rust mirror** (Codex-implemented, reviewed) | landed, tests green |

Full pytest (392+) and `cargo xtask verify` were green at `da87926`; the two
Step 4 commits after it passed their targeted suites + crate tests/clippy/fmt,
but **run `cargo xtask verify` + full pytest once at the tip before the next
landing** (they were deferred to keep the CPU quiet for the running matrix).

## Step 12 — implemented, matrix in flight, NO VERDICT YET

Implementation (all landed): both builders speak both window scopes behind a
parameter (`mixed_windows`); knob-off output is byte-identical to the
incumbent. Ternary laws asserted in both constructors (377 patterns; 726
decoder / 1458 incidence classes; 2187/2184 orbit laws). `Batch.mixed_windows`
+ `MantisConfig.mixed_windows`; the trunk refuses a scope mismatch;
`window_attention` is a live config field again. The mixed scope replaces the
class-histogram matmuls with per-edge class-row gathers (fp32 scatter-add) —
the histogram economy does not survive 726/1458-wide vocabularies. Spec
records two implementation findings: OWN/OPP/MIXED status is subsumed by the
ternary class; the `MODEL_REPR_VERSION` bump is deferred to bake (wire format
stays binary v3 while the knob exists).

### Owner-decision inputs already measured (speed table, steady windows)

| bench | sps | peak GiB | p50 ms | p95 ms |
|---|---|---|---|---|
| A pinned (default budgets) | **1648.8** | 10.03 | 287 | 617 |
| A matrix (4M/250k) | 3186.0 | 3.32 | 102 | 125 |
| B mixed (matrix) | 1068.1 | 9.17 | 308 | 476 |
| C mixed+waOff (matrix) | 390.3 | 2.42 | 63 | 85 |
| D waOff (matrix) | 2388.5 | 2.42 | 63 | 87 |

Readings for the packet:
- **Instrument re-pin**: the old 205.45 sps ruler was drvfs-I/O-bound — the
  WSL clone had no `runs/corpora`, old benches read the corpus over /mnt/d,
  and `load_corpus` mmaps the npz (`corpus.py:588`), so every prefetch
  page-faulted over 9P. Corpus copied to ext4
  (`~/graft-bench/python/mantisnet/runs/corpora/mnorm-late-v1`); the same
  pinned command now reads **1648.8 sps** — the new ruler, consistent with
  stack-939's 1,743 pos/s. Corpus must live on ext4, always.
- **The mixed idea costs ~3× on the fit path like-for-like** (B 1068 vs
  A 3186 at identical budgets), NOT the catastrophic node-bill projection.
- **C (waOff) slower than B is prefetch starvation, not attention cost**:
  wa-off drops GPU chunk time below the CPU mixed-collate cost; the fit goes
  CPU-bound (window seconds 44.8 vs chunk-sum ~3.2). The D-arm speed levers
  are collate cost + prefetch parallelism, not kernel work.
- Binary at matrix budgets beats the pinned defaults 1.9× (3186 vs 1649):
  the default 8M-pair chunks waste padding. Budget tuning is a free speed
  lever for every arm — packet material.

### Matrix in flight (WSL, detached, idempotent)

Recipe: 2×2 factorial (A baseline / B mixed / C mixed+waOff / D waOff) ×
seeds 0-2; epochs=1 over one deterministic 400k train subset (seed 0);
uniform budgets pair 4M / cell 250k, collect 12M/1.2M; device cuda,
compiled; val split scoring with horizon buckets via `step12_driver.py
evaluate` (runs `evaluate_cell`, `include_state_value=True`).

Done at handoff: all 5 benches; A s0-s2 trained+scored; B s0, B s1
trained+scored. In flight: B s2 training (started 09:22 UTC). Pending: B s2
eval, C×3, D×3 with evals. Runner: `~/step12_matrix.sh` (WSL), logs
`~/step12-artifacts/progress.log`, per-stage logs beside it. Cells:
`~/graft-bench/python/mantisnet/runs/lab/step12-matrix/arm{A,B,C,D}/s{0,1,2}`.

- The runner is **idempotent**: if it dies, relaunch
  `nohup bash ~/step12_matrix.sh > ~/step12-artifacts/nohup3.log 2>&1 &`
  from WSL — complete benches/cells are skipped, partial cells rebuilt.
- **B cells are WDDM-paging slow** (~1-2 h each): arm B training sits at the
  card's edge (torch ~9.3 GiB + ~2.5 GiB desktop ambient ≈ 12 GiB), so WSL
  pages. Math is unaffected; only wall-clock. C/D peaks are 2.4 GiB and run
  at honest pace (C ~25 min/cell CPU-bound, D ~8 min/cell).
- ETA at handoff: B s2 done ~11:00 UTC, C by ~12:30, D by ~13:00.

### Assembling the packet (when cells exist — partial is fine)

From WSL, `cd ~/graft-bench/python/mantisnet`:

```bash
python3 step12_packet.py runs/lab/step12-matrix ~/step12-artifacts
```

Prints: overall primaries (val top-1, critic sign) with per-seed paired
deltas vs A (mean Δ, 95% t-interval n=3, sign counts), guards (top3, v̂ MAE,
state sign/MAE), the **mandatory horizon tables** (per-bucket paired deltas
— a bucket collapse blocks promotion), the speed/VRAM table, params.
Early unpaired glimpses (do not over-read): val top-1 A s0 32.61 / A s1
~32.6 / B s0 32.49 / B s1 33.90; critic sign ~50-51% everywhere after one
400k epoch — if the critic primary stays near chance in all arms, report it
as under-powered at this budget, that is a finding for the owner.

**Present the packet to the owner and ask approve/deny per arm. Nothing
bakes without an explicit owner verdict. Hard 2% gate applies only to a
combination proposed for bake, measured against the re-pinned ruler.**

## Step 4 — builder landed on both sides (surplus-hours work)

Per the overnight ruling (code + tests only): the action-row tables exist
behind an `action_rows` builder parameter in BOTH builders, in both window
scopes — ternary own-slot post1 orbits (assert 729) and the binary graft
composite (2·93+3=189, MIXED rows carry −1/masked). Python oracle
(`7610a7f`): `_action_tables` + `tests/test_action_rows.py` (class laws,
successor-board oracle that actually plays each action, D6 multiset
invariance, ply-0, knob-off). Rust mirror (`1cec048`): Codex-implemented off
the engine's own `windows_through` walk (deliberately a different code path
from the Python line-reader), reviewed, gates re-run.

NOT yet done for Step 4: collation/Batch/wire exposure, the model row
encoder + kernels, `chunk_cost` packing, the §33 alias diagnostic, the
`action_rows` MantisConfig knob, MODEL_REPR_VERSION handling. The future
Rust/Python parity test must also pin the row ORDER across languages (both
sides currently pin (axis-major, slot) independently).

## Pinned measurement facts (updated)

- **Harness**: WSL ext4 clone `~/graft-bench`, venv `~/graft-venv`
  (`UV_PROJECT_ENVIRONMENT=$HOME/graft-venv`), corpus ON EXT4 at
  `runs/corpora/mnorm-late-v1` inside the clone, `lab bench fit --corpus
  mnorm-late-v1 --split val --device cuda --compile --seed 7
  --steady-warmup 20 --steady-measure 50`.
- **Ruler**: 1648.8 sps at default budgets (re-pinned 2026-08-11; the old
  205.45 was corpus-I/O-bound — see instrument re-pin above). Peak 10.03
  GiB; ceiling 10.25 GiB stands. Corpus `mnorm-late-v1`
  (SHA-256 `cd5f5d0a…`, 1M/100k/100k).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is **banned** on this
  harness: arm B training died twice at `pair_tables`' argsort with
  `CUDA driver error: device not ready`; clean without it. Short benches do
  not reproduce it — do not rediscover this the hard way.
- Windows-native benching remains disqualified; eager/fp32 remains
  disqualified.
- fused vs foreach Adam: null at production's layout (±0.05%).

## Operational gotchas (accumulated, all real)

- pytest needs `--basetemp=<scratch>` on this machine (broken ACLs in the
  default temp).
- WSL background launches: a `wsl.exe` wrapper that exits immediately after
  `... & echo` kills the child before it starts — keep the wrapper alive
  (`& sleep 4`) and verify with pgrep. Killed wrappers orphan Linux
  children; use inner `timeout`, and `pkill -f 'patter[n]'` (bracket trick)
  to avoid self-matching.
- Never benchmark collection with a fresh random model; use
  `runs/stack-939/checkpoint_000455.pt`.
- Steady windows need warmup ≥ 20 (recompile pollution below that).
- The deck server (uvicorn, WSL docker) is persistent and holds no
  meaningful GPU memory; the ~2.5-2.9 GiB ambient is Windows desktop apps.
- Codex sessions CAN run cargo (verified tonight) but not uv; the
  delegating session runs the Python suites.
- `lab train`/`lab evaluate` CLI cannot set collect budgets; the matrix
  uses `step12_driver.py` (in the WSL clone) for full recipe control.

## Environments

- Windows worktree: `D:\Hexo-Shrimp-Bot\.claude\worktrees\latest-run-recovery-stack-148e60`
  (uv venv in python/mantisnet; fine for tests, banned for measurement).
- WSL bench clone `~/graft-bench` at `da87926` — after landing new commits:
  `git fetch origin mantisnet-graft && git checkout FETCH_HEAD` from inside,
  then `uv sync --all-groups --reinstall-package hexo-py` if Rust changed.
  NOTE: the clone is 2 commits behind the worktree tip (`7610a7f`,
  `1cec048` are builder-only, no consumer — sync before anything that needs
  them).
- ACT donor code: branch `mantisnet-act` (`cc3edef`); old spec at `b735d27`.

## Next actions for the picking-up agent, in order

1. Check `~/step12-artifacts/progress.log`; if the runner died, relaunch it
   (idempotent). Wait for / collect the remaining cells.
2. Run the packet aggregator; write the packet (tables verbatim, the speed
   table above, the instrument re-pin note, VRAM/params deltas, the
   C-starvation and budget-tuning findings, a recommendation). Present to
   the owner; **ask approve/deny**. No bakes, no reverts without verdicts.
3. After any verdict: follow spec §2 knob lifecycle (bake = delete knob +
   LEGACY_BAKED_KNOBS + MODEL_REPR_VERSION bump + golden regen; revert =
   delete the path). A wa-off bake subsumes Step 3.
4. Run `cargo xtask verify` + full pytest at the tip (deferred, see above).
5. Step 4 continues per spec §4 (model side, collation, kernels) — the
   builder contract on both sides is already landed and tested.

## Memory

Auto-memory `mantisnet-graft-fork.md` mirrors campaign state including
tonight's findings; it is bootstrap, not the record — the spec and
ABLATIONS are the record.
