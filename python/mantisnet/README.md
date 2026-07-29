# mantisnet

The MantisNet network of `docs/MODEL_SPEC.md`, real — builder, trunk, heads,
losses — plus the KLENT training path of `docs/KLENT_DESIGN.md`: the
closed-form policy improvement that replaces tree search, implemented
faithful-first against that document.

**Status: implemented, green, and in production training.** The model
is 1.25 M parameters at the §2 defaults — the spec's 1.2 M plus the
appendix-B Q head — with the spec's §12 obligations as tests; KLENT carries
the design doc's §4.7 obligations as its own. Batch building is Rust and
rayon-parallel (~0.1 ms/position), the forward is `torch.compile`d (~2.1×
over eager), and the pipelined loop holds ~4.9 k samples/s steady on the
4070 Ti at the default operating point (4096 games an iteration, 1024
self-play slots, cap 512) — the Performance section below has the numbers
and the hazards worth knowing.
**Now consumed by a `ModelPackage`:** `crates/models/mantisnet` supplies the
container-side encoder, KLENT-improved evaluator opinion, policy/Gumbel
sessions, diagnostics, and proving checkpoint loads. `hexo-bot` calls this same
live Torch module through its leaf PyO3 boundary. The production KLENT trainer
remains here and its buffer remains in-memory per iteration, as the paper's is;
the package explicitly declines container-side `fit` rather than duplicating
that loop.

## Shape

```
python/mantisnet/
  pyproject.toml      # uv project; torch from the cu128 index, hexo-py by path
  mantisnet/
    __init__.py       # flat re-exports; shared Rust MODEL_REPR_VERSION
    builder.py        # §3-§4, §9: independent reference graphs and collation
    model.py          # §5-§7, §10 + appendix B: trunk, policy/Q/value heads
    attention.py      # fused coordinate-biased attention + SDPA fallback/backward
    decoder.py        # the cell-head incidence pass: segment kernel + scatter fallback
    losses.py         # §6, §7, §10: targets, cross-entropies, decay grouping
    segments.py       # ragged per-position reductions, shared by losses and klent
    klent/
      improve.py      # eq. 3 closed form at gain s: π′, v̂, and the §13 diagnostics
      returns.py      # the sign on mover change, the λ-return
      selfplay.py     # batched collection, acting-time v̂, buffer rules, stats
      train.py        # KlentConfig, the fit epoch, the iteration
      evaluate.py     # the zero-search argmax chooser and simple cross-play
      search.py       # evaluation-only Gumbel root sampling + line halving
      opponents.py    # the evaluation-opponent seam and SealBot adapter
      run.py          # the run driver: config.json, metrics.jsonl, checkpoints
      crossplay.py    # the A7 checkpoint round-robin, the forgetting detector
      sealbot.py      # SealBot CLI and telemetry recording
      telemetry.py    # the run's SQLite capture + the queries over it, + CLI
      hardware.py     # the GPU/process/host counter trace behind an iteration
      inspect.py      # the policy debugger: one checkpoint's view of one position
      graft.py        # one-time converter: a scalar-critic checkpoint to the factored head
    deck/
      __main__.py     # the uvicorn entry point the container service runs
      app.py          # FastAPI routes, SSE, static SPA, play/match orchestration
      service.py      # run registry, child lifecycle, read-only queries, checkpoint LRU
      state.py        # schema-versioned deck.db annotations, probes, presets, match jobs
  tests/              # the two specs' obligations, one file per concern
  bench/
    bench_forward.py  # builder and forward throughput at spec defaults
    bench_loop.py     # the loop's own stages: sweep, a real collect, a real fit
```

## Deck

`mantisnet.deck` is the LAN-trusted control surface described by
`docs/DECK_SPEC.md`. It runs in the training container, owns port 8000, serves
the built React SPA, and launches training as child processes. Start it through
the Compose `deck` service; there is deliberately no Windows-native serving
path and no authentication layer.

The run registry treats a directory below `runs/` as a run exactly when it has
`config.json`. It consumes the driver's public artifacts rather than reaching
into the loop:

| Artifact | Deck contract |
| --- | --- |
| `config.json`, `invocations.jsonl` | resolved knobs, versions, and resume history |
| `status.json` | at-most-once-per-second collect/fit/eval heartbeat |
| `telemetry.db` | query-only SQLite connection for every dashboard read |
| `metrics.jsonl` | durable human-readable iteration record; not re-parsed by the API |
| `checkpoint_*.pt` | eager inference, at most two checkpoints resident |
| `STOP`, `CHECKPOINT` | graceful lifecycle requests written by the deck |
| `deck-console.log` | captured child stdout/stderr and the SSE log source |

Run annotations, saved probes, play presets, and match-job history live in
`runs/deck.db`. That database is WAL, schema-versioned, and refused on a version
mismatch. It never shares tables or transactions with telemetry. The only
telemetry write initiated by the deck is a completed SealBot match, through
`klent.sealbot.record_match`; all query endpoints open `telemetry.db` in SQLite
`mode=ro` with `query_only` enabled.

The service module map:

| Module | Role |
| --- | --- |
| `deck.app` | REST models/routes, structured errors, run and match SSE, reference-attention and D6 lab endpoints, SPA fallback |
| `deck.service` | registry state machine, child process boundary, sentinels, checkpoint inference LRU, authoritative play sessions |
| `deck.state` | the deck-owned SQLite schema and its small persistence API |

The API uses `DECK_RUNS_ROOT`, `DECK_FRONTEND_DIST`, `DECK_DEVICE`, and
`SEALBOT_ROOT` in the container. Analysis is one position at a time; match
requests are capped at 64 games and only one job runs at once.

Run everything from this directory:

```
uv sync                              # venv, hexo-py wheel via maturin, torch cu128
uv run pytest                        # the whole suite
uv run python bench/bench_forward.py # throughput on CPU and the local GPU
```

Inside the training container the locked environment is already `/opt/venv`
on PATH and the repository is mounted at `/workspace`, so the same commands
drop their `uv run` — `python -m pytest -q`. See `docker/README.md`.

## Module map

| Module | Role |
| --- | --- |
| `builder` | `build` (raw §11 inputs to a `PositionGraph`), `from_position` (the `hexo_py` wrapper), `collate` (graphs to one `Batch` of index tensors). It is the independent parity reference and imports `MODEL_REPR_VERSION` from the shared Rust owner. |
| `model` | `MantisConfig` (the §2 named parameters), `MantisNet`, `ModelOutput`. `trunk`, `cell_heads`, and `value_head` are separate so a caller pays only for the heads it reads; `policy_head` is the one-head entry the argmax chooser wants. `cell_heads` composes the factored critic's two logits into Q; fitting takes the raw pair instead. |
| `decoder` | `aggregate` (the pass over the decoder incidence both cell heads read) and `head_matrix` (a head's projection and embedding tables folded into the matrix reading an aggregate row). Holds no parameters. |
| `losses` | `value_target` (two-hot projection), `value_loss`, `policy_loss` (segmented CE over ragged engine-order logits), `param_groups` (§10 decay split). |
| `segments` | The ragged per-position reductions everything above and below shares. |
| `klent` | The KLENT baseline: operator, returns, collection, fitting, evaluation, and the run's telemetry capture. See below. |

## Design notes

- **One production encoder, one independent reference, and a parity detector
  between them.** `crates/models/mantisnet/src/encoder.rs` is the shared Rust
  implementation and the sole owner of `MODEL_REPR_VERSION`; the container
  package calls it directly and `hexo-py` is a thin binding over it. The Python
  builder (`build`/`collate`) remains the normative reference: it never
  calls the engine's window walk, which is what keeps `windows_through` an
  *independent* oracle for §12.1 — a builder built on the engine's enumeration
  would agree with it by construction, the deleted-detector failure `CLAUDE.md`
  warns about. The production path
  (`collate_positions`/`collate_prefixes`) is rayon-parallel Rust with the GIL
  released, ~16× the Python path at batch 256, and *allowed* to use the
  engine's walk precisely because `test_rust_builder.py` holds it exactly equal
  to the Python output, field for field — the §12.7-style detector an
  independent reference owes.

- **34 canonical patterns, not the spec draft's 32.** The 62 nonempty, nonfull
  6-bit masks fold to `(62 + 6 palindromes) / 2 = 34` orbits under reversal.
  Raised and corrected in `MODEL_SPEC.md` §3.2; a test pins the count so the
  table cannot silently disagree with the doc again.

- **Two index conventions the spec left to the implementation**, fixed in
  `builder.py`'s docstring: attention buckets are `d-1` clamped, then `SELF`,
  then `TOKEN`, with `TOKEN` winning the token-token pair; the one stoneless
  position (ply 0) takes the background clamp bucket 7.

- **Batching is concatenation plus two padded layouts.** Message passing runs
  on concatenated entities with `index_add_`/`index_select` and both cell
  heads on one segment reduction over the same concatenation — no padding, no
  waste. Attention and the value readout run padded per
  position with the token at slot 0, masked block-diagonal by construction.
  Distance buckets are computed in-forward from padded coordinates —
  elementwise arithmetic, not index discovery — because shipping the
  `(P, T, T)` bucket tensor over PCIe would cost more than recomputing it.

- **`MLP([a; b])` is implemented as two linears** (`_PairMlp`): a linear over a
  concatenation is the sum of two linears, so the 2H-wide inputs of `MLP_W`,
  `MLP_S`, and `MLP_P` are never materialised. In the policy head the token
  half runs per *position* and is gathered to cells afterwards. Identical
  arithmetic and parameter count, ~10% off the batch forward; the batching-
  equivalence and D6 tests are what license calling it identical.

- **The forward allocates no default-dtype buffers.** Scatter targets derive
  their dtype from what is scattered into them, which is what lets the same
  code run fp32 and under bf16 autocast (§10); the value decode is fp32
  unconditionally so every consumer sees the same scalar.

- **`losses.policy_loss` refuses a target that does not sum to 1** per
  position, and `value_target` refuses outcomes outside `[-1, 1]` — both are
  silent-corruption inputs a training loop would otherwise absorb without a
  symptom.

## KLENT

`mantisnet/klent/` implements `docs/KLENT_DESIGN.md`'s baseline — the paper's
algorithm at the design doc's forced deviations and nothing else, checked
against the authors' reference implementation
([KazukiOhta/klent](https://github.com/KazukiOhta/klent)). The accepted items
of `KLENT_PROPOSALS.md` (the λ_intra split, the Bernoulli critic, the dual
controller) are deliberately not in it: the design doc lists them as diffs to
be decided, and a faithful baseline has to exist before a deviation from it
can be measured. Self-play runs from the empty board, period: no seeded
line-builder prefixes, no warm-start heuristic phase, no seed-cut anneal, and
no external-bot grounding inside collection (owner decision, 2026-07-28).
If a cold start starves (an untrained policy essentially never finishes a
game — the design doc's §5 premise), the sanctioned bootstrap is a prefit on
a foreign corpus *before* KLENT begins, not machinery inside the loop.

- **The model KLENT trains is trunk + policy head + Q head.** The Q head is
  the §6 decoder shape with its own parameters (spec appendix B). Per legal
  cell it emits mover-win and magnitude logits, then acting composes
  `Q = (2·sigmoid(p_logit) − 1)·sigmoid(m_logit)` in fp32. Fitting uses
  taken-action BCE-with-logits losses `p_loss` against the λ-return's sign
  and `m_loss` against its absolute value; these replace the scalar
  `q_loss` metric. This separates winner classification from discounted
  distance under `gamma < 1`; at `gamma=1, lam_ret=1`, magnitude learns one
  and the critic reduces to `2p−1`. The §7 value head is outside the loss,
  per the paper's no-V-head ablation, and `v̂ = E_{π′}[Q]` supplies the
  bootstrap. The forward is split into `trunk` plus per-head methods so the
  loop never computes the readout it never reads.
- **The operator carries a critic gain, `q_scale`.**
  `π′ ∝ softmax((s·Q + τ·log π)/(τ + λ))` with `s = q_scale`; `s = 1` is
  eq. 3 verbatim. The gain exists because the operator's sharpening has to
  outweigh the flattening prior exponent `τ/(τ+λ) < 1`, and whether it does
  depends on the spread of Q across a position's moves against `τ+λ`. The
  scalar tanh critic's overconfident magnitudes cleared that bar implicitly;
  the factored critic's calibrated magnitudes in contested positions do not.
  It applies *only* inside the softmax — v̂ averages the unscaled Q, so
  returns and the magnitude target stay in (−1, 1). `improved_policy` takes
  it with no default, for the same reason it takes τ and λ with none: π′ is
  a function of it, and a reader assuming 1 would misreport a run acted at
  another gain. `KlentConfig.q_scale` (default 1.0), `--q-scale`, and the
  acting paths that thread it: `Collector`, `gumbel_choose`,
  `inspect_position`, the SealBot CLI, and the deck service.
- **A scalar-era checkpoint is converted once, explicitly.**
  `python -m mantisnet.klent.graft OLD.pt NEW.pt` reshapes the critic
  readout to two rows, zeroing the new tensors and their Adam moments and
  resetting that parameter's step, which preserves the trained trunk and
  MantisNet's zero-init contract; everything else must match the current
  architecture exactly. The training loaders stay strict — a pre-factoring
  checkpoint is refused rather than adapted. Only the deck's inspection path
  reads the head's width off the state dict, which is what keeps a
  pre-factoring run browsable on the same axes.
- **The sign follows mover change, read off `moves_remaining`** — K1, the
  design doc's most likely catastrophic bug. The detector it prescribes is a
  test here: the phase-derived sign against the engine's own reported movers
  over ~1800 random plies, plus first-stone and second-stone win fixtures
  (K2) through the return recursion.
- **The Count Up Game is the algorithmic anchor.** A two-placements-per-turn
  synthetic solved exactly by backward induction; the KLENT iteration through
  the real `improved_policy` must land on the quantal-response fixed point,
  and episodes scored by the real sign/λ-return machinery must average back
  to `Q*`. K1, K3, and K5 all move the fixed point and fail it.
- **The buffer rules are the design doc's, with no cases added:** capped
  episodes dropped whole (K4 — the reference implementation's NaN-masked
  unterminated tail, as a whole-episode drop), terminal positions never
  samples, `v̂` captured at acting time (K6), and fitting refuses a sample
  whose stored π′ no longer matches its position's legal count. States are
  stored as move prefixes and rebuilt by replay (§12).
- **Collection goes through one seam**: `evaluate(batch) -> (policy_logits,
  q_values)` on CPU. Training wraps the network; the pipeline tests wrap a
  scripted line-extender (`tests/heuristic.py`), which is how the buffer
  rules are testable without a trained model.
- **The default coefficients are paper-verified, not carried.** `τ = 0.1`
  (reverse KL) and `λ = 0.03` (entropy) per the paper's eq. 2 — the design
  doc's original pair was transposed and is corrected — `λ_ret = 0.939`
  (`e^{-1/16}`, the paper's 8-turn horizon at Hexo's two placements per
  turn), `γ = 1.0` (the reference objective's undiscounted return), and
  `q_scale = 1.0` (eq. 3 verbatim). The *reference recipe* the runs are on
  is not the default set: `--gamma 0.99 --lam 0.01 --lam-ret 0.939`, τ = 0.1,
  with `--q-scale 2.0` under ablation for the factored critic. `γ < 1` is
  what ranks faster wins above slower ones — the conversion pressure an
  infinite no-draw board lacks — and `λ = 0.01` is what stops the prior
  exponent flattening decided positions. `docs/KLENT_RUN_PLAN.md` §2/§3
  record the verification and the measured history.
- **Evaluation search is Gumbel sequential halving over deterministic
  lines.** The root samples at most 16 candidates by policy logit plus
  Gumbel noise, then spends the simulation budget deepening surviving lines
  by interior `π′` argmax. The in-loop default is 32 simulations; zero is
  exact policy argmax for historical offline comparisons. This is
  evaluation-only: collection and fitting import no search code, and the
  KLENT operator remains the only training-time improvement.
- **A run is its directory.** `python -m mantisnet.klent.run --out runs/<name>
  --iterations N` writes `config.json` (knobs + versions), `metrics.jsonl`
  (strict JSON, one row per iteration: the §13 metrics including the
  v̂-vs-outcome calibration that watches the §9 bias), `invocations.jsonl`
  (every process that touched the run, with its resolved knobs),
  `telemetry.db` (the Telemetry section below: every game, every self-play
  ply, every evaluation match, and the machine's counters), and
  resumable checkpoints; `--resume` continues after a crash and refuses a
  checkpoint from other versions; `--init-from` forks a fresh run (own
  seed, iteration 0) from a trained checkpoint. `--eval-every N` plays the
  SealBot match in-driver (`--sealbot <root>`, uncapped `--eval-time 0.1`
  by default, optional `--eval-depth`, and `--eval-sims 32`) and merges the
  score into that iteration's metrics row, with an eval RNG derived
  from (run seed, iteration) so the training trajectory is identical with
  evaluation on or off. `--starve-limit` ends a collapsed run with a
  checkpoint instead of a burned night, and `mantisnet.klent.crossplay`
  plays the A7 checkpoint round-robin. `docs/KLENT_RUN_PLAN.md` is the
  operational plan, §3 of it the measured history, around this driver.
- **`mantisnet.klent.sealbot` is the default anchored evaluation** — seat-balanced
  paired matches against [SealBot](https://github.com/Ramora0/SealBot), an
  independent C++ alpha-beta bot for this exact game, from a machine-local
  checkout (`--sealbot <root>`; build its `minimax_cpp` there first —
  MSVC via `setup.py build_ext --inplace` works). It stands in the paper's
  anchored-external-opponent role: it shares no code or training history
  with this repo, which is what makes its score a strength measurement
  self-play conditioning cannot flatter. Its rules implementation is held
  to agree with `hexo-engine` on every placement and winner (a live
  second-implementation oracle; setting `SEALBOT_ROOT` enables the tests),
  its moves are asserted legal, and `hexo_py` stays authoritative.
  The match itself accepts an opponent identity/config plus a batched
  chooser; a future champion network needs one adapter and then works in
  both the driver and offline sweeps without another match loop.
  The in-loop opponent is its uncapped iterative-deepening search at
  0.1 s/turn. `--max-depth` remains an offline weaker-rung knob, while
  offline `--sims` defaults to zero so old argmax curves stay comparable;
  `--run <dir> --every N` writes a strength curve to
  `sealbot_curve.jsonl`. Run plan §4 carries the measured scores.

## Telemetry

Every run writes `runs/<name>/telemetry.db` beside `metrics.jsonl` — one
SQLite file per run, WAL, one transaction per iteration. It is not optional
and has no flag: a run that cannot say what it did is not worth having, and
one code path is cheaper than two. `metrics.jsonl` stays the §13 record;
the database is the queryable substrate a dashboard is built on.

**The storage decision: π′ is not stored.** The improved policy is
`|A_legal|` floats per ply — kilobytes where a ply's whole row is tens of
bytes — and it is *exactly* reproducible from a checkpoint and a move
prefix. So the database keeps the four scalars a query needs (π′'s KL to
π_θ, its normalized entropy, its maximum, and its value at the sampled
move) and `klent.inspect.inspect_position` recomputes the array on demand,
through the training path's own loader, builder, and closed form. Every
other shape here follows from having chosen scalars over arrays.

| Table | One row per | Carrying |
| --- | --- | --- |
| `runs` | process that trained into the run | resolved knobs + versions, the database's mirror of `invocations.jsonl` |
| `iterations` | iteration | the §13 metrics as their own columns (so a threshold query needs no JSON parse), the row verbatim in `metrics_json`, and the hardware trace |
| `games` | game, self-play *and* evaluation | winner, length, capped, eval seat/opening, and the move list as a blob |
| `plies` | self-play ply | mover, `moves_remaining`, `legal_count`, the taken `rank`, and the five acting-time scalars |
| `opponents` | opponent *at a setting* | `name` + `config_json`; uncapped SealBot at 0.1 s and a depth-capped rung are two opponents |
| `eval_matches` | match | score, Wilson interval, Elo, per-seat split — keyed by opponent and by iteration or checkpoint |
| `crossplay` | A7 pairing | replaced wholesale by each run of it, as `crossplay.json` is |

- **Moves pack as little-endian `int16` `(q, r)` pairs**, four bytes a ply,
  defined once in `pack_moves`/`unpack_moves`. Coordinates outside `int16`
  are refused rather than wrapped.
- **The opponent is a row, not a column.** Nothing SealBot-specific reaches
  the schema: its variant, depth cap, and time limit identify an
  `opponents` row, so a stronger engine arriving later registers itself the
  same way and its strength curve is the same query.
- **Resume supersedes rather than duplicates.** A resume starts at its
  checkpoint's iteration, which can be behind what was recorded; those rows
  and their games and plies are dropped as the run redoes them. Offline
  evaluation matches are keyed by checkpoint, not by the loop's position,
  and survive.
- **The hardware trace is a background thread** (`hardware.py`) reading NVML
  and psutil once a second — GPU utilization, power, temperature, both the
  NVML and torch views of VRAM, process CPU/threads/RSS, host RAM —
  aggregated to mean/max columns per iteration. It touches no model, no
  RNG, and no database; a sensor failure stops it and is re-raised on the
  training thread at the next drain rather than becoming quiet zeros.
- **Reading is a module, not a service.** `iteration_series`,
  `search_games`, `fetch_game`, `calibration` (reliability by v̂, by ply, or
  by game length), `blunders` (v̂ swings across a ply, in the mover's own
  frame), `opening_atlas` (symmetry-reduced by `canonical_opening` — the
  rules are D6-invariant about the origin, so counting raw move lists would
  split each opening twelve ways), `strength_curve`, `crossplay_matrix`.
  `python -m mantisnet.klent.telemetry --run runs/<name> summary | games |
  game <id>` is the operator's sanity check over the same functions.
- **The game browser's orders are indexed, one index per order, and the
  indexes arrive on the write path.** `games_browse_<order>` leads with
  `kind` and then repeats that `GAME_ORDERS` entry's sort key column for
  column and direction for direction, so a page walks fifty index entries
  instead of sorting the run's whole history for each request: measured on
  a 575,342-game run, 484 ms → 0.05 ms warm, and 44 s → milliseconds
  through the deck's mount. Sharing one index between an order and its
  reverse depends on a planner heuristic that differs between SQLite
  releases, so each order gets its own; the four cost ~56 MB on a run that
  size. They are *not* part of `SCHEMA_VERSION` — an index is derived from
  rows already in the file and stores no format of its own — so
  `_index_games` applies them idempotently whenever a **writer** opens a
  run, which is how a database written before they existed gains them. The
  deck reads `telemetry.db` read-only and cannot create them itself.

**Measured cost** (1024 games × ~30 plies, the operating point): the write
is **~72 ms an iteration**, ~1% of a 5–7 s iteration, on the driver thread
between the fit and the wait for the next collection — and it draws nothing
from the training RNG, which `tests/test_telemetry.py` pins by running the
same seed with the writer stubbed out and comparing `metrics.jsonl` line for
line. **The per-ply scalars are stored quantized** (schema v2, owner
directive 2026-07-28): integers in units of 1e-4, defined once as
`telemetry._Q` — the five `REAL` columns were 40 bytes of a ~71-byte row
and SQLite has no `float32`; as 2–3-byte varints the same row is ~40
bytes, and 5e-5 of rounding is far below every consumer's noise floor.
Writers multiply and readers divide inside `telemetry.py`; nothing outside
sees an integer. Measured ~65 B/ply on a short-game synthetic corpus
(games-row overhead is heavy at 11 plies a game; the production mix sits
lower). There are still no migrations — a v1 database is refused — but a
finished run's history cannot be replayed, so `--run <dir> convert`
regenerates a v1 file as v2 in place, deliberately and once, keeping the
original as `telemetry.db.v1.bak`. Never against a live writer. The
remaining levers if disk hurts again: dropping a scalar column, or
recording plies for a sample of games rather than all.

```python
from mantisnet import MantisConfig, MantisNet
from mantisnet.klent import KlentConfig, collect_episodes, episode_samples, fit
import numpy as np, torch

model = MantisNet(MantisConfig()).to("cuda")
cfg = KlentConfig(device="cuda", autocast=True, games_per_iteration=32)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
rng = np.random.default_rng(0)
episodes, metrics = collect_episodes(model, cfg, rng)  # f, KL, H/log|A|, …
samples = [s for e in episodes for s in episode_samples(e, cfg.lam_ret)]
if samples:
    metrics.update(fit(model, samples, opt, cfg, rng))
```

## Deliberately absent

| Omitted | Why |
| --- | --- |
| Python-owned encoder / evaluator / sessions | These now live in the Python-free `crates/models/mantisnet` package. `hexo-py` binds the package's shared encoder core, and `hexo-bot` injects only the live Torch forward. KLENT's production training loop deliberately needs none of the container session machinery — no search during training is the algorithm's point. |
| The aux window head (spec appendix A) | Optional by spec, and adding it later touches no input — it reads trunk output. |
| `KLENT_PROPOSALS.md`'s accepted items | Diffs against a baseline that must exist first. Each is a small, named change when wanted. |
| Records / runner integration for the buffer | Design doc §12/§14. The in-memory per-iteration buffer is the paper's own shape; persistence arrives with B2's per-move blob, not with a private writer here. |
| Container-side `fit` | The package returns a loud unsupported-operation error. Production fitting is `mantisnet.klent.run`; migrating it is an owner decision, and no partial trainer is scaffolded. |
| A second checkpoint format | Python training continues to write the authoritative Torch `.pt`. `hexo-bot init` seals that same file with the package manifest, τ/λ metadata, and a probe hash; there is no exported model or translated weight format. |
| A second container search | Shared `hexo-search::GumbelSession` supplies package evaluation. The Python Gumbel line search in `klent/search.py` remains the training loop's own eval search and the container's parity reference — one implementation per side of the boundary, proven equal by fixture. |
| Further hand-written Triton kernels | Two have landed. Block attention computes distance buckets and the learned per-head bias in-kernel and stops each row at its live key prefix, but its fixed 64×64, four-warp, three-stage launch improved the complete long-position forward only ~1.04×, below the 1.4× target. The decoder aggregation is the one that paid: a single-warp segment reduction over the incidence, 1.39–1.82× on the forward. Dense SDPA and an `index_add_` scatter remain as the CPU/failed-shape fallbacks and for fit's recompute backward. The profile now puts the trunk's §5.1/§5.2 message passing on top at ~19 %, and everything below it is under 10 %. The same segment reduction applies to §5.1 directly (`inc_window` is emitted in window order) but not to §5.2, whose `inc_stone` is not sorted — that half would need a permutation the builder does not currently produce. |
| FlexAttention for the §4.1 distance bias | Measured out (2026-07-27): a `score_mod` computing buckets from coordinates in-kernel was built and proven exactly equivalent (outputs ~2e-6, `dist_bias` grads ~2e-8), but at this model's sizes it ran **5× slower in fit and 2.7× slower in collection** for ≤0.2 GiB saved — once batches are budget-packed, attention is no longer the memory driver, and flex under dynamic shapes needed 128-padded lengths plus an eager block mask to compile at all. Revisit only if H or D_MAX grow enough to make the bias tensor dominant again. |

## Performance

Measured on the 4070 Ti / 12-core host, batch 256 over the random-playout
pool (worst-case-dense positions):

| Path | Throughput |
| --- | --- |
| Batch build, Rust (`collate_positions`, all cores) | ~9.5 k pos/s (~0.10 ms/pos) |
| Batch build, Python reference (single thread) | ~0.6 k pos/s |
| Forward, compiled, bf16 autocast (random-playout shape mix) | ~11.7 k pos/s (21.9 ms/batch) |
| Forward, eager, bf16 autocast | ~6.1 k pos/s |
| Collection (1024 slots, 4096-game quota, trained policy) | ~144 k samples in 59.88 s from empty compiler caches / 17.55 s warm (2.42 k / 8.16 k samples/s) |

**Fused attention is a measured negative against its 1.4× whole-forward
target** (2026-07-28). The target-card sweep chose a fixed 64×64 tile,
four warps, and three stages. Complete compiled bf16 forward time was:

| Stones | Cohort | Dense SDPA | Fused | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 256 | 33.1 ms | 32.9 ms | 1.01× |
| 50 | 1024 | 136.0 ms | 134.0 ms | 1.01× |
| 200 | 256 | 112.9 ms | 111.1 ms | 1.02× |
| 200 | 1024 | 455.1 ms | 445.4 ms | 1.02× |
| 400 | 256 | 214.6 ms | 206.4 ms | 1.04× |
| 400 | 1024 | 860.1 ms | 822.8 ms | 1.05× |

At 400 stones peak allocation fell from 2.32 to 2.05 GiB, but removing
the dense bias and padded-key work moved the complete forward only ~4%,
so the latency target was not met. Collection shows the cache tradeoff:
the dense-SDPA baseline was 41.8 s / 3.48 k samples/s cold; fused attention
was 59.88 s / 2.42 k cold but 17.55 s / 8.16 k warm, with 15.76 s of warm
network busy time. A real compiled fit epoch was effectively flat: 8,215
samples in 0.86 s (9,569 samples/s) dense versus 7,784 in 0.82 s (9,487
samples/s) fused, a normalized -0.9%; the two receipts used separate
collector corpora rather than a paired sample set.

**The shared decoder aggregation is a measured 1.39–1.82× on the forward**
(2026-07-28). Both cell heads walk one incidence table and differ only in
their parameters, and a linear map commutes with a sum, so each head's
projection moves out from under the gather-scatter: one pass aggregates the
window rows, the slot-class counts, and the background bucket into a row per
cell, and a head folds its projection and both embedding tables into the
single matrix that reads that row. Compiled bf16 `_policy_q` time was:

| Stones | Cohort | Twice-run | Shared | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 256 | 44.5 ms | 24.5 ms | 1.82× |
| 50 | 1024 | 142.0 ms | 100.5 ms | 1.41× |
| 200 | 256 | 116.6 ms | 83.1 ms | 1.40× |
| 200 | 1024 | 468.4 ms | 335.3 ms | 1.40× |
| 400 | 256 | 217.3 ms | 156.7 ms | 1.39× |
| 400 | 1024 | 871.3 ms | 624.8 ms | 1.39× |

Peak allocation fell with it — 3.04 to 1.96 GiB at 50 stones / cohort 1024,
2.07 to 1.66 at 200 / 256 — since the two `(N_c, H)` fp32 accumulators are
gone.

**Sharing the incidence is not where that came from.** Inductor had already
fused the two heads' scatters into one launch that reads the index tensors
once and issues two `atomic_add` streams, so folding them by hand is worth
1.12–1.15× and no more: the cost was never the duplicated indexing, it was
2×H-wide fp32 atomics into two zeroed accumulators. Removing it takes
aggregating H-wide *once* and projecting afterwards, and then a Triton
segment reduction — the builder emits decoder entries in cell order, so a
cell's run sums in registers and stores once, with no atomics and no zero
fill. The gather-scatter's share of device time at cohort 256:

| Stones | Before | After |
| ---: | ---: | ---: |
| 50 | 9.51 ms of 32.7 (29.1 %) | 0.62 ms of 22.1 (2.8 %) |
| 400 | 49.2 ms of 214.9 (22.9 %) | 3.04 ms of 154.4 (2.0 %) |

**The loop converts that unevenly.** A real compiled fit epoch went 32,864
samples in 4.18 s (7,871 samples/s) to 33,043 in 3.54 s (9,346 samples/s), a
normalized 1.19×. Collection did not move: 15.61 s / 9.3 k samples/s against
14.75–15.25 s / 9.5–9.8 k, inside run-to-run spread. Its median game is 32
plies, and a lockstep step at that size is bound by the host–device transfer
and the Python around it rather than by decoder device time — the same
orchestration ceiling the pipelined loop already met. The largest single
kernel is now the trunk's own §5.1/§5.2 message passing, eight launches per
forward at 18.8 % of device time at both 50 and 400 stones.

Two loop-level facts behind those numbers. **Choosers are batched**
(`choose(positions, rng) -> moves`): `play_match` advances every live game
per lockstep step with one collate and one forward per side, which is the
~20× on evaluation. **Sampling is one uniform per game** against the
stored π′ CDF rather than a per-game `rng.choice`.

**The loop is pipelined: collection overlaps fitting** (2026-07-28, after
a phase-level profile). The sequential loop plateaued at ~3.2 k samples/s
whatever the game count — CPU orchestration with the GPU at ~6% duty. The
fixes, in measured order of value: iteration ``i+1``'s collection runs on
a worker thread against a weight snapshot while iteration ``i`` fits on
the live model (the corpus runs one fit behind the strict alternation —
an algorithmic property, documented in `run_training`); the per-game
sampling loop is one flat CDF + one vectorized `searchsorted` per chunk;
fit preps each chunk one ahead on a worker and accumulates its loss
scalars on-device (one host sync per iteration, not per chunk).

**Collection is a persistent auto-reset cohort** (2026-07-28, design doc
§16 item 15). The old drain-to-empty lockstep ran until its single longest
game ended — same-corpus iterations measured 2.7 s → 144 s, ~30 % of
collection spent with under 6 % of the cohort alive. `Collector` keeps
`envs` slots permanently full (a finished game's slot restarts from the
empty board that step), stops at a completion quota, and carries in-flight
games to the next call. Inside each step, chunks pipeline across three
lanes — collate worker → GPU → sampling worker — so the CPU phases hide
behind the forward (measured: 6.1 s of CPU busy against 4.4 s of
wall-clock overhang at the operating point). Net from the collector rewrite:
**~671 samples/s production average before → ~4.9 k samples/s steady after
(~7×), with iteration wall clock bounded by construction.** On the fused
attention tree the same harness reaches 8.16 k samples/s warm (2.42 k/s from
empty compiler caches). The warm loop remains network-bound (~90% of wall),
but the attention sweep above shows that forward-path share is not the same
as attention's share.

Two threading facts the pipeline depends on. **Every compiled-callable
invocation holds one lock** (`train._gpu_lock`): a dynamo (re)compile on
one thread while the other runs Python was measured to stretch a
seconds-long compile into a minutes-long GIL-thrashing crawl — under the
lock the other side sleeps, and sporadic new-shape-bucket recompiles cost
seconds again. And **the first processed iteration stays sequential**, so
both graphs compile without a concurrent collector. Cold-cache processes
also transiently over-reserve VRAM during compile/autotune (once measured
18.9 GiB reserved at a 6.6 GiB allocated peak); warm steady state sits at
~5 GiB for both.

`KlentConfig.compile` turns on one `torch.compile(dynamic=True)` graph shared
by collection and fitting. Sizes inside the forward come from tensor shapes,
not the `Batch`'s ints, so one symbolic graph serves every batch shape; the
attention custom op remains an opaque call inside that graph, including
during fit. Its first dtype/head-dimension execution compiles the fixed
kernel, and a failed shape is warned once and permanently routed through
SDPA. The first process *on a machine* still pays graph compilation — a
historical full collection-plus-fit cold compile at training sizes measured
~15 min under Windows Triton, disk-cached thereafter — plus one extra
specialisation the first time a 0/1-sized dimension appears.

**VRAM is budgeted, not hoped for.** Every network batch — fit chunk or
collection cohort — is packed under two `KlentConfig` knobs before it is
built: `pair_budget` caps padded attention work and, during fit, the dense
recompute-backward pair tensors; `cell_budget` caps decoder rows. In no-grad
collection the fused kernel no longer materialises a per-pair bias, so its
pair budget is primarily a work/latency bound. `batch_size` is a per-step
*maximum* and the peak is set by config rather than by whatever the corpus
contains. Fit chunks are packed length-sorted (a chunk pads to its own
longest sample, so one 500-stone position can no longer square-inflate a
mixed chunk); collection cohorts split in game order, which the
chunking-invariance test pins. Measured on the iteration-0 worst-case corpus
(~5.5 k legal cells/sample, games to ply 500) at the defaults:
**collection peaks at 0.36 GiB, a fit epoch at ~2.9 GiB — and runs ~2×
faster** than the unpacked batch-256 fit, because homogeneous-length chunks
stop paying padding. The sample buffer itself never touches VRAM: samples
are numpy arrays and move prefixes in host memory.

One hazard remains worth knowing: **Windows VRAM overruns fail slow, not
loud.** The driver spills to system RAM at PCIe speed instead of raising
OOM — a ~50× slowdown with no error. The budgets exist to keep the run out
of that regime; Linux (the deploy target) raises OOM honestly.

**Platforms.** The destination is Docker on Linux for *everything* —
training, self-play, and evaluation, not just serving — and the
environment exists: `docker/` at the repo root (see `docker/README.md`).
On the pre-fused-attention tree,
identical code on the same card ran collection **1.44× faster in the
container than Windows-native** (5.0 k vs 3.5 k samples/s,
cold-compile-inclusive). The fused kernel above was implemented and
measured Windows-native; remeasure that platform ratio before using it as
a current projection. Windows-native runs remain a convenience for tests
and quick checks; there the `triton-windows` dependency is marked
`sys_platform == 'win32'`. Outside the container under bare WSL, keep the
two trees separate from the Windows ones: build `hexo-py` with
`CARGO_TARGET_DIR=target-wsl` (the repo's convention) and give uv its own
environment, e.g. `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/mantisnet uv sync`.

## Connections

- `python/hexo-py` supplies positions and the engine-order legal list — the
  builder's whole input, per `MODEL_SPEC.md` §11 — and the whole game surface
  KLENT's self-play and matches run on.
- `docs/MODEL_SPEC.md` is normative for the model; where code and spec
  disagree, that is a finding to raise, and §3.2's pattern count was one.
  `docs/KLENT_DESIGN.md` governs the training path, faithful first.
- `hexo-model` / `crates/models/mock` show the package shape this model will
  eventually be wrapped in.
