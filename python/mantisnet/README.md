# mantisnet

The MantisNet network of `docs/MODEL_SPEC.md`, real — builder, trunk, heads,
losses — plus the KLENT training path of `docs/KLENT_DESIGN.md`: the
closed-form policy improvement that replaces tree search, implemented
faithful-first against that document.

**Status: implemented, green, and past its first training run.** The model
is 1.25 M parameters at the §2 defaults — the spec's 1.2 M plus the
appendix-B Q head — with the spec's §12 obligations as tests; KLENT carries
the design doc's §4.7 obligations as its own. Batch building is Rust and
rayon-parallel (~0.1 ms/position), the forward is `torch.compile`d (~2.1×
over eager), and a full KLENT iteration at the design's settings (64 games,
cap 512) runs in ~36 s at its iteration-0 worst on the 4070 Ti — the
Performance section below has the numbers and the two hazards worth knowing.
**Not yet a `ModelPackage`:** no encoder, no evaluator, no sessions, and no
record/runner integration — the KLENT buffer is in-memory per iteration, as
the paper's is.

## Shape

```
python/mantisnet/
  pyproject.toml      # uv project; torch from the cu128 index, hexo-py by path
  mantisnet/
    __init__.py       # flat re-exports, MODEL_REPR_VERSION
    builder.py        # §3-§4, §9: graphs, index tables, collation
    model.py          # §5-§7, §10 + appendix B: trunk, policy/Q/value heads
    losses.py         # §6, §7, §10: targets, cross-entropies, decay grouping
    segments.py       # ragged per-position reductions, shared by losses and klent
    klent/
      improve.py      # eq. 3 closed form: π′, v̂, and the §13 diagnostics
      returns.py      # the sign on mover change, the λ-return
      seeds.py        # the line-building seeder / fixed opponent
      selfplay.py     # batched collection, acting-time v̂, buffer rules, stats
      train.py        # KlentConfig, the fit epoch, the iteration
      evaluate.py     # argmax π_θ, seat-balanced matches, the anchor match
      run.py          # the run driver: config.json, metrics.jsonl, checkpoints
      crossplay.py    # the A7 checkpoint round-robin, the forgetting detector
      sealbot.py      # the external yardstick: matches vs a SealBot checkout
  tests/              # the two specs' obligations, one file per concern
  bench/
    bench_forward.py  # builder and forward throughput at spec defaults
```

Run everything from this directory:

```
uv sync                              # venv, hexo-py wheel via maturin, torch cu128
uv run pytest                        # the whole suite
uv run python bench/bench_forward.py # throughput on CPU and the local GPU
```

## Module map

| Module | Role |
| --- | --- |
| `builder` | `build` (raw §11 inputs to a `PositionGraph`), `from_position` (the `hexo_py` wrapper), `collate` (graphs to one `Batch` of index tensors). Owns `MODEL_REPR_VERSION` and every index convention. |
| `model` | `MantisConfig` (the §2 named parameters), `MantisNet`, `ModelOutput`. `trunk` and the three head methods are separate so a caller pays only for the heads it reads. |
| `losses` | `value_target` (two-hot projection), `value_loss`, `policy_loss` (segmented CE over ragged engine-order logits), `param_groups` (§10 decay split). |
| `segments` | The ragged per-position reductions everything above and below shares. |
| `klent` | The KLENT baseline: operator, returns, seeding, collection, fitting, evaluation. See below. |

## Design notes

- **Two builders, one representation, and a parity detector between them.**
  The Python builder (`build`/`collate`) is the normative reference: it never
  calls the engine's window walk, which is what keeps `windows_through` an
  *independent* oracle for §12.1 — a builder built on the engine's enumeration
  would agree with it by construction, the deleted-detector failure `CLAUDE.md`
  warns about. The production path (`collate_positions`/`collate_prefixes`) is
  Rust in `hexo-py`: rayon-parallel with the GIL released, ~16× the Python
  path at batch 256, and *allowed* to use the engine's walk precisely because
  `test_rust_builder.py` holds it exactly equal to the Python output, field
  for field — the §12.7-style detector a second implementation owes. Both are
  covered by one `MODEL_REPR_VERSION`.

- **34 canonical patterns, not the spec draft's 32.** The 62 nonempty, nonfull
  6-bit masks fold to `(62 + 6 palindromes) / 2 = 34` orbits under reversal.
  Raised and corrected in `MODEL_SPEC.md` §3.2; a test pins the count so the
  table cannot silently disagree with the doc again.

- **Two index conventions the spec left to the implementation**, fixed in
  `builder.py`'s docstring: attention buckets are `d-1` clamped, then `SELF`,
  then `TOKEN`, with `TOKEN` winning the token-token pair; the one stoneless
  position (ply 0) takes the background clamp bucket 7.

- **Batching is concatenation plus two padded layouts.** Message passing and
  both MLP heads run on concatenated entities with `index_add_`/`index_select`
  — no padding, no waste. Attention and the value readout run padded per
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
algorithm at the design doc's seven forced deviations and nothing else. The
accepted items of `KLENT_PROPOSALS.md` (the λ_intra split, the Bernoulli
critic, the dual controller) are deliberately not in it: the design doc lists
them as diffs to be decided, and a faithful baseline has to exist before a
deviation from it can be measured.

- **The model KLENT trains is trunk + policy head + Q head.** The Q head is
  the §6 decoder shape with its own parameters (spec appendix B); the §7
  value head is outside the loss, per the paper's no-V-head ablation, and
  `v̂ = E_{π′}[Q]` supplies the bootstrap. The forward is split into `trunk`
  plus per-head methods precisely so the loop never computes the readout it
  never reads.
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
  episodes dropped whole (K4), seeded prefix plies never recorded, terminal
  positions never samples, `v̂` captured at acting time (K6), and fitting
  refuses a sample whose stored π′ no longer matches its position's legal
  count. States are stored as move prefixes and rebuilt by replay (§12).
- **Seeding is the line builder**, the checkpoint-free source the design doc
  names, and the same scoring is the warm-start evaluator
  (`--warm-iterations`) and the fixed evaluation opponent. The cut anneal is
  earned and mechanical (`--anneal`): the ceiling deepens while measured `f`
  holds and backs off when it falls, recorded per row as `seed_cut_hi` —
  a static cut was measured to park the corpus on trivial endgame stubs
  while strength died. Its known gap: `f` measures termination, not
  competence; the successor is a competence-gated walk.
- **Collection goes through one seam**: `evaluate(batch) -> (policy_logits,
  q_values)` on CPU. Training wraps the network; the pipeline tests wrap a
  scripted line-extender, which is how the buffer rules are testable without
  a trained model.
- **The default coefficients are paper-verified, not carried.** `τ = 0.1`
  (reverse KL) and `λ = 0.03` (entropy) per the paper's eq. 2 — the design
  doc's original pair was transposed and is corrected — and
  `λ_ret = e^{-1/16}`, the paper's 8-turn horizon at Hexo's two placements
  per turn. Operationally the first runs use `--lam-ret 1.0`: with a young
  Q, a 0.94-weight bootstrap erases the warm start (measured — the
  run plan's training-night record); the λ-return returns when v̂ has
  earned trust. `docs/KLENT_RUN_PLAN.md` §2/§3 record all of it.
- **A run is its directory.** `python -m mantisnet.klent.run --out runs/<name>
  --iterations N` writes `config.json` (knobs + versions), `metrics.jsonl`
  (strict JSON, one row per iteration: the §13 metrics including the
  v̂-vs-outcome calibration that watches the §9 bias), `invocations.jsonl`
  (every process that touched the run, with its resolved knobs — the anneal
  path changes them on resume), and resumable checkpoints; `--resume`
  continues after a crash and refuses a checkpoint from other versions.
  `--eval-every N` plays `argmax π_θ` against the line builder at pinned
  noise (or a frozen checkpoint via `--eval-anchor`) — seat balanced, caps
  scored ½ and kept visible — with an eval RNG derived from (run seed,
  iteration) so the training trajectory is identical with evaluation on or
  off. `--starve-limit` ends a collapsed run with a checkpoint instead of a
  burned night, and `mantisnet.klent.crossplay` plays the A7 checkpoint
  round-robin. `docs/KLENT_RUN_PLAN.md` is the operational plan, §3 of it
  the measured history, around this driver.
- **`mantisnet.klent.sealbot` is the external yardstick** — seat-balanced
  paired matches against [SealBot](https://github.com/Ramora0/SealBot), an
  independent C++ alpha-beta bot for this exact game, from a machine-local
  checkout (`--sealbot <root>`; build its `minimax_cpp` there first —
  MSVC via `setup.py build_ext --inplace` works). Its rules implementation
  is held to agree with `hexo-engine` on every placement and winner (a live
  second-implementation oracle; setting `SEALBOT_ROOT` enables the tests),
  its moves are asserted legal, and `hexo_py` stays authoritative.
  `--max-depth` caps its search for graded rungs; `--run <dir> --every N`
  writes a strength curve to `sealbot_curve.jsonl`. First measurement
  (2026-07-28): the overnight-3 endpoint loses 0/64 even at depth 1 — the
  run plan §4 has the table and what it says about racing vs defending.
- **Opponent grounding puts that opponent in the corpus**
  (`--ground-fraction`, with `--sealbot`/`--ground-depth`): that fraction
  of each iteration's games seats a depth-capped SealBot on one
  (alternating) side, unseeded. Its whole turns enter the move list but
  never the records — the buffer holds only the model's decisions, judged
  by an outcome a real opponent enforced. Grounded returns are pure
  Monte-Carlo whatever `lam_ret` says (the λ-return's bootstrap chain
  breaks at unrecorded opponent plies), and a capped grounded game is a
  draw (g = 0) rather than dropped: surviving a killer is an outcome, and
  the only gradient toward defense while wins are out of reach. The f
  stats stay self-play-only so the anneal keeps its signal; grounded games
  report `f_grounded` and a per-iteration `grounded_score` (`gnd` on the
  console) instead. Grounding is off during warm iterations, and
  `--init-from` forks a fresh run (own seed, iteration 0) from a trained
  checkpoint — how ablation arms share a parent.

```python
from mantisnet import MantisConfig, MantisNet
from mantisnet.klent import KlentConfig, iterate
import numpy as np, torch

model = MantisNet(MantisConfig()).to("cuda")
cfg = KlentConfig(device="cuda", autocast=True, games_per_iteration=32)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
metrics = iterate(model, opt, cfg, np.random.default_rng(0))  # f, KL, H/log|A|, losses
```

## Deliberately absent

| Omitted | Why |
| --- | --- |
| Encoder / evaluator / sessions | Wiring MantisNet to `hexo-search`'s seam is the Python-backed package of `CONTAINER_SPEC.md`, a change to be made there, not here. KLENT's training loop deliberately needs none of it — no search during training is the algorithm's point. |
| The aux window head (spec appendix A) | Optional by spec, and adding it later touches no input — it reads trunk output. |
| `KLENT_PROPOSALS.md`'s accepted items | Diffs against a baseline that must exist first. Each is a small, named change when wanted. |
| Records / runner integration for the buffer | Design doc §12/§14. The in-memory per-iteration buffer is the paper's own shape; persistence arrives with B2's per-move blob, not with a private writer here. |
| Checkpoint I/O | The manifest and probe protocol are `hexo-model`'s, and arrive with the package. |
| Test-time Gumbel MCTS | Design doc §15: the paper's best number, but it measures the search, not the algorithm. |
| Hand-written Triton kernels | Measured out, for now: after `torch.compile` (which generates fused Triton kernels itself) the forward is no longer the bottleneck, and the remaining costs are memory-bound scatters Inductor already fuses. Revisit if a profile ever shows one kernel dominating. |
| FlexAttention for the §4.1 distance bias | Measured out (2026-07-27): a `score_mod` computing buckets from coordinates in-kernel was built and proven exactly equivalent (outputs ~2e-6, `dist_bias` grads ~2e-8), but at this model's sizes it ran **5× slower in fit and 2.7× slower in collection** for ≤0.2 GiB saved — once batches are budget-packed, attention is no longer the memory driver, and flex under dynamic shapes needed 128-padded lengths plus an eager block mask to compile at all. Revisit only if H or D_MAX grow enough to make the bias tensor dominant again. |

## Performance

Measured on the 4070 Ti / 12-core host, batch 256 over the random-playout
pool (worst-case-dense positions):

| Path | Throughput |
| --- | --- |
| Batch build, Rust (`collate_positions`, all cores) | ~9.5 k pos/s (~0.10 ms/pos) |
| Batch build, Python reference (single thread) | ~0.6 k pos/s |
| Forward, compiled, bf16 autocast | ~9.4 k pos/s (27 ms/batch) |
| Forward, eager, bf16 autocast | ~4.4 k pos/s |
| Collection, steady state (256 games, trained policy) | ~9.3 k pos/s (~0.35 s) |
| KLENT iteration, steady state (256 games) | ~1.1–1.5 s at 3–4.5 k samples (~3 k samples/s end to end) |
| Anchor eval, 128 games | ~0.3 s (was ~6 s sequential) |
| KLENT iteration (64 games, cap 512, iteration 0, untrained) | ~36 s — the capped-tail worst case |

Three loop-level facts behind those numbers. **Choosers are batched**
(`choose(positions, rng) -> moves`): `play_match` advances every live game
per lockstep step with one collate and one forward per side, which is the
~20× on evaluation. **Seed prefixes generate on a worker thread** during
the previous iteration — they depend on nothing the model learns, so their
~0.5 s leaves the critical path (`generate_prefixes`, seeded off the main
stream for resume-reproducibility). **Sampling is one uniform per game**
against the stored π′ CDF rather than a per-game `rng.choice`.

**End-to-end throughput plateaus at ~3.2 k samples/s** (measured 2026-07-28
at games 256 / 512 / 1024: ~1.1 s / ~2.0 s / ~4.2 s per iteration, same
samples/s): the loop is CPU-orchestration-bound, so more games per
iteration raises GPU duty (spikes reach ~99% at 1024, VRAM ~4.4 GiB) but
not throughput, while halving policy-improvement rounds per hour each
doubling. 512 is the operating point. Depth-1 SealBot grounding at
fraction 0.25 costs ~nothing (~1 ms/turn). The next real lever is
pipelining collection against fitting, not more games.

`KlentConfig.compile` turns on one `torch.compile(dynamic=True)` graph shared
by collection and fitting. Sizes inside the forward come from tensor shapes,
not the `Batch`'s ints, so one symbolic graph serves every batch shape; the
first process *on a machine* pays the compile — measured ~15 min cold under
Windows Triton at training sizes, disk-cached thereafter — plus one extra
specialisation the first time a 0/1-sized dimension appears.

**VRAM is budgeted, not hoped for.** Every network batch — fit chunk or
collection cohort — is packed under two `KlentConfig` knobs before it is
built: `pair_budget` bounds the attention's materialised per-pair bias (the
axis quadratic in stone count) and `cell_budget` bounds decoder rows (the
axis linear in legal cells), so `batch_size` is a per-step *maximum* and the
peak is set by config rather than by whatever the corpus contains. Fit
chunks are packed length-sorted (a chunk pads to its own longest sample, so
one 500-stone position can no longer square-inflate a mixed chunk);
collection cohorts split in game order, which the chunking-invariance test
pins. Measured on the iteration-0 worst-case corpus (~5.5 k legal
cells/sample, games to ply 500) at the defaults: **collection peaks at
0.36 GiB, a fit epoch at ~2.9 GiB — and runs ~2× faster** than the unpacked
batch-256 fit, because homogeneous-length chunks stop paying padding. Both
knobs scale the peak roughly linearly if a card needs to give back more.
The sample buffer itself never touches VRAM: samples are numpy arrays and
move prefixes in host memory.

One hazard remains worth knowing: **Windows VRAM overruns fail slow, not
loud.** The driver spills to system RAM at PCIe speed instead of raising
OOM — a ~50× slowdown with no error. The budgets exist to keep the run out
of that regime; Linux (the deploy target) raises OOM honestly.

**Platforms.** The deploy target is Linux (WSL2 / the container of
`CONTAINER_SPEC.md`), where the torch wheel bundles Triton; the
`triton-windows` dependency is marked `sys_platform == 'win32'` and exists
only for the Windows dev box. Under WSL, keep the two trees separate from the
Windows ones: build `hexo-py` with `CARGO_TARGET_DIR=target-wsl` (the repo's
convention) and give uv its own environment, e.g.
`UV_PROJECT_ENVIRONMENT=$HOME/.venvs/mantisnet uv sync`.

## Connections

- `python/hexo-py` supplies positions and the engine-order legal list — the
  builder's whole input, per `MODEL_SPEC.md` §11 — and the whole game surface
  KLENT's self-play and matches run on.
- `docs/MODEL_SPEC.md` is normative for the model; where code and spec
  disagree, that is a finding to raise, and §3.2's pattern count was one.
  `docs/KLENT_DESIGN.md` governs the training path, faithful first.
- `hexo-model` / `crates/models/mock` show the package shape this model will
  eventually be wrapped in.
