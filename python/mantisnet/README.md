# mantisnet

The MantisNet network of `docs/MODEL_SPEC.md`, real — builder, trunk, heads,
losses — plus the KLENT training path of `docs/KLENT_DESIGN.md`: the
closed-form policy improvement that replaces tree search, implemented
faithful-first against that document.

**Status: implemented and green.** The model is 1.25 M parameters at the §2
defaults — the spec's 1.2 M plus the appendix-B Q head — with the spec's §12
obligations as tests; KLENT carries the design
doc's §4.7 obligations as its own. On the 4070 Ti a steady-state KLENT
iteration (32 seeded games → buffer → fit) runs in ~3 s: batch building is
Rust and rayon-parallel (~0.1 ms/position) and the forward is
`torch.compile`d (~2.1× over eager) — the Performance section below has the
numbers. **Not yet a `ModelPackage`:** no
encoder, no evaluator, no sessions, no checkpoints, and no record/runner
integration — the KLENT buffer is in-memory per iteration, as the paper's is.

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
      selfplay.py     # batched collection, acting-time v̂, the buffer rules
      train.py        # KlentConfig, the fit epoch, the iteration
      evaluate.py     # argmax π_θ, seat-balanced matches
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
  names, and the same chooser is the fixed evaluation opponent. Annealing the
  prefix cut toward zero is an operator decision driven by the reported `f`,
  not an automated schedule — the metric is first-class, the controller is
  not yet earned.
- **Collection goes through one seam**: `evaluate(batch) -> (policy_logits,
  q_values)` on CPU. Training wraps the network; the pipeline tests wrap a
  scripted line-extender, which is how the buffer rules are testable without
  a trained model.

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

## Performance

Measured on the 4070 Ti / 12-core host, batch 256 over the random-playout
pool (worst-case-dense positions):

| Path | Throughput |
| --- | --- |
| Batch build, Rust (`collate_positions`, all cores) | ~9.5 k pos/s (~0.10 ms/pos) |
| Batch build, Python reference (single thread) | ~0.6 k pos/s |
| Forward, compiled, bf16 autocast | ~9.4 k pos/s (27 ms/batch) |
| Forward, eager, bf16 autocast | ~4.4 k pos/s |
| KLENT iteration, steady state (32 games, cap 200) | ~3 s (was ~30 s eager + Python builder) |

`KlentConfig.compile` turns on one `torch.compile(dynamic=True)` graph shared
by collection and fitting. Sizes inside the forward come from tensor shapes,
not the `Batch`'s ints, so one symbolic graph serves every batch shape; the
first process pays the compile (tens of seconds, partly cached across runs),
plus one extra specialisation the first time a 0/1-sized dimension appears.

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
