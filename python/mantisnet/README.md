# mantisnet

## Purpose

`python/mantisnet` requires Python 3.12 or later and contains the MantisNet
Torch model, independent graph-builder oracle, losses, Hexo KLENT training loop,
telemetry store, benchmarks, and control-deck backend. The training path uses a policy head and one
`tanh`-bounded scalar action value per legal cell, composed from a per-position
baseline and a policy-centered advantage. Algorithm obligations are in
`docs/KLENT_FOR_HEXO.md`; measured experiment outcomes are in
`docs/ABLATIONS.md`.

## Public surface

The top-level `mantisnet` package exports:

| Item | Contract |
| --- | --- |
| `MantisConfig`, `MantisNet`, `ModelOutput` | Model configuration, module, and outputs |
| `PositionGraph`, `Batch` | One encoded graph and one ragged model batch |
| `build`, `from_position`, `collate` | Independent Python graph path |
| `collate_positions`, `collate_prefixes` | Shared Rust encoder path |
| `policy_loss`, `value_loss`, `value_target` | Training losses and targets |
| `param_groups` | Decay/no-decay optimizer groups |
| `MODEL_REPR_VERSION`, `NUM_PATTERNS` | Representation constants |

Core modules:

| Module | Contract |
| --- | --- |
| `builder` | Graph construction, Rust batch conversion, and collation |
| `model` | Stone/window trunk, policy decoder, dueling action-value head, value head |
| `attention` | Fused attention operation plus reference path |
| `decoder` | Shared legal-cell incidence aggregation |
| `segments` | Ragged segment reductions |
| `losses` | Policy and distributional value losses |
| `klent.improve` | Closed-form improved policy |
| `klent.selfplay` | Persistent-slot collection and samples |
| `klent.train` | Network evaluation, packing, collection, and fit |
| `klent.run` | Checkpointed iteration driver and CLI |
| `klent.graft` | Scalar-critic checkpoint conversion, measured and enforced |
| `klent.telemetry` | SQLite writes, queries, and CLI |
| `klent.search` | Gumbel line search used by Python evaluation |
| `klent.opponents` | Opponent seam, SealBot adapter, `shared_openings`, `wilson`, `elo` |
| `klent.evaluate` | Policy argmax and the two-chooser lockstep match loop |
| `klent.headtohead` | Paired cross-run checkpoint-vs-checkpoint match and CLI |
| `deck.app` | FastAPI routes and SPA serving |
| `deck.service` | Run registry, inference cache, and play sessions |
| `deck.state` | Deck-owned reviews, probes, presets, and match jobs |

`MantisNet.forward(batch)` returns raw policy logits, scalar action values,
distributional state value outputs, and the decoded state value. KLENT
collection and fitting call `trunk` plus `cell_heads` and do not consume the
state-value head.

The `mantisnet.klent` package exports collection, returns, improvement, fit,
evaluation, run, telemetry, and inspection entry points.

## Run / test

From `python/mantisnet`:

```sh
uv sync --all-groups
uv run pytest
```

Run a bounded CPU training invocation:

```sh
uv run python -m mantisnet.klent.run \
  --out runs/readme-smoke \
  --iterations 1 \
  --games 8 \
  --envs 8 \
  --checkpoint-every 1 \
  --device cpu \
  --no-compile
```

Resume an existing run to a new total iteration count:

```sh
uv run python -m mantisnet.klent.run \
  --out runs/readme-smoke \
  --iterations 2 \
  --games 8 \
  --envs 8 \
  --device cpu \
  --no-compile \
  --resume
```

Inspect telemetry and run benchmarks:

```sh
uv run python -m mantisnet.klent.telemetry --run runs/readme-smoke summary
uv run python bench/bench_forward.py
uv run python bench/bench_loop.py sweep --device cpu --stones 20 --cohorts 16 --iters 1
```

Compare two checkpoints from different runs, paired:

```sh
uv run python -m mantisnet.klent.headtohead \
  --a runs/one/checkpoint_001000.pt \
  --b runs/two/checkpoint_001000.pt \
  --pairs 64 --sims 32 --tau 0.1 --lam 0.01 \
  --out h2h.json
```

Run the deck backend directly:

```sh
uv run uvicorn mantisnet.deck.app:app --host 0.0.0.0 --port 8000
```

## Connections

- [`docs/MODEL_SPEC.md`](../../docs/MODEL_SPEC.md) defines the model and
  representation contract.
- [`docs/KLENT_FOR_HEXO.md`](../../docs/KLENT_FOR_HEXO.md) defines KLENT's Hexo
  adaptation and training obligations.
- [`docs/KLENT_PAPER.md`](../../docs/KLENT_PAPER.md) records the source
  algorithm.
- [`docs/ABLATIONS.md`](../../docs/ABLATIONS.md) owns measured run outcomes.
- [`docs/DECK_SPEC.md`](../../docs/DECK_SPEC.md) defines the control-deck API.
- `python/hexo-py` supplies engine positions and the shared Rust encoder.
- `crates/models/mantisnet` consumes the same checkpoint and encoder semantics
  from the native container.
- `frontend` consumes `mantisnet.deck` over HTTP and SSE.
- `docker` supplies the CUDA training and deck environment.

## Invariants & gotchas

- Model weights are fp32; CUDA network calls use bf16 autocast when configured.
- Legal-cell outputs are ragged and follow engine canonical legal order.
- Policy and action-value decoders share incidence aggregation but own separate
  projections, embeddings, and output MLPs.
- Action values are scalar and bounded to `(-1, 1)` by `tanh`.
- An action value is a per-position baseline plus the decoder's advantage,
  centered on the detached raw policy of the same forward.
- The policy, advantage, and critic-baseline output layers initialize to zero.
- KLENT improvement consumes raw policy logits and scalar action values.
- The KLENT fit objective trains policy cross-entropy and the taken action's
  squared return error; the critic baseline has no loss of its own.
- KLENT fitting does not train or read the distributional state-value head.
- Stored training states are move prefixes and are rebuilt through `hexo_py`.
- Capped self-play episodes are excluded from the fitting buffer.
- `games` is a completed-game quota; `envs` is the number of persistent slots.
- Fit and collection batches obey separate attention-pair and legal-cell
  budgets.
- Checkpoints contain model state, optimizer state, completed iteration, and
  NumPy RNG state.
- A fresh run refuses a nonempty output directory.
- `--resume` requires a checkpoint and is exclusive with `--init-from`.
- A pre-dueling scalar-critic checkpoint enters only through
  `python -m mantisnet.klent.graft`, which requires `--tau`, `--lam`, and
  `--manifest`, and writes nothing unless it preserved the parent's ordering of
  `Q`, carried every shared parameter bitwise, and composed appendix B on them.
- `config.json` records current run fields; `invocations.jsonl` records every invocation and resume.
- `metrics.jsonl` is append-only; replay after resume can repeat an iteration number.
- `telemetry.db` is schema-versioned and stores run, iteration, game, ply,
  evaluation, and cross-play data.
- `status.json` is the deck heartbeat and phase-progress surface.
- `STOP` requests shutdown after the current iteration and a durable checkpoint.
- `CHECKPOINT` requests a checkpoint at the next commit point.
- Every match plays `games / 2` shared random openings from both seats; the ply
  cap counts the opening's placements, and a capped game scores one half.
- Both crossplay choosers are deterministic, so the openings are the whole
  source of a pairing's diversity.
- A head-to-head pair shares its opening and its generator, derived from
  `(seed, pair index)`, and is reproducible on its own; the same seed gives the
  same result.
- A head-to-head reports the paired standard error beside the unpaired one, an
  exact sign test over the decisive pairs, and every capped game in `warnings`.
  Pairs that all carry one `d` have no spread to estimate, so such a match
  reports no Elo interval and warns instead.
- A head-to-head at `--sims 0` never consults `--tau`/`--lam` and records them
  as absent; a searched one requires both.
- A head-to-head refuses two checkpoints that differ in rules, action order, or
  Torch version, and names `klent.graft` as the bridge for a representation
  difference.
- `wilson` takes a total score and returns rates; `elo` takes a rate.
- Deck queries open telemetry read-only; deck-owned state is in `runs/deck.db`.
- The inference cache holds at most its configured checkpoint capacity.
- CUDA compilation is enabled by the CLI unless `--no-compile` is set.
