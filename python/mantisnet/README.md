# mantisnet

## Purpose

`python/mantisnet` requires Python 3.12 or later and contains the MantisNet
Torch model, independent graph-builder oracle, losses, Hexo KLENT training loop,
telemetry store, benchmarks, and control-deck backend. The training path uses a policy head and one
`tanh`-bounded scalar action value per legal cell, the latter decoded from the
critic's own private tail over the trunk output. Algorithm obligations are in
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
| `model` | Stone/window trunk, policy decoder, critic tail, action-value decoder, value head |
| `attention` | Fused attention operation plus reference path |
| `decoder` | Per-head legal-cell incidence aggregation |
| `segments` | Ragged segment reductions |
| `losses` | Policy and distributional value losses |
| `klent.improve` | Closed-form improved policy |
| `klent.selfplay` | Persistent-slot collection and samples |
| `klent.train` | Network evaluation, packing, collection, and fit |
| `klent.run` | Checkpointed iteration driver and CLI |
| `klent.graft` | Measured conversion of a pre-critic-tail checkpoint |
| `klent.telemetry` | SQLite writes, queries, and CLI |
| `klent.search` | Gumbel line search used by Python evaluation |
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

Convert a checkpoint trained before the critic tail, measuring the conversion:

```sh
uv run python -m mantisnet.klent.graft OLD.pt NEW.pt \
  --tau 0.1 --lam 0.01 --manifest graft.json
```

Inspect telemetry and run benchmarks:

```sh
uv run python -m mantisnet.klent.telemetry --run runs/readme-smoke summary
uv run python bench/bench_forward.py
uv run python bench/bench_loop.py sweep --device cpu --stones 20 --cohorts 16 --iters 1
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
- The action-value head reads a critic-private tail over the trunk's window
  rows and global token; the policy and state-value heads read the trunk's own
  rows.
- Each cell head therefore runs its own incidence aggregation, and owns its
  projection, embeddings, and output MLP.
- Action values are scalar and bounded to `(-1, 1)` by `tanh`.
- The policy, action-value, and critic-tail output layers initialize to zero,
  so a fresh model's cell heads are zero and its critic tail is the identity.
- KLENT improvement consumes raw policy logits and scalar action values.
- The KLENT fit objective trains policy cross-entropy and the taken action's
  squared return error.
- KLENT fitting does not train or read the distributional state-value head.
- Stored training states are move prefixes and are rebuilt through `hexo_py`.
- Capped self-play episodes are excluded from the fitting buffer.
- `games` is a completed-game quota; `envs` is the number of persistent slots.
- Fit and collection batches obey separate attention-pair and legal-cell
  budgets.
- Checkpoints contain model state, optimizer state, completed iteration, and
  NumPy RNG state.
- `klent.graft` is the only path from a pre-critic-tail checkpoint: it adds the
  tail's keys, remaps Adam state by parameter name, and refuses to write unless
  every parent tensor still equals the source file's bit for bit and a seeded
  probe set shows the grafted action values are the parent's.
- The graft manifest records the count and SHA-256 of the parent tensors it
  carried unchanged alongside the probe's measurements.
- A fresh run refuses a nonempty output directory.
- `--resume` requires a checkpoint and is exclusive with `--init-from`.
- `config.json` records current run fields; `invocations.jsonl` records every invocation and resume.
- `metrics.jsonl` is append-only; replay after resume can repeat an iteration number.
- `telemetry.db` is schema-versioned and stores run, iteration, game, ply,
  evaluation, and cross-play data.
- `status.json` is the deck heartbeat and phase-progress surface.
- `STOP` requests shutdown after the current iteration and a durable checkpoint.
- `CHECKPOINT` requests a checkpoint at the next commit point.
- Deck queries open telemetry read-only; deck-owned state is in `runs/deck.db`.
- The inference cache holds at most its configured checkpoint capacity.
- CUDA compilation is enabled by the CLI unless `--no-compile` is set.
