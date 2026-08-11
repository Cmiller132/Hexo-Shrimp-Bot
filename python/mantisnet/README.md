# mantisnet

Python 3.12+. The MantisNet package contains a board-game neural network
architecture, the KLENT reinforcement-learning training loop, a frozen-corpus
supervised laboratory harness, and the Shrimp Control Deck telemetry dashboard. Algorithm obligations are in
`docs/KLENT_FOR_HEXO.md`; measured experiment outcomes are in
`docs/ABLATIONS.md`.

## The model: MantisNet

MantisNet is a graph network whose nodes are stones, live win windows, and one
global token. It has no cell grid, no coordinate inputs, and no empty-cell
nodes.

A **window** is six consecutive cells along one hex axis. A window is **live**
when it contains at least one stone and stones of only one colour; by default
mixed windows are excluded. The transient `mixed_windows` config knob
(MANTIS_GRAFT_SPEC Step 12) widens the scope to every nonempty candidate
window under ternary slot patterns (empty/own/opponent, 377 canonical
classes) with correspondingly wider joint slot-class tables (726 decoder,
1458 incidence); batches must be built with the model's scope, and the trunk
refuses a mismatch. The input representation is D6-invariant by construction
under either scope: every input -- stone colour (own/opponent relative to the
side to move), window pattern, joint slot classes, hex-distance buckets, and
`moves_remaining` -- is invariant under the twelve board symmetries.

The **trunk** interleaves bipartite message passing (stones to/from windows)
with self-attention over the stone set biased by hex distance. A cell-pass
relay lets windows exchange state through shared empty cells, and a
window-attention layer (the live `window_attention` knob, on by default)
types window pairs by colinear/crossing relations.

Three heads read the trunk output:

- **Policy decoder**: one raw logit per legal cell, routed through live windows
  or a background nearest-stone path.
- **Action-value decoder**: three categorical logits per legal cell (positive,
  negative, zero return), composed into Q in (-1, 1) and committed mass M.
  Acting ranks by the mass-normalized score Q-tilde.
- **State-value head**: multi-query attention over window embeddings, outputting
  a binned distribution over [-1, 1] decoded to a scalar.

Policy and action-value decoders share a parameter-free incidence aggregation
pass but own separate projections, embeddings, and output MLPs. All three
heads produce outputs in engine legal-move order.

### Named parameters

| Symbol | Meaning | Default |
|---|---|---|
| H | embedding width | 128 |
| B | trunk blocks | 4 |
| A | attention heads | 4 |
| F | FFN expansion factor | 2 |
| D_MAX | hex-distance clamp | 12 |
| Q | value-readout queries | 4 |
| K | value bins (odd) | 65 |
| P_H | policy/action-value MLP hidden width | 128 |
| V_H | value MLP hidden width | 128 |

### Batching

Positions batch by concatenation with per-position index offsets. Message
passing never crosses positions; attention is masked block-diagonal. The
builder emits stone tables, window tables with identities, incidence lists with
joint slot classes, legal-cell decoder tables, and `moves_remaining`. All index
tensors are precomputed; the forward contains no data-dependent index
discovery.

### Versioning

`MODEL_REPR_VERSION` (model-owned, currently 3) covers the builder and every
feature encoding. `ACTION_ORDER_VERSION` (engine-owned) governs legal-move
indexing. Either bump invalidates checkpoints.

## KLENT training

The `mantisnet.klent` package trains the model's trunk, policy head, and
categorical action-value head through the KLENT algorithm. The state-value head
is not trained by KLENT.

### Iteration loop

Each iteration collects a buffer of on-policy self-play, fits one epoch against
it, then discards it. `run.py` drives the loop, persisting checkpoints,
telemetry, and metrics. A run directory contains `config.json`,
`invocations.jsonl`, `metrics.jsonl`, `telemetry.db`, periodic checkpoints,
and `status.json`.

Sentinel files control the driver: `STOP` requests a checkpoint and clean exit
after the current iteration; `CHECKPOINT` requests a checkpoint at the next
commit point without stopping.

### Collection

`Collector` maintains a persistent cohort of environment slots. Slots advance
in lockstep and restart from the empty board when a game ends. Each ply records
the legal rank, improved-policy vector, acting-time v-hat, mover, phase, and
four improved-policy diagnostics. Capped episodes have no winner and contribute
no training samples.

### Improvement and returns

The closed-form improvement step ranks legal actions by the mass-normalized
score Q-tilde and computes the improved policy and v-hat:

    pi'(a|s) proportional to exp[(Q-tilde(s,a) + tau * log pi_theta(a|s)) / (tau + lambda)]
    v-hat(s) = E_{a ~ pi'}[Q(s,a)]

Lambda-returns follow mover-change signs: +1 at a FirstStone ply (same mover
places again), -1 at Opening or SecondStone plies (turn handover).

### Fitting

The objective is policy cross-entropy against pi' plus the taken action's
categorical cross-entropy against `(max(G,0), max(-G,0), 1-|G|)`. The reported
Q squared error is a detached diagnostic. Fitting uses `mantisnet.fitloop`,
the shared packed epoch engine.

### Evaluation

In-driver evaluation runs seat-balanced matches against configured opponents
at a configurable iteration cadence. `evaluate.py` provides the policy-argmax
chooser and the two-chooser match loop. `search.py` provides batched Gumbel
sequential-halving search for evaluation-time policy improvement.

### Opponents

`opponents.py` defines the opponent seam. Concrete adapters:

- **SealBot**: an independent C++ alpha-beta bot, accessed through a subprocess
  adapter with its own rules oracle.
- **SeatOpponent**: a native subprocess seat following the wire protocol,
  batching all waiting slots into each request.

`shared_openings` generates seat-paired opening schedules. `wilson` and `elo`
compute confidence intervals and ratings.

### Checkpoint conversion

`graft.py` converts a slot-class/scalar-critic checkpoint to joint-class/
trinomial representation, applying both representation changes and remapping
Adam moments. `trigraft.py` converts a joint-class scalar checkpoint to the
trinomial critic without changing its decoder.

### Head-to-head and crossplay

`headtohead.py` runs paired checkpoint-vs-checkpoint matches with shared
openings and Gumbel search, reporting paired standard errors, sign tests, and
Elo differences.

`crossplay.py` runs a round-robin tournament between independent subprocess
seats. The referee owns every authoritative position, draws shared openings,
alternates seats, and adjudicates outcomes. Output is one JSON manifest with
per-pairing statistics and anchored Bradley-Terry ratings.

### Telemetry

`telemetry.py` stores and queries run data in `telemetry.db`: iteration
metrics, self-play and evaluation games, ply traces, evaluation matches,
crossplay results, and hardware aggregates. WAL mode permits concurrent
readers. Schema-versioned with explicit refusal on mismatch.

## Shared fitting engine

`fitloop.py` is the loss- and architecture-agnostic packed epoch engine shared
by KLENT and lab cells. It owns epoch mechanics: the packing loop (via
`ChunkCost`), optimizer grouping, pipelined CPU preparation, sample-weighted
accumulation, and the post-step parameter check. Callers own batch
construction, their loss, and -- through `ChunkCost` -- what a chunk of their
architecture costs.

MantisNet's `chunk_cost` reads the limits naming its binding quantities --
padded attention pairs and legal cells -- and ignores the rest.

## Laboratory harness

The `mantisnet.lab` package is a frozen-corpus supervised benchmark and model-
measurement harness. All commands run from `python -m mantisnet.lab`.

### Frozen corpora

A corpus freezes a window of completed self-play games from a run's telemetry
into a durable pair: `manifest.json` and `corpus.npz`. Games are shuffled and
partitioned into train, validation, and test splits. Positions are sampled
uniformly from legal prefix lengths. Every sampled game is replay-verified
through `hexo_py.Position`. Loading verifies the archive SHA, format version,
rules version, and action-order version.

### Supervised cells

One cell is one variant and one seed. The recipe trains all production heads
with equal weight using policy cross-entropy, categorical critic
cross-entropy, and (where the architecture holds it) state-value cross-entropy.
Chunk packing, optimizer grouping, and prefetch are supplied by
`mantisnet.fitloop`. An optional EMA of the weights is maintained alongside.

### Variant registry

The variant registry maps a name to a factory, collation path, configuration
dataclass, and description. `mantis` is MantisNet with the Rust prefix builder.
Configuration overrides are typed against the chosen variant's dataclass.

### Evaluation and reports

Evaluation produces imitation top-1 and top-3 accuracy overall and per
distance-from-end bucket, plus per-bucket sign accuracy, MAE, and mean
prediction for the v-hat and (where present) state-value channels. Reports
aggregate scores across seeds, with optional per-seed paired differencing
against a named baseline arm.

### Checkpoint families

The family registry identifies a checkpoint structurally from its model key
set, native critic-readout width, and decoder-table row count. Configuration
is inferred from state-dict tensor shapes, including the live Step 12 knobs
(`mixed_windows` from the window-table rows, `window_attention` from the
presence of §5.1c tensors). Shipped scoreable families: `trinomial-joint`,
`bipolar-joint`, `scalar-joint`, `scalar-slot`, `bipolar-slot`, and
`factored-slot`. Slot-class decoder tables are expanded to 93 joint classes
at load time; mixed-scope checkpoints carry the ternary joint tables
natively.

### Measurement commands

| Command | Purpose |
|---|---|
| `freeze` | Select, replay-verify, and persist a frozen corpus |
| `train` | Run one or more supervised cells |
| `evaluate` | Produce the metric block for a lab or production checkpoint |
| `report` | Aggregate scores across variants and seeds |
| `bench forward` | Time build, collate, and model forward |
| `bench collect` | Instrument a real collector by phase |
| `bench fit` | Time production fitting or a supervised epoch |
| `bench sweep` | Measure collate, forward, and improve over depth and cohort cells |
| `profile trunk` | Attribute trunk stages |
| `profile decode` | Attribute decoder heads and kernels |
| `profile seam` | Attribute transfer, forward, and composition around network evaluation |
| `profile fit` | Profile optimizer steps and bucket kernel self-time |
| `mass` | Measure committed mass, Q/M behavior, and acting-floor sensitivity |
| `check` | Run D6, batch-parity, decoder-coverage, and builder contracts |
| `smoke` | Tiny end-to-end freeze, CPU cell, evaluation, and report |

## Control deck

The Shrimp Control Deck is a LAN-served dashboard over run telemetry,
checkpoint inspection, play, and match control. It consists of
`mantisnet.deck`, a FastAPI service, and `frontend/`, a Vite + React SPA. The
supported serving environment is Docker.

### Architecture

The deck service binds one port. The API serves the built SPA as static files.
`frontend/` development uses Vite's dev server with an `/api` proxy.

```
browser -- :8000 -- /            static frontend build
                 -- /api/...     REST (JSON)
                 -- /api/runs/{run}/events   SSE
```

### Run registry and lifecycle

A run is a directory under `runs/` containing `config.json`. States: `active`,
`stopped`, `completed`, `starved`. External runs (launched from a terminal)
appear read-only-live with full telemetry. The deck launches training as child
processes and controls them through sentinel files and process signals.

### Telemetry queries

Thin GET endpoints wrap the telemetry module's query helpers: iteration series,
game search, game detail, calibration, blunders, opening atlas, strength
curves, and crossplay matrices. The horizon endpoint returns distance-from-end
buckets for critic sign accuracy.

### SSE

Typed events (`iteration`, `heartbeat`, `eval`, `checkpoint`, `log`,
`lifecycle`) produced by polling the DB, heartbeat file, checkpoint directory,
and child console stream. Clients render event-stream panels and refresh
queries on `iteration`.

### Inference and play

An LRU cache (default capacity 2) of loaded checkpoints serves position
analysis. Play sessions maintain server-side authoritative state through
`hexo_py` replay: human, checkpoint, SealBot, or random seats. Match jobs run
checkpoint-vs-checkpoint or checkpoint-vs-SealBot sets in a background thread,
recording results to telemetry.

### Deck-owned state

`runs/deck.db` stores game tags and review notes, saved lab probes, play
presets, and match queue history. Deck state never touches `telemetry.db`.

## Public surface

The top-level `mantisnet` package exports:

| Item | Contract |
|---|---|
| `MantisConfig`, `MantisNet`, `ModelOutput` | Model configuration, module, and outputs |
| `PositionGraph`, `Batch` | One encoded graph and one ragged model batch |
| `build`, `from_position`, `collate` | Independent Python graph path |
| `collate_positions`, `collate_prefixes` | Shared Rust encoder path |
| `policy_loss`, `value_loss`, `value_target` | Training losses and targets |
| `param_groups` | Decay/no-decay optimizer groups |
| `MODEL_REPR_VERSION`, `NUM_PATTERNS`, `DEC_CLASSES`, `TERN_PATTERNS`, `TERN_DEC_CLASSES`, `TERN_OCC_CLASSES` | Representation constants |

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

Resume an existing run:

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

Run the deck backend:

```sh
uv run uvicorn mantisnet.deck.app:app --host 0.0.0.0 --port 8000
```

## Connections

- [`docs/KLENT_FOR_HEXO.md`](../../docs/KLENT_FOR_HEXO.md) defines KLENT's Hexo
  adaptation and training obligations.
- [`docs/KLENT_PAPER.md`](../../docs/KLENT_PAPER.md) records the source
  algorithm.
- [`docs/ABLATIONS.md`](../../docs/ABLATIONS.md) owns measured run outcomes.
- `python/hexo-py` supplies engine positions and the shared Rust encoder.
- `crates/models/mantisnet` consumes the same checkpoint and encoder semantics
  from the native container.
- `frontend` consumes `mantisnet.deck` over HTTP and SSE.
- `docker` supplies the CUDA training and deck environment.

## File listing

### Root package (`mantisnet/`)

| File | Description |
|---|---|
| `__init__.py` | Public exports: model, builder, losses, representation constants |
| `model.py` | `MantisConfig`, `MantisNet` trunk, policy/action-value/state-value heads, `ModelOutput` |
| `builder.py` | Position-to-graph representation, Rust batch conversion, collation, version constants |
| `attention.py` | Fused coordinate-biased multi-head attention with Triton kernel and reference path |
| `decoder.py` | Shared legal-cell incidence aggregation for policy and action-value heads |
| `message_passing.py` | Fused incidence aggregation for the two trunk message-passing directions |
| `relay.py` | Cell-pass relay: windows exchange state through shared empty cells via Triton segment kernels |
| `window_pairs.py` | Window-pair relation tables (colinear/crossing) and typed window attention |
| `segments.py` | Ragged segment reductions: ids, sum, max, log-softmax over CSR offsets |
| `losses.py` | Policy cross-entropy, distributional value cross-entropy, value target projection, param groups |
| `fitloop.py` | Architecture-agnostic packed epoch engine: chunk packing, prefetch, fitting loop |
| `d6.py` | The twelve D6 board symmetries as maps on axial coordinates |

### KLENT training (`mantisnet/klent/`)

| File | Description |
|---|---|
| `__init__.py` | Package exports: collection, improvement, fitting, evaluation, telemetry |
| `run.py` | Checkpointed iteration driver, CLI, sentinel handling, run artifacts |
| `selfplay.py` | Persistent-slot self-play collector, episode and sample dataclasses |
| `improve.py` | Closed-form KLENT improved policy and v-hat computation |
| `returns.py` | Mover-change signs and lambda-return recursion |
| `train.py` | Network evaluation, packing, collection, and fit for one iteration |
| `evaluate.py` | Policy-argmax chooser and the two-chooser match loop |
| `search.py` | Batched Gumbel sequential-halving line search for evaluation |
| `opponents.py` | Opponent seam, SealBot and SeatOpponent adapters, shared openings, wilson, elo |
| `sealbot.py` | SealBot evaluation CLI, telemetry recording, checkpoint-curve orchestration |
| `seat.py` | Strict subprocess client for independent wire-protocol seats |
| `headtohead.py` | Paired cross-run checkpoint-vs-checkpoint match and CLI |
| `crossplay.py` | Round-robin referee for independent subprocess seats |
| `graft.py` | Slot-class/scalar to joint-class/trinomial checkpoint conversion |
| `trigraft.py` | Joint-class scalar to trinomial critic checkpoint conversion |
| `telemetry.py` | SQLite run telemetry: writes, queries, schema versioning, CLI |
| `inspect.py` | Single-position policy/value inspection via the Rust builder |
| `hardware.py` | Background GPU, process, and host counter sampling for telemetry |

### Laboratory harness (`mantisnet/lab/`)

| File | Description |
|---|---|
| `__init__.py` | Package exports: corpus format version, distance buckets, variant registry |
| `__main__.py` | Command-line interface for all lab commands |
| `corpus.py` | Frozen corpus freeze, load, replay verification, and sample access |
| `train.py` | Supervised lab-cell fitting through the production model and fit engine |
| `evaluate.py` | Packed imitation and outcome-horizon evaluation for lab and KLENT checkpoints |
| `report.py` | Cross-seed score aggregation with optional paired-difference baselines |
| `variants.py` | Variant registry: MantisNet presets with typed config overrides |
| `families.py` | Structural checkpoint-family registry, config inference, slot-table expansion |
| `bench.py` | Benchmarks for building, collection, and fitting |
| `check.py` | D6 invariance, batch parity, decoder coverage, and builder contract checks |
| `cohort.py` | Production-shaped position cohorts from real collection or corpus replay |
| `profile.py` | Stage-attribution profiles over production-shaped cohorts |
| `mass.py` | Committed mass, Q/M behavior, and acting-floor sensitivity probes |

### Control deck (`mantisnet/deck/`)

| File | Description |
|---|---|
| `__init__.py` | Package export: `create_app` |
| `__main__.py` | CLI entry point for the deck service |
| `app.py` | FastAPI routes: REST endpoints, SSE, lifecycle, play sessions, SPA serving |
| `service.py` | Run registry, inference cache, play sessions, match jobs, child-process management |
| `state.py` | Deck-owned SQLite persistence: game reviews, probes, presets, match history |
