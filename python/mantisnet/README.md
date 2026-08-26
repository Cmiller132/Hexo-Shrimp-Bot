# mantisnet

Python 3.12+. The MantisNet package contains a board-game neural network
architecture, the KLENT reinforcement-learning training loop, a frozen-corpus
supervised laboratory harness, and the Shrimp Control Deck telemetry
dashboard. Algorithm obligations are in `docs/KLENT_FOR_HEXO.md`.

## The model: MantisNet

MantisNet is a graph network whose nodes are stones, nonempty win windows, and
four invariant state latents. `cell_latents` gives covered cells persistent
state; `cell_nodes` extends that state to every legal cell. Both consume only
mover-relative cell fields and D6-canonical relative geometry; absolute cell
coordinates never enter the model.

A **window** is six consecutive cells along one hex axis. Every nonempty
candidate window is represented under ternary slot patterns
(empty/own/opponent, 377 canonical classes), with 726 decoder classes and
1458 incidence classes. The input representation is D6-invariant by
construction: every input -- stone colour (own/opponent relative to the
side to move), window pattern, joint slot classes, displacement orbits, and
`moves_remaining` -- is invariant under the twelve board symmetries.

The **trunk** interleaves bipartite message passing (stones to/from windows)
with content-only self-attention over the stone set. A cell-pass
relay — or, in the production configuration, persistent typed cell state —
lets windows exchange state through shared empty cells. In every block the four
latents read the real windows, self-mix, and broadcast back to those windows;
their final normalized mean is the global context consumed by the heads.

With `cell_nodes=True`, uncovered legal cells initialize from the
nearest-stone-distance embedding, while covered cells retain the learned-base
initialization. In every block cells also attend to all stones within radius 8
through exact orbit-48, source-owner, and on-axis classes.
`cell_node_scope="all"` is the default and sends those radius edges to every
legal cell; `"uncovered"` sends them only to cells with no decoder incidence.
The scope filters edges only: every legal cell retains its latent and its
nearest-distance feature. `cell_adjacency=True` is a separate sub-knob adding
directed distance-one cell messages with axis-shared weights, and its
destination cells follow the same scope. Both
`cell_adjacency=True` and the non-default scope are refused when they would be
inert without `cell_nodes`.

`cell_structure=True` (refused without `cell_latents`) gives covered cells a
structured start and the cell stage a nonlinear second residual:

- **Structured init.** `cell_base` stays as the bias every covered cell gets,
  and three invariant static encodings of the cell's own coverage join it: the
  sum over its containing live windows of a learned row per 726-class decoder
  incidence class, its nearest-stone bucket (the same 10-row table uncovered
  cells initialize from), and a bucketed coverage count — how many live
  windows contain the cell, 1..18 folded into eight buckets: 1, 2, 3, 4, 5-6,
  7-9, 10-13, 14-18. The count is the cell's decoder incidence degree, so
  nothing new rides the wire. Uncovered cells are unchanged.
- **Nonlinear update.** The window read and the radius read still apply as
  two additive residuals in sequence, exactly as they do off the knob. On top
  of them one more residual MLP runs over `[LN(c0); read]` — a 2H→H layer,
  ReLU, then H→H — where `c0` is the block's incoming cell state and `read`
  sums the same two projected read outputs the additive path applied. Without
  `cell_nodes` there is no radius read and the second input is the window
  read alone. The window read-back and the adjacency pass are untouched.

Both new static tables and the MLP's output layer are zero at init, so a fresh
knob-on model computes the knob-off function apart from the nearest-stone row
covered cells newly read off an existing table. Zeroing that table makes the
two builds byte-identical, which is what the no-op-at-init test asserts.

Three heads read the trunk output:

- **Policy decoder**: one raw logit per legal cell. Each action's 18
  post-placement windows are gathered, typed by 729 joint `(post, slot)`
  classes, ReLU'd through a shared first layer, summed per cell in fp32, and
  read through a policy-specific extension matrix.
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
| Q | value-readout queries | 4 |
| K | value bins (odd) | 65 |
| P_H | policy/action-value MLP hidden width | 128 |
| V_H | value MLP hidden width | 128 |

### Batching

Positions batch by concatenation with per-position index offsets. Message
passing never crosses positions; attention is masked block-diagonal. The
builder emits stone tables, window tables with identities, incidence lists with
joint slot classes, legal-cell decoder tables, action-row classes and reverse
views, cell invariant fields, radius edges, adjacency edges, and
`moves_remaining`. All index tensors are mandatory. Device-side CSR views are
derived once per forward through opaque operations so compiled execution has
no graph break.

### Versioning

`MODEL_REPR_VERSION` (model-owned, currently 9) covers the builder and every
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

`trigraft.py` converts a joint-class scalar checkpoint to the trinomial
critic without changing its decoder.

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
distance-from-end bucket; per-bucket sign accuracy, MAE, and mean prediction
for the v-hat and (where present) state-value channels; and the three fit
losses scored per sample — policy NLL of the played move, trinomial critic
cross-entropy at the played move, and state-value cross-entropy — overall and
per bucket. A cell's per-epoch validation row carries the same three losses.
Scores are written beside the cell as `scores-<split>.json`
(`scores-<split>-ema.json` for the EMA weights), so one cell can hold val and
test scores at once; `scores_format` 2. Reports aggregate one split's scores
across seeds (mean and sample SD per metric).

### Screen protocol v2

`python -m mantisnet.lab.screen` holds the supervised screen protocol as code:
`train` fits one cell under the protocol's recipe (4 epochs over the whole
realized train split, lean budgets, EMA 0.995); `evaluate` scores a cell on
val and test with raw and EMA weights; `verdict` judges arms against the
fixture. The fixture is the baseline configuration at six seeds, which sets
the per-metric seed SD and the critic/policy correlation; an arm is three
seeds. The composite `S = (2·z_critic + z_policy) / sqrt(5 + 4ρ)`, with each
`z` the arm's mean improvement in the fit loss over the fixture divided by
`sd·sqrt(1/3 + 1/6)`, is unit-normal under no effect. `S ≥ 2` with the policy
no more than one SE worse is `keep`; `S ≥ 2` otherwise is `policy-regressed`;
`S ≤ −2` is `negative`; two cells in a critic basin (optimist or agnostic at
1–4 plies from the end) make the arm `pathology-prone` regardless. Speed and
other benefits are decided outside the module.

### Checkpoint families

The family registry identifies a checkpoint structurally from its model key
set, native critic-readout width, and decoder-table row count. Configuration
is inferred from state-dict tensor shapes; head count needs a per-head
tensor (a cell-stage bias table) or the recorded-config hint. Cell state is one
stage, so an all-cell profile is named `cell_nodes` alone — except under
`cell_structure`, whose own requirement puts `cell_latents` back on. The
latent base, per-block latent cycle, and row encoder are required. Shipped
scoreable families are `trinomial-joint`, `bipolar-joint`, and
`scalar-joint`; older representation families are rejected.

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
| `alias` | Report structural alias groups before/after the action-row inputs |
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
| `MODEL_REPR_VERSION`, `TERN_PATTERNS`, `TERN_DEC_CLASSES`, `TERN_OCC_CLASSES` | Representation constants |

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
| `cell_latents.py` | Typed legal-cell/window attention table derivation |
| `cell_nodes.py` | Radius/adjacency edge plans on the typed cell-attention kernels |
| `attention.py` | Fused block-diagonal multi-head attention (§5.3) with Triton kernels and reference path |
| `row_encoder.py` | Action-row encoder: 729-class post-placement window rows for both decoder heads |
| `window_latents.py` | Fused ragged window-latent read/broadcast cycle for the state latents |
| `optim.py` | Fused-Adam execution policy, recorded and reapplied across checkpoint loads |
| `message_passing.py` | Fused incidence aggregation for the two trunk message-passing directions |
| `relay.py` | Cell-pass relay: windows exchange state through shared empty cells via Triton segment kernels |
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
| `trigraft.py` | Joint-class scalar to trinomial critic checkpoint conversion |
| `telemetry.py` | SQLite run telemetry: writes, queries, schema versioning, CLI |
| `inspect.py` | Single-position policy/value inspection via the Rust builder |
| `hardware.py` | Background GPU, process, and host counter sampling for telemetry |

### Laboratory harness (`mantisnet/lab/`)

| File | Description |
|---|---|
| `__init__.py` | Package exports: corpus format version, variant registry |
| `__main__.py` | Command-line interface for all lab commands |
| `corpus.py` | Frozen corpus freeze, load, replay verification, and sample access |
| `train.py` | Supervised lab-cell fitting through the production model and fit engine |
| `evaluate.py` | Packed imitation, outcome-horizon, and per-sample loss evaluation for lab and KLENT checkpoints |
| `report.py` | Cross-seed score aggregation (mean and sample SD per metric) |
| `screen.py` | Screen protocol v2: the arm recipe, the fixture noise floor, and the composite verdict |
| `variants.py` | Variant registry: MantisNet presets with typed config overrides |
| `families.py` | Structural checkpoint-family registry and config inference |
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
