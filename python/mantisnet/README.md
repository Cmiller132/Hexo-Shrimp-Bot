# mantisnet

## Purpose

`python/mantisnet` requires Python 3.12 or later and contains the MantisNet
Torch model, independent graph-builder oracle, losses, Hexo KLENT training loop,
shared fitting engine, frozen-corpus laboratory, telemetry store, and
control-deck backend. The training path uses a policy head and one
action value per legal cell, composed from a three-outcome categorical critic,
and the committed-mass-normalized acting score π′ ranks by.
Algorithm obligations are in `docs/KLENT_FOR_HEXO.md`; measured experiment
outcomes are in `docs/ABLATIONS.md`.

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
| `MODEL_REPR_VERSION`, `NUM_PATTERNS`, `DEC_CLASSES` | Representation constants |

Core modules:

| Module | Contract |
| --- | --- |
| `builder` | Graph construction, Rust batch conversion, and collation |
| `model` | Stone/window trunk, policy decoder, action-value decoder and its composition, value head |
| `attention` | Fused attention operation plus reference path |
| `decoder` | Shared legal-cell incidence aggregation |
| `segments` | Ragged segment reductions |
| `losses` | Policy and distributional value losses |
| `fitloop` | Loss-agnostic packed epoch engine shared by KLENT and lab cells |
| `lab` | Frozen corpora, supervised cells, evaluation, reports, benchmarks, profiles, probes, and contract checks |
| `lab.families` | Structural checkpoint-family registry, config inference, slot-table expansion, and historical critic composition |
| `klent.improve` | Closed-form improved policy |
| `klent.selfplay` | Persistent-slot collection and samples |
| `klent.train` | Network evaluation, packing, collection, and fit |
| `klent.graft` | Slot-class/scalar to joint-class/trinomial conversion, measured and refusing |
| `klent.trigraft` | Joint-class/scalar to trinomial conversion, measured and refusing |
| `klent.run` | Checkpointed iteration driver and CLI |
| `klent.telemetry` | SQLite writes, queries, and CLI |
| `klent.search` | Gumbel line search used by Python evaluation |
| `klent.opponents` | Opponent seam, SealBot and native-seat adapters, `shared_openings`, `wilson`, `elo` |
| `klent.seat` | Shared strict §3.1 subprocess client, participant format, wire codec, and response validation |
| `klent.evaluate` | Policy argmax and the two-chooser lockstep match loop |
| `klent.headtohead` | Paired cross-run checkpoint-vs-checkpoint match and CLI |
| `klent.crossplay` | Round-robin referee for independent §3.1 subprocess seats |
| `deck.app` | FastAPI routes and SPA serving |
| `deck.service` | Run registry, inference cache, and play sessions |
| `deck.state` | Deck-owned reviews, probes, presets, and match jobs |

`MantisNet.forward(batch, mass_floor)` returns raw policy logits, acting scores,
composed action values,
distributional state value outputs, and the decoded state value. KLENT
collection and fitting call `trunk` plus the cell heads and do not consume the
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

Convert one slot-class/scalar-critic checkpoint to this build's composed
joint-class/trinomial representation. The evidence sidecar is written beside the
new checkpoint with a `.json` suffix:

```sh
uv run python -m mantisnet.klent.graft \
  runs/source/checkpoint_000151.pt runs/forked/checkpoint_000151.pt \
  --tau 0.1 --lam 0.01
```

Convert a v2 joint-class scalar checkpoint without changing its decoder:

```sh
uv run python -m mantisnet.klent.trigraft \
  --old runs/joint-939/checkpoint_000300.pt \
  --new runs/grafts/joint939-300-tri.pt \
  --tau 0.1 --lam 0.01 --mass-floor 0.2
```

Inspect telemetry and run the consolidated laboratory modes:

```sh
uv run python -m mantisnet.klent.telemetry --run runs/readme-smoke summary
uv run python -m mantisnet.lab --help
uv run python -m mantisnet.lab bench forward --device cpu --batch 16 --iters 1
uv run python -m mantisnet.lab bench sweep --device cpu --depths 20 --cohorts 16 --iters 1
uv run python -m mantisnet.lab smoke
```

Freeze a run's completed self-play games and train three supervised seeds:

```sh
uv run python -m mantisnet.lab freeze \
  --run runs/joint-mnorm --iters 100 149 --name mnorm-late-v1
uv run python -m mantisnet.lab train \
  --corpus mnorm-late-v1 --sweep representation-v1 --variant mantis \
  --model-kw h=96 blocks=6 heads=4 --seeds 3 --device cpu
uv run python -m mantisnet.lab evaluate \
  --cell runs/lab/representation-v1/mantis+blocks6+h96+heads4/s0 \
  --corpus mnorm-late-v1 --device cpu
uv run python -m mantisnet.lab report --sweep runs/lab/representation-v1
```

Anchor in-driver evaluation against one native §3.1 seat by putting exactly one
normal participant entry in `eval-seat.json`:

```json
[
  {
    "id": "foreign-anchor",
    "command": ["python", "-u", "foreign_seat.py"],
    "hello": {
      "checkpoint": "weights/foreign.bin",
      "variant": "search:visits=64"
    }
  }
]
```

```sh
uv run python -m mantisnet.klent.run \
  --out runs/seat-anchored \
  --iterations 100 \
  --games 64 \
  --envs 64 \
  --eval-every 25 \
  --eval-games 64 \
  --eval-seat eval-seat.json \
  --sealbot /path/to/SealBot
```

The participant ID, `command` argv, `hello.checkpoint`, and `hello.variant` are
all required and are not filled from the environment. A positive
`--eval-every` requires `--eval-seat`, `--sealbot`, or both. When both are
present, `--eval-games` applies to each opponent and both receive the same
opening/model-RNG schedule. Each metrics row stores an attributed
`eval_results` entry per opponent, and telemetry stores one match row per
opponent.

Compare two checkpoints from different runs, paired:

```sh
uv run python -m mantisnet.klent.headtohead \
  --a runs/one/checkpoint_001000.pt \
  --b runs/two/checkpoint_001000.pt \
  --pairs 64 --sims 32 --tau 0.1 --lam 0.01 \
  --out h2h.json
```

Run a seat-protocol round robin from a participant list:

```json
[
  {
    "id": "epoch-1000",
    "command": ["hexo-bot", "serve", "--package", "mantisnet"],
    "hello": {
      "checkpoint": "runs/one/native/checkpoint-1000",
      "variant": "policy"
    }
  },
  {
    "id": "foreign-engine",
    "command": ["python", "-u", "foreign_seat.py"],
    "hello": {
      "checkpoint": "weights/foreign.bin",
      "variant": "search:visits=64"
    }
  }
]
```

```sh
uv run python -m mantisnet.klent.crossplay \
  --participants participants.json \
  --pairs 32 \
  --anchor epoch-1000=0 \
  --cap 512 \
  --out crossplay.json
```

Commands are argv arrays and are launched without a shell. Participant IDs are
referee keys, not seat identities; the seat supplies its name, package version,
optional encoder version, resolved variant, digest, and optional restriction in
`welcome`. At least one repeated `--anchor ID=RATING` is required. Ratings and
anchor values are Bradley–Terry natural log-odds.

The output is one strict, atomically replaced JSON manifest. It records the
exact `hello` and `welcome` for every participant, every game's opening, seats,
plies and adjudication, paired statistics for every unordered pairing, a
row-perspective full matrix, and anchored Bradley–Terry ratings with standard
errors. Listing the same serve command with each run checkpoint is the
within-run checkpoint sweep; there is no checkpoint-scanning execution path.

Run the deck backend directly:

```sh
uv run uvicorn mantisnet.deck.app:app --host 0.0.0.0 --port 8000
```

## Connections

- [`docs/MODEL_SPEC.md`](../../docs/MODEL_SPEC.md) defines the model and
  representation contract.
- [`docs/KLENT_FOR_HEXO.md`](../../docs/KLENT_FOR_HEXO.md) defines KLENT's Hexo
  adaptation and training obligations.
- [`docs/LAB_SPEC.md`](../../docs/LAB_SPEC.md) defines frozen corpora,
  supervised cells, metrics, artifacts, and measurement modes.
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
- The critic emits positive, negative, and zero logits per legal cell. Their
  fp32 softmax gives `Q = p_pos - p_neg` and `M = 1 - p_zero <= 1`.
- The policy and action-value output layers initialize to zero, so initial
  policy logits and action values are exactly zero.
- KLENT improvement ranks by Q over the position's floored maximum committed
  mass and averages unscaled Q for v-hat.
- The KLENT fit objective trains policy cross-entropy plus the taken action's
  categorical cross-entropy against `(G⁺, G⁻, 1−|G|)`.
- `critic_ce` reports that trained critic term; `q_loss` is a detached measured
  diagnostic. Neither has a dedicated telemetry column.
- A slot-class/scalar-critic checkpoint loads only after
  `python -m mantisnet.klent.graft` applies both representation changes; that
  conversion refuses to write unless the exact joint checks and the measured
  trinomial checks all hold.
- KLENT fitting does not train or read the distributional state-value head.
- Stored training states are move prefixes and are rebuilt through `hexo_py`.
- Capped self-play episodes are excluded from the fitting buffer.
- `games` is a completed-game quota; `envs` is the number of persistent slots.
- Fit and collection batches obey separate attention-pair and legal-cell
  budgets.
- Corpus freezes open source telemetry strictly read-only, refuse a schema
  mismatch, replay-verify every sampled game, and pin rules and action order.
- Frozen-corpus loading verifies the archive SHA and version pins before
  exposing samples.
- A lab cell is fresh-only: a nonempty cell directory is refused and there is
  no resume path. Every score identifies the exact corpus SHA.
- Every lab `--checkpoint` consumer uses the family registry. Shipped scoreable
  families are `trinomial-joint`, `bipolar-joint`, `scalar-joint`,
  `scalar-slot`, `bipolar-slot`, and `factored-slot`; the shape-identical two-row
  slot families require `--family`. `tail-slot` and `duel-slot` are identified
  by name and refused until they have runnable composition-parity evidence.
- Checkpoints contain model state, optimizer state, completed iteration, and
  NumPy RNG state.
- A fresh run refuses a nonempty output directory.
- `--resume` requires a checkpoint and is exclusive with `--init-from`.
- `config.json` records current run fields; `invocations.jsonl` records every invocation and resume.
- `metrics.jsonl` is append-only; replay after resume can repeat an iteration number.
- `telemetry.db` is schema-versioned and stores training and in-driver
  evaluation data, with one match row per configured opponent. Seat crossplay
  writes its standalone result manifest, not run telemetry.
- `status.json` is the deck heartbeat and phase-progress surface.
- `STOP` requests shutdown after the current iteration and a durable checkpoint.
- `CHECKPOINT` requests a checkpoint at the next commit point.
- Every evaluation, head-to-head, or crossplay pairing uses shared random
  openings from both seats. The ply cap counts the opening's placements, and a
  capped game scores one half.
- `klent.crossplay` and `SeatOpponent` share the one strict client in
  `klent.seat`; neither imports participant code.
- A `SeatOpponent` batches all newly waiting slots into one lazy `open`; every
  `decide` request names all slots still waiting in that chooser round. A
  declared `restriction_exhausted` refusal forfeits its slot and retries the
  survivors without advancing their move cursors; any other refusal, bad
  response, attestation failure, or dead child aborts the evaluation.
- Seat-opponent strength is attributed by the `welcome` resolved variant and
  digest. Its scored forfeits remain explicit in `eval_results.forfeits`.
- The crossplay referee imports no participant. It holds every authoritative
  `hexo_py.Position` and sends one `decide` per participant per scheduling round
  containing all games then waiting on that seat.
- Crossplay sends the three version constants exported by `hexo_py` to every
  seat independently. It does not require participants' welcome identity,
  encoder, variant, digest, or restriction to agree.
- A slot-local `restriction_exhausted` refusal from a seat that declared a
  welcome restriction, or an illegal action, forfeits that game with its
  complete cause preserved. Any other slot-local fault, a connection refusal,
  malformed response, or dead subprocess aborts the tournament rather than
  becoming a score.
- The crossplay seed fixes referee openings only. §3.1 carries no participant
  RNG seed, so a repeated seed is not a whole-tournament reproducibility claim.
- A head-to-head pair shares its opening and its generator, derived from
  `(seed, pair index)`, and is reproducible on its own; the same seed gives the
  same result.
- A head-to-head reports the paired standard error beside the unpaired one, an
  exact sign test over the decisive pairs, and every capped game in `warnings`.
  Pairs that all carry one `d` have no spread to estimate, so such a match
  reports no Elo interval and warns instead.
- A head-to-head at `--sims 0` never consults `--tau`/`--lam` and records them
  as absent; a searched one requires both.
- `--temperature` scales the root Gumbel for both seats, which is exactly
  drawing the root order from `softmax(logits / T)`. `0` searches
  deterministically and `1` is the unscaled draw. It is recorded in the
  manifest, because two temperatures are two measurements, and it is refused at
  `--sims 0`, where no Gumbel is drawn to scale.
- A head-to-head refuses two checkpoints that differ in rules, action order, or
  Torch version, and names `klent.graft` as the bridge for a representation
  difference.
- `wilson` takes a total score and returns rates; `elo` takes a rate.
- Deck queries open telemetry read-only; deck-owned state is in `runs/deck.db`.
- The inference cache holds at most its configured checkpoint capacity.
- CUDA compilation is enabled by the CLI unless `--no-compile` is set.
