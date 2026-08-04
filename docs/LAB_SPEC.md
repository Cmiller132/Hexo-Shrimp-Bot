# MantisNet laboratory specification

**Status: normative for `mantisnet.lab`.** This document defines frozen
supervised corpora, lab cells, checkpoint scoring, aggregation, and the shared
measurement commands. Model semantics remain owned by
[`MODEL_SPEC.md`](MODEL_SPEC.md); the production fitting and improvement
semantics remain owned by [`KLENT_FOR_HEXO.md`](KLENT_FOR_HEXO.md).

The public command is run from `python/mantisnet`:

```text
python -m mantisnet.lab COMMAND ...
```

Lab artifact formats are not backward compatible. Readers accept the one
format named here and refuse all other versions.

## 1. Frozen corpus format v1

A corpus directory contains exactly the durable pair `manifest.json` and
`corpus.npz`. The conventional location is `runs/corpora/<name>/`.

### 1.1 Source and selection

One freeze reads one run's `telemetry.db`. The SQLite connection uses URI
read-only mode (`mode=ro`) and the stored telemetry schema must equal the
running build's `telemetry.SCHEMA_VERSION`. Freeze neither creates nor mutates
the source database.

Selected rows satisfy all of:

- `kind = 'selfplay'`;
- `winner IS NOT NULL`, which excludes capped and evaluation games; and
- `iteration` lies in the requested inclusive interval.

An empty selection is an error. Moves are decoded only by
`telemetry.unpack_moves`.

Games, not positions, are shuffled and partitioned. The default fractions are
0.90 train, 0.05 validation, and 0.05 test. A recorded split seed makes the
shuffle deterministic and no game may occur in two splits.

Within each split, positions are sampled uniformly without replacement from
all legal prefix lengths `t` in `[0, game_length)`. The default requested
counts are 1,000,000 train, 100,000 validation, and 100,000 test positions. A
split with fewer available plies contributes all of them; requested and
realized counts are both recorded.

### 1.2 Replay verification

Freeze replays every sampled game once with `hexo_py.Position`. At a sampled
prefix it records the rank of the game's next move in `Position.legal_moves()`
and refuses a move absent from that list. After the final move, the replay must
be terminal and its winner must equal telemetry's winner.

Each sampled prefix stores:

| Field | Type | Definition |
| --- | --- | --- |
| `game` | int32 | Index into the corpus game arrays |
| `t` | int32 | Number of moves already applied |
| `rank` | int32 | Played move's engine-legal-order rank at `t` |
| `mover` | int8 | `current_player` at `t` |
| `z` | int8 | `+1` when `mover == winner`, otherwise `-1` |
| `dist` | int32 | `game_length - t` |

### 1.3 `corpus.npz`

The compressed NumPy archive has these game-level arrays:

| Array | Type and shape | Definition |
| --- | --- | --- |
| `moves` | int16 `(total_plies, 2)` | All `(q, r)` moves, concatenated |
| `offsets` | int64 `(n_games + 1,)` | CSR boundaries into `moves` |
| `winner` | int8 `(n_games,)` | Terminal engine winner |
| `source_game_id` | int64 `(n_games,)` | Telemetry primary key |
| `split` | int8 `(n_games,)` | Train/validation/test as 0/1/2 |

For each split prefix `train_`, `val_`, and `test_`, it also contains the six
sample arrays `game`, `t`, `rank`, `mover`, `z`, and `dist` with the types in
§1.2. No materialized position or graph is stored. Replay and
`collate_prefixes` happen while a fitting chunk is prepared on the prefetch
worker.

Archive construction is deterministic: identical source rows, seeds, window,
fractions, and requested counts produce byte-identical `corpus.npz` bytes.

### 1.4 `manifest.json`

The strict manifest records:

- `format_version`, exactly `1`, and the corpus `name`;
- UTC `created` time;
- `source`: run directory, telemetry schema version, inclusive iteration
  window, and selection predicate;
- `split`: seed, fractions, and realized game and ply counts per split;
- `samples`: sampling seed plus requested, available, and realized counts per
  split;
- `RULES_VERSION` and `ACTION_ORDER_VERSION` from `hexo_py`; and
- `corpus_sha256`, the SHA-256 digest of the complete `corpus.npz` file.

Loading verifies the archive digest, format, rules version, and action-order
version before exposing arrays. An action-order change invalidates ranks even
when the rules themselves do not change. A rules change also invalidates the
replayed game corpus. There is no conversion path between corpus versions.

## 2. Variants and parameter budgets

The variant registry maps a name to a factory, collation path, description,
and whether that path uses the shared Rust encoder. Format v1 ships only the
`mantis` variant: `MantisNet(MantisConfig(**model_kw))` with Rust
`collate_prefixes`.

`--model-kw key=value ...` is typed against the `MantisConfig` dataclass.
Unknown, duplicate, malformed, or incorrectly typed keys are errors. The
registry does not contain placeholder representations.

When a parameter budget is supplied, construction counts every model
parameter before training. A count outside the inclusive interval
`budget * (1 ± tolerance)` refuses the cell and reports the count and bounds.

## 3. Supervised cell recipe

One cell is one variant and one seed. The conventional directory is
`runs/lab/<sweep>/<cell>/s<seed>/`. Seeds run sequentially. A fresh cell
refuses a nonempty directory and has no resume operation.

The seed is applied with `torch.manual_seed` before model construction and to
a separate `numpy.random.default_rng` for epoch order. Adam is constructed over
`model.parameters()` at learning rate `1e-3`. The learning-rate schedule is
`constant` unless the recipe selects `cosine`, which anneals from the full rate
at epoch 1 toward zero past the final epoch; the applied rate is recorded in
each metrics row. Default epochs and effective
batch size are eight and 4096; attention-pair and legal-cell limits are the
production `KlentConfig` fit limits. CUDA uses bf16 autocast when requested;
composition and losses remain fp32. Optional compilation uses one dynamic-shape
graph.

An optional EMA of the weights at decay `ema_decay` is maintained per optimizer
step and saved beside the final checkpoint.

Every sampled position trains all production heads with equal weight:

1. Policy cross-entropy uses a one-hot vector at stored `rank` across the full
   legal set and `losses.policy_loss`.
2. The selected critic row uses categorical cross-entropy against
   `(z positive, z negative, 1 - |z|)`. For this corpus, `z` is decisive and
   the target is either `(1, 0, 0)` or `(0, 1, 0)`.
3. The state-value bin logits use `losses.value_loss` against `z`.

Chunk packing, optimizer grouping, sample weighting, prefetch, and nonfinite
refusal are supplied by `mantisnet.fitloop`, the same epoch engine used by
KLENT. Validation is packed under the production collection budgets and
reports imitation top-1 plus state-value sign accuracy and MAE.

## 4. Cell artifacts

Each cell contains:

| Artifact | Contract |
| --- | --- |
| `config.json` | Variant and normalized overrides, corpus name and SHA, recipe and 1:1:1 loss weights, seed, compatibility versions, parameter count, and lab cell format |
| `metrics.jsonl` | One flushed strict-JSON row per epoch with the applied learning rate, three losses, fit steps, seconds, samples/second, and validation metrics |
| `checkpoint_final.pt` | Model state, variant identity and overrides, corpus SHA, versions, parameter count, and `lab_cell_format: 1` |
| `checkpoint_ema.pt` | EMA model state in the same checkpoint format; present only when the recipe sets `ema_decay > 0` |
| `scores.json` | §5 metric block for this checkpoint and corpus |
| `scores_ema.json` | §5 metric block for the EMA checkpoint; present only when the recipe sets `ema_decay > 0` |

The checkpoint is a lab-cell format, not a production KLENT checkpoint. Lab
loading reconstructs its registered variant and requires exact identity,
versions, parameter count, and corpus digest.

## 5. Evaluation metrics and scores

Evaluation consumes a named corpus split, defaults to `test`, uses no gradient,
and packs inference under the production collection budgets.

Distance buckets partition every sampled `dist` exactly once:

```text
1–4, 5–8, 9–12, 13–16, 17–24, 25–32, 33–48, 49–64, 65+
```

Imitation reports top-1 and top-3 accuracy overall and per distance bucket.
When a legal set has fewer than three actions, top-3 means membership in all
available actions up to three.

A horizon channel reports for each bucket:

- `n`, the number of positions;
- sign accuracy against `z`;
- mean absolute error `mean(|prediction - z|)`;
- mean prediction; and
- mean absolute prediction.

Lab cells expose two channels: the trained scalar `state_value` decode and
`v_hat`. Production KLENT checkpoints expose only `v_hat`; their state-value
head is untrained and must not be scored. For both checkpoint kinds,

```text
v_hat = improved_policy(
    policy,
    compose_acting_q(critic, offsets, mass_floor),
    compose_q(critic),
    offsets,
    tau,
    lam,
).v_hat
```

The defaults are `tau=0.1`, `lam=0.01`, and `mass_floor=0.2`, and every score
records the resolved flags.

`scores.json` identifies its corpus by name and SHA, its split, checkpoint path
and SHA, versions, parameter count, checkpoint kind, variant identity, and
metric blocks. Cell evaluation replaces the cell's `scores.json`; production
checkpoint evaluation requires an explicit output path. Scores always name the
corpus SHA.

## 6. Reports

A report aggregates score files by variant identity across seeds. It reports
parameter count, training samples/second, imitation top-1, and each horizon
channel's sign accuracy by distance bucket as mean plus sample standard
deviation. Inputs with different corpus SHA values or split names are refused.
The command writes `report.json` and emits a plain-text table; it creates no
plot.

## 7. Measurement and contract modes

All modes draw positions from the shared cohort layer: either live nonterminal
positions after lockstep production `Collector` self-play or sampled corpus
prefixes replayed through the engine.

| Command | Contract |
| --- | --- |
| `freeze` | Select, replay-verify, and persist corpus format v1; `--dry-run` writes nothing |
| `train` | Run one or more sequential supervised cells |
| `evaluate` | Produce the common metric block for a lab or production checkpoint |
| `report` | Aggregate compatible scores across variants and seeds |
| `bench forward` | Time Python build, Python collate, Rust build+collate, and model forward |
| `bench collect` | Instrument a real collector by phase with bindings restored on every exit |
| `bench fit` | Time production KLENT fitting or the shared supervised corpus epoch |
| `bench sweep` | Measure collate, forward, and improve/sample over depth and cohort-size cells |
| `profile trunk` | Attribute replicated trunk stages and refuse drift from `model.trunk` |
| `profile decode` | Attribute eager trunk and cell heads, optional compiled total, and decoder kernels |
| `profile seam` | Attribute transfer, forward, and composition/return around network evaluation |
| `profile fit` | Profile a window of real optimizer steps inside the fit engine and bucket kernel self-time by family |
| `mass` | Measure the loaded family's committed mass, Q/M behavior, and acting-floor sensitivity |
| `check` | Run D6, batch-parity, decoder-coverage, and applicable Python/Rust builder contracts |
| `smoke` | Run a tiny synthetic-telemetry freeze, CPU cell, evaluation, and report end to end |

`check` applies all 11 nonidentity D6 transforms and requires value and mapped
policy equality at absolute tolerance `1e-5`. Batched versus single inference
uses `1e-6`. Decoder coverage requires one output per legal cell and identifies
background cells exactly as legal cells belonging to no live window. A variant
declaring the Rust path must match the independent Python builder field for
field before it is eligible for training experiments.

## 8. Production checkpoint families

Every lab command that accepts `--checkpoint` loads through the ordered family
registry. The registry identifies a checkpoint structurally from its complete
model key set, native critic-readout width, and decoder-table row count. A named
`--family` must itself claim the checkpoint. Zero matches refuse with the full
registry and this compatibility contract; multiple matches refuse with all tied
candidates and require `--family`. Stored `RULES_VERSION` and
`ACTION_ORDER_VERSION` must equal the running engine. A differing
`MODEL_REPR_VERSION` or Torch version is recorded but is not a reason to refuse
measurement.

Configuration is inferred from the state-dict tensors, never from current
defaults: `h`, block count, attention heads and distance clamp, FFN factor,
policy/value hidden widths, value-query count, and value-bin count all come from
their corresponding shapes. Dropout is zero for evaluation. Three-row
slot-class decoder tables are expanded at load time to the 93 joint classes;
each joint row copies slot class `min(s, 5-s)` of either member of its reversal
orbit. Native critic readouts are not transformed.

The shipped families and their exact fp32 compositions are:

| Family | Native critic | Decoder table | Q | M | Scoreable |
| --- | --- | --- | --- | --- | --- |
| `trinomial-joint` | 3 logits | 93 joint rows | `softmax(z)[+] - softmax(z)[-]` | `1 - softmax(z)[0]` | yes |
| `bipolar-joint` | 2 logits | 93 joint rows | `sigmoid(z[+]) - sigmoid(z[-])` | `sigmoid(z[+]) + sigmoid(z[-])` | yes |
| `scalar-joint` | 1 logit | 93 joint rows | `tanh(z)` | `abs(Q)` | yes |
| `scalar-slot` | 1 logit | 3 slot rows | `tanh(z)` | `abs(Q)` | yes, after table expansion |
| `bipolar-slot` | 2 logits | 3 slot rows | `sigmoid(z[+]) - sigmoid(z[-])` | `sigmoid(z[+]) + sigmoid(z[-])` | yes, after table expansion |
| `factored-slot` | 2 logits | 3 slot rows | `(2*sigmoid(z_p)-1) * sigmoid(z_m)` | `sigmoid(z_m)` | yes, after table expansion |
| `tail-slot` | 1 logit plus `q_tail.*`, `q_tail_ln.*` | 3 slot rows | unknown in this tree | unknown | no |
| `duel-slot` | 1 logit plus `mlp_qbase.*` | 3 slot rows | unknown in this tree | unknown | no |

`bipolar-slot` and `factored-slot` intentionally tie because their key sets and
shapes are identical. Scoring either therefore requires its explicit family.
`tail-slot` and `duel-slot` are named so refusal is stable and informative:
their private forward semantics live only on retired branches `83e5f13` and
`4c8bed8`, respectively, and no composition-parity test against runnable
historical code is possible in this tree.

For every scoreable family, acting uses
`Q_tilde = Q / max(max_b M(s,b), mass_floor)`. This includes scalar families,
where `M = abs(Q) <= 1`; it is documented semantics, not a special case.
`v_hat` is the result of `improved_policy(policy, Q_tilde, Q, offsets, tau,
lam).v_hat`. Production checkpoints never score the untrained state-value head.

This registry is the checkpoint compatibility boundary. A new critic
parameterization, decoder key, or head format must land with a family entry and
a composition-parity test before its checkpoints are scoreable. Merely having
loadable tensor shapes is insufficient. The named-but-unscoreable tail and duel
families are the normative refusal example.
