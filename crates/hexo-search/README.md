# hexo-search

## Purpose

`hexo-search` defines the evaluator boundary and nonblocking decision sessions
used for batched model play. It provides direct-policy, PUCT, and Gumbel
sequential-halving sessions without owning a model, feature encoder, evaluator,
selector policy, thread pool, or device. Drivers interleave sessions and batch
the leaf evaluations they request.

## Public surface

The crate root re-exports:

| Item | Contract |
| --- | --- |
| `Evaluation` | Priors plus side-to-move value for one position |
| `Encoder` | Appends one position encoding to a byte arena |
| `Evaluator` | Appends one answer per encoded batch item |
| `EncodedBatch` | Reusable ragged byte arena and offsets |
| `DecisionSession` | `begin` / `pump` / `resume` / `take_decision` / `reseed` |
| `LeafId` | Opaque session-scoped evaluation request ID |
| `SessionStatus` | `AwaitingEvals` or `Decided` |
| `PolicySession` | One root evaluation per decision |
| `MctsConfig`, `MctsSession` | PUCT tree with bounded visits and in-flight leaves |
| `GumbelConfig`, `GumbelSession`, `GumbelTrace` | Sequential-halving line search, including root Gumbel temperature |
| `Child`, `SearchOutcome` | Completed root-search view |
| `SelectFromPolicy`, `SelectFromSearch` | Package-owned decision policies |
| `SplitMix64` | Session sampling stream |

The nonblocking drive contract is:

1. Call `begin(&position)` on a live `Position`.
2. Call `pump`, encoding each `(LeafId, &Position)` in its callback.
3. Batch encoded items and call one `Evaluator::evaluate`.
4. Deliver each answer with `resume`.
5. Repeat until `SessionStatus::Decided`.
6. Obtain the complete runner `Decision` with `take_decision`.

`EncodedBatch` provides `new`, `with_capacity`, `push_with`, `push_bytes`,
`item`, `iter`, `bytes`, `offsets`, `len`, `is_empty`, and `clear`.

`Evaluation.priors[i]` corresponds to `Position::nth_legal(i)`.
`Evaluation.value` is from the current player at the evaluated leaf.

`GumbelConfig::temperature` is the finite, nonnegative scale applied to every
root Gumbel draw before it is added to the root log prior. For positive
temperature, ranking `log_prior + temperature * gumbel` draws the root order
from `softmax(log_prior / temperature)`. `0.0` removes root noise and makes
the prior order deterministic, while `1.0` leaves every draw unscaled. The
implementation draws first and scales second, so changing temperature does not
change the RNG stream and the committed cross-language fixtures remain
bit-identical at `1.0`.

## Run / test

From the repository root:

```sh
cargo test -p hexo-search
cargo test -p hexo-search --test topology
cargo test -p hexo-search --test policy
cargo test -p hexo-search --test mcts
cargo test -p hexo-search --test gumbel
cargo doc -p hexo-search --no-deps
```

Verify the committed Python/Rust Gumbel fixture from `python/mantisnet`:

```sh
uv run python ../../crates/hexo-search/tests/fixtures/regenerate_gumbel.py --check
```

Run all workspace gates:

```sh
cargo xtask verify
```

## Connections

- `crates/hexo-engine` supplies position reads, canonical action order, and
  reversible `Search`.
- `crates/hexo-runner` supplies the complete `Decision` output.
- `crates/hexo-model` binds package-owned encoders, evaluators, and sessions.
- `crates/models/mock` and `crates/models/mantisnet` provide concrete
  selectors and evaluators.
- `crates/hexo-bot` pumps sessions on workers and owns the device batcher.
- `python/mantisnet/mantisnet/klent/search.py` supplies the independent Gumbel
  parity implementation.

## Invariants & gotchas

- `DecisionSession` is `Send`, object-safe, and nonblocking.
- `begin` accepts a `Position`, so a seat can search its mirror without owning
  or constructing a runner `Game`.
- A session owns its search-position copy and never mutates the caller's
  position.
- The position passed to `pump` exists only for the callback duration; encode it
  before returning.
- `LeafId` values are never reused within a session.
- Delivering an unknown or already-consumed `LeafId` is a contract violation.
- Priors must be finite, nonnegative, in canonical legal order, and match the
  leaf's legal count.
- Value must be finite and within `[-1, 1]` from the leaf's side to move.
- Consecutive tree plies may have the same mover; backup signs compare players
  and do not use depth parity.
- Terminal descendants are resolved by the engine and do not request network
  evaluation.
- A terminal root is invalid input to `begin`.
- `MctsConfig.visits` and `max_in_flight` are nonzero.
- `MctsConfig.c_puct` must be finite and nonnegative.
- The MCTS root evaluation is outside the descent visit budget.
- Virtual loss remains attached to an in-flight path until its answer resumes.
- Gumbel search counts line-deepening simulations; its root evaluation is
  outside the simulation budget.
- `GumbelConfig::temperature` must be finite and nonnegative; zero is
  deterministic and one preserves the unscaled Python/Rust fixture.
- Gumbel candidate and survivor ties preserve canonical root order.
- Selectors author the action and optional diagnostics; sessions do not supply
  a default selection policy.
- Selector legality is adjudicated by the runner.
- `EncodedBatch` is ragged and encoders may only append to its arena.
- `Evaluator::evaluate` must append exactly one answer per batch item, in batch
  order.
- `reseed` replaces the session's sampling stream without changing its search
  configuration.
