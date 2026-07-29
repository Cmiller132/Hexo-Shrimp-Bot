# hexo-runner

## Purpose

`hexo-runner` is the authoritative match-orchestration state machine around one
`hexo-engine::Position`. It records accepted plies, issues generation-tagged
decision requests, validates replies, and adjudicates all non-rule outcomes.
It owns no player handles, loop, transport, persistence, search, or model code.

## Public surface

The crate root re-exports:

| Module | Consumer-facing contract |
| --- | --- |
| `decision` | `Budget`, `Decision`, `Failure`, `Reply` |
| `game` | `Game`, `GameSpec`, `FailurePolicy`, `PlyRecord`, `Step`, `Transition` |
| `outcome` | `MatchResult`, `WinReason`, `DrawReason`, `NoContest` |
| `error` | `SubmitError` |

`PROTOCOL_VERSION` versions the runner-level request, reply, and result
semantics used by manifests and record shards.

A driver follows this state machine:

```rust
use hexo_runner::{Game, GameSpec, Reply, Step};

let mut game = Game::new(GameSpec::default());
match game.step() {
    Step::NeedDecision { seat, generation, budget, .. } => {
        # let _ = (seat, generation, budget);
        // Obtain a Reply from the named seat, then call game.submit.
    }
    Step::Finished(result) => {
        # let _ = result;
    }
}
```

`Game` provides:

- `new`, `spec`, and read-only `position`;
- `plies` and `prefix` for accepted placement history;
- `result` and `step` for state observation;
- `submit(generation, reply)` for the only state transition.

`Reply::Place` carries a `Decision`; `Reply::Resign` and `Reply::Failed` carry
non-placement outcomes. `Transition` reports the accepted placement and current
result after submission.

`MatchResult` distinguishes:

- `Decisive { winner, reason }`;
- `Drawn { reason }`;
- `NoContest { reason }`.

## Run / test

From the repository root:

```sh
cargo test -p hexo-runner
cargo test -p hexo-runner --test game
cargo doc -p hexo-runner --no-deps
cargo check -p hexo-runner
```

Run the complete workspace gates:

```sh
cargo xtask verify
```

The crate is a library state machine and has no standalone executable.
Use `hexo-player::Table` for a blocking loop or the `hexo-bot` driver for
batched model sessions.

## Connections

- `crates/hexo-engine` owns legal moves, wins, canonical position state, and
  position hashing.
- `crates/hexo-player` provides a blocking driver around `Game`.
- `crates/hexo-search` authors `Decision` values through nonblocking sessions.
- `crates/hexo-records` serializes `GameSpec`, `MatchResult`, and `PlyRecord`.
- `crates/hexo-bot` owns the concurrent driver and process lifecycle.
- The normative runner obligations are in
  [`docs/CONTAINER_SPEC.md`](../../docs/CONTAINER_SPEC.md).

## Invariants & gotchas

- `Game` owns the single canonical position and exposes no mutable position
  reference.
- `submit` is the only public operation that can advance a game.
- `Game::step` returns either one outstanding decision request or the finished
  result.
- Every decision request carries the current seat, generation, and budget.
- The runner states the budget; the driver and seat enforce it.
- A reply with a stale or future generation is rejected without changing the
  game.
- A placement decision must echo the hash of the position the seat used.
- A hash mismatch is a retryable `SubmitError::Desync`; the game remains live
  at the same generation.
- An illegal placement is adjudicated as a match result, not returned as a
  submission error.
- Engine representation-limit failures produce `NoContest::EngineLimit`.
- `FailurePolicy` controls whether a seat failure forfeits or produces a
  no-contest result.
- Diagnostics are opaque bytes and are stored verbatim in `PlyRecord`.
- `prefix()` decodes the accepted `ActionId` sequence; `Position` itself has no
  history.
- Consecutive accepted plies may belong to the same seat.
- A placement win is adjudicated before the ply cap.
- The ply cap is applied at a completed turn boundary.
- `GameSpec::ply_cap` is nonzero by type.
- `MatchResult::is_contested` classifies verdicts; it does not classify
  suitability as training data.
- Result reasons retain the seat, action, engine error, or desync hashes needed
  to interpret the outcome.
