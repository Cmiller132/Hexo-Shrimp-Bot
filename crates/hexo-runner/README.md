# hexo-runner

Match orchestration: the authoritative game, and the policy that decides how a
match ends.

**Status: the game state machine is implemented.** Transport, players, records
on disk, and the binary are not.

## Shape

Pure Rust library crate, depends only on `hexo-engine`.

```
crates/hexo-runner/
  Cargo.toml
  README.md
  src/
    lib.rs          # crate root, flat re-exports, PROTOCOL_VERSION
    game.rs         # Game, GameSpec, Step, Transition, PlyRecord, FailurePolicy
    decision.rs     # Reply, Decision, Failure, Budget
    outcome.rs      # MatchResult, WinReason, DrawReason, NoContest
    error.rs        # SubmitError
  tests/
    game.rs         # adjudication, the guards, and both drive shapes
```

## Module map

| Module | Role |
| --- | --- |
| `game` | `Game` owns the one canonical `Position` and is the only code that advances it. A state machine: `step()` says what it wants, `submit()` says what happened. |
| `decision` | What a seat comes back with — a placement, a resignation, or a driver-reported failure — and the `Budget` it was told it had. |
| `outcome` | `MatchResult` and the adjudication vocabulary. Everything the engine refuses to model. |
| `error` | `SubmitError`: a submission the game would not act on. |

There is deliberately no `player.rs` and no `Player` trait. See below.

## Design notes

- **`Game` is a state machine, not a loop.** It never blocks, holds no player
  handle, and has no transport, clock, or I/O. It cannot block because there is
  nobody to block on.

  The obvious alternative is a `Player` trait the runner calls, blocking inside
  `decide` until the seat answers. That makes a game equal to a thread — ten
  thousand self-play games is ten thousand OS threads — and worse, it forecloses
  batching, because every thread is blocked inside its search on a
  single-position evaluation with nothing left to coalesce into the batch a GPU
  wants.

  Inverting it gives both shapes from one type: a fifteen-line loop for one game
  on one thread, or one actor sweeping thousands of `Game` values for everyone
  in `NeedDecision` and handing that whole set to a batched evaluator. This is
  *more* synchronous than the callback design — no `async fn`, no executor, no
  futures, no channels.

- **Players never hold a mutable handle to canonical state.** Enforced by
  ownership, not convention: `Game::position()` returns a shared borrow and
  there is no mutable counterpart. `submit` is the only way to advance.

- **A seat is handed a move prefix, not a position.** `Game::prefix()` is a move
  list the seat replays with `Position::replay`. A container cannot be handed a
  `Position`, and board-shaped construction is the rule-bypass hole the engine
  refuses to reopen.

- **Two guards on every submission, both structural.** `generation` stops a late
  or duplicated reply from playing a move chosen for a position the game has
  moved past — reachable the moment decisions are batched or cross a process
  boundary. The echoed `zobrist` catches a seat whose mirror has drifted, on the
  ply it drifts rather than at the end of a corrupted game.

- **An illegal move is a result, not an error.** `SubmitError` means *nothing
  happened*; the submission was unusable and the same request is still
  outstanding. A seat that plays illegally has lost, and that comes back as a
  `MatchResult`. Putting it in the error type would push adjudication into every
  driver.

  The exception is a refusal the engine reports as its own limit rather than a
  rule violation. `MoveError::is_rule_violation` tells the two apart, and a
  `NoContest::EngineLimit` blames nobody.

- **Three result arms, not two.** The previous implementation had `COMPLETED`
  and `ABORTED`, so a game that legitimately hit its action cap was recorded
  identically to one where a player segfaulted — both unusable as training
  signal and neither distinguishable from the other. `MatchResult` splits by
  whose fault it was, and `is_contested()` is the query that was impossible
  before.

- **Diagnostics are opaque and actually persisted.** A seat attaches bytes; the
  game stores them verbatim and never parses them. The previous runner had this
  field, documented it as reaching the record, and never read it — so every
  model package wrote its own training shards on a path that bypassed the runner
  entirely.

## Not built yet

| Thing | Blocked on |
| --- | --- |
| Wire protocol and transport | C1, C2 — line-delimited stdio over a handshake pinning protocol, rules, and action-order versions |
| On-disk record format | C4. `PlyRecord` is the in-memory shape; nothing serialises it yet |
| The binary and its subcommands | C3 |
| Seed ownership | B4. Deliberately absent rather than decorative — the previous runner carried a seed that reproduced nothing |

## Connections

- Depends on `hexo-engine` for rules and state.
- A driver — in-process, subprocess, or container — sits between `Game` and the
  seats. The runner never learns what a model is.
