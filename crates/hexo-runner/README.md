# hexo-runner

Match orchestration: the authoritative game, and the policy that decides how a
match ends.

**Status: the game state machine is implemented.** Transport is not. The rest of
what used to be missing here moved *out* of the crate rather than into it: the
seats are `hexo-player`'s and `hexo-search`'s, the on-disk record format is
`hexo-records`', and the binary is `hexo-bot`.

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
| `game` | `Game` owns the one canonical `Position` and the record of the game, and is the only code that advances either. A state machine: `step()` says what it wants, `submit()` says what happened. |
| `decision` | What a seat comes back with — a placement, a resignation, or a driver-reported failure — and the `Budget` it was told it had. |
| `outcome` | `MatchResult` and the adjudication vocabulary. Everything the engine refuses to model. |
| `error` | `SubmitError`: a submission the game would not act on. |

There is deliberately no `player.rs` and no `Player` trait *here*. See below. The
trait a driver drives lives in `hexo-player`, which depends on this crate — the
arrow points that way and never back.

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

- **`Game::plies` is the game's history, not a mirror of one.** `Position` keeps
  no move list, so this record is the only one: it is what a game is written out
  as, and `Game::prefix()` decodes it back into placements. Nothing here
  cross-checks a second copy on the board, because there is no second copy —
  what is checked instead is that the record replays into the canonical position,
  which is a claim about two independently built things.

- **A remote seat is handed a move prefix, not a position.** `Game::prefix()` is
  a move list the seat replays with `Position::replay`. A container cannot be
  handed a `Position`, and board-shaped construction is the rule-bypass hole the
  engine refuses to reopen. An in-process driver hands the whole `Game` to the
  seat, which reads `position()` directly and needs no mirror.

- **Two guards on every submission, both structural.** `generation` stops a late
  or duplicated reply from playing a move chosen for a position the game has
  moved past — reachable the moment decisions are batched or cross a process
  boundary. The echoed `zobrist` catches a mirror whose *board content* has
  drifted, on the ply it drifts rather than at the end of a corrupted game. The
  echo is the **seat's** attestation — its own mirror's hash, if it keeps one —
  and a driver must never fill it in from the canonical position, which would be
  the check vouching for itself. A refused desync leaves the game live at the
  same generation, so a driver that can resync its seat may retry; one that
  cannot reports `Failure::Desync` and the failure policy adjudicates.

  Neither guard sees move order within a turn. A mirror that replays an
  opponent's non-winning two-stone turn in the opposite order reaches the same
  occupied cells and the same mover, so it produces the same hash and submits
  cleanly. Catching that needs a digest over the record, which belongs to the
  wire protocol when it lands (C1, C2) rather than to a second hash here.

- **An illegal move is a result, not an error.** `SubmitError` means *nothing
  happened*; the submission was unusable and the same request is still
  outstanding. A seat that plays illegally has lost, and that comes back as a
  `MatchResult`. Putting it in the error type would push adjudication into every
  driver.

  The exception is a refusal the engine reports as its own limit rather than a
  rule violation. `MoveError::is_rule_violation` tells the two apart, and a
  `NoContest::EngineLimit` blames nobody.

  Either way the result keeps its evidence: `WinReason::IllegalMove` carries the
  `ActionId` played and the exact `MoveError` it was refused with, as
  `NoContest::EngineLimit` carries its seat and error. A verdict whose reasons
  were discarded is one an operator cannot debug and a training pipeline cannot
  filter on, and both facts were in hand at the point the game ended.

- **Three result arms, not two.** A completed/aborted split cannot say what a
  training pipeline needs to know: a game that legitimately hit its action cap
  and a game where a player crashed are both unusable as signal, but only the
  second is anybody's fault. `MatchResult` splits by whose fault it was, and
  `is_contested()` answers "did this match reach a verdict" directly.

  It does not answer "is this game usable as training data". A forfeit —
  `WinReason::Timeout`, `Crash`, `Protocol`, or `Desync` — is a decisive, contested
  result, and it is real evidence in a match: a seat that cannot answer has
  lost. But it says nothing about the play on the board, and the stones on it
  are an abandoned game. A consumer selecting positions to learn from matches on
  `WinReason`, not on `is_contested()` alone.

  `GameSpec::ply_cap` is nonzero by type. Adjudication checks a placement's win
  before the cap, so a win on the capping placement is a win, not a draw. The
  cap is only tested on a placement that ended the mover's turn, so a cap
  falling mid-turn stops the game one placement later rather than giving one
  seat a half turn. Driver failures retain the `Player` and `Failure` in
  `NoContest::SeatFailure`; the no-contest policy declines to charge the
  failure to either seat without erasing where it happened.

- **Diagnostics are opaque and actually persisted.** A seat attaches bytes; the
  game stores them verbatim and never parses them. The field is only worth
  having if it reaches the record — a diagnostics channel that is documented but
  dropped pushes every model package onto its own shard-writing path that
  bypasses the runner entirely.

  `PlyRecord` stores facts that vary by placement. The match-wide thinking
  budget remains in `GameSpec` rather than being repeated in every record.

## Not built yet

| Thing | Blocked on |
| --- | --- |
| Wire protocol and transport | C1, C2 — line-delimited stdio over a handshake pinning protocol, rules, and action-order versions |
| Seed ownership | B4. Deliberately absent rather than decorative — replay determinism comes from the stored action list, so a seed field would reproduce nothing. `hexo-search`'s sessions take a seed and `hexo-bot` draws one from entropy per game; nothing mints or records one |

## Connections

- Depends on `hexo-engine` for rules and state.
- A driver — in-process, subprocess, or container — sits between `Game` and the
  seats. The runner never learns what a model is. `hexo-bot`'s driver is the
  in-process one, and it advances `Game` directly.
- `hexo-records` is the byte layout of the shapes defined here — `GameSpec`,
  `MatchResult`, `PlyRecord` — and adds no field this crate does not have.
