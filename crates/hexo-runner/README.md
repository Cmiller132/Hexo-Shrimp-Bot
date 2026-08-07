# hexo-runner

Match-orchestration state machine for Hexo. `hexo-runner` wraps one
`hexo-engine::Position`, accepts seat replies, adjudicates outcomes, and
records the placement history. It is a library crate with no player handles,
event loop, transport, persistence, search, or model code.

## Components

### `Game` and `GameSpec`

`Game` is the single authoritative match. It holds a `Position`, a generation
counter, the ply record, and the current result. `GameSpec` parameterizes the
match: a ply cap (`NonZeroU32`), a `Budget` communicated to seats, and a
`FailurePolicy` that controls whether a seat failure forfeits or produces a
no-contest.

A driver calls `Game::step()` to obtain either a `Step::NeedDecision` (with
seat, generation, budget, zobrist hash, and ply count) or a `Step::Finished`.
The only state transition is `Game::submit(generation, reply)`, which returns a
`Transition` describing the accepted placement and the resulting match state.
Read-only accessors expose the spec, position, ply records, and prefix.

### `Decision`, `Budget`, `Reply`, `Failure`

A seat's answer is a `Reply`: `Place(Decision)`, `Resign`, or
`Failed(Failure)`. `Decision` carries an `Action`, the zobrist hash the seat
chose from, and optional opaque diagnostics bytes. `Budget` is the resource
limit communicated to a seat (unlimited, node count, visit count, or wall
time). `Failure` describes why a seat produced no placement (timeout, crash,
protocol error, or desync).

### `MatchResult`, `WinReason`, `DrawReason`, `NoContest`

`MatchResult` is the terminal outcome: `Decisive` (with winner and
`WinReason`), `Drawn` (with `DrawReason`), or `NoContest` (with `NoContest`
reason). `WinReason` covers six-in-a-row, resignation, illegal move, timeout,
crash, protocol violation, and desync. `DrawReason` covers the ply cap.
`NoContest` covers engine representation limits and seat failures under a
no-contest policy. `MatchResult::is_contested()` distinguishes contested
verdicts from no-contests; `winner()` extracts the winning seat when there is
one.

### `SubmitError`

Submissions the game refuses: the game has already ended (`Finished`), the
generation token is stale (`StaleGeneration`), or the seat's position hash
disagrees with the canonical one (`Desync`). Implements `Display` and `Error`.

### `PROTOCOL_VERSION`

A `u32` constant (currently `2`) that versions the runner decision/result model
and the native seat message set referenced by `docs/CONTAINER_SPEC.md` section 3.1.
Manifests, record shards, and the seat handshake use it.

## Connections

- **`hexo-engine`** provides `Position`, `Action`, `ActionId`, `Applied`,
  `Player`, `TurnPhase`, `MoveError`, and `HexCoord` -- the board and rules
  layer this crate wraps.
- **`hexo-player`** provides a blocking driver loop around `Game`.
- **`hexo-search`** authors `Decision` values through nonblocking sessions.
- **`hexo-records`** serializes `GameSpec`, `MatchResult`, and `PlyRecord`.
- **`hexo-bot`** owns the concurrent driver and process lifecycle.
- **`docs/CONTAINER_SPEC.md`** is the normative spec for runner obligations.

## Files

| File | Description |
| --- | --- |
| `src/lib.rs` | Crate root: module declarations, re-exports, `PROTOCOL_VERSION`, and the driver usage example. |
| `src/game.rs` | `Game`, `GameSpec`, `FailurePolicy`, `PlyRecord`, `Step`, and `Transition`. |
| `src/decision.rs` | `Budget`, `Decision`, `Failure`, and `Reply`. |
| `src/outcome.rs` | `MatchResult`, `WinReason`, `DrawReason`, and `NoContest`. |
| `src/error.rs` | `SubmitError` with `Display` and `Error` implementations. |
