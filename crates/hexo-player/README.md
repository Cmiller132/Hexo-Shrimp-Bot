# hexo-player

## Purpose

`hexo-player` defines the blocking in-process seat interface and the loop that
drives seats against `hexo-runner::Game`. It supports human adapters, scripted
bots, transport adapters, and model wrappers that can answer in one call. It
does not contain a concrete player, search algorithm, evaluator, or transport.

## Public surface

The crate root re-exports:

| Item | Contract |
| --- | --- |
| `Player` | `choose(&mut self, &Game) -> Decision` |
| `Model` | Separate self-play and evaluation move methods |
| `Mode` | `SelfPlay` or `Eval` dispatch mode |
| `ModelPlayer<M>` | Binds a `Model` value to one `Mode` |
| `Table<P>` | Owns one `Game` and two seats |
| `sweep` | Advances every unfinished table by at most one placement |

`Player` receives a shared view of the complete `Game` and returns the complete
`Decision`. `Box<P>` implements `Player` when `P` does, so heterogeneous seats
can use `Table<Box<dyn Player>>`.

`Model` requires:

```rust
fn self_play_move(&mut self, game: &Game) -> Decision;
fn eval_move(&mut self, game: &Game) -> Decision;
```

`ModelPlayer::new(model, mode)` performs dispatch only. Its accessors are
`mode`, `model`, and `into_model`.

`Table` provides:

- `new(GameSpec, [P; 2])`;
- `game()` and `result()` read access;
- `seat(Seat)` and `into_seats()`;
- `step()` for one placement;
- `run()` for a complete bounded game.

Minimal drive shape:

```rust
use hexo_player::{Table, sweep};
use hexo_runner::GameSpec;

# fn drive<P: hexo_player::Player>(seats: [P; 2]) {
let mut tables = [Table::new(GameSpec::default(), seats)];
while sweep(&mut tables) > 0 {}
assert!(tables[0].result().is_some());
# }
```

## Run / test

From the repository root:

```sh
cargo test -p hexo-player
cargo test -p hexo-player --test table
cargo doc -p hexo-player --no-deps
cargo check -p hexo-player
```

Run all workspace gates:

```sh
cargo xtask verify
```

The crate contains no binary target and has no standalone run command.
Consumer examples and doctests run through `cargo test` and `cargo doc`.

## Connections

- `crates/hexo-runner` supplies `Game`, `GameSpec`, `Decision`, `Reply`, and
  match results.
- `crates/hexo-engine` supplies the two seat identifiers.
- `crates/hexo-search` is the nonblocking session interface used by batched
  model-backed drivers.
- `crates/hexo-bot` drives `DecisionSession` values rather than this blocking
  interface.
- `src/player.rs` defines the generic blocking seat.
- `src/model.rs` defines mode-specific model dispatch.
- `src/table.rs` owns game driving and desync failure conversion.

## Invariants & gotchas

- A seat receives `&Game`; it cannot mutate canonical state directly.
- The current mover is `game.position().current_player()`.
- The budget is `game.spec().budget`; the seat is responsible for honoring it.
- Move history is available through `game.plies()` and `game.prefix()`.
- A `Decision` includes the chosen action, the position hash attestation, and
  optional diagnostics.
- The seat authors the hash attestation; the driver does not replace it.
- The driver submits illegal actions and lets the runner adjudicate them.
- `Model` has distinct required methods for self-play and evaluation.
- `ModelPlayer` adds no sampling, search, legality check, or diagnostics.
- `Table` owns two independent seat values indexed by engine `Player::index`.
- `Table::step` asks only the seat named by `Game::step`.
- A refused desynchronized placement is converted to `Failure::Desync` because
  `Table` has no transport resynchronization mechanism.
- A generation read from the same `Game` is submitted unchanged.
- `Table::run` terminates because `GameSpec::ply_cap` is nonzero.
- `sweep` skips finished tables and returns the number still running.
- The blocking `Player` interface cannot expose intermediate evaluation
  requests for cross-game batching; use `hexo-search` for that shape.
