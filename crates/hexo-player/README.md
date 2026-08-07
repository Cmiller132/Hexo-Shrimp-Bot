# hexo-player

Blocking in-process seat interface for Hexo games. `hexo-player` defines the
`Player` trait (what a seat must do), the `Model` trait (what a trainable seat
must do), and `Table` (the loop that couples two seats to one `hexo-runner`
game). It contains no concrete player, search algorithm, evaluator, or
transport.

## Components

### Player

The `Player` trait requires a single method, `choose(&mut self, &Game) ->
Decision`. The seat receives the complete `Game` and returns a complete
`Decision`. A blanket impl on `Box<P>` allows heterogeneous seats via
`Table<Box<dyn Player>>`.

### Model / Mode / ModelPlayer

`Model` is a trait for trainable players that separates self-play and evaluation
policies into two required methods: `self_play_move` and `eval_move`. `Mode` is
an enum (`SelfPlay` or `Eval`) selecting which policy to use. `ModelPlayer<M>`
binds a `Model` value to a `Mode` and implements `Player` by dispatching to the
appropriate method. Its accessors are `mode()`, `model()`, and `into_model()`.

### Table

`Table<P>` owns one `Game` and an array of two `P` seats. It provides:

- `new(GameSpec, [P; 2])` to start a game;
- `game()` and `result()` for read access;
- `seat(Seat)` and `into_seats()` for player access;
- `step()` to advance by one placement;
- `run()` to drive a game to completion.

`step` asks the current seat for a decision, submits it to the game, and
converts a generation desync into `Failure::Desync`. `run` loops `step` until
the game ends.

### sweep

The free function `sweep(&mut [Table<P>])` advances every unfinished table by
one placement and returns the count still running. `while sweep(&mut tables) > 0
{}` drives a batch of games to completion.

## Dependencies

- `hexo-engine` supplies the `Player` seat identifiers (`P0`, `P1`).
- `hexo-runner` supplies `Game`, `GameSpec`, `Decision`, `Reply`,
  `MatchResult`, `Step`, `SubmitError`, and `Failure`.

## Files

| File | Description |
| --- | --- |
| `src/lib.rs` | Crate root; declares modules and re-exports `Player`, `Model`, `Mode`, `ModelPlayer`, `Table`, and `sweep` |
| `src/player.rs` | The `Player` trait and its `Box<P>` blanket impl |
| `src/model.rs` | The `Model` trait, the `Mode` enum, and the `ModelPlayer` adapter |
| `src/table.rs` | `Table` game driver, desync handling, and the `sweep` batch helper |
| `tests/table.rs` | Integration test for `Table` |
