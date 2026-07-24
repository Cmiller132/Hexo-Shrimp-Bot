# Hexo-Shrimp-Bot

A ground-up rebuild of the Hexo engine and bot-training framework, succeeding
`Hexo-BotTrainer-hexgt`. The game rules are unchanged; the architecture is not.

> **Status: scaffold.** Both crates are empty. Only the engine and the match
> runner are in scope right now — no models, no search, no training, no Python.

## Layout

```
Hexo-Shrimp-Bot/
  Cargo.toml              workspace root; shared version, edition, lint policy
  crates/
    hexo-engine/          authoritative rules and game state
    hexo-runner/          match orchestration and player communication
  docs/
    SUGGESTIONS.md        open design proposals, not yet decided
  .github/workflows/
    ci.yml                fmt, clippy, test on every push
```

Each crate carries its own `README.md` with a module table and design notes.

## Design decisions

Settled. Open questions live in [docs/SUGGESTIONS.md](docs/SUGGESTIONS.md).

**The game is unchanged.** Infinite hex board, axial coordinates, centre
opening, two placements per turn with a win check after each, six-in-a-row
windows, frontier-radius legality. The rules are being reimplemented, not
redesigned.

**Rust owns the engine and the match loop.** Where the Rust/Python boundary
ultimately falls is deliberately undecided — Python earns its place in plenty of
roles, and picking the line before there is anything to put on either side of it
would be guessing.

**PyO3 will be a leaf crate, never a feature flag.** When Python bindings
arrive they get their own crate that depends on the logic crates. No logic crate
ever mentions PyO3, even optionally. This keeps `cargo test` free of any Python
toolchain, keeps compiles fast, avoids Cargo feature-unification surprises, and
leaves `hexo-engine` compilable to `wasm32` — which is what would let a web
frontend run the real rules instead of a reimplementation.

**The engine owns canonical state; players get their own copy.** Only the runner
may advance the authoritative position. Players receive a position of their own
and search it with make/unmake. Enforced by ownership rather than convention:
the canonical state is private, handed out as an owned copy or a shared borrow,
never as a mutable handle.

**The board is a recentred dense grid.** Play is local and contiguous even
though the board is unbounded, so a dense array with an origin offset that grows
and recentres replaces the previous sparse hash map. Array indexing instead of
hashing on every neighbour query, bitboard shifts for window updates, and
`clone` as a flat memcpy — which matters because copying positions is on the
critical path of the design above.

**One placement is the atom, not one turn.** A turn is two placements, but a win
is checked after each, so a turn can end after the first. Making the placement
the unit keeps that out of the record format, the wire protocol, and every
future policy head.

**Players are handed a position once, then fed a move stream.** No fresh copy
per turn: the player keeps its own mirror and applies the moves the runner
sends. This costs O(1) per ply instead of O(board), it is the only handoff shape
a container can use, and the move stream *is* the game record, so recording
falls out for free.

**The position carries an incremental Zobrist hash.** Maintained through the
same delta stack as `apply` / `undo`. Near-free at runtime, and it is what lets
the runner catch desync with a remote player by exchanging one number per ply.

**Models are independent and own their own encoding.** A model may be written in
Rust and depends only on the engine's read surface. Neither the engine nor the
runner ever learns what a model is.

**The runner is a library, not a service.** A containerised bot carries the
engine *and* the runner inside it, so it can drive its own self-play games
without an outside orchestrator, while the same library also backs a host
orchestrator running matches between containers. One authority implementation,
two deployment shapes. Exactly one authority exists per game — a container
answering someone else's protocol runs as a player and does not adjudicate.

**One build backend, one workspace.** Cargo workspace for Rust; when Python
arrives, a `uv` workspace and maturin — not the three backends and manual
`PYTHONPATH` assembly of the previous repo.

## Build

Requires Rust 1.85+ (edition 2024). Currently on 1.95.

```sh
cargo build
cargo test
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
```

Building the same tree from both Windows and WSL collides on `target/`. Set
`CARGO_TARGET_DIR=target-wsl` on the WSL side; both are gitignored.

## Docs

| Doc | What it is |
| --- | --- |
| [docs/SUGGESTIONS.md](docs/SUGGESTIONS.md) | Open design proposals with status, rationale, and trade-offs. Accepted items graduate to this README. |
