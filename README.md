# Hexo-Shrimp-Bot

A ground-up rebuild of the Hexo engine and bot-training framework, succeeding
`Hexo-BotTrainer-hexgt`. The game rules are unchanged; the architecture is not.

> **Status: `hexo-engine` implemented, `hexo-runner` still a scaffold.** The
> engine ships the full rule machine, make/unmake search, and its test suite;
> the runner crate is empty. Only the engine and the match runner are in scope
> right now — no models, no search, no training, no Python.

## Layout

```
Hexo-Shrimp-Bot/
  Cargo.toml              workspace root; shared version, edition, lint policy
  crates/
    hexo-engine/          authoritative rules and game state
    hexo-runner/          the authoritative game and adjudication policy
    hexo-reference/       frozen copy of the previous engine; a test oracle only
  docs/
    ENGINE_SPEC.md        normative implementation target for hexo-engine
    ENGINE_RL_AUDIT.md    review findings on readiness for parallel self-play
    OPEN_DECISIONS.md     questions that block the engine and the runner
    SUGGESTIONS.md        open design proposals, not yet decided
  .github/workflows/
    ci.yml                fmt, clippy, msrv, docs, test, wasm32 on every push
    nightly-smoke.yml     the deep smoke run, scheduled
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
hashing on every neighbour query, and bitboard shifts for window updates.
`clone` copies five flat buffers — four grid planes and the move history — with
no pointer chasing and no per-cell work.

**One placement is the atom, not one turn.** A turn is two placements, but a win
is checked after each, so a turn can end after the first. Making the placement
the unit keeps that out of the record format, the wire protocol, and every
future policy head.

**Players are handed a move prefix once, then fed a move stream.** No fresh copy
per turn: the player replays the prefix into its own mirror and applies the moves
the runner sends. This costs O(1) per ply instead of O(board), and it is a move
list rather than a serialised position because a container cannot be handed one —
`Position` has no `serde` impl, deliberately, since board-shaped construction is
what re-opens the rule-bypass hole. The old engine had exactly that and its
`Board` deserialiser skipped the turn rules.

**The position carries its move history, and there is exactly one hash.**
`history()` is what makes a position writable as a record and rebuildable by
`replay()`. The Zobrist hash stays *position-only* — it covers stones, owners,
mover, phase, and the terminal bit, not move order. Hexo transposes
structurally, since a turn's two stones are playable in either order and reach
the same position, so a history-sensitive key would forfeit a 2x merge per turn
of search. A model whose features depend on move order reads `history()` and
mixes it into its own cache key. The old repo kept a separate, history-sensitive
hash for exactly that reason and had to document it as process-internal and
never persisted; this one crosses the container boundary and is what lets the
runner catch desync by exchanging one number per ply.

**One canonical action ordering, owned by the engine, in both directions.**
`legal_rank` and `nth_legal` are the mapping a policy head is indexed by. If
self-play, training, and serving each derive it themselves they agree only by
coincidence, and a divergence is silent — the network keeps training, against
scrambled targets.

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

**The previous engine is kept as a test oracle.** `hexo-reference` is a frozen,
dependency-free copy of `Hexo-BotTrainer-hexgt`'s rules crate, and
`cargo test --workspace` drives random legal games through both engines and
compares legality, terminal results, turn state, and window ownership ply by
ply. It exists so "the rules are unchanged" is a checked property rather than a
claim, and it must never be edited to make a test pass — a divergence is a
finding, and the rewrite is not automatically the correct side.

## Build

Requires Rust 1.88+ (let chains). Developed on 1.95, and the floor is gated by
the `msrv` CI job rather than declared and hoped for.

```sh
cargo build
cargo test --workspace
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo clippy --release --all-targets -- -D warnings
cargo build -p hexo-engine --target wasm32-unknown-unknown
```

The release lint is a separate gate: `debug_assertions` is off in release, so
helpers whose only callers are `#[cfg(debug_assertions)]` become dead code that
the debug lint cannot see.

The `wasm32` build is a gate too. Nothing in the native build would catch a
`std::time` call, a threading primitive, or a PyO3 dependency creeping into
`hexo-engine`, and any of those costs the crate its ability to run the real
rules in a browser. `rustup target add wasm32-unknown-unknown` first.

Building the same tree from both Windows and WSL collides on `target/`. Set
`CARGO_TARGET_DIR=target-wsl` on the WSL side; both are gitignored.

## Docs

| Doc | What it is |
| --- | --- |
| [docs/ENGINE_SPEC.md](docs/ENGINE_SPEC.md) | **Normative.** The single implementation target for `crates/hexo-engine`: rules, state, storage, growth policy, error precedence, invariants, and test obligations. |
| [docs/OPEN_DECISIONS.md](docs/OPEN_DECISIONS.md) | Questions that *must* be answered to build the engine and runner, grouped by what they block. |
| [docs/SUGGESTIONS.md](docs/SUGGESTIONS.md) | *Optional* design proposals with status, rationale, and trade-offs. Accepted items graduate to this README. |
