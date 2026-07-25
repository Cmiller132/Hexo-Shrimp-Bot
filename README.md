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
  xtask/                  the verification gates, defined once; `cargo xtask`
  docs/
    ENGINE_SPEC.md        normative implementation target for hexo-engine
    ENGINE_RL_AUDIT.md    review findings on readiness for parallel self-play
    OPEN_DECISIONS.md     questions that block the engine and the runner
    SUGGESTIONS.md        open design proposals, not yet decided
  .github/workflows/      which runner executes which gate, and nothing more
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

**The engine's internal design is settled, and recorded with the engine.** Four
decisions constrain how `hexo-engine` is built rather than how the workspace is
shaped, so [crates/hexo-engine/README.md](crates/hexo-engine/README.md) argues
them and this list only names them: the board is a recentred dense grid, not a
sparse map; one placement is the atom, not one turn; the position carries its
move history and there is exactly one, position-only, Zobrist hash; and the
engine owns one canonical action ordering in both directions, `legal_rank` and
`nth_legal`. The invariants that protect each of them are documented there too.

**Players are handed a move prefix once, then fed a move stream.** No fresh copy
per turn: the player replays the prefix into its own mirror and applies the moves
the runner sends. This costs O(1) per ply instead of O(board), and it is a move
list rather than a serialised position because a container cannot be handed one —
`Position` has no `serde` impl, deliberately, since board-shaped construction is
what re-opens the rule-bypass hole. The old engine had exactly that and its
`Board` deserialiser skipped the turn rules.

**The hash crosses the container boundary.** `zobrist()` is position-only, which
is an engine decision argued with the engine — but it is what lets the runner
catch desync by exchanging one number per ply. The previous repo's
history-sensitive hash had to be documented as process-internal and never
persisted; this one does not.

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

Requires Rust 1.88+ (let chains) — the floor is declared in `Cargo.toml` and
gated rather than hoped for. Developed on 1.95.

```sh
cargo xtask verify     # every gate CI runs, in the order CI runs it
cargo xtask            # list the gates and what each one catches
cargo xtask lint       # just one
```

The gates are defined in `xtask/src/main.rs` and nowhere else, so a local run
and CI cannot disagree about what they are. Several are not redundant with
`cargo test` in ways that are easy to assume away — the release profile sees
dead code the debug profile deletes, rustdoc is not checked by clippy, the MSRV
floor is a promise nothing else tests, and the `wasm32` target is what keeps a
`std::time` call or a PyO3 dependency out of `hexo-engine`. Each gate explains
itself when it fails.

Two need a toolchain you may not have: `rustup target add
wasm32-unknown-unknown`, and the MSRV toolchain named in `Cargo.toml`.

Building the same tree from both Windows and WSL collides on `target/`. Set
`CARGO_TARGET_DIR=target-wsl` on the WSL side; both are gitignored.

## Docs

| Doc | What it is |
| --- | --- |
| [docs/ENGINE_SPEC.md](docs/ENGINE_SPEC.md) | **Normative.** The single implementation target for `crates/hexo-engine`: rules, state, storage, growth policy, error precedence, invariants, and test obligations. |
| [docs/OPEN_DECISIONS.md](docs/OPEN_DECISIONS.md) | Questions that *must* be answered to build the engine and runner, grouped by what they block. |
| [docs/SUGGESTIONS.md](docs/SUGGESTIONS.md) | *Optional* design proposals with status, rationale, and trade-offs. Accepted items graduate to this README. |
| [docs/KLENT_DESIGN.md](docs/KLENT_DESIGN.md) | How the KLENT algorithm (Ota et al., ICML 2026) would work on Hexo: the placement-level MDP, the value target, the action space, the training corpus, and what it asks of the engine and runner. Specification sketch, not normative. |
