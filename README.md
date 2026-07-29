# Hexo-Shrimp-Bot

A Hexo engine and bot-training framework, built ground up.

> **Status: the container stack runs through its first real package.** The
> engine ships
> the full rule machine, make/unmake search, and its test suite; the runner ships
> the nonblocking game state machine and the result and adjudication model;
> `hexo-player` ships the seat seam a human, a scripted bot, or a transport
> adapter plugs into. Above them, `hexo-search` ships the evaluator seam and the
> nonblocking decision sessions, `hexo-records` the on-disk shard format,
> `hexo-model` the model-package trait and the probe that holds a checkpoint
> honest, `crates/models/mock` the deterministic package built to it, and
> `hexo-bot` the binary, whose `init`, `train`, and `match` subcommands run.
> `crates/models/mantisnet` is the first network-backed package: it shares one
> Rust encoder with `python/hexo-py`, turns the live network's raw heads into
> KLENT's π′/v̂ opinion, uses policy self-play and shared Gumbel evaluation,
> and seals and proves the Python trainer's Torch checkpoints. PyO3/Torch live
> only in the `hexo-bot` executable boundary; logic crates remain Python-free.
> MantisNet deliberately declines container-side `fit`: its production trainer
> remains the faithful KLENT loop under `python/mantisnet`, pending an owner
> decision to migrate that loop. The `mantisnet-train` image and compose
> environment live under `docker/`, and the control deck — the LAN dashboard
> that launches and watches runs, `docs/DECK_SPEC.md` — runs in that same
> image, serving the `frontend/` SPA.

## Layout

```
Hexo-Shrimp-Bot/
  Cargo.toml              workspace root; shared version, edition, lint policy
  crates/
    hexo-engine/          authoritative rules and game state
    hexo-runner/          the authoritative game and adjudication policy
    hexo-player/          the player seam, and the loop that drives games
    hexo-search/          the evaluator seam, and the nonblocking sessions
    hexo-records/         the on-disk shard format, written once, read strictly
    hexo-model/           the model-package trait, the manifest, and the probe
    models/               model packages, one crate each: `mock`, `mantisnet`
    hexo-bot/             the binary: `init`, `train`, and `match`
  xtask/                  the verification gates, defined once; `cargo xtask`
  python/
    hexo-py/              PyO3 engine/shared-encoder bindings; detached leaf crate
    mantisnet/            the MantisNet model and KLENT trainer, plus the control deck
  frontend/               the deck's React SPA, built once and served by the deck
  docker/                 the mantisnet-train image and WSL compose environment
  docs/
    ENGINE_SPEC.md        normative implementation target for hexo-engine
    MODEL_SPEC.md         normative target for the MantisNet network
    KLENT_DESIGN.md       how KLENT works on Hexo; the training path's governing doc
    DECK_SPEC.md          normative target for the control deck: API, lifecycle, SPA
    KLENT_PROPOSALS.md    an external KLENT suggestion set, reviewed; two items since applied
    KLENT_RUN_PLAN.md     the operational plan: settled questions, shakeout, path to production
    ENGINE_RL_AUDIT.md    review findings on readiness for parallel self-play
    CONTAINER_SPEC.md     how a bot is packaged, deployed, and run
    OPEN_DECISIONS.md     what is still undecided, and where the settled answers went
    SUGGESTIONS.md        open design proposals, not yet decided
  .github/workflows/      which runner executes which gate, and nothing more
```

Each crate carries its own `README.md` with a module table and design notes.

## Design decisions

Settled. Open questions live in [docs/SUGGESTIONS.md](docs/SUGGESTIONS.md).

**The rules are fixed, and the architecture is what is being designed.**
Infinite hex board, axial coordinates, centre opening, two placements per turn
with a win check after each, six-in-a-row windows, frontier-radius legality.
`docs/ENGINE_SPEC.md` states them normatively; nothing in this workspace is
free to reinterpret them.

**Rust owns the engine and the match loop; the live forward is the one Python
crossing.** MantisNet's package injects Python-free `ForwardLoader` and
`Forward` traits into its evaluator. The `hexo-bot` executable implements them
with PyO3, loads the production `MantisNet` module, and crosses the interpreter
once per batch. Encoder bytes and KLENT improvement semantics stay in Rust.

**PyO3 is a leaf dependency, never a feature flag.** `hexo-bot` carries the
embedded boundary unconditionally, and `python/hexo-py` is the detached binding
over the same Rust encoder core. No logic crate ever mentions PyO3, even
optionally. Logic-crate tests therefore need no Python runtime, Cargo cannot
feature-unify Python into them, and `hexo-engine` remains compilable to
`wasm32` — which is what lets a web frontend run the real rules instead of a
reimplementation. Building the full workspace, which includes the leaf binary,
uses the selected Python interpreter.

**The engine owns canonical state; players get their own copy.** Only the runner
may advance the authoritative position. Players receive a position of their own
and search it with make/unmake. Enforced by ownership rather than convention:
the canonical state is private, handed out as an owned copy or a shared borrow,
never as a mutable handle.

**The engine's internal design is settled, and recorded with the engine.** Four
decisions constrain how `hexo-engine` is built rather than how the workspace is
shaped, so [crates/hexo-engine/README.md](crates/hexo-engine/README.md) argues
them and this list only names them: the board is a recentred dense grid, not a
sparse map; one placement is the atom, not one turn; the position is a value with
no move history of its own and exactly one, position-only, Zobrist hash; and the
engine owns one canonical action ordering in both directions, `legal_rank` and
`nth_legal`. The invariants that protect each of them are documented there too.

**A remote seat is handed a move prefix once, then fed a move stream.** It
replays the prefix into its own mirror and applies the moves the runner sends, so
it pays O(1) per ply rather than O(board) and never needs a fresh copy per turn.
A move list rather than a serialised position, because a container cannot be
handed one: `Position` has no `serde` impl, deliberately. Board-shaped
construction is a rule-bypass hole — a deserialiser that accepts a bare cell list
reconstructs a position without ever running the turn rules that could have
produced it. The prefix is derived from the runner's record, which is the game's
history — the position keeps none. An in-process seat is handed the whole `Game`
and needs no mirror; the mirror lands with the transport that requires it.

**The hash crosses the container boundary.** `zobrist()` is position-only, which
is an engine decision argued with the engine — but it is what lets the runner
catch desync by exchanging one number per ply. A history-sensitive hash would
have to be documented as process-internal and never persisted; this one is safe
to store and to compare across processes.

**Models are independent and own their own encoding.** A model may be written in
Rust and depends only on the engine's read surface. Neither the engine nor the
runner ever learns what a model is.

**A model owns its move selection, and must implement two modes.** `hexo-player`
states the seam and supplies none of it — no sampler, no temperature, no argmax.
`Model` has two required methods with no defaults: `self_play_move`, which must
vary, and `eval_move`, which is greedier but not deterministic. A non-model seat
— a human, a scripted bot — implements the plainer `Player` trait and never sees
a mode. [crates/hexo-player/README.md](crates/hexo-player/README.md) argues both.

**A model is a package, and the container's whole knowledge of it is one trait.**
Packages live under `crates/models/`, one crate each; each implements
`ModelPackage` and owns its encoder, its evaluator, its two session modes, its
selection policies, its diagnostics format, its weight format, and its `fit`, so
adding one is a new crate and one registry entry and nothing below learns
anything. A checkpoint is weights plus a manifest, and loading one is *proving*
it: the probe hash is recomputed over the bytes the evaluator actually returns
and a mismatch refuses the load.
[crates/hexo-model/README.md](crates/hexo-model/README.md) argues the trait and
the probe; [crates/models/mock/README.md](crates/models/mock/README.md) is the
deterministic exemplar the whole loop is exercised against, and
[crates/models/mantisnet/README.md](crates/models/mantisnet/README.md) states
the real package's shared encoder, injected forward, sessions, diagnostics,
checkpoint metadata, and explicit external-fit refusal.

**The evaluator seam is three types, and no session blocks.** The package's
encoder runs worker-side, so a leaf position never crosses a thread and the
queues carry bytes; the package's evaluator answers a whole batch batcher-side,
so one crossing serves hundreds of leaves. A seat's search is a state machine
that hands out the leaves it wants and returns, which is what makes a game in
flight cost bytes rather than a thread.
[crates/hexo-search/README.md](crates/hexo-search/README.md) argues both, and
settles [docs/SUGGESTIONS.md](docs/SUGGESTIONS.md) S3.

**Records are one binary format with one implementation, read strictly.** A
shard is written once by one writer and renamed into place, and the reader
refuses any file it cannot fully account for rather than offering a lenient
mode; `verify` replays the record through the engine, which is the detector
parsing cannot be. Python will read shards through this code rather than growing
a parser of its own, because two parsers are two definitions of the format.
[crates/hexo-records/README.md](crates/hexo-records/README.md) argues it.

**The runner is a library, not a service.** A containerised bot carries the
engine *and* the runner inside it, so it can drive its own self-play games
without an outside orchestrator, while the same library also backs a host
orchestrator running matches between containers. One authority implementation,
two deployment shapes. Exactly one authority exists per game — a container
answering someone else's protocol runs as a player and does not adjudicate.

**The binary ships the subcommands that work.** `hexo-bot init` atomically
seals a package checkpoint; `train` is a whole training run — batched
self-play, fit, checkpoint, evaluation — in one long-lived process; and
`match` is the local head-to-head that pits two checkpoints, or two search
shapes over one checkpoint, against each other. One driver serves both game
paths. `serve` and `play` are absent rather than stubbed: both are entirely wire
protocol and there is no wire protocol yet, so a stub would publish a command
line before the thing behind it is designed and its guessed flags would become
the constraint the real implementation has to argue its way out of.
[crates/hexo-bot/README.md](crates/hexo-bot/README.md) argues the boundary and
all three commands.

**One build backend per language, one dependency graph per tree.** Cargo owns
the Rust workspace; `python/mantisnet` is a locked `uv` project and maturin
builds the detached `hexo-py` extension. `PYO3_PYTHON` and `HEXO_PYTHON` select
one interpreter for the embedded binary rather than assembling a private model
path in a logic crate.

**The engine is checked against oracles written independently of it.**
`Position::audit()`, the independent oracles in `crates/hexo-engine/tests/common`,
and the frozen golden vectors each detect a class the others cannot — in
particular the symmetric bugs that apply and un-apply identically, which no
round-trip or invariant test can see. Making one of them agree with the
implementation by construction deletes a detector rather than fixing anything.

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
PyO3 or other native-only dependency out of `hexo-engine`. Each gate explains
itself when it fails.

Two need a toolchain you may not have: `rustup target add
wasm32-unknown-unknown`, and the MSRV toolchain named in `Cargo.toml`.

Building the same tree from both Windows and WSL collides on `target/`. Set
`CARGO_TARGET_DIR=target-wsl` on the WSL side; both are gitignored.

## Docs

| Doc | What it is |
| --- | --- |
| [docs/ENGINE_SPEC.md](docs/ENGINE_SPEC.md) | **Normative.** The single implementation target for `crates/hexo-engine`: rules, state, storage, growth policy, error precedence, invariants, and test obligations. |
| [docs/ENGINE_RL_AUDIT.md](docs/ENGINE_RL_AUDIT.md) | Review findings on readiness for massively parallel self-play, with each one's resolution recorded next to it. A snapshot with annotations, not a live plan. |
| [docs/CONTAINER_SPEC.md](docs/CONTAINER_SPEC.md) | **Normative** for the container crates: how a bot is packaged, deployed, and run — one binary, where state lives, and what a long-lived training process has to guarantee. Trued up through the MantisNet package, embedded live-forward boundary, shared Gumbel session, checkpoint sealing, and image. |
| [docs/OPEN_DECISIONS.md](docs/OPEN_DECISIONS.md) | Questions that *must* be answered before the code depending on them exists. A settled one leaves, and the file records where its answer went — the engine's, the runner's, and the record format's are all settled; seeds and the wire protocol are not. |
| [docs/SUGGESTIONS.md](docs/SUGGESTIONS.md) | *Optional* design proposals with status, rationale, and trade-offs. Accepted items graduate to the `README.md` of whatever they decided. |
