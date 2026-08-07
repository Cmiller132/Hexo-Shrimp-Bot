# hexo-py

PyO3 extension crate that exposes the Hexo engine's read and replay surface to
Python. It provides position management (create, replay, advance, inspect),
legal-move enumeration in canonical engine order, window geometry queries, and
batch graph encoding for MantisNet model input. The installed module name is
`hexo_py`.

## Components

### Position

A Python wrapper around `hexo_engine::Position`. Positions are created empty or
by replaying a sequence of `(q, r)` placements. The wrapper exposes:

- **Construction:** `Position()` for the empty board; `Position.replay(moves)`
  to construct by replaying a placement sequence.
- **Mutation:** `advance(q, r)` applies one legal placement; `copy()` clones the
  position.
- **Stone inspection:** `stones()` returns `(q, r, player)` tuples in canonical
  order; `stone_count` gives the total.
- **Legal moves:** `legal_moves()` lists all legal `(q, r)` placements in the
  engine's canonical action order; `nth_legal(index)` retrieves one by rank;
  `legal_count` gives the count.
- **Game state:** `current_player` (0 or 1), `moves_remaining` (derived from the
  engine's `TurnPhase`), `is_terminal`, `winner` (player index or `None`).
- **Hashing:** `zobrist` returns the incremental Zobrist hash.
- **Window geometry:** `windows_through(q, r)` returns the engine's win-window
  walk as `(axis, start_q, start_r, mask_p0, mask_p1)` tuples, where bit `k` of
  a mask is the cell `k` steps from the window start along the axis.

### Batch builders

- `build_batch(positions)` encodes a list of positions into the MantisNet graph
  representation, returning a dictionary of NumPy arrays whose keys match
  `mantisnet.builder.Batch` field names. Releases the GIL and uses Rayon for
  parallel encoding.
- `build_batch_prefixes(games, ts)` replays each game's first `ts[i]` placements
  and encodes the resulting positions in parallel. This is the fitting path,
  where stored positions are move prefixes.

### Version constants

The module re-exports version constants that checkpoints pin:

- `RULES_VERSION` and `ACTION_ORDER_VERSION` from `hexo-engine`.
- `LEGAL_RADIUS` from `hexo-engine`.
- `MODEL_REPR_VERSION` from `hexo-model-mantisnet`.
- `PROTOCOL_VERSION` from `hexo-runner`, the third version in the container
  handshake.

## Build

The crate has its own Cargo workspace, detached from the root workspace. It is
built by maturin from the Python side:

```sh
cd python/mantisnet
uv run --with maturin maturin develop --release -m ../hexo-py/Cargo.toml
```

The extension uses `abi3-py312`, targeting CPython 3.12 and later. Tests live in
the MantisNet package:

```sh
cd python/mantisnet
uv sync
uv run pytest tests/test_rust_builder.py
```

A direct Rust check uses the crate's own manifest:

```sh
cargo check --manifest-path python/hexo-py/Cargo.toml
```

## Connections

- `crates/hexo-engine` supplies the wrapped `Position` and rule/action-order
  versions.
- `crates/models/mantisnet/src/encoder.rs` supplies the graph encoder behind
  `build_batch` and `build_batch_prefixes`.
- `crates/hexo-runner` supplies `PROTOCOL_VERSION`.
- `python/mantisnet/mantisnet/builder.py` defines an independent Python parity
  implementation; `python/mantisnet/tests/test_rust_builder.py` asserts the two
  produce identical output.
- The model architecture is described in `python/mantisnet/README.md`.

## Files

| File | Description |
| --- | --- |
| `src/lib.rs` | PyO3 module: `Position` wrapper, batch-builder bindings, version exports |
| `Cargo.toml` | Crate manifest with dependencies on `hexo-engine`, `hexo-model-mantisnet`, `hexo-runner`, PyO3, and numpy |
| `pyproject.toml` | Python package metadata and maturin build configuration |
