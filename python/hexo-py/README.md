# hexo-py

## Purpose

`hexo-py` is the PyO3 extension that exposes the engine read/replay surface and
the shared Rust MantisNet encoder to Python. It permits positions to be created
empty, replayed, advanced legally, inspected, and collated for model input. It
does not expose internal board construction or mutable engine storage.

## Public surface

The installed module name is `hexo_py`.

`hexo_py.Position` provides:

| Member | Contract |
| --- | --- |
| `Position()` | Empty engine position |
| `Position.replay(moves)` | Construct by replaying `(q, r)` placements |
| `advance(q, r)` | Apply one legal placement |
| `copy()` | Clone the current position |
| `stones()` | Canonically ordered `(q, r, player)` rows |
| `legal_moves()` | Canonical `(q, r)` legal-action order |
| `nth_legal(index)` | Legal move at one canonical rank |
| `windows_through(q, r)` | Engine window geometry and occupancy masks |
| `legal_count` | Number of legal actions |
| `current_player` | Side to move as `0` or `1` |
| `moves_remaining` | Placements remaining in the current turn |
| `is_terminal` | Terminal-position flag |
| `winner` | Winning player or `None` |
| `stone_count` | Total stones |
| `zobrist` | Position-only engine hash |

Module functions:

- `build_batch(positions)` builds and collates the shared Rust graph encoding.
- `build_batch_prefixes(games, ts)` replays each move prefix and collates it.

Both functions return dictionaries of NumPy arrays matching
`mantisnet.builder.Batch` field names.

Module constants:

- `RULES_VERSION`;
- `ACTION_ORDER_VERSION`;
- `LEGAL_RADIUS`;
- `MODEL_REPR_VERSION`;
- `PROTOCOL_VERSION`, re-exported from `hexo_runner`. A host orchestrator opens
  every seat with the three versions of `CONTAINER_SPEC.md` §3.1's handshake;
  two of them are the engine's, and this is the third, so a Python orchestrator
  never keeps its own copy of that number.

The extension uses `abi3-py312`, so its binary contract targets CPython 3.12
and later on the same platform.

## Run / test

The normal development path is the parent MantisNet environment:

```sh
cd python/mantisnet
uv sync
uv run pytest tests/test_rust_builder.py
```

Build the extension explicitly with maturin:

```sh
cd python/mantisnet
uv run --with maturin maturin develop --release -m ../hexo-py/Cargo.toml
uv run python -c "import hexo_py; print(hexo_py.RULES_VERSION)"
```

In the container:

```sh
maturin develop --release -m ../hexo-py/Cargo.toml
python -m pytest tests/test_rust_builder.py -q
```

The crate is detached from the root Cargo workspace, so use its own manifest
for a direct Rust check:

```sh
cargo check --manifest-path python/hexo-py/Cargo.toml
```

## Connections

- `crates/hexo-engine` supplies the wrapped `Position` and rule versions.
- `crates/models/mantisnet/src/encoder.rs` supplies the shared graph encoder.
- `python/mantisnet/mantisnet/builder.py` defines the independent Python parity
  implementation and `Batch` shape.
- `python/mantisnet/tests/test_rust_builder.py` compares every emitted array.
- The representation contract is
  [`docs/MODEL_SPEC.md`](../../docs/MODEL_SPEC.md).
- Hexo-specific KLENT state storage is described in
  [`docs/KLENT_FOR_HEXO.md`](../../docs/KLENT_FOR_HEXO.md).
- The extension boundary is covered by
  [`docs/CONTAINER_SPEC.md`](../../docs/CONTAINER_SPEC.md).

## Invariants & gotchas

- The crate has its own Cargo workspace and is not built by root workspace
  commands.
- The root workspace forbids unsafe code; PyO3-generated extension code requires
  this detached build boundary.
- Positions are constructed empty or by replay, never from arbitrary cells.
- Illegal replay or advance operations raise `ValueError`.
- Replay errors identify the refused ply.
- Legal-move arrays use engine canonical action order.
- `moves_remaining` is derived from engine `TurnPhase`.
- `windows_through` returns `(axis, start_q, start_r, mask_p0, mask_p1)` rows.
- Window mask bit `k` describes the cell `k` steps from the window start.
- Batch construction releases the GIL and uses Rayon for position encoding.
- Rust batch output must remain exactly equal to the independent Python builder
  output.
- `MODEL_REPR_VERSION` is owned by the Rust MantisNet package.
- Linux and host-platform Cargo artifacts must use separate target directories.
- The production package forward boundary is the opposite direction: the
  `hexo-bot` executable embeds Python and is not implemented here.
