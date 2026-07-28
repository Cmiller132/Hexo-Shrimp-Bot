# hexo-py

Python bindings for `hexo-engine` — the read surface a model builder needs and
nothing that could bypass the rules — plus a thin binding over the shared Rust
MantisNet encoder.

**Status: implemented.** One class, two batch builders, and four module
constants. It is the extension-module PyO3 leaf: Python calls in here, while
the native `hexo-bot` binary owns the separate embedding boundary that calls
out to live Torch.

## Shape

A cdylib crate with **its own `[workspace]` table**, deliberately outside the
root cargo workspace. Two reasons, both stated where they bind: the workspace
forbids `unsafe_code`, which PyO3's generated code cannot satisfy, and
`CONTAINER_SPEC.md` §4 keeps the workspace free of a Python toolchain — this
crate exists exactly to cross that boundary, so it stands alone. It is built by
maturin from the Python side (`uv sync` in `python/mantisnet` builds it as a
path dependency); `cargo xtask verify` neither builds nor gates it.

```
python/hexo-py/
  Cargo.toml      # own [workspace]; engine + shared MantisNet crate by path,
                  # pyo3 with abi3-py312, numpy for returned arrays
  pyproject.toml  # maturin build backend (release profile), module hexo_py
  src/lib.rs      # Position, the two batch entry points, numpy conversion
```

The graph implementation lives once at
`crates/models/mantisnet/src/encoder.rs`. A path dependency across the detached
workspace boundary is intentional: both Cargo workspaces compile the same Rust
source instead of carrying implementations that merely intend to agree.

Building under WSL: set `CARGO_TARGET_DIR=target-wsl` (the repo's convention —
the Windows and Linux toolchains fight over one `target/`).

## Surface

`Position`, wrapping `hexo_engine::Position` one-to-one:

| Member | Engine call | Notes |
| --- | --- | --- |
| `Position()` | `new` | the empty position |
| `Position.replay(moves)` | `replay` | `[(q, r), ...]`; `ValueError` names the ply on refusal |
| `advance(q, r)` | `advance` | `ValueError` on an illegal move |
| `copy()` | `clone` | |
| `stones()` | `stones` | `[(q, r, player)]`, canonical order |
| `legal_moves()` | `legal_actions` | canonical order — the order priors index |
| `nth_legal(i)` | `nth_legal` | one placement by rank, without the whole list |
| `windows_through(q, r)` | `windows_through` | see below |
| `legal_count`, `current_player`, `moves_remaining`, `is_terminal`, `winner`, `stone_count`, `zobrist` | getters | `moves_remaining` derives from `TurnPhase`: `FirstStone` is 2, `Opening` and `SecondStone` are 1 |

Module constants: `RULES_VERSION`, `ACTION_ORDER_VERSION`, `LEGAL_RADIUS`, and
the shared encoder owner's `MODEL_REPR_VERSION`.

Module functions, the batch builder:

- `build_batch(positions)` — every position's MantisNet graph, built in
  parallel under rayon with the GIL released, collated into one dict of numpy
  arrays keyed by `mantisnet.builder.Batch`'s field names.
- `build_batch_prefixes(games, ts)` — the fitting path: replay each game's
  first `ts[i]` placements, then the same build. A stored position is a move
  prefix (`KLENT_DESIGN.md` §12).

## Design notes

- **The shared Rust encoder is the production twin of `mantisnet.builder`, and
  the Python builder is its oracle.** Two implementations of one representation
  are only tolerable with an exact parity detector between them:
  `mantisnet/tests/test_rust_builder.py` holds every emitted array exactly
  equal to the Python path's, field for field. That test is also what frees
  this implementation to use the engine's own `windows_through` walk — the
  independence that `MODEL_SPEC.md` §12.1 needs is carried by the Python
  builder, which never calls it.

- **Positions are created empty or by replay, never deserialised.** The same
  argument as `ENGINE_SPEC.md` §12 one level down: a board-shaped constructor
  is a rule-bypass hole. Anything Python wants to evaluate, it reaches by
  moves.

- **`windows_through` exists for one caller: the builder-oracle test.**
  `MODEL_SPEC.md` §12.1 wants the model's window enumeration checked against an
  independent walk, and the engine's own is independent precisely because the
  builder never calls it. Exposing it here is what makes the engine usable as
  that oracle. It returns `(axis, start_q, start_r, mask_p0, mask_p1)` per
  window, bit `k` of a mask being the cell `k` steps from the start.

- **`abi3-py312`**: one wheel per platform, any CPython ≥ 3.12, so the crate
  does not rebuild when the interpreter minor-steps.

## Connections

- `python/mantisnet` consumes it: the builder's inputs (`MODEL_SPEC.md` §11)
  and every engine position its tests replay.
- `crates/models/mantisnet` is the container package and the owner of the
  encoder this extension exposes.
- The live-Torch boundary crosses the other way (Rust embedding Python) and
  remains private to the `hexo-bot` executable.
