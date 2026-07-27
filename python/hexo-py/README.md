# hexo-py

Python bindings for `hexo-engine`: the read surface a model builder needs, and
nothing that could bypass the rules.

**Status: implemented.** One class, three module constants. It is the PyO3 leaf
crate the root `README.md` promised — it depends on `hexo-engine` and nothing
depends on it.

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
  Cargo.toml      # own [workspace]; hexo-engine by path, pyo3 with abi3-py312
  pyproject.toml  # maturin build backend, module name hexo_py
  src/lib.rs      # everything
```

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
| `windows_through(q, r)` | `windows_through` | see below |
| `legal_count`, `current_player`, `moves_remaining`, `is_terminal`, `winner`, `stone_count`, `zobrist` | getters | `moves_remaining` derives from `TurnPhase`: `FirstStone` is 2, `Opening` and `SecondStone` are 1 |

Module constants: `RULES_VERSION`, `ACTION_ORDER_VERSION`, `LEGAL_RADIUS`.

## Design notes

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
- The Python-backed `ModelPackage` of `CONTAINER_SPEC.md` §4 is **not** this
  crate and will not be: that boundary crosses the other way (Rust embedding
  Python), lives in `hexo-bot`'s process, and arrives with the package itself.
