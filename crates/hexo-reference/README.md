# hexo-reference — frozen oracle, not a dependency

**This crate is a frozen, verbatim copy of another repository's rules engine.
It exists only as a test oracle. It must never be edited to make a test pass.**

## What it is

A copy of `Hexo-BotTrainer-hexgt/packages/hexo_engine/rust/src`, taken on
**2026-07-24** from that repository at commit
`93d7a7614087054fc9b1001aacb166dafc4878f1`.

That engine is the previous, battle-tested implementation of the Hexo rules.
`crates/hexo-engine` is a ground-up rewrite of the *same* rules with a different
architecture, so the two are independent implementations of one specification.
Running random legal games through both and asserting they agree turns "the
rewrite is correct" from a belief backed by internally-written oracles into a
checked property. `tests/differential.rs` is that check.

The old checkout will not survive; that is why the copy is vendored here rather
than referenced by path. A `path = "E:/..."` dependency would not survive CI or
another machine.

## The rule

If a differential test fails, **the divergence gets reported, not patched**.
Neither engine is presumed correct. In particular the new engine is *not*
automatically right: it is the one that changed. Editing this crate to make a
test go green destroys the only independent evidence in the workspace.

Nothing outside `tests/` may depend on this crate, and nothing here should be
"improved", reformatted for style, or refactored. It is a photograph.

## Exactly what was changed from the original

Only mechanical stripping — no rule, no control flow, and no data structure was
touched. Every change is marked with a `VENDORED CHANGE:` note in the file it
affects.

| Change | Why |
| --- | --- |
| `pybridge.rs` deleted | PyO3 bridge; the crate's `python` feature was off by default anyway, and the bindings are not rules. |
| `snapshot.rs` deleted, with `HexoState::snapshot`, `load_state`, and `StateLoadError` | Serialisation of a move list. Dormant in the original ("No production code serializes snapshots today"), and the differential test replays move lists itself. |
| `serde` derives and the `Board` `Serialize`/`Deserialize` impls deleted | Drops the `serde` dependency. The `Board` impls were the dormant board-shaped deserialisation path that `docs/OPEN_DECISIONS.md` A3 says not to port. |
| `Board::place` and `Board::debug_stones` deleted | Only the deleted serde impl and the deleted unit tests called them. `place_with_delta`, which does the work, is untouched. |
| `ahash::AHashMap`/`AHashSet` → `std::collections::HashMap`/`HashSet` | Drops the `ahash` dependency. A hasher swap only: no public output depends on hash iteration order (legal-move enumeration reads the sorted `LegalMoveStore::ordered` vector, and the differential test never iterates `WindowStore::entries`). |
| `thiserror` derive on `MoveError` → hand-written `Display`/`Error` | Drops the last dependency. The message strings are reproduced character for character. |
| `#[cfg(test)] mod tests` blocks deleted, plus the `#[cfg(test)]`-only `WindowStore::update_for_placement` wrapper | The old crate's own tests needed `serde_json` and `snapshot.rs`. They tested the old engine against itself, which is not what this crate is here for. |

Net effect: **zero dependencies**, the same as `hexo-engine`. Nothing to fetch,
nothing to unify, and the copy builds anywhere the workspace does.

The rules-bearing files — `state.rs` (phase machine, win/freeze), `legal.rs`
(the legal set and the packed action encoding), `rules.rs` (legality
precedence), `tactics.rs` (six-cell windows and win detection), `coord.rs`,
`board.rs` — retain their original logic byte for byte outside the table above.

## Running the differential test

```text
cargo test -p hexo-reference --test differential -- --nocapture
```

Defaults are sized to add roughly two seconds to a debug
`cargo test --workspace`. The heavy sweep is opt-in, in the same style as
`crates/hexo-engine/tests/smoke.rs` and its `HEXO_SMOKE_GAMES`:

| Variable | Meaning | Default |
| --- | --- | --- |
| `HEXO_DIFF_GAMES` | line-building games, which reach a win | 600 |
| `HEXO_DIFF_UNIFORM` | uniform games, which spread the board wide | 20 |
| `HEXO_DIFF_PLIES` | ply bound for a uniform game | 128 |

```text
HEXO_DIFF_GAMES=60000 HEXO_DIFF_UNIFORM=400 HEXO_DIFF_PLIES=400 \
    cargo test --release -p hexo-reference --test differential -- --nocapture
```

Each run prints a one-line summary per driver, including how many plies the two
engines listed the legal moves in the *same order* — an observation, never an
assertion, since the contract is the set and not the order.

## Lints

This crate does **not** take `[lints] workspace = true`. The workspace policy
warns on `missing_docs`, and the vendored sources do not document every public
enum variant or struct field; documenting them would mean editing the frozen
copy. So `missing_docs` is relaxed **for this crate only**, rather than
weakening the policy that `hexo-engine` is held to. `clippy::doc_lazy_continuation`
is relaxed for the same reason: it fires on the vendored prose, and silencing a
style lint beats reflowing a frozen doc comment.

`unsafe_code = "forbid"` is kept — the old engine contains no `unsafe` and
compiles cleanly under it — as is `clippy::correctness = deny`.

## Edition

Pinned to `edition = "2021"`, the edition the original was written for, instead
of inheriting the workspace's 2024. Keeping the copy compiling under its own
edition is part of keeping it verbatim.
