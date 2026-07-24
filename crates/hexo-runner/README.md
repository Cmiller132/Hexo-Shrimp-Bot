# hexo-runner

Match orchestration: the authoritative game loop and player communication.

**Status: EMPTY — scaffold only.**

## Shape

Pure Rust library crate, depends only on `hexo-engine`.

```
crates/hexo-runner/
  Cargo.toml
  README.md
  src/
    lib.rs        # crate root and public surface
```

## Planned module map

Proposed split, not yet built.

| Module | Role |
| --- | --- |
| `game.rs` | Owns the one canonical `State` for a game and the turn loop. The only code that advances canonical state. |
| `player.rs` | The transport-agnostic player interface: seat assignment, position handoff, move request, result notification. |
| `record.rs` | Game record as initial position plus move stream — the same thing the wire protocol carries. |
| `adjudication.rs` | Failure and termination policy: illegal move, timeout, crash, resignation, draw conditions. |
| `error.rs` | Error types for protocol and adjudication failures. |

## Design notes

- **Players never hold a mutable handle to canonical state.** They receive
  their own position and submit candidate moves; the runner validates before
  applying. This is enforced by ownership, not by convention — the canonical
  state is a private field, exposed only as an owned copy or a shared borrow.
- **The interface is designed for a remote player first.** An in-process player
  is the easy case; if the interface only works in-process, it will have to be
  rebuilt the day a player lives in a container.
- **Adjudication policy is explicit and lives here.** What happens on an
  illegal move or a timeout is a rule of the *match*, not a rule of the *game*,
  and it is much cheaper to decide now than to retrofit.

## Connections

- Depends on `hexo-engine` for rules and state.
- Model crates implement the player interface. The runner never learns what a
  model is.
