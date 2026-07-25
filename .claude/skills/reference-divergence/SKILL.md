---
name: reference-divergence
description: Read when the differential test against crates/hexo-reference fails, or when any change would touch that crate. hexo-reference is a frozen copy of the previous engine kept as the workspace's only independent oracle; a divergence is a finding to report, not a test to fix, and neither engine is presumed correct.
---

# A differential divergence

`crates/hexo-reference` is a verbatim, frozen copy of the previous engine's
rules, vendored as a test oracle. `crates/hexo-engine` is a ground-up rewrite of
the same rules, so the two are independent implementations of one specification.
That independence is the entire value, and it is easy to destroy by accident.

## The rule

**Report the divergence. Do not patch it away.** Specifically:

- Never edit `crates/hexo-reference` to make a test pass. Editing it destroys
  the only independent evidence in the workspace, and it does so invisibly —
  the suite goes green and the finding disappears.
- Never weaken, narrow, or `#[ignore]` the differential test for the same
  reason.
- **Neither engine is presumed correct.** `hexo-engine` is the one that changed,
  so it is the likelier suspect — but "the rewrite is newer" is not an argument,
  and the previous engine has its own known defects (see the vendoring table in
  that crate's README for what was already deliberately removed).
- `docs/ENGINE_SPEC.md` is the tiebreak. If the spec settles which behaviour is
  correct, say so and cite the section. If it does not, the divergence is an
  open question for the owner, not a judgement call to make while fixing
  something else.

Nothing outside `tests/` may depend on the crate, and nothing in it should be
"improved", reformatted, or refactored.

## Narrowing one down

The test prints a one-line summary per driver. Re-run it directly for the full
output, and scale it up to find a rare shape:

```text
cargo test -p hexo-reference --test differential -- --nocapture
```

`crates/hexo-reference/README.md` documents the `HEXO_DIFF_*` variables that
scale it — the defaults are sized to add about two seconds to `cargo xtask
test`, and the heavy sweep is opt-in.

A divergence reproduces from the move list in the failure output — both engines
replay a move list, so the case is a game prefix, not a board state.

One thing the summary reports is *not* an assertion: how many plies the two
engines listed the legal moves in the same order. The contract is the set, not
the order. A drop there is an observation.

## Reporting it

State the move prefix that reproduces it, what each engine says, and which
section of `docs/ENGINE_SPEC.md` bears on it (or that none does). That is the
deliverable — a divergence is a finding, and resolving it is a separate,
deliberate decision.
