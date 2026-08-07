---
name: engine-change
description: Read before changing crates/hexo-engine — the rule machine, grid storage, growth policy, Zobrist hashing, window/win detection, the legal-move ordering, or undo. Explains which invariant tier catches which class of bug and when a golden-vector failure means bump the version rather than re-baseline. Not needed for hexo-runner or docs-only work.
---

# Changing hexo-engine

`crates/hexo-engine/README.md` is the normative reference for the engine. Read
the section that governs what you are changing. Where code and README disagree,
that is a finding to raise — not a discrepancy to resolve quietly in either
direction.

## The invariant tiers, and why they are not redundant

Symmetric bugs are the hazard: a wrong disk offset, a wrong hash constant, or
a growth copy with the same wrong index on both sides all apply and un-apply
identically. Round-trip and invariant tests are
structurally blind to them. Three independent detectors exist for that reason:

- **Tier C** — `debug_assert` inside `apply_raw`/`undo_raw` (§10.1). Runs on
  every placement in a debug build, which is why `cargo xtask test` uses the
  debug profile and why the release lint is a separate gate.
- **Tier A** — `Position::audit()` (§10.4), which checks in a defined order.
- **Tier T** — the brute-force oracles in `tests/common` and the frozen golden
  vectors (§10.5).

A change that makes one detector agree with the implementation *by construction*
has deleted a detector, not fixed a bug. If an oracle and the implementation now
share a helper, that oracle has stopped being independent.

## Golden vectors

`tests/golden.rs` pins the Zobrist hash and the canonical legal-move ordering
across processes. Its module doc states the policy and is the authority: a
failure there means a persisted artefact broke — stored game records,
cross-process hash agreement, or checkpoints that indexed a policy head by
legal-move position — and the response is a deliberate `RULES_VERSION` /
`ACTION_ORDER_VERSION` bump. Regenerating the vectors is legitimate only in that
same change, which is why the test's own message says to do both; a re-baseline
that leaves the version where it stood is forbidden, because it retires the
detector and leaves every artefact written under the old version claiming an
agreement it no longer has.

## What a change obliges

§11 lists the per-module unit obligations, the property tests, and the oracle
cross-checks. A new `MoveError` variant or a new reachable precedence pair needs
a test pinning it; §11 also records which precedence pairs are *unreachable* and
therefore deliberately untested, so absence of a test there is not a gap.

Verify with `cargo xtask verify`. If the change touches hashing, ordering,
growth, or win detection, also run `cargo xtask smoke` — the per-push `test`
gate is sized for speed, and those are the classes that only show up over
thousands of playouts.
