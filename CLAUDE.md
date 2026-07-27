# Hexo-Shrimp-Bot

## The game

An infinite hex board in axial coordinates `(q, r)`, with lines along three
axes. `P0` opens at the origin; after that each player places two stones per
turn. A placement is legal if the cell is empty and within 8 hex steps of some
occupied cell. Six or more of your own stones in a row along one axis wins,
checked after *every* placement — so a turn can end on its first stone. No
draws, passes, or captures, and stones are permanent. `hexo-engine` is
authoritative; `docs/ENGINE_SPEC.md` is the normative target.

## Verifying

`cargo xtask verify` runs every gate CI runs. `cargo xtask` with no argument
lists them and says what each one catches; `xtask/src/main.rs` is where they are
defined. Nothing else states a gate's command line, so do not assemble one from
a README — that drift is exactly what the xtask exists to prevent.

## Gotchas

**A green `cargo test` is not a green build.** Four of the eight gates catch a
class the test run cannot see at all. Report work as verified only after
`cargo xtask verify`.

**Symmetric bugs are the hazard that matters here.** A wrong disk offset, a
wrong hash constant, or a growth copy with the same wrong index on both sides
all apply and un-apply identically — so no
round-trip or invariant test can see them. `Position::audit()`, the independent
oracles in `crates/hexo-engine/tests/common`, and the frozen golden vectors are
the only detectors. They are not redundant with each other, and making one of
them agree with the implementation *by construction* silently deletes a
detector rather than fixing anything.

**`docs/ENGINE_SPEC.md` is normative for `hexo-engine`.** Where the code and the
spec disagree, that is a finding to raise, not a discrepancy to quietly resolve
in whichever direction is less work.

**`target/` collides between Windows and WSL.** Set
`CARGO_TARGET_DIR=target-wsl` on the WSL side; both are gitignored.

## How to write code here

Simple means not building for requirements that don't exist yet; it does not
mean the cheap version of the work in front of you. Where the harder approach is
genuinely more robust, take it — and finish it, rather than leaving a
half-implementation.

Trim as you go. When something is upgraded the old version comes out in the same
change — no dual paths, no compatibility shims, no deprecated names kept alive.
One implementation per job: no reference-vs-fast variants of the same logic, no
registry of dead experiments. Formats are not backward compatible; bump them and
regenerate the data rather than teaching a reader two shapes.

Fail loudly. Missing or unexpected input is an error, never a silently
substituted default.

Comments say what the code does and why it does it that way, not the story of
how it got there. Keep each crate's `README.md` current. Prune `docs/` when a
document stops being true — a settled question moves to a `README.md` and leaves
the open-questions file — but prune deliberately, not aggressively.
