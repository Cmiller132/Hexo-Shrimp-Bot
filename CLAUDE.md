# Hexo-Shrimp-Bot

## The game

An infinite hex board in axial coordinates `(q, r)`, with lines along three
axes. `P0` opens at the origin; after that each player places two stones per
turn. A placement is legal if the cell is empty and within 8 hex steps of some
occupied cell. Six or more of your own stones in a row along one axis wins,
checked after *every* placement — so a turn can end on its first stone. No
draws, passes, or captures, and stones are permanent. `hexo-engine` is
authoritative; `docs/ENGINE_SPEC.md` is the normative target.

## How to write code here

Keep it simple, and keep it consistent with what is already there. Simple means
not building for requirements that don't exist yet; it does not mean the cheap
version of the work in front of you. Where the harder approach is genuinely more
robust, take it — and finish it, rather than leaving a half-implementation.

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
