# MantisNet-ACT v4 — deviations from the build specification

`docs/MANTIS_ACT_SPEC.md` is normative. Every point at which the implementation
departs from it is recorded here, with the section it departs from and the
reason. A reader of the spec must read this file too; nothing is deviated from
silently.

## §20 — same-turn second-placement modeling is not implemented

Owner ruling, on cost. Partner rows were the largest tensor family in the
representation and the bulk of the build: measured on real stack-939 self-play
positions at ply 161 they were about 84,000 rows, 4.5 MiB of a 7.6 MiB position
graph, and 20.7 ms of a 29.2 ms build — 71% of the time to encode a position.
Weighed against what they buy, they are not worth it. §20 is therefore dropped
whole rather than kept behind a flag: there is no pair module, no pair tensor,
and no `pair_scope` field, so a configuration can neither ask for partner
modeling nor ask for its absence.

Absent as a consequence, exactly:

- §20 entire — the scopes of §20.1, the collinear enumeration of §20.2, the
  evidence rows of §20.3, the tactical scope of §20.4, the controls of §20.5,
  and §7's `(dst_action, partner_coord, evidence_kind, window_identity)` row
  order.
- §3's required component 10. Components 1–9 and 11–15 are unaffected.
- §6's `use_action_pair_messages`, `pair_scope`, and `pair_max_distance`.
  `architecture_hash` changes with the field set; ACT has no checkpoints yet, so
  nothing on disk is invalidated.
- §22's action-block step 2. The block's other five steps are unchanged and in
  their given order.
- §25's seven `pair_*` fields, and with them the `pair_offsets` tensor the §25
  entry below adds.
- §26's `pair evidence rows` packer line and the `O(E_pair)` term of the
  complexity target.
- §29's `full_no_pair`, which no longer names an architecture distinct from
  `full_act_v4`. Every other preset of §29 is present.
- §30's tests 15, 16, and 17, and §31's test 9.
- §33's `pair evidence` line of the structural alias signature. The other four
  lines still compose it.
- §34's `pair rows` and `newly legal prospective partner rows` telemetry, and
  the per-phase `pair row counts` split. The other per-phase diagnostics stay.
- §35's ablations 7 and 8.
- §36's Stage A `prospective pair builder` and Stage D `same-turn partner
  messages`.
- §37's acceptance criterion 9.
- §38's "Partner modeling remains internal" and "Recommended partner mode
  includes newly legal second-placement cells", which now describe nothing.

Untouched, each being a neighbour of §20 rather than a part of it:

- §13.1's three-way OPENING/FIRST/SECOND phase and its §13.2 FiLM. The phase is
  a §13 feature: the model still needs to know which placement of a turn it is
  on, and `moves_remaining` still decides the KLENT return sign.
- §19's counterfactual action encoder in full — all 18 post-placement rows per
  legal action, `action_window_index`, `action_post1_class`,
  `action_pre_status`, and §19.3's deterministic tactical vector. It is now the
  only place the model learns what a placement does, which raises its weight
  rather than lowering it.
- §21's action-set latents, and §24's auxiliary heads including action
  auxiliaries 5 and 6, "winning second partner exists" and "winning second
  partner count". Those are labels computed deterministically from the board and
  a hypothetical placement; they read no pair row and never did.

Measured on the same 24 real stack-939 self-play games, `full_act_v4`, best of
three builds per position:

| ply | build before | build after | bytes before | bytes after |
|----:|-------------:|------------:|-------------:|------------:|
|  21 |     15.39 ms |     3.29 ms |   3287.7 KiB |   665.1 KiB |
|  61 |     20.58 ms |     4.86 ms |   4886.3 KiB |  1409.9 KiB |
| 121 |     25.44 ms |     7.05 ms |   6597.5 KiB |  2441.0 KiB |
| 161 |     29.21 ms |     8.53 ms |   7744.2 KiB |  3130.7 KiB |

## §28 — representation version is a new constant, not a bump of the old one

The spec asks to "bump `MODEL_REPR_VERSION` to the next repository value".
`MODEL_REPR_VERSION` is a Rust constant in `crates/models/mantisnet/src/lib.rs`
that gates the legacy encoder and is checked by every existing checkpoint load.
Bumping it invalidates every MantisNet checkpoint, which §32 and §37.13 forbid
in the same document.

ACT therefore takes the next repository value, 4, as its own constant
`MANTIS_ACT_REPR_VERSION` in `mantisnet/models/mantis_act/config.py`.
`MODEL_REPR_VERSION` stays at 3 and continues to describe the legacy
representation. This satisfies §2's requirement to "use a new model
representation version" and leaves §32 and §37.13 intact.

## §5 — the builder is split further than the suggested layout

§5 gives a "suggested layout" with a single `builder.py`. The builder is split
into `windows.py`, `cells.py`, and `actions.py`, with `builder.py` retained as
the orchestration entry point. §5's own instruction is to "add a new
package rather than expanding one monolithic file"; the split follows that
instruction further than the illustrative file list does.

## §6 — `axis_pool_mode` added to the config block

§12.5 requires a selectable invariant head pooling mode
(`"mean"` or `"learned_attention"`), but the §6 dataclass omits the field. It is
added with default `"learned_attention"`, the mode §12.5 recommends.

## §29 — the parameter-matched control preset is named

§16 and §29 both require a parameter-matched extra-FFN control alongside
`full_with_typed_window_attention`, but neither names it. It is added to
`PRESETS` as `full_extra_ffn_control`.

## §2, §25 — KLENT dispatch seam

§2 and §25 require the external `network_evaluate` interface to be unchanged.
It is. Internally, `mantisnet/klent/train.py::_policy_q` reaches into
`model.trunk(batch)` and `model.cell_head_logits(w, g, batch)` — a shape
contract specific to MantisNet's stone/window trunk that ACT's
cell/window/action/latent trunk does not have. Both architectures instead expose
`policy_q(batch) -> (policy_logits, critic_logits)` and `_policy_q` calls that.
The change is confined to one private function; every caller of
`network_evaluate` is unaffected.

## §9.2, §10.1, §19.2 — the class-count checks raise rather than `assert`

`python -O` strips `assert`. A table reaching a builder with the wrong number of
classes trains a silently aliased embedding that no downstream test can detect,
which is the exact failure mode `CLAUDE.md` names as the hazard here. The checks
raise `AssertionError` from a helper, so the failure type and the spec's wording
survive and only the stripping is gone.

## §11.1 — `orbit_table` refuses `d_max > 12`

§11.2 places `RELATION_FAR` at id 48, immediately above the radius-12 orbits. A
radius-13 table has more than 48 classes and would silently collide with the
reserved band, so a wider radius is a relation-id-space change rather than a
table parameter and is refused with a message naming the value.

## §29 `full_coarse_geometry` — on/off-axis is a binary flag

The coarse relation is `2 * distance_bucket + off_axis`, not the specific axis
index. Putting the axis index in the class would make the relation depend on
absolute axis identity, which §11.3 and §12.2 forbid. The specific axis remains
available separately as the route, exactly as on the orbit path.

## §25 — packed container naming and contents

The batch is `PackedACTBatch` rather than `PackedACTInput`, carries every §25
field name verbatim, and adds the CSR offset tensors §25 omits but §26 requires
(`adjacency_offsets`, `radius_offsets`, `position_count`).
`cell_qr`, `cell_is_occupied`, and `window_id` stay on the per-position graph and
are not collated: coordinates are builder metadata and §7 forbids the model
seeing them.

## §8.3 — `legal_to_cell_index` may be all `-1`

Under `cell_scope = "occupied_only"` no legal cell is a node, so the mapping is
entirely sentinel. §29 requires that preset to construct, and §25's convention
already gives `-1` the meaning "no such entity". The graph validator refuses a
*mixture* of named and sentinel entries, which is a real detector: a legal cell
is empty, so `occupied_only` omits all of them and the other two scopes hold all
of them. A mixture is a half-built node set.

## §6 — the three numeric-feature toggles produce zero-width blocks

`use_window_numeric_features`, `use_global_numeric_features`, and
`use_action_tactical_features` make their feature block absent rather than a
column of zeros that a learned projection would still read. §25 leaves those
widths free. Consumers must handle an `(N, 0)` block.

## §13.3 — the global vector is eight named scalars

§13.3 lists six bullets, the last naming three window fractions. The resolved
vector is `log1p(stones)`, own and opponent stone fraction, `log1p(legal)`,
`log1p(windows)`, and the own-live/opponent-live/mixed window fractions. A zero
denominator yields `0.0` rather than a NaN, and the count it divides is carried
separately so the pair cannot read as a real ratio.

# Findings that qualify the specification

These are not deviations. They are measured facts about Hexo that change what a
spec decision buys, and they belong with the contract.

## §8.1 — `window_and_legal` and `occupied_and_legal` are the same node set

Provably, under Hexo's rules: every cell of a nonempty six-cell window lies
within five steps of one of its stones, and the legal radius is eight, so every
empty cell of a persistent window is already a legal cell. Measured equal at
plies 20, 60, and 120, where `n_cells == stones + legal` exactly.

The §8.1 distinction between those two scopes is therefore vacuous on this game.
`occupied_only` is the only cell scope that changes anything, and it changes the
model rather than only its cost. There is no cheaper middle scope to retreat to.

## §4 — mixed windows are the majority of the window set in real play

Measured on stack-939 self-play positions:

| ply | windows | mixed | mixed share | mean nearest-stone distance |
|----:|--------:|------:|------------:|----------------------------:|
|  20 |     252 |    86 |       34.1% |                        1.24 |
|  60 |     594 |   287 |       48.3% |                        1.13 |
| 120 |     969 |   575 |       59.3% |                        1.04 |
| 160 |    1198 |   758 |       63.3% |                        1.03 |

So `window_scope = "nonempty"` roughly **doubles** the window family over
`live`, and by ply 120 the majority of the board's window structure is invisible
to a live-only representation. This is the measurement that motivates §4's
default, and §35.2 is a structural ablation rather than a marginal one.

**Benchmarks must use real self-play positions, not random playouts.** Uniformly
random legal play scatters stones about 2.7 cells apart and puts mixed windows at
4%; trained self-play is contact play at about 1.03 and puts them above 60%.
Every node and edge family scales off that density, and in opposite directions —
random positions have five times more legal cells than a real position of the
same depth, while carrying a fifteenth of the mixed windows. A cost table or a
budget tuned on random playouts describes a board this model will never see.

## Scope: the old MantisNet is kept

The instruction that opened this work was to rewrite rather than keep legacy
code. The spec that followed it requires the opposite for the old architecture
specifically: §7, §5, §32, and §37.13 all state that MantisNet remains
independently importable, selectable, and unaffected, because it is the control
every ablation in §35 is measured against.

Both are honoured as follows: `mantisnet/models/mantis_act/` is written clean,
with no shim, no shared module, and no compatibility path back to MantisNet;
MantisNet itself is left exactly as it is. Nothing in the new package imports
from the old model, and the two share only the engine.
