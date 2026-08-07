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

Two modules the list does not name hold the fused kernels §36 permits once the
reference exists: `segment_message.py` behind `messages.py` and
`latent_attention.py` behind `latents.py`. Each is the Triton implementation of
one module's inner loop plus the torch formulation it is held against, and
neither is importable from the package's `__init__`. They are separate files
because a kernel, its CSR views, its guards and its parity reference are as
long as the module that calls them, not because they are a second interface.

## §6 — `axis_pool_mode` added to the config block

§12.5 requires a selectable invariant head pooling mode
(`"mean"` or `"learned_attention"`), but the §6 dataclass omits the field. It is
added with default `"learned_attention"`, the mode §12.5 recommends.

## §29, §35 — the extra-FFN control cannot be parameter-matched with `ffn_mult`

§16 and §29 both require a parameter-matched extra-FFN control alongside
`full_with_typed_window_attention`, but neither names it. It is `PRESETS`'
`full_extra_ffn_control`. It is **not** parameter-matched, and with the levers
§6 defines it cannot be made so.

Typed window attention adds 19,424 parameters to each of the four state blocks
— two norms at 176, the invariant stream's bias-free q/k/v at 12,288 and its
output projection at 4,160, the same pair on the axis stream at 1,728 and 600,
one `(num_heads, 48)` class bias per stream at 192 each, and the residual's two
LayerScales at 88 — so **77,696** in all. `full_extra_ffn_control` adds
**336,440**, which is 4.33 times the quantity it is supposed to hold constant:

| arm | parameters | vs `full_act_v4` |
| --- | ---: | ---: |
| `full_act_v4` | 1,726,468 | — |
| `full_with_typed_window_attention` | 1,804,164 | +77,696 (+4.50%) |
| `full_extra_ffn_control` | 2,062,908 | +336,440 (+19.49%) |

`ffn_mult` is the only FFN width §6 exposes, it is global — the state trunk's
cell and window FFNs, the action encoder's row and pool MLPs, the action blocks
and both head bodies all scale on it — and it is an integer. The control's
reachable budgets are therefore 0, +336,440, +672,880, and so on; 77,696 is not
among them and the granularity is larger than the target. This is a property of
§6's field set, not of the implementation, which is why the residual is stated
here rather than closed.

What would close it is one config-level lever, and the arithmetic is already
known. An extra window-only `EquivariantFFN` stage in each state block — which
is literally what §35's ablation 13 says the control is, "no typed attention
plus parameter-matched FFN" — costs 19,216 per block at `ffn_mult = 2`, leaving
a residual of 832 over the four blocks: 1.1% of the added budget and 0.05% of
the model, against the present control's 433%. Making it exact needs the
stage's two hidden widths free rather than tied to `ffn_mult`, and
`352 + 129 * hidden_inv + 49 * hidden_axis = 19,424` has the single sensible
solution `hidden_inv = 111`, `hidden_axis = 97`. Both options are a change to
`config.py`, which this change does not own.

Until one lands, `full_extra_ffn_control` is a *width* control and not a
parameter-matched one, and ablation 13 must be read as such: it separates
"a direct typed window path" from "19.5% more parameters spent on width",
not from "the same parameters spent on width".

## §2, §25 — KLENT dispatch seam

§2 and §25 require the external `network_evaluate` interface to be unchanged,
and it is. The problem was internal: `klent/train.py::_policy_q` reached into
`model.trunk(batch)` and `model.cell_head_logits(w, g, batch)` — a shape contract
specific to MantisNet's stone/window trunk, which ACT's cell/window/action/latent
trunk does not have.

Both architectures now expose
`policy_q(batch) -> (policy_logits, critic_logits)`, and `_policy_q` calls that
and nothing else. The change to `klent/train.py` is confined to that one private
function; `network_evaluate`'s signature, its return triple, and every one of
its callers are untouched.

`MantisNet.policy_q` is the one edit to `mantisnet/model.py`, and it is purely
additive: it composes the model's own `trunk` and `cell_head_logits` in the
order `_policy_q` used to, changes no existing method, no tensor shape, and no
checkpoint layout, and the old model's full test suite is unaffected. This
satisfies §32's "old MantisNet behavior and checkpoints remain unaffected"
and §37.13.

## §7, §16 — window identities reach the device, as a join key and nothing else

§7 says "coordinates and window identities are builder metadata only. Do not
embed raw coordinates or absolute axis IDs." §16's typed collinear/crossing path
is a relation between two window *identities*: whether two windows are on one
line and how far apart their starts are, or where the one cell their two lines
cross sits in each of their spans. There is no way to have the path without the
identities somewhere.

`collate` therefore carries `window_id` into `PackedACTBatch`, and
`messages.window_window_edges` joins the pairs on the device once per forward.
`cell_qr` stays behind; nothing else was added. The prohibition §7 actually
states is honoured exactly: no coordinate and no absolute axis id is ever an
embedding index or a parameter selector. What the identities produce is the
48-class collinear/crossing vocabulary, every class of which is a D6 invariant —
`tests/act/test_act_window_attention.py` replays a real game through all twelve
transforms and requires the whole edge set to map window for window with its
classes unchanged, which is the property the arm's shared per-class bias table
rests on.

Deriving the edges beside the model rather than in the builder is a cost
decision, not a convenience. A window carries 132 to 302 typed partners across
plies 21 to 161 of real self-play, so the edge views are two orders of magnitude
larger than the identities they are a join of: at ply 161 the four int64 arrays
are 10.6 MiB per position against the 3.06 MiB the whole position graph
occupies. The identities cross the bus at 24 bytes a window; the edges never
cross it at all.

The join, its class folds and the flash kernels that reduce over them are
`mantisnet/window_pairs.py`'s, imported rather than reimplemented. That module
is MantisNet's §5.1c path, it is board-level rather than architecture-level, and
a second copy of it under `mantis_act/` would be two implementations of one job.
`WINDOW_WINDOW_RELATIONS` is read off it for the same reason.

One bound moved. The pair key identifies a line by its offset from the origin —
`start_r`, `start_q`, or, on the QR axis, `start_q + start_r` — and packs that
offset by `2**16` into a `2**17` field, so a *sum* of two coordinates has to stay
inside `2**16`. `windows.WINDOW_COORD_LIMIT` is therefore `2**15`, the int16
range the engine's own move coordinates live in, where it used to be `2**20` for
this package's own identity packing alone. It is now one number for the
representation rather than one per packer, and `packed.py` bounds `window_id`
against it on every graph, however built. Real play sits within a few hundred of
the origin, because the board grows at most eight cells a move.

### What the arm costs (§16)

§16 requires the arm's parameter and time cost to be reported. Measured on eight
real stack-939 self-play prefixes per ply, whole model through `policy_q`, bf16
autocast on a 4070 Ti, best of eleven after five warm-up steps:

| ply | windows | pair edges | per window | radius edges | fwd base | fwd typed | step base | step typed | peak base | peak typed |
|----:|--------:|-----------:|-----------:|-------------:|---------:|----------:|----------:|-----------:|----------:|-----------:|
|  21 |   1,863 |    246,513 |      132.3 |       66,273 |  63.4 ms |   72.1 ms |  169.6 ms |   183.9 ms |  686 MiB |   724 MiB |
|  61 |   4,384 |    952,184 |      217.2 |      208,812 |  63.4 ms |   68.7 ms |  174.7 ms |   186.6 ms | 1094 MiB |  1187 MiB |
| 121 |   6,957 |  1,954,147 |      280.9 |      424,727 |  63.5 ms |   69.2 ms |  178.8 ms |   187.8 ms | 1401 MiB |  1555 MiB |
| 161 |   8,779 |  2,652,111 |      302.1 |      571,841 |  65.0 ms |   71.5 ms |  171.9 ms |   185.6 ms | 1666 MiB |  1873 MiB |

Parameters are +4.50%. On a batch of eight the step is 1.08x and the forward
1.10x, which understates the arm: at that size both models are launch bound and
the attention hides inside the launch overhead. What it actually costs shows up
as the chunk grows, because the pair family does not. At ply 121:

| chunk | base | typed | ratio |
| ---: | ---: | ---: | ---: |
|  8 |  44.8 pos/s |  41.3 pos/s | 0.92x |
| 16 |  90.9 pos/s |  78.7 pos/s | 0.87x |
| 32 | 166.1 pos/s | 115.3 pos/s | 0.69x |

Two structural costs sit behind that. The pair family is the largest in the
representation by a wide margin — 4.6x the radius family at ply 161, where the
radius family was previously the largest thing in a position — and it grows
faster than linearly in windows, because a denser board puts more windows within
reach of each other. And the joins are data-dependent: the step's device
synchronisations go from 2 to 10, undoing most of the reduction the §26/§27
entry below records. Both are properties of `window_pairs.pair_tables`, which
MantisNet's own §5.1c arm already pays for; making them sync-free is a change to
that module, not to this one.

`use_full_cell_attention=True` and `phase_conditioning="token_only"` remain
refused by name in `state_trunk.refuse_unimplemented_paths`. §29 names no preset
that asks for either.

## §6 — the model is below the 2.5-4M parameter target

§6 asks the default model to "target roughly 2.5–4 million trainable
parameters". Built at §6's own widths and depths, `full_act_v4` holds
**1,726,468**:

| subsystem | parameters | share |
| --- | ---: | ---: |
| state trunk (§18) | 1,266,168 | 73.3% |
| action encoder (§19, §21, §22) | 330,344 | 19.1% |
| policy/critic heads (§23) | 129,956 | 7.5% |

Left as measured rather than padded. §6 fixes `d_inv=64`, `d_axis=24`,
`d_rel=24`, four state blocks, two action blocks, and one private block per
head; reaching 2.5M would mean widening something the spec names a value for,
and choosing which one is a model decision rather than an implementation one.
The single largest lever is `d_inv`, which every subsystem scales on: the
latent passes, the incidence messages, and both private adapters are all
quadratic in it. Raising it is an owner decision, and this entry is where the
shortfall is recorded until one is made.

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
`cell_qr` and `cell_is_occupied` stay on the per-position graph and are not
collated: coordinates are builder metadata and §7 forbids the model seeing them.
`window_id` is collated, and is the one name past §25's list — see the §7, §16
entry above for why the pair join needs it and what it is and is not allowed to
do with it.

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

## §26, §27 — every bound the forward used to read back is stated on the host

§26 budgets the forward's *work*; it says nothing about where a bound is
checked, and the first implementation checked each one where it was used. On
CUDA that is not free: `int(t.min())`, `bool(t.all())`, `int(offsets[-1])` and a
`repeat_interleave` that has to discover its own output length each stall the
host until the queue drains. Measured on real stack-939 self-play at ply 161,
batch 16, the model's `policy_q` forward performed **166 device
synchronisations**, 136 of them in the state trunk, on a step that was already
host-bound — roughly 100 ms of CPU against 82 ms of GPU.

Every one of them was a *check*, a *size*, or a *structural property*, and each
class is answered differently. Nothing is deleted for being expensive; a check
that stays is a check that still catches what it caught.

**Bounds move to the host stage that already owns loud failure.**
`ACTGraph._validate` bounds every column of every family in numpy, per position,
before a tensor exists, and it runs from `__post_init__`, so it reaches every
producer rather than the ones that remember to call it. The model-side re-reads
of those same columns were redundant work rather than a second detector, and
each construction site now names the packer line that covers each of its
columns. One check was *not* covered and is added: `_check_consistency` refuses
a §15.2 radius edge whose source cell is empty — a semantic claim whose
violation produces a perfectly in-range relation and is invisible to every
shape, dtype and round-trip check downstream. `collate`'s `_refuse_crossing`
survives for a different reason than it was first given; see the §7/§25/§26
entry below.

**Sizes come from shapes the host already holds.**
`latent_attention.row_positions` now takes the family's row count and passes it
as ATen's `output_size`. That is not a hint traded for a check: ATen refuses a
value disagreeing with the offsets' own cumulative total
(`result_size == cumsum_ptr[size - 1]`, reached by both the CPU and the CUDA
kernel), so the equality that `LatentPass._ordered`, `ActionEncoder.forward` and
`ActionHeads` each read back separately is now enforced by the call that
consumes it, on every call rather than once per pass. `RaggedStream` carries the
resulting vector beside its offsets, so the trunk builds one per family per
forward instead of twenty.

**Structural properties travel with the family that has them.** `TypedEdges`
gains two required host-side booleans. `dst_sorted` says the rows arrive
destination-ascending — §7's order, checked per graph by
`_check_ordering` — so `message_plan` adopts the destination view instead of
probing the largest family for an answer §7 already fixes. `fully_routed` says
every row carries a real axis, which is `packed`'s bound on `window_axis` and
`adjacency_axis`. Neither has a default: a wrong flag is a silent fault, so a
caller must state it, and each of the three builders states it with the file and
line of its guarantee.

**One quantity had nowhere host-side to come from, so the packer records it.**
`radius_orbit`'s ceiling is the one index space in this representation that a
configuration resizes, which is why `_VALUE_RANGES` leaves it open above. The
model still has to refuse a batch built for a wider §11.2 relation space than
its own, and nothing else can: `RelationGatedMessage._check` compares the model's
config against itself. `collate` therefore records `radius_orbit_bound`, taken in
numpy where the arrays already are, and `messages.radius_edges` compares two
host-side integers. This is the one field §25 does not name that this phase adds.

Measured on the same real stack-939 prefixes, RTX 4070 Ti, bf16 autocast,
`full_act_v4`, `policy_q` — the KLENT seam — forward and forward+backward:

| subject | ply | batch | syncs | launches fwd/step | GPU fwd/step ms | wall fwd/step ms |
|---|---:|---:|---:|---:|---:|---:|
| model | 161 | 8 | 166 → **2** | 4789 → 4332 / 11032 → 10575 | 20.81 → 19.62 / 67.67 → 63.76 | 83.6 → 77.0 / 236.7 → 219.7 |
| model | 161 | 16 | 166 → **2** | 4792 → 4333 / 11012 → 10553 | 37.52 → 36.25 / 113.69 → 112.66 | 88.8 → 94.0 / 240.4 → 226.6 |
| trunk | 161 | 8 | 136 → **2** | 3804 → 3431 / 8620 → 8247 | 19.51 → 16.18 / 49.48 → 62.64 | 65.5 → 58.3 / 172.5 → 164.4 |
| trunk | 161 | 16 | 136 → **2** | 3806 → 3431 / 8614 → 8239 | 26.51 → 25.38 / 81.80 → 80.75 | 69.6 → 67.4 / 172.6 → 174.1 |

Peak allocated is unchanged to within 0.7 MiB everywhere. The backward performed
no synchronisation before this phase and performs none now.

**The two that remain are irreducible without a change to the packed format.**
`incidence_edges`' `mask.nonzero()` discovers how many window slots the cell
scope represents; `radius_edges`' `(axis >= 0).nonzero()` discovers how many
displacements lie on an axis. Neither is a check, both are once per batch rather
than once per block, and no host tensor carries either count. They would go if
`collate` emitted the incidence edge list and the routed radius subset directly
— it holds both in numpy — which is a §25 format change and a separate piece of
work.

**What got weaker, stated plainly.** Two refusals that raised a named
`ValueError` from Python now fail elsewhere:

- A family whose offsets do not end at its row count trips ATen's
  `result_size == cumsum_ptr[size - 1]`. On CPU that is a `RuntimeError`; on
  CUDA it is an asynchronous device-side assert naming `Repeat.cu:20`. Loud, and
  now checked on every call rather than once per pass, but less legible than the
  message it replaced.
- `TypedEdges` no longer refuses an out-of-range index. `packed.py:363-372` and
  `packed.py:674-680` cover every column of every family this package builds; a
  hand-built `TypedEdges` in a test is not covered.

A third was withdrawn. `PhaseFiLM`'s range check was removed on the argument
that the host-side gate was strictly stronger, and the argument was true above
the vocabulary and false below it: torch's advanced indexing *wraps*, so
`phase_row[-1]` is the last phase's row — in range, wrongly typed, and refused
by nothing at any stage. `-1` is this representation's own sentinel, so the
value is not hypothetical. The selection is now `index_select`, which
bounds-checks in both directions and costs no synchronisation to do it, and the
host-side gate is what makes the refusal a named one on the path that matters;
see §13.1 below.

The property this rests on has its own test rather than being an argument:
`test_the_forward_stalls_the_host_at_most_twice` counts the trunk forward's
synchronisations on a real batch — of `full_act_v4`; §16's typed window
attention arm pays eight more for its data-dependent join, which the §7, §16
entry above measures — and
`test_the_structural_flags_describe_the_families_they_travel_with` and
`test_section_7_orders_every_family_but_the_reverse_incidence` hold both
declarations to the data they claim to describe. Held against a reconstruction
of the pre-change tree with deterministic algorithms pinned, `policy_q`'s
forward and all 3,360 parameter gradients are **bit-identical** at batch 8 and
16 on real positions.

## §13.2, §14 — two modules evaluate the spec's formula somewhere cheaper

§13.2 and §14 fix what these two modules compute. Neither says where the
arithmetic happens, and both were computing it in the most expensive place
available. The functions are unchanged; the parameters, their count, and the
initialisation distribution are unchanged.

**§13.2's FiLM is a function of the phase class, so it is evaluated on the
classes.** `PhaseFiLM` gathered an embedding row per entity and ran the phase
MLP and both output projections over every cell and window row — three GEMMs
and an embedding over 18,410 rows to produce at most three distinct results,
since §13.1 gives the phase three classes and a budget-packed batch usually
carries one. The chain now runs on `embed.weight` itself, which *is* the table
of embedded classes, and the per-entity step is a one-hot row selection and the
affine. The selection is a matmul rather than a gather deliberately: a gather's
backward is `embedding_dense_backward`, which sorts tens of thousands of indices
to accumulate them into three rows, where `one_hot.T @ grad` is a matmul over a
three-column operand with no sort and no atomic. The one-hot is exact in every
float dtype, so it selects the row rather than approximating it.

**§14's update MLP is a linear over a concatenation, so it is one linear.**
`_PairMLP` held two input linears whose results were added — the standard trade
that avoids materialising the wide input. Under autocast that trade is the wrong
way round. An `nn.Linear` call casts its weight, its bias and its input, runs the
GEMM, and casts three more times in its backward; measured on this trunk
`aten::copy_` is **61% of every launch a step makes** (2,836 forward and 3,086
backward of 9,776) and those parameter casts are most of it. A second input
linear therefore costs about twelve launches where the concatenation that
removes it costs one. It is not extra memory either: autograd saves one
`(N, d_a + d_b)` operand where the split form saved a `(N, d_a)` and a
`(N, d_b)` one.

Measured on real stack-939 self-play prefixes, RTX 4070 Ti, bf16 autocast,
`full_act_v4`, the state trunk, both formulations alternating in one process so
the allocator and clock state are shared:

| | ply 161 batch 8 | ply 161 batch 16 |
|---|---|---|
| peak allocated MiB | 1227.1 → **1213.9** | 2381.6 → **2366.5** |
| launches, forward | 3952 → **3888** | 3944 → **3880** |
| launches, backward | 5872 → **5512** | 5872 → **5512** |
| launches, step | 9824 → **9400** | 9816 → **9392** |
| device syncs, step | 2.7 → 2.7 | 2.7 → 2.7 |
| GPU ms, step (min of 5 traces) | 48.5 → 61.9 | 80.7 → **79.8** |
| wall ms, step (min of 9 rounds) | 144.9 → **142.0** | 147.9 → **142.2** |
| wall ms, step (median) | 154.5 → 156.1 | 150.3 → **146.1** |

Launch counts are exact and reproduce across processes: −424 a step, −4.3%, at
both batch sizes. Wall clock follows them down by 2 to 4%, which is what a
host-bound step should do. The device-time column is the one to distrust: a
host-bound step lets the GPU drop clocks between launches, and at batch 8 the
per-trace medians swing ±14 ms in adjacent rows and cancel. Neither change moves
GPU work materially, which is the expected result — they delete host launches,
not arithmetic.

**How the identity was established, since two of these numbers are the whole
argument.** Per module, in float64: the rewritten `PhaseFiLM`'s forward and
input gradients are **bitwise identical** to the formulation it replaced and its
parameter gradients agree to 1e-15; `_PairMLP` agrees to 5e-16 throughout.
Whole trunk, in float64, at plies 21/61/121/161 and batches 4 and 8: the forward
is **bitwise identical** at every configuration, 90% of the 1,205 live parameter
gradients are bitwise identical, the p99 is 1.4e-12, and the worst is 9.6e-8,
concentrated on the latent read's value-projection biases whose gradient is a
heavily cancelling sum over eighteen thousand rows.

Comparing the two paths against each other in bf16 was not treated as a gate,
because it is not a question with an answer — two summation orders of a
cancelling sum differ by tens of percent in three decimal digits whether or not
either is wrong. The question that does have an answer is which is closer to the
truth, and against a float64 reference at ply 161 batch 8 the two paths are
equidistant: forward median error 5.08e-3 rewritten against 5.14e-3 replaced,
gradients 4.87e-3 against 4.85e-3, both at bf16's own epsilon. A null control
confirms the trunk is bitwise deterministic run to run in fp32, so the float64
differences above are caused by the change and not by the hardware.

**What was found and not acted on**, so the next phase does not re-derive it:

- `state_edges` still runs inside `StateTrunk._run`, and the eight CSR message
  plans are still built lazily inside block 0. Measured with the harness's
  plan-cache probe at ply 161 batch 16, that is **616 launches a step (6.3%)**
  and one of the three remaining forward syncs — block 0 costs 1,917 launches
  against blocks 1–3's 1,312. It is not fixed by caching on `PackedACTBatch` as
  such: a training batch is used once, so the work only leaves the step if the
  collater builds the edges and their plans as CPU tensors and `to()` moves
  them. That is a §25 packed-format change — `collate` has to take the config,
  and `TypedEdges`, `MessagePlan` and `StateEdges` each need to be device
  movable — and `CLAUDE.md` says a format change is bumped and regenerated
  rather than bolted on, so it is its own change and not a rider on this one.
- `incidence_reduce="attention"` materialises an `(E, d)` tensor, and it is
  **off the default path**: `config.py:135` defaults it to `"sum"` and no preset
  in `PRESETS` sets it. It is a §14 ablation, correctly deprioritised, and
  `messages.py`'s own docstring already says so — "the price of an ablation, not
  of the default". Left alone.
- Every **key projection's bias in every latent attention** is a structurally
  dead parameter: a softmax is invariant to a constant shift of its scores, a
  key bias shifts them all equally, and `q . b` cancels. Their gradient is
  exactly zero — 48 of the trunk's 1,253 parameter tensors, found because they
  were the only ones a relative-error metric could not divide by. Raised, not
  removed: which of §17's projections carry a bias is a spec question.

## §19.2, §22.1, §27 — the action stack's two fused ops, and the one dtype they change

§19.2 fixes what the eighteen post-placement rows compute and §22.1 what an
action reads from the state latents. Neither says where the arithmetic happens.
Both were computing it as whole-tensor primitives over a family that is linear
in the legal action count — 285,012 rows at ply 161 batch 16 — so every
intermediate was an `(M, 64)` tensor autograd kept alive and nothing outside the
module ever read. The functions, the parameters, their count and their
initialisation are unchanged.

**§19.2's row encoder is two registered ops** (`post_rows.py`). The sentinel
gather resolves `-1` inside the kernel instead of concatenating the shared base
onto the window table, and everything from the gathered row to the gate's
product — the LayerNorm, the value projection, the relation's bias and gate
projections, the sigmoid and the multiply — is one op whose backward re-derives
the forward rather than storing it. The two-layer row MLP after it is left as
ordinary GEMMs: its weights are the only ones here wide enough that a fused
parameter gradient spills, measured at 24 ms against 4 for the same arithmetic.

The padded table is what the gather removes, and it is the backward that pays
for it: 78% of a real ply-161 batch's rows are sentinel (89% at ply 21), so a
pad row makes 221,688 rows scatter their gradient onto one address, in whatever
precision autocast chose, with CUDA emulating the bf16 atomic it has no
instruction for. Measured in isolation on that shape, the invariant stream's
gather is **1.20 ms → 0.39** forward-and-backward and the axis stream's
**0.72 → 0.33**.

**§22.1's read is `latent_attention.latent_broadcast`, not a second attention.**
Its context is one position's latents — a configured constant — which is §17.4's
broadcast exactly, with `R` context rows and `C` equal to 1 for the invariant
stream and 3 for the axis stream. The module previously gathered that context
onto every action and materialised an `(N, R, heads, head_dim)` fp32 score and
value tensor per stream per block. `CLAUDE.md` allows one implementation per
job, so the op that already exists is the one used, and `_attend` is gone.

**The one deviation, and it is §27's direction.** The fused row op emits the
promoted accumulator, as `at_least_fp32` means everywhere else in this package;
the eager chain emitted whatever autocast chose for the projections, which is
bf16. So under bf16 autocast §19.2's gated product is now fp32, and the row MLP
does the rounding instead. It is strictly the more accurate order — one rounding
of an fp32-exact product rather than three roundings of its terms — and it is
what makes the fp32 parity below meaningful, but it costs one `(M, 64)` fp32
write and its read back, which is most of why the backward's device time is flat
rather than lower.

Measured on real stack-939 self-play prefixes, RTX 4070 Ti, bf16 autocast,
`full_act_v4`, whole model, both formulations alternating in one process so the
allocator and clock state are shared, and with the trunk untouched in both:

| ply 161 | batch 8 | batch 16 |
|---|---|---|
| peak allocated MiB, step | 1820.2 → **1670.0** | 3616.4 → **3305.3** |
| peak allocated MiB, forward | 217.1 → 217.1 | 436.7 → 436.7 |
| launches, step (whole model) | 12035 → **11905** | 12028 → **11896** |
| launches, step (this change's modules) | 933 → **803** | 934 → **802** |
| launches, forward (this change's modules) | 345 → **283** | 344 → **282** |
| device syncs, step | 2.2 → 2.2 | 2.2 → 2.2 |
| GPU ms, step (this change's modules) | 11.07 → **8.77** | 20.49 → **17.18** |
| GPU ms, step (whole model) | 72.79 → **65.61** | 113.89 → **110.12** |
| wall ms, step (median of 40, alternating) | 221.6 → 226.0 | 223.2 → 226.6 |

Launch counts are exact and reproduce across processes: −132 a step at both
batch sizes, which is −14% of what these modules launch and −1.1% of the step.
Split by module at batch 16: §19.2's encoder 288 → 216 launches and 14.93 →
13.25 ms, of which the forward is 5.59 → **3.99**; §22.1's read 492 → 432 and
4.27 → **2.68 ms**. The forward peak does not move because a `no_grad` forward
frees the intermediates anyway — the memory this change saves is saved
activations, which is the step column.

**Wall clock did not move**, and that is the honest reading rather than a
measurement problem: −132 launches out of 12,028 is 1.1%, and the alternating
medians differ by less than the 2% this GPU drifts within one run. The step is
host-bound on more than launch submission.

**How parity was established.** Three layers, in `tests/act/test_act_post_rows.py`
and one test in `tests/act/test_act_action_encoder.py`. A per-row Python oracle
in float64, written from §19.2's own lines and sharing no indexing with either
implementation, holds the torch reference on boards small enough to loop over.
`gradcheck` in float64 — which falls back to the reference by signature — checks
the analytic reference backward by finite differences. The kernels are then held
against that reference on the builder's real row grids at both stream widths and
in both channel modes; the §22.1 substitution is held against the gathered chain
it replaced, restated literally in the test, which is what catches a transposed
key or a softmax over the wrong dimension that a shared op cannot catch for
itself.

End to end on real prefixes at plies 21/61/121/161 and batches 8 and 16, with
*both* halves of this change put back — the eager row encoder and the gathered
§22.1 read — every one of 1,647 tensors per configuration, both logit families
and every parameter gradient in the model, agrees in fp32 to better than 2e-4
relative; the worst meaningful disagreement over all 13,176 comparisons is
**4.3e-5**, and the isolated ops' own reassociation is 5e-7. The only quantities
excluded are the ones whose true value is zero — the latent attentions' key
biases, recorded above — which are checked against the model's gradient scale
instead.

In bf16 the same sweep leaves 2 comparisons of 13,176 outside 6%, both of them
one of those zero gradients. The largest real disagreement is 2.2%, and that is
not a question with an answer: against the same model run in fp32, on the two
private-adapter context gradients that move most, the eager path is 0.5871 away
and the fused path 0.5849, and across all 1,645 parameters the fused path is
further from fp32 on 238 of them — a coin flip, which is what equal accuracy
looks like at bf16's epsilon.

**What was found and not acted on:**

- `nn.Embedding`'s backward is still 62 of §19.2's 166 backward launches:
  `post1` (729 classes) and `pre_status` (4) each cost 31 `embedding_dense_backward`
  launches sorting 285,012 indices into their table. The four-class table is a
  `one_hot @ W` away from being a matmul, as §13.2's FiLM already is above; the
  729-class one is not, because its one-hot is 285,012 × 729. Folding both into
  the fused op instead needs a class-sorted CSR of the row grid so the table
  gradient stays a deterministic segment reduction, and that is a per-batch
  sort — the same `PackedACTBatch` format change the trunk's note above defers.
- `StateContextBroadcast` still runs eight `nn.Linear` calls per block, four of
  them over 96 rows, and each costs about twelve launches under autocast for
  the parameter casts. `k_inv`/`v_inv` share an input and `q`/`o` are the only
  two that touch an action-sized tensor. Fusing them is the same job as the
  latent passes' forty linears, and belongs with that change rather than beside
  it.
- `window_rows` returns a `WindowRows` rather than an `EquivariantState`
  because a row's axis half is a single channel, which the container's shape law
  correctly refuses. **That does not obstruct fusion** and was not changed: the
  ops work on flat `(M, D)` tensors and the container is a Python-level shape
  law above them. It has to stay on the forward path, though — the §12.2
  negative control in `test_act_action_encoder.py` overrides `window_rows` and
  requires the forward to read it — which is why the gather and the row gate are
  two ops rather than one, and why the `(M, D)` gathered tensor still exists.

## §26, §2 — the packer limit is the architecture's law, so the seam carries it

§26 asks for the packer limits to be extended to cover ACT's families. They
could not be: `fitloop.pack_chunks` did not have limits to extend, it had
MantisNet's law written into the loop — `(positions + 1) * width²` against
`pair_budget`, and summed legal cells against `cell_budget`, where `width` is
the chunk's longest position. Neither term exists in this architecture. ACT
pads nothing, so it has no quantity that is quadratic in a chunk's longest
position, and a limit of `2^63` on a term ACT does not have is a sentinel, not
a budget.

So the loop and the law were separated. `fitloop` keeps the position cap, the
deterministic descending pack order and the singleton rule — a sample too large
for any chunk is still fitted alone rather than dropped — and takes a
`ChunkCost` for everything that is a property of a representation's memory.
`mantisnet/builder.py::PaddedPairChunkCost` is MantisNet's law, moved out of the
loop unchanged and still under the same test; `packed.py::ACTChunkCost` is this
one's.

The KLENT seam grew by two methods for the same reason `policy_q` exists: the
fitter holds `(game, ply)` pairs and no representation at all.

- `collate_prefixes(games, ts)`. What a batch of stored prefixes *is* depends on
  the configuration here — `window_scope` and `cell_scope` decide the node set,
  `d_max` the relation vocabulary — so the model that will consume the batch is
  what builds it. `klent/train.py::_rebuild` previously called MantisNet's Rust
  builder by name, which is why the seam was incomplete: `_policy_q` was
  architecture-neutral but the tensor it was handed was not.
- `chunk_cost(stones, legal, budgets)`. Both arguments are known from a stored
  sample without building its graph, which is what a packer needs, since the
  graph is what packing decides the size of.

`FitBudgets` therefore records every limit the trainer offers and each
architecture reads its own: `pair_budget` and `cell_budget` are MantisNet's,
`graph_cell_budget` is ACT's. `MantisNet.collate_prefixes` and
`MantisNet.chunk_cost` are additive in the sense §32 and §37.13 require — they
delegate to the module-level builder and to the law lifted verbatim out of the
loop, and change no existing method, tensor shape, or checkpoint layout.

`fitloop._PREFETCH_DEPTH` moved from four to two on the measurement below: it is
both the queue depth and the concurrent-worker count, and this builder is Python
where MantisNet's is Rust.

Self-play collation is still MantisNet's `collate_positions`, so there is no ACT
collection path to size from live play. The lab's no-grad passes are a
collection path all the same, and they now run this architecture, so
`KlentConfig.collect_graph_cell_budget` exists and is read: it is
`3 * ACT_GRAPH_CELL_BUDGET`, tripled by the same rule that makes MantisNet's
`collect_pair_budget` and `collect_cell_budget` triples of their fit limits. A
no-grad chunk holds no backward graph, and nothing else about the quantity
changes, so the ratio is inherited rather than measured again. It is not a
field nothing reads: `lab/train.py::pack_inference_chunks` packs validation and
evaluation with it.

That packer no longer contains a law of its own. It was MantisNet's — padded
attention pairs against `collect_pair_budget` — written out a second time
beside `PaddedPairChunkCost`, and it is now `fitloop.pack_chunks` over the
model's own `chunk_cost` under the collection budgets. On MantisNet the two
produce the same chunks by construction: the packer's sort was already
descending padded width, its break condition already
`(size + 1) * width² > pair_budget or cells > cell_budget`, and its first
sample already fixed the width.

## §33 — the signature is hashed rather than compared, and the geometry is sampled

§33 asks for a command that "hashes each legal action's builder-side structural
signature". The first implementation hashed a Python object it built per action
— four nested tuples, about 200 items — and then described the geometry of
*every* legal action so that a sample could be read against it. Measured on real
stack-939 prefixes at ply 161, that was **38.2 ms of signature and 42.4 ms of
geometry against a 9.6 ms build**: eight times the cost of the thing it
describes, on a diagnostic whose whole purpose is to be run over a corpus.

Three changes, and each is a change to how a fixed quantity is computed rather
than to what it is:

- **Each of §33's four lines is one 64-bit polynomial hash over a padded
  table.** Codes are stored one above their natural value so a pad is `0` and
  contributes nothing to the sum, which is what keeps a digest a function of
  the line's own content rather than of the width of the table that held it —
  two graphs of different sizes still agree on an action they describe
  identically. The canonical order under the group is by digest: a per-axis
  group is hashed and the three digests sorted, which drops the absolute axis
  id exactly as sorting the contents did. The pass is `np.lexsort`,
  `np.bincount`, and one multiply-and-sum per table, with no per-action Python.
- **`ActionSignatures.lines` is one uint64 column per line, not a tuple of
  dicts**, and `value` is derived from them in `__post_init__` rather than
  passed in, so a signature cannot carry a digest its own lines disagree with.
  `digest(lines)` is replaced by `combine(lines)`, which folds the four columns
  at once; deleting a line — §33's negative control — is replacing a column
  with a constant and recombining.
- **`alias_report` takes a callable that describes a group's rows**, so the
  geometry of an action is computed only for the groups actually sampled.
  `graph_geometry` gained a `rows` argument and is otherwise unchanged.

The digests are different numbers than the tuple hash produced. Nothing stores
them: they are compared within one report and, in the D6 test, between two
reports of the same run.

| ply | legal | build | signature before | signature after | grouping | sampled geometry | whole-set geometry |
|----:|------:|------:|-----------------:|----------------:|---------:|-----------------:|-------------------:|
|  21 |   505 | 2.6 ms |  9.95 ms | **0.80 ms** | 0.03 ms | 0.03 ms | 11.1 ms |
|  61 |   892 | 5.2 ms | 21.40 ms | **1.89 ms** | 0.04 ms | 0.04 ms | 24.7 ms |
| 121 |   998 | 7.9 ms | 30.46 ms | **3.17 ms** | 0.04 ms | 0.04 ms | 33.3 ms |
| 161 |  1159 | 9.6 ms | 38.22 ms | **4.27 ms** | 0.05 ms | 0.05 ms | 42.4 ms |

So a whole §33 report at ply 161 is **4.4 ms against a 9.6 ms build**, down from
80.6 ms — 45% of the build rather than 840% of it, which is the difference
between a diagnostic that is run and one that is quoted.

**What got weaker, stated plainly.** A 64-bit hash can collide where a tuple
comparison could not, and a collision reports two unlike actions as an alias
group. At the scale this runs — 17,461 legal actions over the register's own
24-position sweep — the birthday probability is about 8e-15, and a collision is
visible rather than silent: `test_what_alias_the_full_model_does_retain`
requires every member of a group to share the orbit multiset the geometry line
hashes, which a collision would break. The full corpus sweep reproduces the
figures the tuple implementation recorded exactly — 24 groups, 95 aliased
actions, largest eleven — and `test_act_diagnostics.py` holds the two passes to
the same *partition* on eight real positions and on the planted board below,
against an exact tuple oracle that shares no arithmetic with the hash.

**One field was added to the geometry descriptor, because §33's own reading
demanded it.** §33 asks a sampled group for its "differing omitted geometry",
and every field of the descriptor described the radius-`d_max` disk — which is
what the orbit line already hashes, so on a group the orbit line put together
they can only ever report "identical". `omitted_stones` is the multiset of
`(hex distance, colour)` of the stones *outside* that disk. It is distance and
colour rather than an orbit id because §11.1 caps the orbit table at radius 12
and the omitted geometry is by definition beyond it.

## §34 — the auxiliary accuracy needs labels, so §24.1's labels exist

§34 splits three model diagnostics by OPENING/FIRST/SECOND. Two are functions of
the forward's own outputs. The third, `action auxiliary accuracy`, is a function
of a *label*, and `heads.py` computes none — §24 heads emit logits and the mask
of rows that carry a label, and the label was the training stage's, of which
this architecture has none for auxiliaries. The line was therefore
unimplementable as the package stood.

`mantis_act/aux_labels.py` is the missing half: §24.1's six labels for every
legal action, deterministic, no search, derived from the eighteen pre- and
post-placement window codes `actions.py` already gathers. It is a separate
module from `heads.py` because one is torch over a packed batch and the other
numpy over a position's own tables, and it is not called by `builder.py`
because a label is a training quantity and the builder emits the
representation.

Three readings §24.1's field list leaves open, resolved in the module's
docstring and repeated here because they are contract: `own_max_occupancy` is
an occupancy and not a threat, so it counts own stones whatever else the window
holds; `opponent_threats_hit` counts the post-placement rows whose window held
four or five opponent stones and no own stone, which is exactly the pair §19.3
feeds as `opponent_five_windows_hit` and `opponent_four_windows_hit`; and
`winning_partner_count` counts distinct *cells*, so two windows sharing one
empty cell are one second placement. A placement that wins outright has partner
count `0`: the turn ends on a win, so there is no second placement to label.

The two partner labels need no partner representation, which is why they
survived §20's removal. A cell wins as a second placement exactly when some
window holds five own stones and it after the first placement, and such a
window either already held five own stones and one empty — and then that empty
cell is one of the actions `win_now` marks, the same set for every first
placement — or is one of the first placement's own eighteen rows, holding five
own stones and no opponent stone afterwards.

`§34`'s fourth per-phase line, **`pair row counts`, is omitted**. §20 is gone by
owner ruling, so the column would be zero for every phase, and a zero column
reads as a measurement rather than as an absence. The §20 entry above already
lists the per-phase split among what that ruling removed; this entry is where
the *emitting* code says so.

Two more resolutions §34 leaves to the implementation:

- **Q standard deviation is taken over the top eight policy actions**
  (`TOP_POLICY_ACTIONS`). §34 says "among top-policy actions" without a count.
  The question the line asks is whether the critic separates the moves the
  policy would consider; over the whole legal set the spread is dominated by
  the halo, which the policy never ranks.
- **A phase the batch does not hold is not reported**, and a model holding an
  auxiliary head with no label supplied is refused by name rather than reported
  with the accuracy column quietly missing.

Cost, RTX 4070 Ti, real stack-939 prefixes, `full_act_v4` with both partner
auxiliaries built:

| stage | ply 61, batch 16 | ply 161, batch 16 |
|---|---:|---:|
| forward | 46.8 ms | 53.9 ms |
| §34 phase split | **2.95 ms** | **2.86 ms** |
| §24.1 labels, all 16 positions | 26.7 ms | 37.2 ms |

The split is 5–6% of the forward it describes and is cheap enough to run every
epoch. The labels are **2.3 ms per position at ply 161, about 27% of that
position's build**, and most of that is the window enumeration `position_aux_
labels` repeats because it takes a position rather than a built graph. A
training stage that wanted them every step would pass the `ActionTables` the
builder already holds to `action_aux_labels` and pay only the row arithmetic;
nothing needs that yet, so nothing threads it.

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

## §26 — the binding quantity is cells plus occupied cells, and it is not the largest family

Measured on real stack-939 self-play prefixes, `MantisACT.policy_q` forward and
backward under bf16 autocast at `full_act_v4`, peak allocated against batch size
at a fixed ply. Peak is affine in the batch, so its slope is the per-position
cost; dividing that slope by each §26 quantity says which one the memory is
actually proportional to (MiB per unit):

| divisor                       | ply 21  | ply 61  | ply 121 | ply 161 | spread |
|-------------------------------|---------|---------|---------|---------|--------|
| **cells + occupied cells**    | 0.1551  | 0.1561  | 0.1558  | 0.1550  | 0.7%   |
| cells                         | 0.1608  | 0.1677  | 0.1744  | 0.1767  | 9.9%   |
| legal actions                 | 0.1670  | 0.1813  | 0.1981  | 0.2054  | 23%    |
| post-action rows (18 × legal) | 0.00928 | 0.01008 | 0.01101 | 0.01141 | 23%    |
| window-slot incidences        | 0.0624  | 0.0405  | 0.0321  | 0.0297  | 110%   |
| cell adjacency edges          | 0.0282  | 0.0292  | 0.0302  | 0.0306  | 8.5%   |
| radius-12 occupied edges      | 0.0110  | 0.0052  | 0.0033  | 0.0028  | 290%   |

**The largest family is not the binding one.** Radius edges outnumber cells 62:1
at ply 161 and drive almost none of the memory, because the fused segment
message recomputes them in backward rather than holding them. A budget expressed
in radius edges would be wrong by a factor of four across the ply range, and
would have to be measured from a built graph, which a packer cannot do.

**The winning unit is computable from a stored sample.** `cells == stones +
legal` holds exactly on this game (the §8.1 finding above), so cells plus
occupied cells is `2 × stones + legal actions` — a sample's ply and the length
of its stored π′, both known before anything is built. That is why the budget is
in this unit rather than in cells alone, which is a further 10% loose and no
cheaper to compute.

Peak allocated per unit is 0.1535 MiB at chunk sizes large enough for the fixed
constant to wash out (measured 0.1531–0.1553 over 24 to 64 positions at ply 161
and over 16 to 96 positions on a ply-1-to-200 mixture).

## §26 — the ACT budget, and why it is where it is

`ACT_GRAPH_CELL_BUDGET = 48,000` units. Throughput through the KLENT seam
against chunk size, on a card with about 11 GiB free:

| ply-161 chunk | units  | peak MiB | reserved | step ms | positions/s |
|--------------:|-------:|---------:|---------:|--------:|------------:|
| 8             | 10,103 |   1,670  |   1,732  |   182   |   44.0      |
| 16            | 20,986 |   3,306  |   3,416  |   175   |   91.4      |
| 24            | 31,531 |   4,909  |   5,342  |   182   |  132.0      |
| 32            | 42,208 |   6,546  |   7,092  |   220   |  145.3      |
| 40            | 54,013 |   8,440  |   9,168  |   275   |  145.3      |
| 48            | 64,951 |  10,080  |  10,864  |   414   |  116.0      |
| 56            | 75,329 |  11,595  |  12,480  |   905   |   61.9      |

The step is *flat* from 8 to 24 positions — 182, 175, 182 ms — which is the
whole argument for filling the batch: below the knee every extra position is
free, because the step's cost is its launch count and not its arithmetic.
Throughput plateaus at 42,000 units and the card runs out at 65,000, where the
allocation is served over PCIe under WDDM rather than refused. The budget sits
between the plateau and the cliff, at roughly 7.4 GiB of a 12 GiB card.

On a mixed-ply buffer, which is what a collected iteration holds, the same
budget gives about 46 positions per chunk and 190 positions/s, against 122
positions/s at the batch of 16 the architecture was previously exercised at.

## §26 — one preparation worker cannot feed the ACT fit path, and four is worse than two

This builder is Python and numpy; MantisNet's is Rust and releases the GIL.
Positions built per second on the same box, on real prefixes:

| builder   | ply 21 | ply 61 | ply 121 | ply 161 |
|-----------|-------:|-------:|--------:|--------:|
| ACT       |  361   |  169   |   101   |    86   |
| MantisNet | 5,710  | 4,134  | 3,542   | 3,113   |

and against `fitloop._PREFETCH_DEPTH`, which is both the queue depth and the
worker count, on a mixed-ply chunk:

| concurrent workers | 1 | 2 | 3 | 4 | 6 | 8 |
|--------------------|---|---|---|---|---|---|
| ACT positions/s (chunk 48)  | 127 | 212 | — | 151 | 101 | 98 |
| ACT positions/s (chunk 96)  | 122 | 216 | 213 | 154 | — | — |
| MantisNet positions/s (chunk 304) | 7,306 | 7,263 | — | 8,058 | 7,791 | 7,947 |

**One ACT worker does not keep up**: 86 positions/s built at ply 161 against 145
the GPU can consume there, and 127 built on a mixture against 190 consumed.
**Two workers do** — 212 to 216, above the GPU either way — and four does not,
at 151 to 154. The loss at four is GIL contention plus
numpy thread oversubscription, and it reproduces at both chunk sizes. MantisNet
is indifferent: its builder is 4.5x faster than its own GPU step at every worker
count, so the constant is doing nothing for it.

The same ladder measured over a whole `klent.train.fit` epoch of 512 real
prefixes, which is the quantity that decides the constant:

| `_PREFETCH_DEPTH`     |   1  |   2  |   3  |   4  |   6  |
|-----------------------|-----:|-----:|-----:|-----:|-----:|
| ACT samples/s         | 108.6| **114.6** | 113.5 |  98.8 |  82.1 |
| MantisNet samples/s   | 1497 | 1457 | 1465 | 1503 | 1483 |

MantisNet is flat across the whole ladder — its builder is four times faster
than its own GPU step at every depth — and ACT peaks at two. The constant is
therefore two. It stays one number rather than becoming a queue depth and a
worker count separately: depth three is within noise of two, so there is no
measured demand for a deeper queue behind two workers, and a knob with no
measurement behind it is speculation.

## §13.1 — the phase selection is bounds-checked, not wrapped

`PhaseFiLM` selects its modulation row with `phase_row.index_select(0, id)`
rather than `phase_row[id]`. The two agree on every id the vocabulary holds and
disagree on every one it does not: advanced indexing wraps a negative subscript,
so `-1` reads the *last* phase's row and the model receives the SECOND phase's
scale and bias with nothing raised at any stage — the symmetric fault
`CLAUDE.md` names, since applying and un-applying it is identical and no
round trip, shape, or invariant check can see it. `-1` is not a hypothetical
value here: it is the representation's one sentinel, carried by
`window_cell_index`, `legal_to_cell_index` and `radius_axis_or_neg1`, so a
future site that gathered a phase through a sentinel-bearing column would land
on it.

`index_select` refuses both ends. It costs no synchronisation, which is what the
removed min/max check cost — twenty host stalls a forward — so the §26 property
that the forward stalls the host at most twice is untouched. On the host the
`IndexError` is translated into a `ValueError` naming the field and the range;
on CUDA an out-of-range gather is an asynchronous assert, which is why the
*named* refusal for anything reaching the module through a `PackedACTBatch` is
still the host-side one at `packed.py:491-502` — that one runs on Python ints,
also cross-checks the phase against `moves_remaining` and the occupied-cell
count, and now runs from `ACTGraph.__post_init__` rather than from whichever
builder remembered to call it.

## §7, §25, §26 — validation is structural, and `collate` owns only its own arithmetic

`ACTGraph._validate` runs from `__post_init__`. Before, it was a public method
builders called before returning, `collate` documented that its inputs "are
assumed to have passed" it, and `collate` never called it — so the whole
host-side defence was conditional on which producer built the graph, a thing
neither `collate` nor the model can see.

Two consequences, both taken rather than papered over:

- **`cell_nearest_bucket` is closed in `_VALUE_RANGES`.** It was open above,
  with the ceiling stated in one builder helper (`cells._bounded_buckets`) on
  the reasoning that a bucket vocabulary "belongs to the module that emits
  them". It does not: `NEAREST_BUCKETS` is `LEGAL_RADIUS + 2`, fixed by the
  rules and not by any configuration, so it is a schema bound and it now lives
  in `packed.py` beside the field table with the helper deleted. Verified
  before the change: `validate()` accepted `9999` and `CellEmbedding` then died
  with an unnamed `IndexError` on CPU and an uncatchable device-side assert on
  CUDA. It now raises `cell_nearest_bucket must be <= 9`, naming field and row.
- **`collate` re-checks nothing a graph states about itself.** Its two
  per-value floor checks in `_to_global` were a second copy of `_INDEX_FIELDS`'
  floors, and adding a family's own offset to an index already inside that
  family's range cannot move it out of its position's slice — so
  `_refuse_crossing` cannot fire on anything a graph can carry either. It is
  kept, because it is not redundant with the graph's bounds at all: it is the
  check on *collation's own arithmetic*, and a field shifted by the wrong
  family's offsets is a fault that applies and un-applies identically and
  leaves every index in range for the batch. Its test corrupts the offsets
  rather than the graph, which is what it was always detecting.

## §31, §36 — the position law runs on the device as well as the host

The D6 suite's three detectors are not equally device-sensitive. Two read
parameters and modules, so a device tells them nothing. The third reads
arithmetic, and this package's arithmetic is not the same code on both devices:
`segment_message` and `latent_attention` dispatch to Triton only when their
inputs are on CUDA. The suite ran on the host alone, so §31 had never been put
to a fused kernel, and a kernel that read a fixed absolute axis channel instead
of the row's own would have passed CI.

The law now runs once per available device on the same six samples. Measured on
an RTX 4070 Ti: 0 of the comparisons over budget, worst drift 12% of its slack,
and 192 `segment_message` plus 144 `latent_attention` fused acceptances counted
over the run — the acceptance count is asserted, so a silent fallback to the
torch reference cannot pass as a device run. The forbidden constructions are
introduced under the kernels too, and all five the law declares still fire
there: `constant_axis_route` on 62 comparisons at 36,365x its tolerance,
`absolute_axis_embedding` on 47 at 526x, `per_channel_latent_base` on 33 at
413x, `fixed_order_concat` on 13 at 215x, `per_axis_bias` on 10 at 260x.

## §24.1 — four of the six auxiliaries almost never fire on trained self-play

Measured with `aux_labels.action_aux_labels` over 144 real stack-939 positions —
16 games at plies 21 through 181 — **119,344 legal actions**:

| auxiliary | class distribution |
|---|---|
| `own_max_occupancy` | 1: 56.9%, 2: 26.7%, 3: 11.4%, 4: 4.2%, 5: 0.8% |
| `opponent_threats_hit` | 0: 99.960%, 1: 0.028%, 2: 0.012%, 3: 0.001% |
| `own_five_windows_after` | 0: **100%** |
| `win_now` | 0: **100%** |
| `winning_partner_exists` | 0: **100%** |
| `winning_partner_count` | 0: **100%** |

The zeros are not a bug and they are not an accident of the sample; they are
what strong play *is* on this game. **The mover never holds an unanswered live
four**, because the opponent has just moved and blocking one is forced — so a
window the mover can take to five own stones with one empty cell left does not
exist at the moment it is the mover's turn. Every one of the 969 actions that
reaches five own stones in a window does so in a window that already holds an
opponent stone, which is a dead shape. The same asymmetry explains the one
family that does fire in the other direction: the *opponent's* live fours are
still on the board 0.04% of the time, because it is the mover who has not
blocked yet.

So three of §24.1's six auxiliaries have a constant label on every real
training state, and a fourth is one part in 2,500. Training a head on a constant
label teaches the model the prior and nothing else. This is recorded rather than
acted on — §24 makes every auxiliary optional and off by default, and §35 names
no ablation for them, so the measurement is what a future ablation would need
before it is worth running. `test_act_diagnostics.py` exercises all six on a
crafted engine-legal game instead, where both sides hold a live four.

## §23.3, §29 — the lab scores the critic it trains, and the state-value term is conditional

The lab's third loss term and `state_value` score channel both
read a binned state-value head. §29 gives `full_act_v4` none, and §23.3 makes
one optional. Two ways out were available and both were wrong: adding the head
to the ACT presets so the harness has something to read changes the
architecture to fit the measuring instrument, and deleting the term for
everyone changes what MantisNet — the control of every §35 ablation — is
trained and scored on, retroactively invalidating every number already in
`ABLATIONS.md`.

Neither was needed, because ACT is not missing a critic. `policy_q` returns the
action-value categorical logits, which is the critic KLENT actually trains and
the one every previous corpus comparison in this project was decided on
(stack-939 against joint-939 was exactly that comparison). That is `v_hat`, the
channel §5 already composes from those logits, and it is the arms' shared
ground.

So: the critic channel is scored off `policy_q`'s logits for every
architecture, and the state-value loss term and score channel exist exactly
where that head does. `has_state_value_head` is what each architecture answers —
`True` for MantisNet, `False` here — and an ACT cell's `config.json` records the
two weights it trains rather than three. §23.3's head is refused in
`supervised_heads` rather than silently unscored when it is enabled: it emits a
scalar in `[-1, 1]` over the state latents, not the bin logits `losses.value_loss`
reads, so instantiating it under this recipe would count parameters that
nothing trains and no channel reports.

The supervised pass reaches all of this through `supervised_heads`, a second
seam beside `policy_q` rather than a caller of it. Both quantities come off one
trunk output, and the lab running the trunk once for the logits and again for
the value would double the cost of every MantisNet cell — including its backward
graph — to obtain tensors the first pass already had.

The refactor is numerically inert on MantisNet, which is the property that
matters, since it is what every recorded lab number is comparable against. A
CPU cell retrained after the change produced a byte-identical
`checkpoint_final.pt` (sha256 `d67bdb6b…`) with identical per-epoch losses, and
re-scoring two unchanged checkpoints produced byte-identical `scores.json`.

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
