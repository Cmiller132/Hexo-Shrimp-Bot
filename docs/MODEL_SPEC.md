# MantisNet — model specification

The reference description of the MantisNet network: what it plays, what it
sees, how it computes, what it outputs, and how it is trained. It is written
for readers outside the project; no prior knowledge of the codebase is
assumed. Code comments cite this document's section numbers (`§5.1c`,
`appendix B`, …), so the numbering is part of the interface and is kept
stable.

Authoritative sources, where finer detail is needed: the rules engine is
specified in `crates/hexo-engine/README.md`; the training algorithm's
obligations in `docs/KLENT_FOR_HEXO.md`; the implementation in
`python/mantisnet/mantisnet/` with `model.py` as the root.

## 1. Overview

MantisNet evaluates positions of Hexo, a two-player placement game on an
infinite hexagonal board in axial coordinates `(q, r)`. Player 0 opens with a
single stone at the origin; thereafter each player places two stones per
turn. A placement must be an empty cell within 8 hex steps of some occupied
cell. Six or more of a player's stones consecutive along one of the board's
three axes win, checked after every placement — a turn can end on its first
stone. There are no draws, passes, or captures, and stones are permanent.

The network is a graph model, not a grid model: the board is unbounded, so
there is no canvas to convolve over. Its node sets are

- the **stones** on the board,
- the live **windows** — every six-cell line segment currently holding at
  least one stone,
- four persistent **state latents** carrying position-level context, and
- the **legal cells**, as persistent latent nodes carrying per-cell state
  through the trunk.

A trunk of B identical blocks passes messages between these sets and applies
attention within them. Three heads read the result: a policy over the legal
cells, a categorical action-value (the training-time critic), and a
distributional state value. All inputs are relative — own/opponent rather
than black/white, offsets rather than coordinates — and invariant under the
board's twelve symmetries, so the network cannot distinguish a position from
its rotation, reflection, or colour swap.

## 2. Named parameters

| Symbol | Meaning | Default |
| --- | --- | --- |
| H | embedding width, used everywhere | 128 |
| B | trunk blocks | 4 |
| A | attention heads | 4 |
| F | FFN expansion factor | 2 |
| Q | state-value readout queries | 4 |
| K | state-value bins (odd, so an exact-zero bin exists) | 65 |
| P_H | decoder MLP hidden width | 128 |
| V_H | state-value MLP hidden width | 128 |

Architecture knobs (validated jointly at construction):

| Knob | Default | Meaning |
| --- | --- | --- |
| `window_attention` | `True` | the §5.1c typed window-pair attention stage |
| `claim_reach` | `5` | §5.1c crossing-join reach; `{0, 5}`, a path selector |
| `cell_latents` | `False` | persistent covered-cell latents replacing the §5.1b relay |
| `line_pass` | `False` | whole-line window attention in the §5.1c slot |
| `cell_nodes` | `False` | extends persistent cell state to every legal cell, with radius edges |
| `cell_node_scope` | `"all"` | radius/adjacency edge destinations: `all` or `uncovered` |
| `cell_adjacency` | `False` | directed distance-one cell↔cell messages (requires `cell_nodes`) |
| `action_tactical` | `False` | encodes the §4.4 tactical scalars into the action row through a small MLP |
| `action_latents` | `False` | two invariant latents read/mix/broadcast over each legal set (§6) |
| `global_magnitude` | `False` | feeds three whole-position magnitude features into the §3.3 latent init |
| `orbit_vectors` | `False` | adds the §4.1 content term to the stone-attention bias |

The production training configuration at this writing is `cell_latents=True,
cell_nodes=True, cell_node_scope="all", window_attention=False`; the default
configuration keeps window attention on and cell state off. Parameter count
is 4,804,581 at defaults (pinned by tests) and 5,197,093 at the production
configuration. `orbit_vectors` adds `B × A × 51 × H/A` parameters on top of
whatever else is on — 26,112 at either of those shapes.

## 3. Input entities

### 3.1 Stones

One node per stone. The only per-stone feature is ownership relative to the
side to move (own / opponent, a 2-row embedding). Stone coordinates never
enter node features; geometry acts only through the relational structures
below and the displacement orbits of §4.1.

### 3.2 Live windows

A **window** is six consecutive cells along one of the three axes. A window
is **live** — and becomes a node — when it is nonempty: it contains at
least one stone of either player, including windows holding both players'
stones that can no longer be completed. Each live window's slot pattern
(per-slot: empty / own / opponent) is encoded under reversal
canonicalization into a vocabulary of **377 ternary patterns** — every
nonempty length-6 ternary string up to reversal — and the pattern embedding
is the window's initial state. A stone lies in up to 18 windows (six per
axis); a cell likewise.

### 3.3 State latents

Four position-level latent rows replace the single "global token" of earlier
designs. They initialize as a learned 4×H base plus an embedding of
`moves_remaining` (1 or 2 stones left in the current turn — the only
whole-position scalar the incumbent model consumes). Each block lets them
attend with the stones (§5.3), read the window set, mix among themselves, and
broadcast back to the windows (§5.4). After the final block their normalized
mean is the global context row `g` the heads consume.

With `global_magnitude` on, the init carries one further term, added
identically to all four rows — the channel is whole-position, so the slots
are not given separate copies of it to specialize from. That row sums three
whole-position magnitude features, read through two learned maps:

- the position's **window-pattern histogram**: the count of its live windows
  in each of the 377 canonical patterns of §3.2, under `log1p`, read through
  a learned 377×H table (the dense count vector is the additive term, so the
  table carries no bias);
- `log1p` of the position's **stone count** and `log1p` of its **legal-cell
  count**, the pair read through one learned linear.

Each is a count over D6-invariant inputs, so the channel preserves §8. The
table and the linear are zero-initialized, so a knob-on model begins at
exactly the knob-off function.

### 3.4 Legal cells

The engine supplies the legal placements in a canonical order (**engine
legal order**); every head's output is aligned to it. A legal cell is
**covered** when at least one live window contains it — equivalently, when
it has decoder incidence — and **uncovered** otherwise (no stone within
five cells along any axis line).

With `cell_latents` on, covered cells become persistent latent nodes: each
starts from one shared learned H-row (`cell_base`) and accrues identity
through typed reads (§5.1b). With `cell_nodes` on, *every* legal cell is a
node: uncovered cells initialize from one invariant feature embedding — the
bucketed nearest-stone distance (10 buckets) — while covered cells keep the
learned-base initialization, and all cells additionally read radius-edge
context from stones (§5.1b). The `cell_node_scope` knob restricts which cells
receive radius/adjacency edges (`all`, or only `uncovered` ones); every legal
cell keeps its latent and features regardless of scope.

### 3.5 Excluded inputs

Deliberately absent: absolute coordinates, axis identity as a feature, move
history, and ply number. The model is also **magnitude-blind at the position
level** — `moves_remaining` is the only whole-position quantity the latents
start from, so nothing tells them how large the position is beyond what the
message passing carries node by node. Everything the model knows arrives
through the entities above and the encodings of §4.

Two knobs breach that boundary on purpose, each stated where it enters:
`action_tactical` admits the hand-crafted tactical scalars of §4.4 into the
action row, and `global_magnitude` admits the three whole-position magnitude
features of §3.3 into the latent init. Both are off by default, and both are
zero-initialized so that turning one on changes nothing until training moves
its parameters.

## 4. Feature encodings

### 4.1 Displacement orbits

Stone self-attention (§5.3) is biased by the D6 orbit of each query–key
displacement: the same frozen **orbit-48** vocabulary the cell-node edges
use (§4.2), covering every displacement of hex distance 1..12, then FAR for
pairs beyond radius 12, SELF on the diagonal, and TOKEN for any pair that
touches a latent row — 51 bias rows per head.

The **static** part of the bias is learned in two pieces: a coarse table over
the 23 (distance, on-axis) classes of distances 1..12 plus FAR, SELF and
TOKEN (`axis_bias`, 26 rows), and a per-orbit residual (`orbit_bias`, 48
rows); an orbit's static row is its class row plus its residual, and
FAR/SELF/TOKEN are class rows alone. Both are zero-initialised, so the table
starts as a (distance, on-axis) table whose class rows pool gradient over
every displacement at a distance, and the residual adds orbit resolution
where the data asks for it.

With the `orbit_vectors` knob the bias gains a **content** part:

    score[m, n] = q[m]·k[n]/√d + static[h, bucket] + q[m] · orbit_vec[h, bucket]

`orbit_vec` is a learned vector per (head, bucket), 51 × H/A, that the
querying row projects onto, so what a head makes of a displacement depends on
the stone that is asking rather than on the geometry alone. Only the query
enters — the key's content already reaches the score through `q·k` — and the
term carries no scale factor of its own. It is zero-initialised, so a model
with the knob on begins exactly at the static table.

The kernel is handed the whole bias, either as the static table or, with the
knob on, as a table **per query row** of shape (positions, heads, rows, 51)
that the model composes before the call; the kernel knows only "a table
indexed by (row, bucket)" and its gradient is that table's bucket histogram —
pooled over the tile for the static width, per row for the wide one. Padding
is a finite sentinel column appended at compute time, not a learned row. The
orbit is symmetric under negating the displacement, so the static bias is
symmetric in the pair; the content term is not, since only the query
contributes it. The kernels gather the orbit from a constant 25×25
displacement table generated from the orbit function, never hand-written.

### 4.2 Cell-node vocabularies

Cell nodes read stones within hex radius 8 through **typed radius edges**.
The edge type is the product of three invariant classifications, 48 × 2 × 2
= **192 radius classes**:

- the **orbit-48** class: the source-to-destination offset vector's orbit
  under the twelve board symmetries, a frozen 48-class vocabulary generated
  from the axial transforms and independently checked against a
  cube-coordinate oracle — the one geometry vocabulary of the model, shared
  with the stone attention of §4.1;
- whether the source stone is **own or opponent**;
- whether source and destination are **on a shared axis**.

The optional adjacency pass uses a single class: all six distance-one
neighbours share one relation row (the per-axis split is emitted by the
builder for a future equivariant route but collapsed here). Nearest-stone
distance for cell features uses 10 buckets.

### 4.3 Ternary window classes and the incidence fold

Windows relate to the stones and cells they contain through **incidence
lists** typed by joint classes that fold in the window's whole ternary
pattern:

- stone ↔ window messages (§5.1, §5.2) use **1458 occupancy classes** —
  the joint of the window's canonical pattern and the stone's slot;
- cell ↔ window structures (the decoder incidence of §6 and the typed cell
  reads of §5.1b) use **726 decoder classes** — the joint of the pattern
  and the empty candidate slot.

Both vocabularies are folded under simultaneous reversal of pattern and
slot, so a window and its mirror image produce identical classes. Window
pair relations (§5.1c) use a **48-class** vocabulary typing
colinear/crossing geometry; `claim_reach` selects the crossing join — `5`
joins window pairs through any claimed cell within the donor geometry's
reach, `0` restricts to pairs sharing an in-span cell. The whole-line pass
uses a **13-class** colinear vocabulary at unbounded offset.

### 4.4 Action rows

Every legal action is also encoded by its consequences: the 18 windows the
placed stone would occupy, each typed by the joint of its **post-placement**
ternary pattern and the placement's slot — **729 classes** under the same
reversal fold. A window that would be entirely empty but for the new stone
is an EMPTY row; the three EMPTY orbits share learned rows, and a cell's
EMPTY contribution reduces to per-orbit counts times those shared rows. An
uncovered cell's action encoding is therefore exactly its EMPTY-count
signature — all uncovered cells at the same far-ring geometry share one row,
which is the intended semantics. The encoder is shared by both decoders
(§6): one projection and class table feed a ReLU, the summed hidden rows
pass through a per-head extension matrix.

The builder also emits eleven deterministic **tactical scalars** per legal
action, derived from the same 18 rows and a global census of opponent-only
threat windows: immediate win; max own count after and max opponent count
before (each /6); own five- and four-window counts after, opponent five- and
four-windows hit (each /18); opponent five- and four-windows remaining
globally (capped at 8); a blocks-all-immediate-threats flag; and the
nonempty-row fraction. All values are D6-invariant and in [0, 1]. They ride
the wire as fp32 and enter the model only under the `action_tactical` knob,
through one small zero-initialized MLP added to the action row.

## 5. Trunk

Embeddings (§3) enter B identical pre-norm residual blocks. One block runs,
in order:

1. **§5.1 window ← stones** — each window aggregates its stones (sum, not
   mean: the count is signal) plus its 1458-class row sum, through a
   two-input MLP residual.
2. **§5.1b cell state** — with cell state on: cells read their (at most 18)
   containing windows under the 726-class typing with per-class score biases
   and class value rows; with `cell_nodes`, cells then read stones through
   the 192-class radius edges (and optionally distance-one neighbours);
   finally windows read back from their (at most 5) empty candidate cells,
   bias-typed. A full window has no empty cells and reads zero. With cell
   state off, a parameter-tied transient relay lets windows exchange state
   through shared empty cells instead.
3. **§5.1c window attention** — multi-head attention over each window's
   typed pair relations (48 classes), when `window_attention` is on; the
   whole-line pass (13 classes) runs in the same slot when `line_pass` is
   on.
4. **§5.2 stone ← windows** — the mirror of §5.1: stones aggregate their
   windows plus the class row sum.
5. **§5.3 stone self-attention** — attention over `[latent rows; stones]`,
   block-diagonal per position, biased by the §4.1 orbit table and, with
   `orbit_vectors`, its content vectors. The four latent rows attend as
   ordinary rows under the TOKEN bucket.
6. **§5.4 window-latent cycle** — the latents read the window set
   (attention over each position's real windows), mix among themselves
   (dense 4×4 attention), and broadcast back to the windows.
7. **FFN** — one shared feed-forward over stones and latents.

Every stage is residual with pre-LayerNorm; every ragged attention runs its
scores, softmax, and weighted sums in fp32 regardless of autocast. After the
last block a single shared LayerNorm produces the head inputs: stone rows,
window rows, the mean-pooled latent context `g`, and — with cell state on —
the refined cell rows scattered into engine legal order, uncovered cells
carrying the (normalized) learned base row.

## 6. Legal-cell decoders

Both cell heads read the same per-cell input rows:

- with cell state on, the trunk's refined cell latents;
- otherwise, a one-shot parameter-free aggregation over the decoder
  incidence: each covered cell sums its windows' rows; uncovered cells read
  zero.

With `action_latents` on, the shared action-row encoding first passes a
read/mix/broadcast cycle: two invariant latents per position read the
position's rows under an fp32 segment softmax, self-mix densely, and
broadcast back — permutation-invariant context about the alternatives,
without quadratic action attention. Keys are bias-free and the broadcast
output is zero-initialized, so the cycle starts as the identity.

Each head then forms, per legal cell, `lin(rows) + class_row_sum(e_head) +
extension(action_rows)` — its own projection of the shared input, its own
726-class embedding summed over the cell's decoder incidence, and its own
extension of the shared §4.4 action-row encoding — feeds the sum through the
cell half of a two-input MLP whose position half reads `g`, and reads out:

- the **policy head**: one raw logit per legal cell; the acting policy is
  the per-position softmax;
- the **action-value head** (appendix B): three categorical outcome logits
  per legal cell.

Both MLP outputs are zero-initialized, so an untrained model plays uniformly
over legal cells and assigns exactly zero action value everywhere. Outputs
are in engine legal order; nothing is emitted for illegal cells.

## 7. State-value head

A multi-query attention readout: Q learned query rows attend over `[g;
window rows]` per position, the concatenated readouts feed an MLP, and the
output is a softmax distribution over K bins whose centers span [-1, 1].
The scalar value is the distribution's expectation, decoded in fp32 inside
the forward so every consumer sees the same number. The state-value head is
trained only on the supervised path (below); the self-play loop trains the
action-value head instead and never reads this one.

## 8. Symmetry

Every input is invariant under the board's twelve symmetries (the D6
rotations and reflections composed with reversal where applicable): stone
ownership, canonical window patterns, folded incidence classes, distance
buckets, orbit-48 edge classes, and `moves_remaining`. The network therefore
computes identical outputs, up to the engine's legal-order permutation, for
a position and any of its transforms — a property enforced by replay tests
that transform the move history through the engine and demand the permuted
outputs back.

## 9. Batching

Positions batch by concatenation with per-position index offsets; message
passing never crosses positions, and attention is masked block-diagonal.
The builder emits stone tables, window tables with identities, incidence
lists with folded classes, legal-cell decoder tables, action-row classes
and reverse views, cell fields, radius and adjacency edges, and
`moves_remaining`; all index tensors are mandatory, and a missing input is
an error rather than a silently substituted default. The production builder
is the Rust encoder (shared with the engine bindings); the Python builder
is its oracle. Relational tables that depend only on window identities —
§5.1c pairs, lines, cell views — are derived once per forward on the
model's device through compiler-opaque operations, so compiled execution
has no graph break.

## 10. Numerics and conventions

Weights are fp32; the forward is written to run under bf16 autocast without
assuming it. Ragged softmaxes, segment reductions, and the categorical
compositions run in fp32 unconditionally. Custom kernels have deterministic
backward passes and CPU reference paths asserted equal at tolerance on CUDA.
The one exception is the §4.1 static bias table: every query tile reduces
into the same 51 rows per head, so its gradient is accumulated with an
atomic and its last bits depend on block scheduling. Under `orbit_vectors`
each query row owns a table row, the reduction becomes a plain store, and
that exception disappears. Embeddings, the latent and cell bases, and the
value queries initialize N(0, 0.02); decoder output layers initialize to zero;
attention key projections and the spec's bare matrices are bias-free (a
shared key bias cancels in softmax), while FFN, MLP, and the remaining
attention linears keep the framework-default bias.

## 11. Interface and versioning

One forward answers every head:

| Output | Shape | Meaning |
| --- | --- | --- |
| `policy_logits` | (N_cells,) | raw, engine legal order per position |
| `q_score` | (N_cells,) | the acting score π′ ranks by (appendix B) |
| `q_values` | (N_cells,) | action values Q in (−1, 1) |
| `value` | (P,) | scalar state value in [−1, 1], fp32 decode |
| `value_dist` | (P, K) | softmax over bins |
| `value_logits` | (P, K) | raw bin logits — what the value loss trains |

`MODEL_REPR_VERSION` (currently **9**) covers the builder and every feature
encoding; `ACTION_ORDER_VERSION` (engine-owned) governs legal-move
indexing; either bump invalidates checkpoints, and loaders refuse
mismatches rather than adapt. Checkpoints record their `model_config`;
configurations from the architecture's knob era are accepted exactly when
they match the baked values (`axis_bias=True`, `off_axis_bias=False`, `d_max=12`,
`cell_pass=True`, `cell_pass_from=0`, `cell_pass_rounds=1`,
`joint_incidence=True`, `mixed_windows=True`, `action_rows=True`,
`state_latents=4`) and refused loudly otherwise.

## 12. Test obligations

The invariants the test suite holds, stated as contract: D6 equivariance by
engine replay of transformed histories; knob-off byte-identity (every
architecture knob at its default produces bit-identical outputs to a build
without the knob's code); CPU/CUDA kernel parity and bitwise-deterministic
backward; decoder coverage (every legal cell scored exactly once, in engine
order); builder parity between the Rust encoder and the Python oracle; and
the parameter-count pins of §2.

## 13. Training (KLENT)

The self-play loop trains the trunk, policy head, and action-value head
with KLENT, a closed-form KL-regularized entropy-tempered improvement
operator. At every acting ply the improved policy is

    pi'(a|s)  proportional to  exp[ (Q~(s,a) + tau * log pi_theta(a|s)) / (tau + lambda) ]

where `Q~` is the mass-normalized acting score of appendix B, and

    v_hat(s) = E_{a ~ pi'}[ Q(s,a) ]

is the acting-time value estimate. Returns are lambda-returns over `v_hat`
with mover-change signs (+1 when the same player places again, −1 at a turn
handover), discounted by gamma. The fit minimizes policy cross-entropy
against `pi'` plus the taken action's categorical cross-entropy against the
target `(max(G,0), max(-G,0), 1-|G|)`; the state-value head is trained only
in the supervised laboratory harness, with a distributional cross-entropy
against the two-hot projection of the outcome onto the bins.

The reference recipe — always passed explicitly; the command-line defaults
remain paper-faithful and differ — is `gamma=0.99, lam=0.01, tau=0.1,
lam_ret=0.939, mass_floor=0.2, lr=1e-3`, with 4096 completed games per
iteration over 1024 environment slots, a 512-ply cap, and batch 4096. Each
iteration collects an on-policy buffer, fits one epoch against it, and
discards it.

## Appendix B — action-value head (the KLENT interface)

The critic is categorical: three logits per legal cell, softmaxed to
`(p_pos, p_neg, p_zero)` over the outcomes of the signed return G. From the
simplex,

    Q = p_pos - p_neg              # in (-1, 1); what the lambda-return targets
    M = p_pos + p_neg = 1 - p_zero # committed mass, in (0, 1)

At the cross-entropy optimum `p_pos = E[G+]` and `p_neg = E[G-]`, so `M`
estimates `E|G|` — how much return the critic commits to this action at
all. Acting ranks by the **mass-normalized score**

    Q~(s,a) = Q(s,a) / max( max_b M(s,b), mass_floor )

one positive divisor per position, which preserves Q's order while
expressing it in units of the position's most committed action; the floor
bounds sharpening when every action puts most probability on zero return.
The composition runs in fp32; fitting scores only the taken action's row
off the same logits acting composed, so the two paths cannot disagree.
