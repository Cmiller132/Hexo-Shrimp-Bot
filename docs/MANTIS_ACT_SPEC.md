# MantisNet-ACT v4.1 — Build Specification

**Status:** normative implementation target for a new architecture alongside the existing MantisNet.

**Architecture ID:** `mantis_act_v4`

**Instruction to the implementation model:** implement the complete configurable superset described here. Do not replace, mutate, or remove the current MantisNet. Add this as a separate selectable architecture, preserve the existing KLENT-facing interface, and implement every major component behind explicit toggles so the requested ablations can be run from configuration.

### v4.1 changelog

- Same-turn second-placement partner rows and messages are removed from the
  architecture and ablation matrix; one placement remains one MDP action.
- The typed-window-attention control is an exactly parameter-matched dedicated
  window FFN stage, with invariant/axis hidden widths 111/97.
- The production parameter count is a reference ceiling for controls, not a
  minimum model-size target; invariant width remains an owner decision.
- `occupied_and_legal` is removed. `window_and_legal` remains distinct under
  `action_relevant` windows.
- Packed batches carry consuming-operation execution plans, a 13-field builder
  fingerprint, and an explicit packed-schema discriminator.
- Key-projection biases are forbidden, checkpoint/config loading is strict over
  dropout, chunk-cost laws are preset-specific, and KLENT collection, search,
  fitting, and evaluation use architecture-neutral seams.
- Ablations use a predeclared supervised survivor screen followed by self-play
  confirmation, and the performance acceptance protocol is fixed pending its
  merged-tree threshold calibration.

---

## 1. Purpose

Build a stronger exactly D6-equivariant model for HEXO that directly represents:

- finite relevant board cells;
- every tactically relevant six-cell line window, including mixed/dead windows;
- line direction through three axis-equivariant channels;
- exact D6 displacement shapes using 48 radius-12 orbit classes;
- global board context through multiple latent tokens;
- the counterfactual result of placing each legal action;
- distinct policy and critic computations after a shared trunk.

The default full model must continue to output one raw policy logit and one action-value prediction per legal cell in exact engine legal-move order.

---

## 2. Existing contracts that remain authoritative

Do not change these contracts:

- engine rules, legal moves, move ordering, and terminal detection;
- one placement is one MDP action;
- `moves_remaining == 2` means the mover keeps the move after this placement;
- `moves_remaining == 1` means control changes after a nonterminal placement;
- policy, Q, acting values, and returns use the current mover’s perspective;
- a terminal successor is never evaluated or bootstrapped;
- stone colors are encoded relative to the side to move;
- the KLENT evaluator consumes flat policy logits, `q_score`, and `q_value` in legal-cell order;
- ordinary checkpoint loading is strict.

The architecture-neutral KLENT boundary is:

```text
collate_positions(positions) -> architecture batch
collate_prefixes(games, plies) -> architecture batch
chunk_cost(sample_metadata, budgets) -> deterministic packing law
policy_q(batch) -> flat policy logits, flat critic logits
supervised_heads(batch) -> declared supervised outputs
has_state_value_head -> bool
```

Collection and search receive `collate_positions` and `chunk_cost` from the
selected architecture; they must not import a concrete builder. Fitting and
corpus evaluation use `collate_prefixes`, `chunk_cost`, and the model methods
above. The external evaluator remains:

```text
evaluate(batch) -> flat CPU (policy_logits, q_score, q_value)
```

All flat action tensors and `legal_offsets` remain in exact engine legal order.
Search consumes only that evaluator contract plus engine positions; it is
independent of the batch representation.

This document changes the model representation, so use the ACT representation,
packed-schema, checkpoint-format, and architecture/config discriminators in
§28. The legacy MantisNet discriminators do not change.

---

## 3. Required full-model components

The `full_act_v4` preset must enable all of the following:

1. all nonempty six-cell windows as persistent nodes, including own-only, opponent-only, and mixed windows;
2. explicit relevant cell nodes, including occupied cells, legal cells, and empty cells inside persistent windows;
3. invariant features plus three axis-equivariant channels;
4. exact 48 D6 displacement-orbit relations for radius 1 through 12;
5. relation-gated cell↔window message passing;
6. sparse local cell geometry;
7. four invariant state latents and two axis-equivariant state latents;
8. placement-phase FiLM conditioning in every block;
9. all 18 post-placement windows through every legal action;
10. two action-set latents;
11. separate policy-private and critic-private adapters;
12. the current three-class categorical critic by default;
13. no direct quadratic attention over all cells, windows, or actions;
14. no state-value head by default.

---

## 4. Clarification: “all windows” on an infinite board

Literal enumeration of every empty window is impossible. Implement these finite scopes:

```python
window_scope: Literal[
    "live",             # current-style one-color nonempty windows only
    "nonempty",         # recommended: every window containing >=1 stone
    "action_relevant",  # nonempty plus empty windows through legal cells
]
```

Recommended default:

```python
window_scope = "nonempty"
```

Under `nonempty`, mixed windows are retained. Empty windows are still represented exactly for action evaluation through the fixed 18-window counterfactual action table, but they are not persistent state nodes.

`action_relevant` is a heavier ablation that also persists every empty window passing through at least one legal cell.

---

## 5. Package layout

Add a new package rather than expanding one monolithic file. Suggested layout:

```text
mantisnet/models/mantis_act/
    __init__.py
    config.py
    symmetry.py
    pattern_classes.py
    builder.py
    packed.py
    equivariant.py
    messages.py
    latents.py
    state_trunk.py
    action_encoder.py
    post_rows.py
    plans.py
    heads.py
    model.py
    diagnostics.py
```

Tests should mirror these modules.

The old model remains independently importable and selectable.

---

## 6. Default configuration

Implement a frozen serializable dataclass. Validate every enum explicitly; unknown values must raise rather than fall back.

```python
@dataclass(frozen=True)
class MantisACTConfig:
    architecture_id: str = "mantis_act_v4"

    # Widths
    d_inv: int = 64
    d_axis: int = 24
    d_rel: int = 24
    num_heads: int = 4
    ffn_mult: int = 2
    extra_window_ffn_hidden_inv: int = 0
    extra_window_ffn_hidden_axis: int = 0

    # Depth
    state_blocks: int = 4
    action_blocks: int = 2
    policy_private_blocks: int = 1
    critic_private_blocks: int = 1

    # Representation
    window_scope: str = "nonempty"
    cell_scope: str = "window_and_legal"
    use_axis_channels: bool = True
    use_global_numeric_features: bool = True
    use_window_numeric_features: bool = True
    use_action_tactical_features: bool = True

    # Geometry
    d6_relation_mode: str = "orbit48"
    d_max: int = 12
    use_cell_adjacency: bool = True
    use_occupied_radius_edges: bool = True
    occupied_radius: int = 12
    route_on_axis_radius_messages: bool = True

    # Message passing
    incidence_message: str = "relation_gated"
    incidence_reduce: str = "sum"
    share_relation_embeddings_across_blocks: bool = True

    # Global communication
    global_mode: str = "latents"
    num_inv_latents: int = 4
    num_axis_latents: int = 2
    num_action_latents: int = 2
    window_window_mode: str = "none"

    # Action modeling
    use_counterfactual_action_windows: bool = True
    use_action_set_latents: bool = True

    # Phase
    use_three_way_phase: bool = True

    # Heads
    head_separation: str = "private_adapters"
    axis_pool_mode: str = "attention"
    critic_type: str = "categorical3"
    enable_state_value_head: bool = False

    # Optional training-only heads
    enable_action_aux_heads: bool = False
    enable_window_fate_head: bool = False

    # Numerics
    norm: str = "layernorm"
    activation: str = "silu"
    dropout: float = 0.0
    layer_scale_init: float = 1e-2
```

The dedicated extra-window FFN stage is absent exactly when both of its hidden
widths are zero. A nonzero stage requires a positive invariant width and, when
axis channels exist, a positive axis width. It is window-only, occurs once per
state block at the same residual location as the optional typed-window path,
and shares no weights across blocks.

The resolved production `full_act_v4` parameter count is the reference ceiling
for controls, not a parity floor that smaller variants must be padded up to.
The default `d_inv=64` remains an owner-selected width; changing it to reach an
arbitrary total is not an implementation decision. Provide a model-summary
function reporting exact parameters by subsystem and named stage.

`dropout` is part of the resolved training semantics. It is included in the
architecture hash and checkpoint config comparison, and ordinary load/resume
requires exact equality. Evaluation still disables dropout through module mode;
it does not obtain permission to load a checkpoint under a different value.

Every resolved configuration field, including training-semantic fields such as
`dropout`, is included in the checkpoint architecture hash and strict config
comparison.

`use_full_cell_attention` and `phase_conditioning` are not configuration fields.
Full-cell attention is not an ACT path, and §13.2 fixes FiLM as the phase
mechanism. Deserialization must refuse those obsolete names and the former
`token_only` value rather than retaining no-op knobs.

---

## 7. Stable builder ordering

For deterministic batching, testing, and diagnostics:

- sort cell nodes lexicographically by `(q, r)`;
- sort persistent windows by `(native_axis, start_q, start_r)`;
- preserve legal actions in engine order, never sorted independently;
- sort ordinary graph edges by `(dst, src, relation)`.

Coordinates are builder/debug metadata only. Persistent window identities may
reach the packed batch solely as the join key for the optional typed-window
path; they are not embedded and never select a parameter. Do not embed raw
coordinates or absolute axis IDs.

---

## 8. Relevant cell nodes

### 8.1 Cell scopes

```python
cell_scope: Literal[
    "occupied_only",
    "window_and_legal",
]
```

Recommended:

```python
cell_scope = "window_and_legal"
```

For `window_and_legal`, the node set is the union of:

- every occupied coordinate;
- every legal coordinate;
- every coordinate appearing in a persistent window.

A cell inside a nonempty six-cell window may be represented even when it is empty and currently illegal. This lets empty intersections and future support participate in the state graph.

For `live` and `nonempty` windows, every empty persistent-window cell is within
five steps of a stone and therefore already legal; in those two scopes the
full node set is exactly occupied plus legal cells. This does **not** make
`window_and_legal` a removable name. Under `action_relevant`, a legal cell at
distance eight may seed an empty persistent window reaching distance thirteen,
so its window cells can be neither occupied nor legal. The removed
`occupied_and_legal` name would therefore be equivalent in two window scopes
and inequivalent in the third.

### 8.2 Cell fields

Store:

```text
coord                     builder/debug only
occupancy                 EMPTY / OWN / OPP relative to mover
is_legal                  bool
is_occupied               bool
nearest_stone_distance    bucketed/clamped
position_index            batch segment
```

Initial invariant embedding:

```text
E_occupancy[3]
+ E_legal[2]
+ E_nearest_distance[bucket]
+ optional numeric MLP
```

No axis may have its own learned parameter. Initial axis channels use a shared learned base replicated three times, plus structural messages.

### 8.3 Legal mapping

Store:

```text
legal_to_cell_index: [num_legal]
```

Output row `j` must always correspond to `legal_moves[j]`.

Under `window_and_legal`, every entry is the represented cell's nonnegative
index. Under `occupied_only`, no empty legal cell is a trunk node, so every
entry is the `-1` sentinel. Consumers branch on the resolved scope and must not
use `-1` as an accidental gather index.

---

## 9. Persistent window nodes

### 9.1 Enumeration

For every placed stone, enumerate:

```text
3 undirected axes × 6 possible occupied slots = 18 windows
```

Deduplicate by:

```text
(native_axis, start_q, start_r)
```

Under `nonempty`, keep every deduplicated window containing at least one stone:

- own-only;
- opponent-only;
- mixed.

A current window containing six own stones or six opponent stones means the state is terminal and must be refused. A full mixed window is allowed.

### 9.2 Ternary pattern classes

Encode slots as:

```text
0 = empty
1 = own
2 = opponent
```

Canonicalize all `3**6 = 729` raw patterns under slot reversal.

Assert:

```text
ALL_WINDOW_PATTERN_CLASSES = 378
NONEMPTY_WINDOW_PATTERN_CLASSES = 377
```

The one additional class is the all-empty pattern and is used only by `action_relevant` persistent windows or explicit empty-window context.

Precompute immutable tables:

```text
raw_code -> reverse_code
raw_code -> canonical_class 0..377
canonical_class -> representative raw pattern
canonical_class -> is_empty
```

Use one 378-row pattern embedding table so all window scopes share the table shape.

### 9.3 Window status and numeric features

Store:

```text
native_axis: 0..2
pattern_class: 0..377
status: EMPTY / OWN_LIVE / OPP_LIVE / MIXED
own_count
opp_count
empty_count
own_max_contiguous_run
opp_max_contiguous_run
```

Initial invariant representation:

```text
E_window_pattern[378]
+ E_window_status[4]
+ optional MLP(normalized counts and runs)
```

The initial axis tensor has shape `[3, d_axis]`. Project the pattern into the native-axis channel. Other channels start at a shared neutral value or zero.

---

## 10. Cell↔window incidence

Every persistent window has six geometric slots. Store:

```text
window_cell_index:      [num_windows, 6]  # -1 when cell scope omits that slot
window_incidence_class: [num_windows, 6]
window_incidence_mask:  [num_windows, 6]
```

The full default `window_and_legal` scope should include all six cells of every persistent window, so its mask is all true.

### 10.1 Exact joint relation classes

Classify:

```text
(ternary pattern, slot index)
```

under joint reversal:

```text
(pattern, slot) -> (reverse(pattern), 5 - slot)
```

Across all 729 patterns:

```text
ALL_CELL_WINDOW_REL_CLASSES = (729 * 6) / 2 = 2187
```

For nonempty patterns only:

```text
NONEMPTY_CELL_WINDOW_REL_CLASSES = 2184
```

Assert both counts. Use one 2187-row relation table; default nonempty windows simply never use the three all-empty incidence orbits.

Do not factor this into separately canonicalized pattern and coarse slot embeddings. The joint orbit is required to avoid exact relation aliases.

---

## 11. Exact 48-class D6 displacement relations

### 11.1 Table generation

For axial displacement `(dq, dr)`:

1. apply all 12 engine-consistent D6 transforms;
2. choose the lexicographically minimum transformed displacement as canonical;
3. enumerate every nonzero displacement with hex distance `1 <= d <= 12`;
4. assign stable IDs sorted by `(distance, canonical_dq, canonical_dr)`.

Assert:

```text
D6_ORBITS_DMAX12 = 48
```

Generate this table from the transform functions. Do not manually enumerate 48 cases.
`d_max` and `occupied_radius` are integers in `1..12`; larger values are refused
rather than silently extending or clipping the frozen relation vocabulary.

### 11.2 Relation IDs

```text
0..47  exact orbit for distance <= 12
48     FAR       # optional dense-attention use
49     SELF
50     LATENT
51     PAD
```

Sparse default paths do not emit edges beyond radius 12. Global latents carry longer-range context.

In `coarse` mode, relation identity is exactly `(hex_distance,
is_on_any_undirected_axis)`. The on-axis component is a binary invariant flag;
the absolute axis number is carried only by the separate equivariant route.

### 11.3 Axis routing

When the displacement lies exactly on one undirected axis, store `edge_axis: 0..2`. The orbit ID is invariant; the axis route permutes under D6.

---

## 12. Three-axis equivariant channels

### 12.1 Representation law

Every cell, window, action, and axis latent has:

```text
h_inv:  [..., d_inv]
h_axis: [..., 3, d_axis]
```

For transform `g`, node mapping `T_g`, and induced axis permutation `pi_g`:

```text
h_inv'(T_g(i)) = h_inv(i)
h_axis'(T_g(i), pi_g(a)) = h_axis(i, a)
```

### 12.2 Forbidden operations

Do not:

- concatenate channels 0/1/2 in fixed order into an unconstrained MLP;
- learn different weights, biases, norms, or base embeddings for absolute axes;
- use absolute axis identity as an embedding lookup.

### 12.3 Allowed operations

- shared linear/MLP/norm applied independently to all three channels;
- route line messages into the structural native axis;
- permutation-equivariant cross-axis mixing;
- symmetric pooling over axes for scalar outputs.

### 12.4 `AxisMix`

Implement:

```python
u_a = LN_axis(x_a)
total = sum_b u_b
other_a = (total - u_a) / 2

delta_a = MLP_axis([
    u_a,
    other_a,
    W_inv_to_axis(LN_inv(h_inv)),
])

x_a += layer_scale_axis * delta_a

axis_summary = sum_a phi_axis(u_a) / 3
h_inv += layer_scale_inv * MLP_inv([
    LN_inv(h_inv),
    axis_summary,
])
```

All operations over `a` share parameters.

### 12.5 Invariant head pooling

Provide:

```python
axis_pool_mode: Literal["mean", "learned_attention"]
```

Recommended learned pool:

```python
score_a = w.T @ tanh(Wa @ x_a + Wi @ h_inv)
weight_a = softmax(score over axes)
axis_pool = sum_a weight_a * x_a
```

This is invariant because the scores and channels permute together.

---

## 13. Phase and global scalar inputs

### 13.1 Three-way phase

Derive:

```text
OPENING = board empty and moves_remaining == 1
FIRST   = moves_remaining == 2
SECOND  = board nonempty and moves_remaining == 1
```

The KLENT return sign still uses the authoritative `moves_remaining`; the three-way ID is only a model feature.

### 13.2 Phase FiLM

In every state block, action block, and private adapter:

```python
scale, bias = PhaseMLP(E_phase[phase_id])
h = scale * h + bias
```

Use separate invariant and axis projections, with axis weights shared across channels. Initialize FiLM to exact identity:

```text
scale = 1
bias = 0
```

FiLM is the only phase-conditioning path and therefore is not a configuration
knob. `use_three_way_phase=False` is the explicit two-way ablation: it folds
OPENING into SECOND while retaining the same FiLM mechanism. There is no
`token_only` mode because this architecture has no phase-token stream.

### 13.3 Global numeric features

Optionally initialize invariant state latents with an MLP over stable state-derived scalars:

```text
log1p(total stones)
own stone fraction
opponent stone fraction
log1p(legal count)
log1p(persistent window count)
fraction of own-live / opponent-live / mixed windows
```

This is exactly eight float32 scalars when enabled. Disabled global, window,
or action-tactical numeric families remain present as arrays with trailing
width zero; the packed schema never omits a field because a feature toggle is
off.

No history, recency, absolute move number, or board origin is used.

---

## 14. Relation-gated messages

Implement a generic module for typed sparse edges.

For relation `r`:

```python
rel = E_rel[r]
value_inv = Wv_inv(LN(src.h_inv))
gate_inv = sigmoid(Wg_inv(rel))
bias_inv = Wb_inv(rel)
msg_inv = gate_inv * value_inv + bias_inv
```

For an edge routed through axis `a`:

```python
value_axis = Wv_axis(LN_axis(src.h_axis[a]))
gate_axis = sigmoid(Wg_axis(rel))
bias_axis = Wb_axis(rel)
msg_axis = gate_axis * value_axis + bias_axis
```

Aggregate by destination in fp32. Default reduction is sum because incidence count is signal. Preserve mean and attention as explicit ablations.

Destination update:

```python
dst.h_inv += MLP_update_inv([LN(dst.h_inv), agg_inv])
dst.h_axis[a] += MLP_update_axis([LN_axis(dst.h_axis[a]), agg_axis[a]])
```

Relation embeddings may be shared across blocks; projections and update MLPs remain block-private.

Legacy additive control:

```text
msg = U @ src + E_relation
```

must remain available as `incidence_message="additive"`.

---

## 15. Local cell geometry

### 15.1 Hex adjacency

Build directed edges between represented cells at hex distance one. Store the structural undirected axis of each edge. Update invariant features and the matching axis channel with axis-shared weights.

### 15.2 Occupied-to-cell radius edges

For every occupied source and represented destination within radius 12, emit:

```text
src
dst
orbit48_id
source occupancy OWN/OPP
on_axis_axis_or_minus1
```

Use the 48 D6 classes in this path. This supplies context to far legal halo cells that occur in no current nonempty six-cell window.

All edge budgets must be measured because radius-12 edges may dominate dense late positions.

### 15.3 Symmetry-safe neighbor selection

Do not use a fixed top-K cutoff with coordinate-order tie breaking. Radius thresholds are safe. A future KNN mode must include all ties at its cutoff.

---

## 16. Optional window↔window attention

Implement the current typed collinear/crossing window-attention path as an optional ablation:

```python
window_window_mode: Literal[
    "none",
    "typed_collinear_crossing",
]
```

Recommended default:

```python
window_window_mode = "none"
```

The full model instead uses:

- explicit shared cell nodes for intersections and forks;
- cell↔window incidence;
- multiple global latents.

The control is a dedicated extra window-only equivariant FFN stage, not a
global `ffn_mult` change. In `full_extra_ffn_control` set:

```text
extra_window_ffn_hidden_inv  = 111
extra_window_ffn_hidden_axis = 97
window_window_mode           = "none"
```

At the default widths its per-block parameter count is exactly:

```text
352 + 129 * 111 + 49 * 97 = 19,424
```

which equals the typed-window-attention stage exactly; over four state blocks
both deltas are 77,696 parameters. Construction and the model-summary test must
assert equality, not merely report approximate matching. Report the two arms'
separate time and memory costs in `docs/ABLATIONS.md`.

---

## 17. Global state latents

### 17.1 Latent tensors

```text
L_inv:  [batch, K_inv, d_inv]
L_axis: [batch, K_axis, 3, d_axis]
```

Recommended:

```text
K_inv = 4
K_axis = 2
```

Invariant latents may have distinct learned identities. Each axis latent has one learned base replicated across all three axis channels.

### 17.2 Read

- invariant latents attend over cell/window invariant states plus a symmetric pool of their axis states and a safe entity-type embedding;
- axis latents attend independently over the matching axis channel of cells/windows;
- query/key/value parameters are shared over axes.

### 17.3 Mix

- invariant latents self-attend across `K_inv`;
- axis latents self-attend across `K_axis` separately for each axis using shared weights;
- apply `AxisMix` to each axis latent;
- invariant↔axis communication uses symmetric axis pooling and shared invariant-to-axis broadcast only.

### 17.4 Broadcast

- cells/windows receive invariant context from invariant latents and pooled axis latents;
- their axis channel `a` receives context from axis latent channel `a`.

### 17.5 Packed implementation

Use ragged packed segment attention. A slow per-position reference implementation may exist for tests, but the training path must not loop over every node in Python.

### 17.6 Key projections have no bias

Every key projection used under a softmax is bias-free. Adding one constant key
bias to every key shifts all scores for a query by the same scalar and cancels
exactly in the softmax, so such a parameter is structurally dead. Query, value,
and output projections retain their specified biases.

The `full_act_v4` named-parameter census contains exactly 34 key-bias tensors
to remove, totaling 1,616 scalars:

| location | passes/blocks | key biases per instance | tensors | scalars |
|---|---:|---:|---:|---:|
| state latent read/mix/broadcast, invariant and axis | 4 | 6 | 24 | 1,056 |
| invariant action-latent read/mix/broadcast | 2 | 3 | 6 | 384 |
| state-to-action invariant and axis broadcast | 2 | 2 | 4 | 176 |
| **total** |  |  | **34** | **1,616** |

The detector enumerates `named_parameters()` and requires that no parameter
whose owning projection is a softmax key ends in `.bias`; a raw substring count
is not sufficient. Removing these tensors is a checkpoint state-dict shape
change and therefore follows §28's format-bump rule.

---

## 18. State trunk block

Each of the four default blocks runs:

```text
1. W <- relation-gated messages from represented window-slot cells
2. C <- relation-gated messages from incident windows
3. C <- local hex-adjacency messages
4. C <- occupied-radius orbit48 messages
5. optional W <- typed window attention
6. state latents read cells and windows
7. latent self-attention and invariant/axis mixing
8. latents broadcast to cells and windows
9. cell-specific AxisMix + FFN
10. window-specific AxisMix + FFN
11. phase FiLM on every residual branch
```

Use pre-norm residual blocks. Cells, windows, invariant streams, and axis streams each require appropriate separate norms.

After the final block, use separate final norms for:

```text
cell invariant
cell axis
window invariant
window axis
invariant latents
axis latents
```

Do not reuse one final norm across entity types.

---

## 19. Counterfactual legal-action encoder

Every legal cell becomes an action embedding after the state trunk.

### 19.1 Base action state

For `window_and_legal`, gather in engine order:

```text
A_inv  = C_inv[legal_to_cell_index]
A_axis = C_axis[legal_to_cell_index]
```

For `occupied_only`, `legal_to_cell_index` is `-1` and the action encoder uses
a shared learned empty/legal base replicated in engine order. It must not index
the final occupied cell by Python-style negative indexing.

### 19.2 Fixed 18 post-placement windows

For every legal action `a`, enumerate:

```text
3 axes × 6 candidate slots = 18 windows
```

Hypothetically insert an own stone at `a`. Classify `(post-placement ternary pattern, candidate slot)` under joint reversal.

Raw count:

```text
6 * 3**5 = 1458
```

Assert:

```text
POST1_REL_CLASSES = 729
```

Builder output should be dense:

```text
action_window_index: [num_legal, 3, 6]  # -1 if no persistent pre-action window
action_post1_class:  [num_legal, 3, 6]
action_pre_status:   [num_legal, 3, 6]  # EMPTY / OWN_LIVE / OPP_LIVE / MIXED
```

For each of the six rows on an axis:

- gather the final persistent window state if present;
- otherwise use a shared learned pre-empty-window state;
- combine with the 729-way post-placement relation;
- use a shared per-row nonlinear encoder;
- sum or attend over the six rows;
- add the result to the corresponding action axis channel.

Symmetrically pool the three axis summaries into `A_inv`.

This path must be used for every legal action, including cells that currently lie in no nonempty window. It replaces the old nearest-distance-only background alias.

### 19.3 Deterministic tactical input vector

When enabled, derive without search:

```text
immediate_win
max own count after / 6
max opponent count before / 6
own five-window count after
own four-window count after
opponent five-windows hit
opponent four-windows hit
opponent five-windows remaining globally
opponent four-windows remaining globally
blocks all current immediate threats flag
mixed windows created / 18
nonempty post-windows / 18
```

Clip/count-normalize fields and encode with a small invariant MLP.

These are deterministic functions of the current state and hypothetical action. They must have an on/off toggle.

---

## 21. Action-set latents

Use two invariant latent queries over the legal action set after counterfactual initialization:

```text
read all actions -> latent self-mix -> broadcast to all actions
```

This gives each action permutation-invariant context about its alternatives without quadratic action attention.

Keep state latents and action latents separate: state latents do not contain post-placement effects.

---

## 22. Shared action blocks

Each of two default action blocks performs:

```text
1. broadcast state latent context
2. optional action-set latent read/mix/broadcast
3. AxisMix
4. invariant and shared-axis FFNs
5. phase FiLM
```

Action embeddings may read state cells, windows, and latents but must not write back into them.

---

## 23. Policy and critic separation

After shared action processing, fork into independent adapters:

```text
A_policy = PolicyPrivateAdapter(A_shared, latents)
A_critic = CriticPrivateAdapter(A_shared, latents)
```

Supported modes:

```python
head_separation: Literal[
    "single_shared_head",      # ablation only
    "separate_output_mlps",
    "private_adapters",        # recommended
]
```

Default to one private equivariant residual block for each head.

### 23.1 Policy output

Use invariant action state plus a permutation-invariant axis pool. Emit one raw logit per legal action. Zero-initialize the final policy output layer.

### 23.2 Critic output

Default:

```text
[z_pos, z_neg, z_zero]
p = softmax_fp32(z)
Q = p_pos - p_neg
M = p_pos + p_neg
```

Zero-initialize the final critic output layer.

Keep a paper-faithful optional head:

```python
critic_type: Literal["categorical3", "scalar_tanh"]
```

Do not silently alter KLENT’s `q_score`/`q_value` operator. Mass-based acting-score scaling remains an explicit KLENT configuration outside the architecture config.

### 23.3 State-value head

Do not instantiate one by default. If enabled, make it an explicit auxiliary/evaluation head over state latents and report its parameters separately.

---

## 24. Optional training-only auxiliary heads

All auxiliary heads have independent named weights. A zero weight means the head is absent unless an explicit debug option requests zero-weight instantiation.

### 24.1 Action auxiliaries

Potential dense labels over every legal action:

1. `win_now` binary;
2. own post-action maximum occupancy, 7 classes;
3. opponent threat windows hit, capped categorical;
4. own five-windows after, capped categorical;
5. winning second partner exists, first-placement states only;
6. winning second partner count, first-placement states only.

If an exact deterministic feature is already fed as input, do not also claim its auxiliary prediction as useful representation learning. Either mask that auxiliary or run the learned-only input ablation.

Any proposal to remove or redefine one of these labels requires a census
stratified over OPENING, FIRST, and SECOND positions from a validated corpus.
A census confined to FIRST phase does not justify a label-contract amendment.

### 24.2 Window-fate auxiliary

Retain the previous future fate head only as an optional experiment and apply it only to live windows. Mixed windows are masked because they are already dead for both players.

---

## 25. Internal packed input/output

The production input is a `PackedACTBatch`. Its graph columns are:

```python
@dataclass
class PackedACTBatch:
    packed_schema_version: int
    position_count: int
    cell_offsets: Tensor
    window_offsets: Tensor
    legal_offsets: Tensor
    adjacency_offsets: Tensor
    radius_offsets: Tensor

    cell_occupancy: Tensor
    cell_is_legal: Tensor
    cell_nearest_bucket: Tensor
    legal_to_cell_index: Tensor

    window_id: Tensor                  # [Nw, 3], metadata/join key only
    window_pattern_class: Tensor
    window_status: Tensor
    window_axis: Tensor
    window_numeric: Tensor
    window_cell_index: Tensor          # [Nw, 6], -1 allowed
    window_incidence_class: Tensor     # [Nw, 6], -1 where masked
    window_incidence_mask: Tensor      # [Nw, 6]

    adjacency_src: Tensor
    adjacency_dst: Tensor
    adjacency_axis: Tensor
    radius_src: Tensor
    radius_dst: Tensor
    radius_orbit: Tensor
    radius_axis_or_neg1: Tensor

    action_window_index: Tensor        # [Na, 3, 6], -1 allowed
    action_post1_class: Tensor         # [Na, 3, 6]
    action_pre_status: Tensor          # [Na, 3, 6]
    action_tactical_numeric: Tensor

    phase_id: Tensor
    moves_remaining: Tensor
    global_numeric: Tensor
    radius_orbit_bound: int

    plans: ACTPlans
    builder_fingerprint: str
```

The graph builder and collation boundary constructs `ACTPlans` before the batch
enters the model hot path. A consuming operator may gather/reduce from these
plans, but may not rediscover them with a sort, `unique`, or histogram in a
forward or backward pass. Plans have these consuming-operation contracts:

| Plan family | Required semantics | Consuming operation |
|---|---|---|
| incidence, adjacency, and radius message CSR | stable destination-, source-, and relation-major views; tied rows retain packed order | `segment_message` relation reducers and their backwards |
| routed radius subset | stable subset of on-axis radius rows with source, destination, relation, and axis | axis-routed radius message path |
| cell/window initial class rows | class-major CSR plus blocks of at most 128 rows, never crossing a class | embedding gather and table-gradient reduction |
| action `post1` and `pre_status` rows | class-major CSR plus the same 128-row block partition | post-row embedding and table-gradient reduction |
| action source-window rows | source-window-major CSR plus explicit sentinel rows | counterfactual post-placement gather/backward |
| action base-cell rows | represented-source-cell-major CSR | base action cell gather/backward |
| row ownership and phase | exact packed-row to position and phase mappings | phase FiLM, heads, and segment routing |
| state/action latent segments | contiguous segment ranges, bases, counts, and row ownership | latent read, mix, and broadcast |

All plan indices and pointers are signed 32-bit integers unless they also serve
as ordinary framework gather indices, in which case they are signed 64-bit.
Every plan sort is stable. Empty optional edge families have well-formed empty
plans rather than absent fields.

`builder_fingerprint` binds a batch and its plans to exactly these ordered
inputs: `architecture_id`, `window_scope`, `cell_scope`,
`use_axis_channels`, `use_global_numeric_features`,
`use_window_numeric_features`, `use_action_tactical_features`,
`d6_relation_mode`, `d_max`, `use_cell_adjacency`,
`use_occupied_radius_edges`, `occupied_radius`, and
`route_on_axis_radius_messages`. Encode each as `name=repr(value)` with a
trailing newline, hash the concatenation with SHA-256, and store the first 16
lowercase hexadecimal digits. Every plan consumer compares the supplied
fingerprint with the model's resolved builder fingerprint before arithmetic.

`packed_schema_version` is `MANTIS_ACT_PACKED_SCHEMA_VERSION = 2` for this
schema. A consumer must refuse a missing or unequal discriminator before using
any graph or plan field.

The model output is:

```python
@dataclass
class ACTOutput:
    policy_logits: Tensor       # flat [total legal]
    critic_logits: Tensor       # flat [total legal, classes]
    q_value: Tensor             # fp32 [total legal]
    q_score: Tensor             # fp32 [total legal]
    legal_offsets: Tensor
    aux: dict[str, Tensor]
```

The existing external `network_evaluate` interface must remain unchanged.

---

## 26. Ragged batching and complexity

Concatenate node families with per-position offsets. No edge may cross positions.

Extend packer limits and telemetry to cover:

```text
positions
cells
windows
window-slot incidences
cell adjacency edges
radius-12 occupied edges
legal actions
post-action rows, exactly 18 per legal action
state-latent scores
action-latent scores
execution-plan rows and pointers by family
```

Default asymptotic work should be approximately:

```text
O(6 * N_windows)
+ O(E_cell_adjacency)
+ O(E_radius12)
+ O(K_state * (N_cells + N_windows))
+ O(18 * N_legal)
+ O(K_action * N_legal)
```

No default path may be quadratic in all cells, windows, or actions.

Profile radius-12 edges carefully; expose `occupied_radius` so radius 6 and 12 can be compared without changing relation semantics.

`ACTChunkCost` is preset-sensitive and must be valid for the resolved builder
configuration. Its unit law is:

| Cell/window scope | Units per position |
|---|---:|
| `cell_scope="occupied_only"` | `2 * occupied_stones` |
| `cell_scope="window_and_legal"`, `window_scope in {"live", "nonempty"}` | `2 * occupied_stones + legal_actions` |
| `cell_scope="window_and_legal"`, `window_scope="action_relevant"` | `graph_cell_count + occupied_stones` |

For `action_relevant`, `graph_cell_count` comes from stored metadata validated
against the builder or from an exact count-only builder projection. It may not
be replaced by `occupied_stones + legal_actions`: a legal cell at distance 8
can seed persistent windows whose cells extend to distance 13. A preset whose
exact count is unavailable must refuse budgeted packing rather than undercharge.

Only `full_act_v4` standardizes the budget-unit thresholds used by the fitting
protocol. Other presets use the same dimensional definition above, but must
declare and tune their own limits; budget numbers are not portable across node
laws.

---

## 27. Numerics and initialization

- parameters stored in fp32;
- bf16 autocast supported;
- segment sums and all softmaxes computed in fp32;
- critic composition in fp32;
- no BatchNorm;
- separate LayerNorms by entity type and stream;
- embeddings, relation tables, and latent bases initialized `N(0, 0.02)`;
- relation attention biases initialized zero;
- policy and critic final layers initialized zero;
- FiLM initialized to identity;
- default LayerScale initialized `1e-2` for fresh training;
- sanctioned function-preserving grafts initialize new branch gates to zero.

Axis bases are learned once and replicated over the three channels.

Numerical qualification is path-specific. Eager, compiled, fused, and
unfused bf16 implementations are each compared with a declared fp32 or fp64
reference and tolerance for that operation. Passing the same tolerance does
not make two bf16 paths "equidistant" from the reference, nor does it permit
one path's error to stand in for another's qualification.

---

## 28. Checkpointing and versioning

- the legacy MantisNet gate remains `MODEL_REPR_VERSION = 3` and is not an ACT
  version field;
- ACT checkpoints carry `MANTIS_ACT_REPR_VERSION = 4` and architecture id
  `mantis_act_v4`;
- packed batches carry `MANTIS_ACT_PACKED_SCHEMA_VERSION = 2` as specified in
  §25; this discriminator is independent of checkpoint representation version;
- save the complete resolved architecture config and stable hash, including
  `dropout` and every other resolved field;
- strict load requires exact config, representation-version, checkpoint-format,
  and architecture-hash agreement;
- do not auto-load old MantisNet checkpoints into this model;
- `ACT_CHECKPOINT_FORMAT` is bumped whenever the state-dict key set, tensor
  shapes, or parameter identity changes. Removing key-projection biases is one
  such change and its implementation must update the format atomically;
- any graft tool must emit a manifest and numerical parity evidence;
- ordinary loading remains strict and conversion-free.

---

## 29. Required named presets

### `full_act_v4`

```text
nonempty windows
window+legal relevant cells
axis channels
orbit48 and radius12 edges
cell adjacency
relation-gated incidence
4 invariant state latents
2 axis state latents
2 action latents
18-window counterfactual action encoder
phase FiLM
private policy and critic adapters
categorical3 critic
no direct typed window attention
no state-value head
```

### `full_no_axis`

Set `d_axis=0`, remove axis latents, and route line messages to invariant features. Do not retain unused axis parameters.

### `full_live_windows`

Full model with current-style live-only persistent windows.

### `full_action_relevant_windows`

Full model with nonempty and empty windows through legal actions persisted.

### `full_no_latents`

Remove state and action latents while retaining local graph paths.

### `full_one_latent`

One invariant state latent; no axis or action latents.

### `full_coarse_geometry`

Replace orbit48 relations with the old distance plus on/off-axis scheme.

### `full_radius6`

Use exact orbit classes but emit occupied edges only through radius six.

### `full_occupied_cells_only`

Only occupied cells participate in the state trunk. Legal actions are created after the trunk; empty window-slot incidences are masked. This is the efficient control against persistent relevant empty cells.

### `full_additive_incidence`

Replace relation-gated messages with `U h + E_r`.

### `full_shared_head`

Remove private adapters and use one shared action representation before the final outputs. Ablation only.

### `full_with_typed_window_attention`

Add typed direct window attention. `window_id` is only the deterministic join
key used to construct the typed edges; it is never embedded or used as a
parameter selector.

### `full_extra_ffn_control`

Keep typed direct window attention disabled. Set
`extra_window_ffn_hidden_inv=111` and
`extra_window_ffn_hidden_axis=97`. All other fields equal `full_act_v4`.
The parameter delta must equal the typed-attention delta exactly, as asserted
by §16; this control may not approximate equality with `ffn_mult` or by changing
`d_inv`.

### `full_no_tactical_inputs`

Disable deterministic tactical action scalars while retaining post-placement pattern encoding.

### `full_no_action_latents`

Disable action-set latent read/broadcast.

### `full_output_only_separation`

Set `head_separation="separate_output_mlps"`: policy and critic have separate
output MLPs but no private equivariant adapter blocks. This is the output-only
arm in §35.

### `full_no_axis_live_windows`

Apply both `full_no_axis` and `full_live_windows` to `full_act_v4`, removing
unused axis parameters exactly. This is the predeclared combined-candidate arm
in §35 rather than an after-the-fact composition.

---

## 30. Required builder and class tests

1. Nonempty window enumeration matches an independent naive oracle.
2. Mixed windows are retained under `nonempty` and absent under `live`.
3. Own-six and opponent-six current states are refused as terminal.
4. There are exactly 378 ternary reversal classes and 377 nonempty classes.
5. There are exactly 2187 all-pattern cell-window joint classes and 2184 nonempty-pattern classes.
6. There are exactly 729 post-one-placement joint classes.
7. Every class is invariant on its reversal orbit and distinct across orbits.
8. There are exactly 48 nonzero D6 displacement orbits through radius 12.
9. Every engine D6 transform induces a valid permutation of the three undirected axes.
10. Transforming a displacement preserves its orbit ID.
11. Every persistent window has six geometric slot entries and valid masks.
12. Every legal move maps to one cell node and one output row.
13. Every legal action has exactly 18 counterfactual post-placement rows.
14. Counterfactual patterns match an independently constructed successor-board oracle.
15. No graph or plan row crosses a batch position.
16. Every plan is byte-identical to an independent stable reference construction.
17. Class-row blocks contain at most 128 rows and never cross a class boundary.
18. A missing or unequal packed-schema discriminator is refused.
19. A builder-fingerprint mismatch is refused before plan arithmetic.
20. Each §26 chunk-cost law matches exact builder counts for its preset family.

---

## 31. Required D6 tests

For randomized and real nonterminal positions, apply all 11 nonidentity transforms and verify:

1. transformed legal coordinates map policy logits correctly;
2. critic logits, Q values, and acting scores map correctly;
3. scalar state outputs are unchanged;
4. invariant cell/window/action states map unchanged by node correspondence;
5. cell/window/action axis states permute by the induced axis permutation;
6. invariant latents remain invariant;
7. axis latents permute correctly;
8. counterfactual action rows map exactly;
9. packed execution plans transform to the same consuming-operation results;
10. batched and single-position forwards agree within pinned tolerance.

Provide a debug forward that can expose selected intermediate tensors. Production forward need not return them.

---

## 32. Numerical and interface tests

- bf16 smoke forward/backward produces finite outputs, gradients, and parameters;
- segment reductions occur in fp32 where required;
- zero output initialization yields identical initial policy logits and zero Q;
- legal output order is asserted directly from `legal_to_cell_index`;
- disabled optional modules make exactly no contribution;
- strict config/version mismatch loading fails loudly;
- model summary parameter totals match `sum(p.numel())`;
- typed window attention and `full_extra_ffn_control` have exactly equal total
  parameter counts, with the expected 19,424 parameters per state block;
- every softmax key projection has `bias=False`, and the named-parameter census
  contains none of the 34 forbidden key-bias tensors;
- changing `dropout` changes the architecture hash and is refused by strict load;
- packed-schema and builder-fingerprint mismatches fail before model arithmetic;
- plan-backed and independent reference consuming operations agree for random
  and real packed chunks;
- old MantisNet behavior and checkpoints remain unaffected.

---

## 33. Structural alias diagnostic

Add a command that computes each legal action's builder-side structural signature:

```text
cell features
incident persistent windows
18 post1 classes
nearby orbit48 relations
```

The 64-bit hash is only an index for candidate groups. Within every hash bucket,
compare the complete canonical signature exactly before declaring an alias.
For 17,461 signatures, the birthday approximation for at least one accidental
collision is approximately `17,461 * 17,460 / (2 * 2^64) = 8.26e-12`.
That probability is small but not observable from aggregate unique-hash counts,
so a hash alone is never structural identity.

Report:

```text
legal action count
unique signature count
number of alias groups
maximum alias group size
coordinates and differing omitted geometry for sampled aliases
```

The full model should not retain a systematic background alias caused solely by nearest-stone distance.

---

## 34. Telemetry and profiling

Report mean/max per position:

```text
cells
windows by status
legal actions
window incidences
cell adjacency edges
radius edges
execution-plan rows and pointers by family
```

Report:

```text
builder time
collation time
forward time
backward time
samples/second
legal actions/second
peak allocated/reserved VRAM
parameters by subsystem
```

Split important model diagnostics by OPENING/FIRST/SECOND phase:

```text
policy entropy
Q standard deviation among top-policy actions
action auxiliary accuracy
```

### 34.1 Production performance gate

The performance qualification uses the named `full_act_v4` preset on an NVIDIA
RTX 4070 Ti-class GPU with 12 GiB VRAM. Run the merged production fit path with
bf16 autocast, `torch.compile` enabled in its production dynamic-shape mode,
optimizer fusion enabled, fitting batch size 512 positions, and two ordinary
CPU prefetch workers. Inputs are a frozen real self-play prefix sample whose
corpus digest, position count, ply distribution, and §26 cost totals are recorded
with the result. No other CPU- or GPU-intensive workload may run concurrently.

Time the complete steady-state fit unit: prefix build, packed graph and plan
construction, container wrapping, host-to-device transfer, compiled forward,
loss, backward, and optimizer update. Warm up through at least 10 completed
updates after compilation, then measure 50 consecutive completed updates.
Synchronize CUDA at the start and end of the measured region, and use CUDA
events for GPU sub-stages. Report median and p95 end-to-end latency, sustained
positions per second, stage medians, and peak allocated and reserved VRAM.
Record GPU model, driver, CUDA, PyTorch, compiler mode, CPU model, Rayon thread
count, prefetch worker count, commit, resolved config hash, packed schema, and
builder fingerprint.

The v4.1 gate remains uncalibrated until the merged-tree measurement replaces
every named placeholder below. A placeholder is not a passing threshold.

| Gate | Required threshold |
|---|---:|
| median end-to-end fit latency | `V4_1_MAX_MEDIAN_FIT_MS_TBD` |
| p95 end-to-end fit latency | `V4_1_MAX_P95_FIT_MS_TBD` |
| sustained throughput | `V4_1_MIN_POSITIONS_PER_SECOND_TBD` |
| peak allocated VRAM | `V4_1_MAX_ALLOCATED_GIB_TBD` |
| peak reserved VRAM | `V4_1_MAX_RESERVED_GIB_TBD` |

---

## 35. Recommended first ablations

Use two predeclared stages. Stage 1 is a supervised screen; Stage 2 is self-play
confirmation. Before Stage 1 begins, declare the supervised metric and direction,
equivalence margin, train/validation/test split, seed set and seed aggregation,
maximum survivor count, survivor cutoff, and deterministic tie-break. The test
split is not consulted while selecting survivors. Apply that rule mechanically;
do not promote an arm because of an unregistered secondary metric.

Only Stage-1 survivors enter Stage 2. Hold KLENT coefficients, collection,
search, fitting, replay lifetime, evaluator settings, simulator budget, and seed
policy fixed. Predeclare the self-play metric, confirmation margin, aggregation,
and tie-break. Report supervised and self-play outcomes in `docs/ABLATIONS.md`,
including strength per simulator evaluation and per wall-clock time.

The initial supervised arms are:

1. `full_act_v4` vs `full_occupied_cells_only` — explicit relevant empty cells.
2. `full_act_v4` vs `full_live_windows` — mixed/all nonempty windows.
3. `full_act_v4` vs `full_no_axis` — three-axis equivariant channels.
4. `full_act_v4` vs `full_coarse_geometry` — 48 exact D6 buckets.
5. radius 12 vs radius 6 — geometry range versus cost.
6. full latents vs one latent vs no latents.
7. relation-gated vs additive incidence.
8. deterministic tactical inputs on vs off.
9. action-set latents on vs off.
10. `private_adapters` vs `full_output_only_separation` vs `full_shared_head`.
11. typed window attention vs `full_extra_ffn_control`.
12. nonempty vs action-relevant persistent windows.
13. `full_no_axis_live_windows` as the combined-candidate arm against
    `full_act_v4`, `full_no_axis`, and `full_live_windows`.

---

## 36. Implementation sequence

### Stage A — pure representation

- config and architecture selection;
- D6 coordinate transforms and axis permutations;
- 378/2187/729 pattern/relation tables;
- 48 orbit table;
- relevant cell builder;
- all nonempty window builder;
- counterfactual 18-window action tables;
- packed collation, schema discriminator, and builder fingerprint;
- all §25 execution plans beside collation;
- independent oracle tests.

### Stage B — equivariant local modules

- invariant/axis state container;
- `AxisMix`;
- three-way phase FiLM;
- relation-gated incidence;
- cell adjacency;
- radius-12 orbit messages.

### Stage C — global state trunk

- packed invariant latents;
- packed axis latents;
- read/mix/broadcast;
- four state blocks;
- debug intermediate-forward path;
- D6 tests.

### Stage D — action model and heads

- 18-window post-placement encoder;
- action-set latents;
- two shared action blocks;
- private policy/critic adapters;
- KLENT evaluator integration.

### Stage E — controls, auxiliaries, diagnostics

- all named presets;
- optional typed window attention;
- optional auxiliaries;
- exact 111/97 parameter-matched control and assertion;
- output-only and combined-candidate presets;
- preset-valid chunk-cost laws;
- profiling;
- structural alias diagnostics.

Correctness comes before custom kernels. Any optimized path must have random-weight and real-checkpoint parity tests against a reference implementation.

---

## 37. Acceptance criteria

Implementation is complete only when:

1. every named preset constructs and runs;
2. all 378/377, 2187/2184, 729, and 48 class-count tests pass;
3. full output D6 equivariance passes;
4. intermediate axis-channel equivariance passes;
5. all legal outputs align exactly with engine order;
6. mixed windows are nodes in the default model;
7. default relevant cells include empty persistent-window cells;
8. every legal action receives all 18 post-placement rows;
9. all execution plans match the reference and are consumed without hot-path sorting;
10. packed schema and builder fingerprint mismatches fail loudly;
11. the KLENT collect/search/evaluate seams are architecture-neutral and the
    external evaluation result contract is unchanged;
12. a bf16 smoke training run is finite under every qualified execution path;
13. the calibrated §34.1 performance gate passes;
14. node/edge/plan/time/memory telemetry is available;
15. the old MantisNet remains untouched and selectable.

---

## 38. Decisions that must not be silently changed

- Default “all windows” means all **nonempty** windows; literal infinite empty windows are not enumerated.
- Mixed windows are represented, not removed.
- Empty cells inside persistent windows are explicit nodes in the full preset.
- Axis direction is represented by equivariant channel permutation, not absolute axis-specific parameters.
- The 48 geometry buckets are generated from all 12 D6 transforms and asserted.
- Every legal action is encoded from its hypothetical post-placement windows.
- Multiple fixed-count latents are the default global path.
- Policy and critic share the trunk but have private adapters.
- The current categorical critic remains default until separately ablated.
- Direct typed window attention is optional, not default.
- Its exact parameter control uses dedicated 111/97 window-FFN widths; neither
  `d_inv` nor the production parameter count is a parity knob.
- `window_and_legal` is retained because it differs from `occupied_only` under
  `action_relevant`; the alias `occupied_and_legal` does not exist.
- Key projections under softmax have no bias.
- Packed plans are bound by both schema discriminator and builder fingerprint.
- Dropout is strict checkpoint/config identity, not an un-hashed runtime override.
- No absolute coordinates, fixed crop, or move-history features are introduced.
