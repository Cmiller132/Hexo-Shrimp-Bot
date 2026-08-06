# MantisNet-ACT v4 — Build Specification

**Status:** normative implementation target for a new architecture alongside the existing MantisNet.

**Architecture ID:** `mantis_act_v4`

**Instruction to the implementation model:** implement the complete configurable superset described here. Do not replace, mutate, or remove the current MantisNet. Add this as a separate selectable architecture, preserve the existing KLENT-facing interface, and implement every major component behind explicit toggles so the requested ablations can be run from configuration.

---

## 1. Purpose

Build a stronger exactly D6-equivariant model for HEXO that directly represents:

- finite relevant board cells;
- every tactically relevant six-cell line window, including mixed/dead windows;
- line direction through three axis-equivariant channels;
- exact D6 displacement shapes using 48 radius-12 orbit classes;
- global board context through multiple latent tokens;
- the counterfactual result of placing each legal action;
- useful same-turn second-placement partners;
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

This document changes the model representation, so use a new model representation version and architecture/config hash.

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
10. same-turn partner messages for first placements;
11. two action-set latents;
12. separate policy-private and critic-private adapters;
13. the current three-class categorical critic by default;
14. no direct quadratic attention over all cells, windows, or actions;
15. no state-value head by default.

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
    use_full_cell_attention: bool = False
    window_window_mode: str = "none"

    # Action modeling
    use_counterfactual_action_windows: bool = True
    use_action_pair_messages: bool = True
    pair_scope: str = "post_action_collinear"
    pair_max_distance: int = 5
    use_action_set_latents: bool = True

    # Phase
    phase_conditioning: str = "film"
    use_three_way_phase: bool = True

    # Heads
    head_separation: str = "private_adapters"
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

The default model should target roughly 2.5–4 million trainable parameters. Provide a model-summary function reporting parameters by subsystem.

Fields changing node sets, relation semantics, tensor shapes, or output semantics must be included in the checkpoint architecture hash.

---

## 7. Stable builder ordering

For deterministic batching, testing, and diagnostics:

- sort cell nodes lexicographically by `(q, r)`;
- sort persistent windows by `(native_axis, start_q, start_r)`;
- preserve legal actions in engine order, never sorted independently;
- sort ordinary graph edges by `(dst, src, relation)`;
- sort action partner rows by `(dst_action, partner_coord, evidence_kind, window_identity)`.

Coordinates and window identities are builder metadata only. Do not embed raw coordinates or absolute axis IDs.

---

## 8. Relevant cell nodes

### 8.1 Cell scopes

```python
cell_scope: Literal[
    "occupied_only",
    "occupied_and_legal",
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

### 11.2 Relation IDs

```text
0..47  exact orbit for distance <= 12
48     FAR       # optional dense-attention use
49     SELF
50     LATENT
51     PAD
```

Sparse default paths do not emit edges beyond radius 12. Global latents carry longer-range context.

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

Toggle:

```python
phase_conditioning: Literal["token_only", "film"]
```

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

If typed window attention is enabled, provide a parameter-matched extra-FFN control preset and report its separate parameter/time cost.

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

Gather in engine order:

```text
A_inv  = C_inv[legal_to_cell_index]
A_axis = C_axis[legal_to_cell_index]
```

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

## 20. Same-turn second-placement modeling

The output remains one placement at a time. Partner modeling only enriches the first-placement action representation and `Q(s, a1)`.

Enable partner messages only for `phase == FIRST` / `moves_remaining == 2`.

Do not emit or apply partner messages for an action that already wins immediately.

### 20.1 Pair scopes

```python
pair_scope: Literal[
    "none",
    "current_legal_collinear",
    "post_action_collinear",
    "post_action_tactical",  # optional heavier mode
]
```

Recommended initial default:

```python
pair_scope = "post_action_collinear"
```

### 20.2 Collinear partner enumeration

For every legal first action `a`, enumerate every empty coordinate `b` on the three axes at signed distance 1 through 5.

- `current_legal_collinear`: retain only currently legal `b`;
- `post_action_collinear`: retain `b` if the engine says it would be legal after placing `a`.

The second mode must include newly opened legal cells that are absent from the current action set.

### 20.3 Pair evidence rows

For each ordered `(a, b)`, enumerate every six-cell window containing both. Emit one evidence row per shared window:

```text
dst_action_index = a
src_action_index = current legal index of b, else -1
pair_axis
pair_distance 1..5
post2_pattern_class  # one of the 377 nonempty reversal classes after adding a and b
src_is_current_legal
```

For current legal `b`, use its pre-pair action embedding as source content. For newly legal `b`, use a shared prospective-partner base plus the post-two-placement pattern relation.

Aggregate pair evidence into the destination action’s matching axis channel and a symmetric invariant summary.

### 20.4 Optional tactical partner scope

`post_action_tactical` additionally includes engine-legal second cells that, after `a`:

- complete or strengthen an own four/five window;
- hit a remaining opponent four/five window.

Noncollinear tactical partner evidence updates the invariant stream. Keep this mode optional because it is more expensive and game-specific.

### 20.5 Controls

Provide:

- pair messages off;
- current-legal only;
- post-action prospective cells;
- optional degree/axis-distribution-preserving random rewiring diagnostic.

---

## 21. Action-set latents

Use two invariant latent queries over the legal action set after counterfactual initialization and pair messages:

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
2. optional same-turn partner messages
3. optional action-set latent read/mix/broadcast
4. AxisMix
5. invariant and shared-axis FFNs
6. phase FiLM
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

### 24.2 Window-fate auxiliary

Retain the previous future fate head only as an optional experiment and apply it only to live windows. Mixed windows are masked because they are already dead for both players.

---

## 25. Internal packed input/output

Suggested packed input:

```python
@dataclass
class PackedACTInput:
    position_count: int
    cell_offsets: Tensor
    window_offsets: Tensor
    legal_offsets: Tensor

    cell_occupancy: Tensor
    cell_is_legal: Tensor
    cell_nearest_bucket: Tensor
    legal_to_cell_index: Tensor

    window_pattern_class: Tensor
    window_status: Tensor
    window_axis: Tensor
    window_numeric: Tensor
    window_cell_index: Tensor          # [Nw, 6], -1 allowed
    window_incidence_class: Tensor     # [Nw, 6]
    window_incidence_mask: Tensor      # [Nw, 6]

    adjacency_src: Tensor
    adjacency_dst: Tensor
    adjacency_axis: Tensor

    radius_src: Tensor
    radius_dst: Tensor
    radius_orbit: Tensor
    radius_axis_or_neg1: Tensor

    action_window_index: Tensor        # [Na, 3, 6]
    action_post1_class: Tensor         # [Na, 3, 6]
    action_pre_status: Tensor          # [Na, 3, 6]
    action_tactical_numeric: Tensor

    pair_dst_action: Tensor
    pair_src_action_or_neg1: Tensor
    pair_axis_or_neg1: Tensor
    pair_distance: Tensor
    pair_post2_pattern: Tensor
    pair_evidence_kind: Tensor
    pair_src_is_current_legal: Tensor

    phase_id: Tensor
    moves_remaining: Tensor
    global_numeric: Tensor
```

Suggested output:

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
pair evidence rows
state-latent scores
action-latent scores
```

Default asymptotic work should be approximately:

```text
O(6 * N_windows)
+ O(E_cell_adjacency)
+ O(E_radius12)
+ O(K_state * (N_cells + N_windows))
+ O(18 * N_legal)
+ O(E_pair)
+ O(K_action * N_legal)
```

No default path may be quadratic in all cells, windows, or actions.

Profile radius-12 edges carefully; expose `occupied_radius` so radius 6 and 12 can be compared without changing relation semantics.

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

---

## 28. Checkpointing and versioning

- add architecture id `mantis_act_v4`;
- bump `MODEL_REPR_VERSION` to the next repository value;
- save the complete resolved architecture config and stable hash;
- strict load requires exact agreement for all shape/semantic fields;
- do not auto-load old MantisNet checkpoints into this model;
- any future graft tool must emit a manifest and numerical parity evidence;
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
post-action collinear partners
phase FiLM
private policy and critic adapters
categorical3 critic
no direct typed window attention
no state-value head
```

### `full_no_pair`

Full model with pair rows/messages disabled.

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

Add typed direct window attention. Also implement an equal-parameter extra-FFN control.

### `full_no_tactical_inputs`

Disable deterministic tactical action scalars while retaining post-placement pattern encoding.

### `full_no_action_latents`

Disable action-set latent read/broadcast.

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
15. Pair evidence uses empty cells, correct distance/axis, and correct shared-window post-two patterns.
16. `post_action_collinear` includes newly legal partner cells whenever the engine does.
17. Pair evidence is absent/masked on second/opening phase and for immediate-winning first actions.
18. No graph edge crosses a batch position.

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
9. pair evidence rows map exactly;
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
- old MantisNet behavior and checkpoints remain unaffected.

---

## 33. Structural alias diagnostic

Add a command that hashes each legal action’s builder-side structural signature:

```text
cell features
incident persistent windows
18 post1 classes
nearby orbit48 relations
pair evidence
```

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
pair rows
newly legal prospective partner rows
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
pair row counts
```

---

## 35. Recommended first ablations

Hold KLENT coefficients, collection, fitting, replay lifetime, and evaluator settings fixed.

1. `full_act_v4` vs `full_occupied_cells_only` — explicit relevant empty cells.
2. `full_act_v4` vs `full_live_windows` — mixed/all nonempty windows.
3. `full_act_v4` vs `full_no_axis` — three-axis equivariant channels.
4. `full_act_v4` vs `full_coarse_geometry` — 48 exact D6 buckets.
5. radius 12 vs radius 6 — geometry range versus cost.
6. full latents vs one latent vs no latents.
7. `full_act_v4` vs `full_no_pair` — same-turn partner modeling.
8. current-legal vs post-action prospective partners.
9. relation-gated vs additive incidence.
10. private adapters vs output-only separation vs fully shared head.
11. deterministic tactical inputs on vs off.
12. action-set latents on vs off.
13. typed window attention vs no typed attention plus parameter-matched FFN.
14. nonempty vs action-relevant persistent windows.

Report both strength per simulator evaluation and strength per wall-clock time.

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
- prospective pair builder;
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
- same-turn partner messages;
- action-set latents;
- two shared action blocks;
- private policy/critic adapters;
- KLENT evaluator integration.

### Stage E — controls, auxiliaries, diagnostics

- all named presets;
- optional typed window attention;
- optional auxiliaries;
- parameter-matched controls;
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
9. first-placement prospective partners include engine-newly-legal cells;
10. the external KLENT evaluation seam is unchanged;
11. a bf16 smoke training run is finite;
12. node/edge/time/memory telemetry is available;
13. the old MantisNet remains untouched and selectable.

---

## 38. Decisions that must not be silently changed

- Default “all windows” means all **nonempty** windows; literal infinite empty windows are not enumerated.
- Mixed windows are represented, not removed.
- Empty cells inside persistent windows are explicit nodes in the full preset.
- Axis direction is represented by equivariant channel permutation, not absolute axis-specific parameters.
- The 48 geometry buckets are generated from all 12 D6 transforms and asserted.
- Every legal action is encoded from its hypothetical post-placement windows.
- Partner modeling remains internal; policy output is still one placement.
- Recommended partner mode includes newly legal second-placement cells.
- Multiple fixed-count latents are the default global path.
- Policy and critic share the trunk but have private adapters.
- The current categorical critic remains default until separately ablated.
- Direct typed window attention is optional, not default.
- No absolute coordinates, fixed crop, or move-history features are introduced.
