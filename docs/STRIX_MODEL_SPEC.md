# hexo-strix `HeXONet` — model architecture specification

**Status: descriptive reference.** This documents the neural network of
[SootyOwl/hexo-strix](https://github.com/SootyOwl/hexo-strix) at commit
`294c454e` (2026-07-14), analyzed from source. It covers the **model only** —
board→graph representation, trunk, heads, exact hyperparameters, and the
inference-time op sequence — not the search or the training loop, which appear
only where they pin down an output's semantics (e.g. what the value head's
target is). The goal is reimplementation-grade: the model should be
rebuildable from this document alone. §14 maps the design onto our engine.

File citations (`path:line`) refer to the strix repo at that commit.

---

## 1. Orientation: the game, and the two architecture generations

Strix plays **HeXO**: an unbounded hex board in axial coordinates `(q, r)`,
starting from a single stone at the origin. Players alternate placing **two
stones per turn**; a placement is legal on any empty cell within
`placement_radius` hex steps of an existing stone; `win_length` own stones in
a row along one of three axes wins, checked after every placement. This is the
same game as ours at the full setting (`win_length = 6`,
`placement_radius = 8`), except strix trains through a **curriculum** of
smaller settings (win 4–6, radius 2–8) and imposes a `max_moves` cap during
self-play (games that hit it are labelled "draw" for training targets only —
the game itself has no draws).

`HeXONet` is a graph neural network: the position becomes a graph whose nodes
are stones, legal cells, and one global "dummy" node; message passing runs
over edges laid along the three win axes; a per-node policy head scores each
legal cell and a stone-pooled value head scores the position. There is
**exactly one graph builder** — the Rust `hexo-mcts/src/axis_graph.rs`; the
Python `graph.py` is a thin tensor wrapper over it — and **three**
implementations of the forward pass that must agree: eager PyTorch
(`model.py`), a TorchScript twin (`scriptable_model.py`), and a
dependency-free pure-Rust forward (`hexo-infer`), held together by committed
parity fixtures.

Two architecture generations coexist in the repo, and it matters which one a
statement is about:

| | **rel2** (deployed production) | **lean-d6** (newest, from-scratch) |
|---|---|---|
| Config | `configs/gine-mini/4l-128p32v-jkcat-rel2.toml` | `configs/gine-mini/4l-128p32v-lean-d6.toml` |
| Conv | `DedupGINEConv` (PyG GINEConv + edge-feature dedup) | `AxisRelationalConv` (tied-weight relational GINE) |
| Node features | 11-dim (relative + threat) | 8-dim (lean: no coords, no `empty` one-hot) |
| Edge features | dense 5-dim `[axis×3, ±dist, src_player]` | integer `edge_type` + unsigned `edge_dist` |
| D6 symmetry | approximate, via training-time augmentation | **exact, by construction**; augmentation off |
| Parameters | 283,970 (~1.1 MB fp32) | 683,210 |
| Rust inference (`hexo-infer`) | supported (this is what ships) | **not supported** |

Both share the same skeleton: 4 layers, `hidden_dim = 128`, pre-norm residual
blocks, JK-cat (heads see all 4 layer outputs concatenated → 512-dim), policy
hidden 128, value hidden 32. The rel2 line is what the deployed
`strixbot-rel2.safetensors` artifact is (the ~284k-param census in §10
reproduces its published 1.1 MB size exactly); the lean-d6 line is the
architectural frontier, built to fix a "symmetric moves rated differently"
bug by making D6 invariance architectural rather than learned.

---

## 2. Board → graph: the node set

Builder: `hexo-mcts/src/axis_graph.rs` (single source of truth; every
consumer — Python training, self-play, Rust serving, wasm — calls it).

```
nodes = placed_stones  ∪  legal_moves  ∪  {dummy}
num_nodes = n_stones + n_legal + 1
```

- **Stones:** every placed stone, both colours, **sorted lexicographically by
  `(q, r)`** (`axis_graph.rs:551-552`).
- **Legal cells:** exactly `game.legal_moves()` — the union of
  `placement_radius`-disks around stones minus occupied cells, with
  `hex_distance(a,b) = max(|Δq|, |Δr|, |Δq+Δr|)` — already sorted
  lexicographically by the engine. No other empties, no frontier heuristic.
- **Dummy (global) node:** exactly one, appended last at index
  `n_real = n_stones + n_legal`, coordinates pinned to `(0, 0)`. It is in
  neither `stone_mask` nor `legal_mask`, so it participates in message
  passing but is invisible to both heads.

**Node ordering is a load-bearing contract**: stones (sorted) then legals
(sorted) then dummy. Legal node *j* sits at index `n_stones + j`, and the
policy logit at position *j* corresponds to `game.legal_moves()[j]`. The
Rust parity suite asserts this ordering directly
(`hexo-infer/tests/parity.rs:93-106`), not just via output values.

Terminal states are a hard error in the builder — never a silent default.

Initial full-game position (`win_length 6`, radius 8): 1 stone + 216 legal
cells + 1 dummy = **218 nodes**.

---

## 3. Node features

The feature layout is **derived from config flags**, not fixed. Columns are
allocated in a fixed order, skipping disabled ones
(`axis_graph.rs:64-95`; Python mirror `config.py:51-68`):

```
base = 7 if relative_stone_encoding else 8
base -= 1 if compact_stone_onehot          # drop the 'empty' one-hot
base -= 2 if not node_coords               # drop norm_q, norm_r
base -= 1 if moves_scope == "graph"        # (declared but unimplemented — see §13)
dim  = base + (4 if threat_features else 0)
```

Concrete layouts:

| Schema | Dim | Columns |
|---|---|---|
| Legacy absolute | 8 | `[p1, p2, empty, to_move, moves, norm_q, norm_r, inv_dist]` |
| Legacy relative | 7 | `[own, opp, empty, moves, norm_q, norm_r, inv_dist]` |
| rel2 (relative + threat) | **11** | relative 7 + `[threat0..threat3]` |
| lean-d6 | **8** | `[own, opp, moves, inv_dist, threat0..threat3]` |

Encodings, exactly:

- **Stone one-hot:** 1.0 in `own`/`opp` (or `p1`/`p2` absolute). Under
  `relative_stone_encoding`, "own" is the side to move (terminal fallback:
  P2). Under `compact_stone_onehot`, an empty cell is simply all-zero in
  both stone columns.
- **`to_move`** (absolute schema only): `+1.0` if P1 to move else `−1.0`
  (including terminal). Replicated on **every** node.
- **`moves`**: `moves_remaining_this_turn() / 2.0` ∈ {0.5, 1.0}. Replicated
  on every node. This is the **only temporal signal in the input** — there
  are no move-number, recency, or history features anywhere.
- **Positional (`norm_q`, `norm_r`)** — legacy schemas only; computed in
  f64, cast to f32 (`axis_graph.rs:205-233`):

  ```
  cq = mean(q over stones);  cr = mean(r over stones)     (0,0 if no stones)
  spread = max(1.0, max over stones of max(|q−cq|, |r−cr|))   # Chebyshev in (q,r), NOT hex distance
  norm_q = (q − cq) / spread ;  norm_r = (r − cr) / spread
  ```

  Applied to stones and legal nodes; the dummy keeps 0.0. The lean schema
  **drops these entirely** — the config calls them "a rotation-symmetry leak
  and largely redundant with the axis-edge geometry" (`config.py:47`).
- **`inv_dist`** — legal nodes only: `1 / max(1, min hex distance to any
  stone)`. Stone nodes keep 0.0.
- **Threat features** (4 dims, `hexo-engine/src/threat.rs:15-66`) — for each
  real node (dummy keeps zeros) and each of the 3 win axes, slide all
  `win_length`-cell windows over the `2·win_length − 1` cells centred on the
  node (the node's own occupant counts):

  ```
  axis_own = max over windows with opp_count == 0 of own_count   (per axis)
  axis_opp = max over windows with own_count == 0 of opp_count
  own_max  = max over axes of axis_own ;  opp_max likewise
  own_axes = #{axes : axis_own ≥ win_length − 2} ;  opp_axes likewise
  features = [own_max/wl, opp_max/wl, own_axes/3, opp_axes/3]
  ```

  "own" = side to move.
- **Dummy node features:** all zero except `to_move` (if that column exists)
  and `moves`. It is the carrier of the global scalars.

---

## 4. Edges

### 4.1 The axis walk

Constants: `WIN_AXES = [(1,0), (0,1), (1,-1)]` (one direction per axis; both
signs walked). Reach: `window = win_length − 1` hops (**game** win length,
not the model's `axis_window`).

For every real node `i` (dummy excluded), each axis, each sign, walk
`d = 1..window` (`axis_graph.rs:320-400`):

```
target = i + d·sign·axis_dir
j = coord_to_idx[target]  — if absent (off-graph), the ray ENDS (no skipping)
unless (prune_empty_edges and both i,j empty):
    emit i→j  with (axis, +d·sign, player(i))
    emit j→i  with (axis, −d·sign, player(j))
stop the ray after j if:
    i is a stone  and  j is an opponent stone      # lines blocked by the enemy
    i is empty    and  j is any stone              # empties only reach nearest stones
```

Consequences: a stone's edges pass through empties and own stones up to and
including the first opponent stone; an empty reaches only its nearest stone
per direction ("the stones carry long-range line information",
`axis_graph.rs:383-385`); a gap outside the legal set truncates the ray.
Because both endpoints walk, connectivity is symmetric even when one side's
forward walk is blocked. Every edge is emitted in both directions; no
self-loops; duplicates (both endpoints discovering the same pair) are
deduplicated by key `(src, dst, axis)` keep-first — provably lossless since
the attributes coincide.

`prune_empty_edges` (default **true**, on in every production config) drops
only the empty↔empty *emissions*; the walk still continues past them.

Scale: axis edges strictly subsume hex adjacency; a pinned test bounds the
axis/hex edge-count ratio below 15 at 50 stones. The stated motivation: **3
axis layers ≈ the receptive field of ~9 hex-adjacency layers** (each layer
reaches `win_length − 1 = 5` hops along every win line), which is the whole
reason the axis representation exists (`hexo-a0/README.md:30`).

### 4.2 Global (dummy) edges

The dummy is bidirectionally connected to **every** real node (`2·n_real`
edges). Legacy: appended into the same `edge_index` with an **all-zero
5-dim attribute** (the all-zero axis one-hot is what downstream code keys on
to recognise them). Relational: routed into a separate
`global_edge_src/dst` relation instead.

### 4.3 Edge features

**Legacy (rel2): dense `(E, 5)` f32** (`axis_graph.rs:366-378`):

| idx | meaning | values |
|---|---|---|
| 0–2 | axis one-hot (`(1,0)`, `(0,1)`, `(1,-1)`) | 0/1 (all zero for dummy edges) |
| 3 | **signed** hop distance | `±d`, `1 ≤ |d| ≤ win_length−1`; antisymmetric across the reverse edge |
| 4 | **source-node player identity** | `+1` P1, `−1` P2, `0` empty |

Two findings on dim 4 (see §13): the repo's own README and `graph.py`
comments call it a "same-colour flag" — it is not; and it is **absolute**
P1/P2 even under `relative_stone_encoding`, a colour-symmetry leak into the
trunk. Its stated purpose: the destination "uses this to know what's sending
it a message — critical for empty nodes to detect opponent threats"
(`axis_graph.rs:349-352`).

**Relational (lean-d6):** `edge_attr` is replaced by

```
edge_type ∈ {0, 1, 2}                      # which axis
edge_dist = |signed_dist|, clamped into [1, axis_window]
global edges → separate relation, no features
```

Sign and `src_player` are deliberately dropped ("both redundant",
`config.py:44`). `axis_window` (default 8) sizes the model's learned
distance-embedding table and decouples the fixed architecture from the
per-curriculum-stage `win_length`; it must satisfy
`axis_window ≥ win_length − 1` across all stages.

### 4.4 Infinite board → finite graph

There is **no windowing, cropping, or padding**. Finiteness comes entirely
from the rules: the node set is the stone cloud plus its radius-disk legal
halo, and grows with play. Coordinates enter the network only via the
centroid-relative, spread-normalised `norm_q/norm_r` (legacy) or not at all
(lean). The curriculum changes graph *topology* per stage (edge reach with
`win_length`, node count with `placement_radius`) while the architecture
stays fixed; node-count variance is absorbed by edge-budget micro-batching
in the trainer.

### 4.5 Batching

Two equivalent paths. (a) PyG `Batch.from_data_list`: standard node-offset
concatenation; each graph keeps its own dummy; edges never cross graph
boundaries; `global_edge_index` is offset too. (b) Production: a flat Rust
collation (`batch_tensors.rs:103-154`) that also **precomputes**
`legal_idx`, `stone_idx`, `stone_batch`, `legal_counts` so the compiled
forward contains no `nonzero()` calls; delivered to Python as native-endian
byte buffers read with a single `torch.frombuffer` per field.

---

## 5. Trunk — `RepresentationNetwork`

`model.py:78-429`. Contract: `(x, edge_index, edge features…) → (N, D)`
per-node embeddings, where `D = L·H` under JK-cat else `H`. Three mutually
exclusive backbones selected by config: **GINE** (rel2), **AxisRelational**
(lean-d6), GATv2 (legacy original; `num_heads` is inert under the other
two).

### 5.1 Stems

- **Node stem:** a single `nn.Linear(node_dim, H)` — no MLP, no embedding
  tables (`model.py:123-124`).
- **Edge stem (legacy axis only):** one shared `nn.Linear(5, H)`
  (`edge_proj`), applied **once per forward** and reused by every layer.
  Edge features therefore pass through *two* linears: the shared `5→H`,
  then each layer's own `H→H` (GINEConv's `lin`).
- **Relational:** no `edge_proj` at all. Edge information enters via a
  per-layer learned embedding table over hop distance plus the integer axis
  type (§5.3). `edge_dist` is clamped once to `[1, axis_window]` before the
  layer loop.
- **Legacy→lean shim:** a relational model handed a legacy graph
  auto-converts — node columns via a registered `_lean_cols` index-select,
  edges via `legacy_edges_to_lean` (axis one-hot → argmax type, `|dist|`,
  all-zero-one-hot rows → the global relation, `src_player` dropped). This
  is numerically identical to native lean input (pinned to `atol 1e-5`) and
  is how self-play/eval legacy graphs feed a relational model.

### 5.2 The GINE layer (rel2 backbone)

Built as PyG `GINEConv` with MLP `Linear(H,H) → ReLU → Linear(H,H)` and
`edge_dim = H`; **`train_eps` is not set**, so `eps` is a **non-trainable
buffer fixed at 0.0** (it still appears in the state dict). Exact per-layer
math (confirmed op-for-op by the Rust forward, `forward.rs:101-134`):

```
m_ij  = ReLU( x_j + lin(e_ij) )          # lin: Linear(H,H); ReLU per message, BEFORE the sum
agg_i = Σ_{j∈N(i)} m_ij                  # plain sum at the DESTINATION
out_i = MLP( (1 + ε)·x_i + agg_i )       # ε ≡ 0 in practice
```

**Edge-attr dedup** (`DedupGINEConv`, `model.py:48-75`): edge attributes are
purely structural, so ~13k edges on a real radius-8 board carry only ~91
distinct 5-vectors. Both edge linears are computed on the unique rows and
gathered (`torch.unique(edge_attr, dim=0, return_inverse=True)`), skipping
~85% of forward FLOPs on the edge path. Equivalence is exact in exact
arithmetic (row-wise linearity) and "allclose, not bitwise" in floats (GEMM
tiling); the dedup is skipped under `torch.compile` tracing (data-dependent
shape) and for GATv2. State-dict keys are unchanged.

### 5.3 The AxisRelationalConv layer (lean-d6 backbone)

`axis_conv.py:50-162`. An R-GCN-style relational conv where the 3 axes are
an edge-type partition processed by **one tied-weight GINE block**, combined
by a permutation-symmetric sum — making the layer *exactly* invariant to
relabelling the axes:

```
dist_feat = dist_embed(edge_dist − 1)            # nn.Embedding(axis_window, H); rows ↔ hops 1..W
agg = Σ_{k=0..2} axis_gine( x, edges of type k, dist_feat[type k] )    # ONE shared GINE, train_eps=True
agg += global_gine( x, global_edges, global_edge_embed )               # UNTIED GINE; learned edge vec
out  = node_update( concat[x, agg] )             # Linear(2H,H) → ReLU → Linear(H,H)
```

Details: both sub-convs are GINEConv with `train_eps=True` (learnable ε here,
unlike the legacy path); each axis call contributes its own `(1+ε)x` self
term, so the self term is counted 3 times in `agg` plus once from the global
branch (deliberate, reproduced by the TorchScript twin);
`global_edge_embed = randn(H)·0.1` is the one deliberately scaled init in
the trunk; the global relation sits **outside** the axis sum so it cannot
break the symmetry; unsigned distances make the embedding D6-safe. Cost:
`9H² + (9+W)·H + 2` params per layer — 149,634 at H=128, W=8, i.e. 3× the
legacy GINE layer.

### 5.4 The residual block

Per layer, **pre-norm** variant (production; `model.py:368-403`):

```
residual = x
x = LayerNorm_i(x)            # eps 1e-5, elementwise affine — LayerNorm everywhere, never BatchNorm
x = conv_i(x, …)
x = layer_scales[i] * x       # only if use_layer_scale (off in both mainline configs); init 1.0
x = x + residual
x = ReLU(x)                   # AFTER the residual add — the residual stream is non-negative from layer 1
x = Dropout(x)                # config.dropout, 0.0 everywhere in production
```

Post-norm variant (`pre_norm=False`, one legacy config): conv → (scale) →
add → norm → ReLU → dropout. With pre-norm, a `final_norm = LayerNorm(H)` is
applied after the stack (placement under JK below). The ReLU-after-residual
is a real architectural property of this network, not an accident.

### 5.5 Jumping Knowledge

`use_jk` + `jk_mode ∈ {sum, cat, max, lstm}`. Per-layer outputs are
collected **after** the full block (post-ReLU/dropout). Production is
**`cat`**:

```
rep = concat( final_norm(h_0), final_norm(h_1), …, final_norm(h_{L−1}) )   # (N, L·H)
```

The **same** `LayerNorm(H)` is applied to each layer slice *before*
concatenation — not one norm over the concatenated vector. This keeps
`final_norm`'s parameters at shape `(H,)` so a no-JK checkpoint loads
directly, and makes the JK-cat head graft forward-identical at step 0 (the
old head weights land on the last slice, zeros elsewhere). `sum` is a
learnable per-layer scalar softmax (init zeros → uniform); `max` is
per-channel max; `lstm` exists but is **not exportable** to TorchScript.
Without JK: `rep = final_norm(h_{L−1})`, dim `H`.

### 5.6 The global node in the trunk

No special update rule — it is an ordinary node (stem, residual stream,
norms). Only its *edges* differ: all-zero 5-dim attrs (legacy — the shared
`edge_proj` maps them to its bias) or the dedicated untied `global_conv`
with a learned edge vector (relational). It functions as a one-hop global
relay carrying `to_move`/`moves_remaining`, and is excluded from both heads
purely by the masks.

---

## 6. Heads

All heads consume the same `(N, D)` trunk output (`D = 512` in both mainline
configs). There is no separate graph readout; the value head's stone pool
*is* the readout. No head has any custom init, dropout, or normalization.
The production forward calls each head's `.mlp` directly — the
`PolicyHead.forward`/`ValueHead.forward` methods are dead code with
*divergent* fallbacks; do not port them (§13).

### 6.1 Policy head

```
gather rows of rep at legal-node indices          # masking BY GATHER — illegal cells never scored
logits = Linear(D, 128) → ReLU → Linear(128, 1)   # raw logits, no softmax, no temperature anywhere
```

Output: flat `(total_legal,)` split per graph by `legal_counts`. Logit *j*
of a graph ↔ `game.legal_moves()[j]` (lexicographic `(q,r)` order) — the
ordering contract of §2. The head scores **single placements**; the
two-stones-per-turn structure is handled by the `moves_remaining` input
feature and by the search re-evaluating the position between the two
placements. Softmax lives entirely in the search.

### 6.2 Value head (scalar — the deployed head)

```
pooled = mean over STONE nodes of rep             # scatter-add / count.clamp(min=1)
                                                  # zero stones → pooled = 0-vector (NOT mean-over-all)
value  = Linear(D, 32) → ReLU → Linear(32, 1) → Tanh     ∈ [−1, 1]
```

Legal nodes and the dummy are excluded from the pool ("value should not be
diluted by hundreds of empty candidate cells", `model.py:496-498`).
**Perspective: side-to-move** — the training target is `+1` if the player to
move at that position eventually won, `−1` if lost, `draw_value` on a
move-cap timeout (default `−0.3`; `0` in current configs — this is what the
"draw penalty" ablation tunes; there is no draw rule in the game).

### 6.3 Distributional / auxiliary heads (ablation-only, none deployed)

- **Binned value** (`value_bins = 65`): replaces the scalar head; same
  pooled input; `Linear(D,32) → ReLU → Linear(32, 65)` (no tanh). Bin
  *centers* = `linspace(−1, 1, 65)` (step 1/32, exact 0 bin), kept as a
  non-persistent buffer. Target: the same scalar outcome projected to a
  **two-hot** distribution (C51-style, exact in expectation); loss is
  cross-entropy. Scalar decode `Σ softmax(logits)·centers` happens
  *in-forward*, so the inference wire format never changes. Motivation:
  avoid tanh-saturation/outlier-gradient pathologies of MSE regression.
- **Horizon heads** (`value_horizons = [4, 12, 32]`, train-only, never
  exported): one full binned value head per horizon, all fed the same
  pooled vector; horizon-*k* target = the position's own outcome if the
  game resolves within *k* **placements**, else 0.0 (neutral) — a
  "resolution imminence" signal, not a bootstrapped value and not
  moves-left.
- **Q head** (`q_head`, train-only, never exported):
  `Linear(D, 64) → ReLU → Linear(64, 1) → Tanh` over the *same gathered
  legal-node rows the policy uses*, element-aligned with the logits. Target:
  per-move MCTS completed-Q (side-to-move), loss masked to visited moves.

Loss-side facts that define output semantics (not training mechanics): the
policy target is the **Gumbel-MCTS improved policy**
`softmax(logits + σ(completedQ))`, not visit counts, trained with KL — with
one edge case: when the forcing solver proves a win at a root with two
placements remaining, the target is two-hot 0.5/0.5 over the winning pair
(the turn is order-invariant). The scalar value trains with MSE. All losses
are per-sample weighted (playout-cap reduction).

---

## 7. Symmetry: augmentation vs invariance by construction

The D6 group (12 transforms fixing the origin) acts on axial coords:

```
rotations:   (q,r) → (-r,q+r) → (-q-r,q) → (-q,-r) → (r,-q-r) → (q+r,-q)
reflections: (r,q), (-q,q+r), (-q-r,r), (-r,-q), (q,-q-r), (q+r,-r)
```

A transform acts on the graph by mapping coords, re-sorting stones and
legals independently (restoring the ordering contract), permuting node rows,
remapping `edge_index`, permuting the edge axis one-hot and flipping the
distance sign per the axis map, recomputing `norm_q/norm_r` from the new
centroid, and permuting the policy vector by the legal-order permutation.
Three implementations (rebuild-from-transformed-board, transform-the-tensors
in Python, batched Rust) are pinned equal for all 11 non-identity transforms
× 4 encodings.

- **rel2** is *approximately* equivariant, trained with random D6
  augmentation (`augment_symmetries = true`) — and carries two genuine
  symmetry leaks: the absolute `src_player` edge channel and the signed
  distance.
- **lean-d6** is *exactly* D6-invariant by construction — tied per-axis
  weights + symmetric sum, unsigned distances, no coordinate features, no
  `src_player` — verified end-to-end on real positions to 1e-4 for all 11
  transforms, with per-move policies matched through the coordinate map.
  Augmentation is turned **off** as redundant. This closes a real observed
  bug ("symmetric moves rated differently", 2026-07-03).

---

## 8. Canonical inference op sequence

As implemented three times and cross-pinned by parity fixtures. The
pure-Rust `hexo-infer/src/forward.rs` is the cleanest statement (rel2
architecture; numbers for the deployed config):

```
0.  x     = input_proj(X)                          # (N,11) → (N,128)
    eproj = edge_proj(unique edge-attr rows)       # (U,5) → (U,128), ONCE, shared by all layers
1.  for each layer i in 0..4:
        h   = LayerNorm_i(x)                       # eps 1e-5, population variance
        le  = lin_i(eproj rows)                    # per-layer H→H on the unique rows
        z   = (1 + ε_i)·h                          # ε_i ≡ 0 (buffer)
        for each edge (s→d): m = h[s] + le[row]; if m > 0: z[d] += m     # fused ReLU+sum
        x   = ReLU( nn2_i(ReLU(nn0_i(z))) + x )    # GINE MLP, then residual add, THEN ReLU
2.  rep = concat over layers of final_norm(h_i)    # shared LayerNorm(128) per slice → (N, 512)
3.  policy: gather legal rows → Linear(512,128) → ReLU → Linear(128,1)   # raw logits
4.  value:  mean over stone rows (zeros if none) → Linear(512,32) → ReLU → Linear(32,1) → tanh
```

Numerics worth pinning: LayerNorm is population-variance with ε=1e-5 inside
the sqrt (PyTorch convention); no softmax anywhere in the model; the only
output nonlinearity is the value tanh; training runs under bf16 autocast
(no GradScaler), inference exports support fp32/bf16/fp16. The Rust `linear`
uses 8-lane accumulators, and that reassociation is the *entire* parity
budget: tiny-model fixtures at `1e-4`, real-checkpoint fixtures at `1e-3`,
plus a pinned bit-fingerprint of the forward.

**`hexo-infer` supported envelope** (loudly rejected otherwise): GINE + axis
graph + pre-norm + (no-JK or JK-cat) + scalar tanh value, legacy 5-dim
edges, node dims 7/8/11/12. Not supported: GATv2, hex graphs, post-norm,
LayerScale, JK sum/max/lstm — and, **without a clean error**, the entire
lean-d6/relational line and binned value heads, which fail at tensor load
with shape/name errors instead (§13).

---

## 9. Concrete hyperparameters

### 9.1 Deployed production — `4l-128p32v-jkcat-rel2`

| Parameter | Value |
|---|---|
| `conv_type` | `gine` (via `DedupGINEConv`) |
| `hidden_dim` | 128 |
| `num_layers` | 4 |
| `pre_norm` | true |
| `dropout` | 0.0 |
| `use_jk` / `jk_mode` | true / `cat` → head input D = 512 |
| `policy_hidden` / `value_hidden` | 128 / 32 |
| `graph_type` | `axis` |
| `prune_empty_edges` | true |
| `threat_features` | true |
| `relative_stone_encoding` | true → node dim **11**, edge dim 5 |
| heads | policy + scalar tanh value only |
| params | **283,970** learnable (+4 ε buffers); 1.08 MiB fp32 |

Context (not model): `win_length 6`, radius 8, `max_moves 300`; Gumbel MCTS
`n_simulations 128`, `m_actions 16`; lr 4e-4 cosine, batch 512, edge budget
250k, buffer 500k; curriculum W1(4/2) → W2(5/2) → S1(6/2) → S2(6/4) →
S3(6/6) → S4(6/8) with SPRT promotion.

Parameter distribution: conv stack 199,168 (70%), policy head 65,793 (23% —
the `[128,512]` first layer alone is 23% of the model, the direct price of
JK-cat), value head 16,449 (6%), stems+norms ~2,560 (1%).

### 9.2 Frontier — `4l-128p32v-lean-d6`

Same skeleton (4 layers, H=128, pre-norm, JK-cat, heads 128/32) with:
`axis_relational true`, `axis_window 8`, `compact_stone_onehot true`,
`node_coords false` → node dim **8**; relational edges; augmentation off;
lr 2e-4; from scratch. Params **683,210** — trunk 88%, of which the four
relational convs are 149,634 each. Variants: `3l-…` (3 layers, D=384),
`-vbins` (+65-bin value), `-vhoriz` (+horizons [4,12,32], loss weight 0.25),
`-qhead` (+Q head, loss weight 0.5), `-one-stage`.

### 9.3 Config naming scheme

`4l-128p32v-jkcat-lean-d6` decodes as: `4l` = 4 layers; **`128p32v` =
`policy_hidden 128` / `value_hidden 32`** (head widths — *not* hidden_dim;
the deliberate asymmetric reallocation "policy is harder than value on
HeXO"); `jkcat` = JK-cat; `nojk`; `rel2` = relative stones + threat
(2026-06-10 revision); `lean-d6` = the relational schema **with augmentation
off** (`d6` means invariance by construction, not D6 augmentation);
`scratch` = no graft; `w12` = curriculum truncated to stages W1+W2;
`layerscale`; `vbins`/`vhoriz`/`qhead` as in §9.2. Size tiers `gine-mini` /
`-midi` / `-full` = hidden 128 / 192 / 256. The `configs/ablations/` sweep
predates default changes and three of its eight configs (`gine`,
`prune-empty-edges`, `fewer-heads` — GATv2 attention heads, not output
heads) are now no-ops against current defaults.

---

## 10. Export format and weight schema

`hexo-safetensors-v1` (`export.py`): the raw `HeXONet` state dict,
`_orig_mod.` compile prefix stripped, all tensors cast to contiguous fp32,
**unfiltered** (train-only heads export too; `hexo-infer` simply never reads
them). Metadata (strings): `format`, `model_config` (full JSON of
`ModelConfig` — the loader reconstructs the architecture from this),
`train_steps`, `source_checkpoint` (the provenance gate the parity suite
enforces), optional stage-specific `game_config`.

Complete tensor schema (rel2 family; `H` hidden, `L` layers, `ND` node dim,
`P`/`V` head hiddens, `D = L·H` under JK-cat else `H`):

| Tensor | Shape |
|---|---|
| `representation.input_proj.{weight,bias}` | `[H,ND]`, `[H]` |
| `representation.edge_proj.{weight,bias}` | `[H,5]`, `[H]` |
| `representation.convs.{i}.eps` | `[1]` (buffer, 0.0) |
| `representation.convs.{i}.lin.{weight,bias}` | `[H,H]`, `[H]` |
| `representation.convs.{i}.nn.0.{weight,bias}` | `[H,H]`, `[H]` |
| `representation.convs.{i}.nn.2.{weight,bias}` | `[H,H]`, `[H]` |
| `representation.norms.{i}.{weight,bias}` | `[H]`, `[H]` |
| `representation.final_norm.{weight,bias}` | `[H]`, `[H]` |
| `policy_head.mlp.0.{weight,bias}` | `[P,D]`, `[P]` |
| `policy_head.mlp.2.{weight,bias}` | `[1,P]`, `[1]` |
| `value_head.mlp.0.{weight,bias}` | `[V,D]`, `[V]` |
| `value_head.mlp.2.{weight,bias}` | `[1,V]`, `[1]` |

(Index 1/3 modules are parameterless ReLU/Tanh.) For the deployed config
this is 50 tensors, 283,974 elements — which independently reproduces the
published "~284k params, 1.1 MB" artifact. These state-dict key names are
the ABI for both export paths.

Init, for completeness: **no custom init anywhere** except
`global_edge_embed = randn·0.1` (relational); everything else is framework
default (Linear kaiming-uniform, LayerNorm 1/0, `jk_weights` 0 → uniform,
`layer_scales` 1.0, `dist_embed` N(0,1) — the one arguably load-bearing
unmodified default). Weight decay: all 1-D params (biases, norms,
`jk_weights`, `global_edge_embed`, ε) excluded; ≥2-D decayed (including
`dist_embed.weight`).

---

## 11. The parity harness (why three implementations stay honest)

Committed, randomly-initialized **tiny fixtures** (H=16, L=2) in two
flavours chosen to cover both branch sets (JK-cat/11-dim/pruned and
no-JK/8-dim/unpruned), each evaluated at four *independently generated*
positions (0/5/9/20 plies; the 9-ply one must land mid-turn so the
`moves_remaining = 1` encoding is pinned). Real-checkpoint fixtures are
gitignored and provenance-gated: the fixture's `checkpoint` field must equal
the loaded model's `source_checkpoint`, and the test *panics* rather than
silently skipping if real weights are absent but pinned. Tolerances: 1e-4
tiny / 1e-3 real, documented as bounding exactly the 8-lane reassociation.
Ordering is asserted directly, not just through values. Additional bitwise
pins: dedup-vs-stock row gather, batched-vs-serial evaluation, and a
whole-forward fingerprint captured before the dedup optimization landed.

---

## 12. TorchScript twin (`scriptable_model.py`)

A parallel hand-written implementation (no PyG) loaded from the `HeXONet`
state dict, engineered for `torch.compile(fullgraph=True)` (pinned to a
single graph, zero breaks): all edge relations unified into
`(edge_index, edge_bucket, edge_dist)` with bucket 3 = global; branchless
`torch.where` edge projection; the distance table projected once and
gathered; the 3-axis tied GINE fused into batched matmuls; `Final[...]`
flags so dead branches constant-fold. Numerically "allclose" (1e-5), not
bitwise — a couple of in-code "bit-identical" comments overstate this.
Known asymmetries: dropout is absent entirely; `jk_mode="lstm"` is rejected;
train-only heads are dropped at load. This is what self-play inference
actually runs (exported atomically so Rust threads never see a torn file);
the wire protocol is exactly `(logits, legal_counts, values)`.

---

## 13. Findings and traps (verified against source; do not copy these)

1. **Edge dim 4 is mis-documented in-repo** — README and `graph.py` call it
   a "same-colour flag"; it is absolute source-player identity
   (`axis_graph.rs:349-378`). A reimplementation from their README would be
   wrong.
2. **rel2 leaks colour and orientation** — absolute `src_player` and signed
   distance survive under relative stone encoding. The lean schema exists
   partly to delete exactly these.
3. **Dead head code disagrees with the live path** — `ValueHead.forward`'s
   zero-stone fallback is mean-over-all-nodes; the production scatter path
   (and the Rust) yields a zero vector. `PolicyHead.forward` is likewise
   never called.
4. **`hexo-infer` fails unhelpfully outside its envelope** — `axis_relational`
   and `value_bins` are not config-rejected; lean-d6/vbins checkpoints die
   at tensor load with shape/name errors, and its node-dim formula ignores
   the lean flags (would compute 11 where the truth is 8).
5. **`moves_scope="graph"` is declared but unimplemented** — the native
   builder rejects it; nothing in the model would re-inject the scalar. A
   silent information deletion if ever wired through.
6. **`hex` + `gine` crashes eager** (`edge_dim=None` dereference) while the
   scriptable twin accepts it — the implementations disagree on legality.
   No shipped config hits it.
7. **Depth-grafting a relational model silently breaks the identity-at-graft
   invariant** — the zero-init helper looks for `conv.nn`, which
   `AxisRelationalConv` doesn't have.
8. **Stale self-descriptions** — the no-JK config's header still claims
   "production-strongest"; the referenced lean-d6 design doc doesn't exist
   in the checkout; the ablation sweep is partly no-ops under current
   defaults. Config headers, not docs, are the surviving design record.
9. **An unstated precision invariant** — the eager value pool allocates in
   default dtype and works only because the trunk's last op is LayerNorm
   (kept fp32 under autocast); a wholesale `.to(bf16)` model would fail.
10. **Augmentation does not support the lean schema** — both augment paths
    are legacy-only; lean-d6 runs sidestep this by not augmenting (exact
    invariance makes it redundant), and a lean run that *did* enable
    augmentation would silently train on legacy-schema batches via the shim
    (numerically equivalent, but easy to misread).

---

## 14. Relevance notes: mapping onto our engine

Same game at the full setting, so the seams line up directly. Where a strix
choice touches a contract our engine already owns:

- **Policy-index ↔ legal-move ordering.** Strix's entire policy head rests
  on `legal_moves()` being engine-sorted lexicographically, and it pins that
  contract with a dedicated test because a violation passes numeric parity
  while mislabelling every move. This is exactly what our
  `ACTION_ORDER_VERSION` (ENGINE_SPEC §9) exists for — any model we build
  indexing a policy by legal-move position inherits the same bump-on-change
  discipline, and deserves the same direct ordering assertion, not just
  output parity.
- **Legality radius.** Their `placement_radius = 8` is our
  `LEGAL_RADIUS = 8`; their initial 218-node graph is our `DISK_CELLS = 217`
  plus the dummy. A graph builder on our side would source the legal set
  from `Position::is_legal` / the frontier, and the "legal cells are the
  candidate nodes" choice means graph size tracks our frontier size — no
  windowing needed, matching our unbounded-board model (the recentred arena
  stays private; only coordinates cross the boundary, and in the lean
  schema not even those).
- **Turn phase.** Their sole temporal input, `moves_remaining/2`, is our
  `TurnPhase` — and note the model never sees which stone of the pair was
  placed first; the search re-evaluates between placements. Their
  solver-proved two-hot 0.5/0.5 policy target encodes the turn's
  order-invariance; our engine's two-placement `advance` semantics would
  feed the same structure.
- **Win axes.** Their `WIN_AXES [(1,0),(0,1),(1,-1)]` are the same three
  lines as our `coord` axes; their edge reach `win_length − 1 = 5` is our
  `WINDOW_LEN − 1`. Their per-node threat features are computed by sliding
  the 6-windows through each cell — precisely the geometry our `window`
  module (`WINDOWS_PER_PLACEMENT = 18`) already enumerates; on our side
  these features could be produced from the same window walk the win check
  uses rather than a separate scan.
- **Curriculum vs fixed rules.** Strix varies `win_length` and radius per
  training stage; our engine deliberately fixes `WINDOW_LEN`/`LEGAL_RADIUS`
  as constants. Adopting a curriculum would require rule parametrization our
  engine doesn't (and per its spec shouldn't casually) expose — whereas the
  strix trick of decoupling the *model* from the stage via `axis_window ≥
  max(win_length) − 1` and per-stage graph topology needs nothing from the
  engine at all. Alternatively, a curriculum over `placement_radius` alone
  is close to free on our side (it's one legality constant), while a
  `win_length` curriculum cuts much deeper (windows, win detection, golden
  vectors).
- **Symmetry.** The lean-d6 lesson generalizes: exact D6 invariance came
  from removing every absolute-orientation input (coords, signed distance,
  absolute colour) and making the axis dimension a summed partition with
  tied weights — cheaper and stronger than augmentation, at ~3× conv
  parameter cost. Our engine's Zobrist/grid layers have no symmetry
  machinery and need none for this; the D6 table in §7 is pure coordinate
  algebra we'd host model-side.
- **Independent oracles.** Their parity discipline (committed tiny
  fixtures, provenance-gated real fixtures, panic-don't-skip, direct
  ordering asserts, a pinned bit-fingerprint) is the same philosophy as our
  golden vectors and `Position::audit()` — worth copying wholesale for any
  model we ship with more than one forward implementation.

What we'd likely *not* copy unchanged: the absolute `src_player` edge
channel (their own frontier deleted it), coordinate node features (ditto),
the dead-code dual head forwards, and the training-only heads' export
asymmetry (their Rust path can't load two of the shipped ablations — fail
loudly at config parse instead, as our style rules require).
