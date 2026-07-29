# MantisNet — model specification

**Status: normative for the model.** This is the implementation target for the
MantisNet network: the input representation, the trunk, the heads, and the
contracts they expose. Search and the training loop are out of scope except
where a training target defines what an output *means*. Game rules are owned
by `docs/ENGINE_SPEC.md`; where this document restates a rule, the engine
spec wins.

Dimensions and table sizes are named parameters (§2). "The default model"
means the model instantiated with the defaults listed there.

---

## 1. Overview

MantisNet is a graph network whose units are **stones** and **live win
windows**, plus one **global token**. It has no cell grid, no coordinate
inputs, and no nodes for empty cells.

- A **window** is a set of six consecutive cells along one of the three
  axes — the atom of the win condition. A window is **live** when it
  contains at least one stone and stones of only one colour. Mixed windows
  can never be completed by either player and are excluded from the model
  entirely: blocking is represented by *absence*, not by a feature.
- The **trunk** interleaves bipartite message passing (stones ↔ the windows
  they occupy) with self-attention over the stone set, biased only by hex
  distance. The window pathway carries all line/tactical structure; the
  attention pathway carries global context in a single hop.
- The **policy** is a decoder, not a trunk node set: each legal cell's logit
  is computed on demand from the live windows passing through it (or from a
  background path when there are none). Trunk cost therefore scales with
  stones and live windows, not with the legal halo.
- The **action-value** head is an independently parameterized decoder with the
  same routing as policy. It emits one tanh-bounded scalar per legal cell
  (appendix B).
- The **value** head reads the board through multi-query attention over the
  window embeddings and outputs a binned distribution over `[−1, 1]`,
  decoded to a scalar in-forward.

Every input to the network is invariant under the 12-element hex symmetry
group D6, so the whole model is D6-invariant by construction (§8).

---

## 2. Named parameters

| Symbol | Meaning | Default |
|---|---|---|
| `H` | embedding width, everywhere | 128 |
| `B` | number of trunk blocks | 4 |
| `A` | attention heads | 4 |
| `F` | FFN expansion factor in the attention sub-block | 2 |
| `D_MAX` | hex-distance clamp for the attention bias table | 12 |
| `Q` | learned value-readout queries | 4 |
| `K` | value bins | 65 |
| `P_H` | policy and action-value decoder MLP hidden width | 128 |
| `V_H` | value MLP hidden width | 128 |
| `DROPOUT` | dropout probability (trunk sub-blocks) | 0.0 |

Fixed constants (not parameters): `WINDOW_LEN = 6` cells per window, 3 axes,
slot classes = 3 (§4.3), `moves_remaining ∈ {1, 2}`.

The default configuration has 1,249,699 parameters: 1,063,648 in the four
trunk blocks and 186,051 across input/final parameters and the three heads.

---

## 3. Input entities

### 3.1 Stones

Every placed stone, both colours. Colour is encoded **relative to the side
to move**: own / opponent. There are no coordinate features, no
move-number or recency features, and no per-stone scalars of any kind; a
stone's initial embedding is a 2-entry table lookup on own/opp.
Differentiation between stones comes entirely from the structure around
them (their windows and their distances to other stones).

### 3.2 Live windows

The window set is:

```
windows = { w : w is a 6-cell axis window,
            w contains ≥ 1 stone,
            all stones in w are one colour }
```

Windows containing stones of both colours (dead) and windows containing no
stones are **not entities**. A window with 6 own stones is a completed win;
such positions are terminal and are never evaluated, so live windows carry
1–5 stones.

Builder: enumerate, for each stone, the 18 windows through it (3 axes × 6
offsets — the same walk the engine's win check performs), deduplicate by
window identity (axis + anchor cell), discard mixed. The window count is
therefore bounded by `18 · n_stones` before deduplication and death.
Terminal positions are a builder error, not a silent default.

Each window's initial embedding is a table lookup on
`(colour, canonical occupancy pattern)`:

- occupancy pattern = the 6-bit mask of which of the window's slots hold a
  stone (slot order along the axis; 1–5 bits set);
- canonicalized under reversal: `canon(m) = min(m, reverse6(m))`, because a
  reflection reverses slot order. There are 34 canonical patterns of 1–5
  bits — the 62 nonempty, nonfull masks fold to `(62 + 6 palindromes) / 2`
  orbits — so the table has `2 × 34 = 68` entries.

### 3.3 The global token

One learned token per position, initialized as a base embedding plus a
2-entry table lookup on `moves_remaining` (how many placements the side to
move still has this turn). This is the model's **only** temporal input.
Side-to-move never appears as a feature: every colour in the input is
already side-to-move relative. The token participates in the attention
sub-block (§5.3) and in all three heads.

### 3.4 Excluded inputs and entities

No empty-cell nodes (the policy decoder covers them, §6). No global
"virtual node" wired into message passing (the token lives in attention).
No coordinates, axis identities, signed distances, or absolute colours
anywhere (§8 depends on this). No history planes. No jumping-knowledge
aggregation.

---

## 4. Feature encodings

### 4.1 Hex distance buckets

For stones `i, j`: `d(i,j) = max(|Δq|, |Δr|, |Δq+Δr|)`, clamped to `D_MAX`.
The attention bias table (§5.3) has, per head, one learned scalar per bucket
`1..D_MAX`, plus two dedicated indices: `SELF` (i = j) and `TOKEN` (any pair
involving the global token). Hex distance is D6-invariant, which is what
makes the attention pathway symmetry-safe.

### 4.2 Nearest-stone buckets (background policy path)

For a legal cell, the hex distance to the nearest stone, clamped to the
legality radius (8): a table of 8 embeddings of width `H`.

### 4.3 Slot classes

A cell occupies slot `s ∈ 0..5` of a window it belongs to. Under reversal
`s ↔ 5 − s`, so the model uses the reversal-invariant **slot class**
`min(s, 5 − s) ∈ {0, 1, 2}` (end / near-end / centre). Slot-class embedding
tables of width `H` appear in each place a stone-or-cell↔window pairing is
encoded; each site owns its own table.

---

## 5. Trunk

`B` identical blocks over the state `(S, W, g)`: stone embeddings
`S ∈ R^{n_s×H}`, window embeddings `W ∈ R^{n_w×H}`, token `g ∈ R^H`. All
sub-blocks are pre-norm residual with `LayerNorm` (ε = 1e-5); dropout (if
`DROPOUT > 0`) is applied to each sub-block's output before the residual
add. After the last block a final `LayerNorm` is applied to `S`, `W`, and
`g` separately (one shared `LayerNorm(H)` module).

Each block, in order:

### 5.1 Window ← stones

For each window `w`, over the stones it contains:

```
agg_w = Σ_{i ∈ w}  ( U · LN(S_i)  +  E_ws[class(i, w)] )
W_w   = W_w + MLP_W( [ LN(W_w) ; agg_w ] )        # MLP_W: 2H → H → H, ReLU
```

Aggregation is a sum, not a mean; the stone count remains represented in its
magnitude.

### 5.2 Stone ← windows

For each stone `i`, over the (≤ 18) windows containing it:

```
agg_i = Σ_{w ∋ i}  ( V · LN(W_w)  +  E_sw[class(i, w)] )
S_i   = S_i + MLP_S( [ LN(S_i) ; agg_i ] )        # MLP_S: 2H → H → H, ReLU
```

### 5.3 Stone self-attention (+ token)

Standard multi-head attention over the `n_s + 1` rows `[S; g]`, pre-norm:

```
Z        = LN([S; g])
logits   = (Z W_q)(Z W_k)^T / sqrt(H/A)  +  bias
bias_h[i,j] = b_h[ bucket(i, j) ]                  # per-head table, §4.1
[S; g]  += (softmax(logits) · Z W_v) W_o
[S; g]  += FFN( LN([S; g]) )                       # H → F·H → H, ReLU
```

Windows do not attend. In a batch, attention is masked block-diagonal per
position (§9).

---

## 6. Policy head (decoder)

One raw logit per legal cell, in **engine legal-move order** — the same
lexicographic `(q, r)` order `Position` exposes; logit index `j` means
`legal_moves[j]`, and this coupling is versioned by `ACTION_ORDER_VERSION`
(ENGINE_SPEC §9). Illegal cells are never scored: masking is by
construction, not by `−inf`.

For each legal cell `a`:

- **Window path** (cell lies in ≥ 1 live window):

  ```
  h_a    = Σ_{w ∋ a, live}  ( P · W_w  +  E_pw[class(a, w)] )     # ≤ 18 terms
  logit  = MLP_P( [ h_a ; g ] )                    # 2H → P_H → 1, ReLU
  ```

  The builder emits, per legal cell, its list of (window index, slot class)
  pairs; the decoder is a gather-sum, never a search.

- **Background path** (cell lies in no live window):

  ```
  h_a    = E_bg[ nearest-stone bucket(a) ]         # §4.2
  logit  = MLP_P( [ h_a ; g ] )                    # same MLP
  ```

The head scores **single placements**. The two-placements-per-turn
structure enters only through the token's `moves_remaining` input; pairing
the two placements of a turn is the search's job (it re-evaluates the
position between them). No softmax and no temperature exist anywhere in the
model; normalization is downstream.

## 7. Value head

Multi-query attention readout over the windows plus the token, then a
binned output:

```
keys/values = LN over rows [ W ; g ]               # token always present ⇒ well-defined even with n_w = 0
r_q         = Attn( query_q, keys, values )        # q = 1..Q learned query vectors, single-head each
v_logits    = MLP_V( [ r_1 ; … ; r_Q ] )           # Q·H → V_H → K, ReLU
```

- Bin centers: `linspace(−1, 1, K)` — uniform, endpoints inclusive, odd `K`
  so an exact-zero bin exists. Centers are derived from config, never
  stored in checkpoints.
- Scalar decode, applied in-forward so every consumer sees the same value:
  `value = Σ softmax(v_logits) · centers ∈ [−1, 1]`.
- **Perspective: side to move.** The training target is the game outcome
  from the position's mover (`+1` eventual win, `−1` loss), projected onto
  the bins as an exact-in-expectation two-hot distribution and trained with
  cross-entropy. Any non-decisive outcome label is defined by the training
  target, not by the model.

---

## 8. Symmetry

Requirement: **the model is exactly D6-invariant** — for every position and
every one of the 12 board symmetries, the value is identical and the policy
maps through the coordinate transform. Training does not use symmetry
augmentation.

The invariance is architectural, and rests on an input audit — every input
is a D6 invariant:

| Input | Why invariant |
|---|---|
| stone colour (own/opp) | colours don't move under board symmetry |
| window occupancy pattern | canonicalized under reversal (§3.2) |
| slot classes | reversal-invariant by definition (§4.3) |
| hex-distance buckets | hex distance is D6-invariant |
| nearest-stone buckets | ditto |
| `moves_remaining` | not geometric |

plus the structural facts that windows map to windows under D6, no feature
names an axis, and all aggregations are unordered sums or softmaxes.
Adding **any** feature that can distinguish two D6-equivalent positions
breaks the guarantee and must be treated as an architecture change, not a
feature tweak.

Test obligation: for all 11 non-identity transforms on real positions,
value equal and per-move policy equal through the transform, to
`atol 1e-5` (exact in exact arithmetic; summation order changes under node
reordering, so bit-equality is not required).

---

## 9. Batching

Positions batch by concatenation with per-position index offsets — stones,
windows, and one token per position. Message passing never crosses
positions (indices are per-position by construction); attention is masked
block-diagonal per position. The builder emits, per position: the stone
table, the window table (colour + canonical pattern), the stone↔window
incidence list with slot classes, the legal-cell decoder table
(per-cell window/slot-class lists or background bucket, in engine order),
and `moves_remaining`. All index tensors are precomputed by the builder;
the forward contains no data-dependent index discovery.

---

## 10. Numerics and conventions

- Weights fp32; training under bf16 autocast is supported and the model
  must not assume otherwise (no default-dtype buffer allocations in the
  forward).
- `LayerNorm` everywhere, ε = 1e-5; no BatchNorm anywhere (batch
  transparency is required).
- Activations: ReLU throughout.
- Init: framework defaults for linears; embedding tables and the learned
  value queries `N(0, 0.02)`; attention-bias tables zero; the token base
  embedding `N(0, 0.02)`.
- Optimizer grouping: parameters with `ndim ≤ 1`, all embedding tables,
  and the attention-bias tables are excluded from weight decay.
- Outputs: raw policy logits; tanh-bounded scalar action values; the state-value
  bin distribution and its scalar decode. The model applies no policy softmax
  and does not clamp the decoded state value.

---

## 11. Interface and versioning

Inputs (from the engine, per position): stone list `(coord, colour)`;
legal-move list in engine order; `moves_remaining`. The builder (§3, §9) is
part of the model, not the engine: the engine exposes rules and ordering,
the model owns its representation.

Outputs (per position): `policy_logits` and `q_values` (one of each per legal
move in engine order), `value` (state-value scalar), `value_dist` (`K`
probabilities), and `value_logits` (`K` raw bin logits). Appendix B defines
the action-value semantics.

Two version constants govern compatibility:

- `ACTION_ORDER_VERSION` (engine-owned): a bump invalidates every
  checkpoint, as the policy indexes legal moves by position.
- `MODEL_REPR_VERSION` (model-owned): covers the builder and every feature
  encoding in §3–§4 (window liveness rule, pattern canonicalization, slot
  classes, bucket tables, incidence layout). Any change to these bumps it
  and invalidates checkpoints. Formats are not backward compatible; there
  is one builder and one schema per version.

---

## 12. Test obligations

1. **Builder oracle:** window enumeration checked against an independent
   walk (the engine's own window geometry qualifies as the independent
   implementation only if the model builder does not call it — otherwise
   write the naive re-derivation in the test).
2. **Ordering:** a direct assertion that policy and action-value index `j`
   maps to `legal_moves[j]` — not inferred from output parity, asserted on
   the decoder table itself.
3. **D6 invariance:** as specified in §8.
4. **Batching equivalence:** batched forward equals per-position forward
   (`atol 1e-6`).
5. **Liveness:** placing an opponent stone into a window removes it from
   the entity set; a position differing only by a dead window's contents
   beyond the mix produces an identical graph.
6. **Decoder coverage:** both cell heads score every legal cell exactly once;
   cells on the background path are exactly those with no live window.
7. If a second forward implementation ever exists, committed random-weight
   parity fixtures with pinned tolerances and provenance-gated
   real-checkpoint fixtures, failing loudly when weights are absent.

---

## Appendix A — auxiliary window head (optional extension)

A training-only head over the final window embeddings. It is not exported
and is absent from the inference interface.

```
aux_logits_w = MLP_A( W_w )        # H → H → 3, ReLU
```

Three-class target per window, judged over the remainder of the game:

| Class | Meaning |
|---|---|
| 0 | the window dies (an opposing stone enters it) |
| 1 | the window stays live to the end without completing |
| 2 | the window completes (is one of the winning windows) |

Loss: cross-entropy, mean over windows, added to the total with weight
`λ_AUX`; 0.0 means the head is absent. The nonzero weight and any per-class
weighting are training configuration and do not change the model.

Enabling or disabling this head does not touch `MODEL_REPR_VERSION` — it
reads the trunk's output and adds no inputs. Checkpoints that carry it
load into models without it by dropping the unmatched tensors loudly (an
explicit allowlist, not silent prefix matching).

---

## Appendix B — action-value head (the KLENT interface)

The action-value head has the §6 decoder shape and emits one scalar per legal
cell in engine legal-move order. It owns a window projection, slot-class
table, background-bucket table, and MLP distinct from the policy decoder's
parameters. The policy and action-value heads may share the parameter-free
pass over the decoder incidence table.

For each legal cell `a`, the head uses the same window/background routing as
§6:

```
h_a     = Σ_{w ∋ a, live} ( Q_W · W_w + E_qw[class(a, w)] )
q_raw   = MLP_Q( [ h_a ; g ] )                    # 2H → P_H → 1, ReLU
Q(s, a) = tanh(q_raw)
```

For a background cell, `h_a = E_qbg[nearest-stone bucket(a)]`. The head applies
`tanh`, so each action value lies in `(−1, 1)`. The KLENT operator and loss cast
these values to fp32 before arithmetic.

The KLENT operator and training contract are specified in
[`KLENT_FOR_HEXO.md`](KLENT_FOR_HEXO.md). The improvement step consumes the
policy logits and action values without an additional gain:

```
π′(a|s) ∝ exp[(Q(s,a) + τ·log π_θ(a|s)) / (τ+λ)]
v̂(s)    = E_{a~π′}[Q(s,a)]
```

Training selects the taken action and minimizes its squared error
`(Q(s,a_taken) − G)²` against the sample's λ-return `G`. Policy
cross-entropy is unchanged.

**Both the policy decoder's and action-value decoder's MLP output layers
initialize to zero**, overriding §10's framework default for those two
layers. Initial policy logits and action values are therefore exactly zero.

This head reads the trunk output and adds no inputs, so it does not change
`MODEL_REPR_VERSION`. The §7 state-value head is neither called nor trained
by the KLENT path.
