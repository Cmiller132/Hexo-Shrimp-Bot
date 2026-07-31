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
- The **legal-cell decoders** score each legal cell on demand from the live
  windows passing through it, or from a background path when there are none.
  Policy and action value have independent parameters but share this routing
  and its joint `(occupancy mask, candidate slot)` class. Trunk cost therefore
  scales with stones and live windows, not with the legal halo.
- The **policy** decoder emits one raw logit per legal cell. The **action-value**
  decoder emits two return-mass logits per legal cell and composes them into one
  action value in `(−1, 1)` (appendix B).
- The **state-value** head reads the board through multi-query attention over the
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
slot classes = 3 and `DEC_CLASSES = 93` (§4.3), critic logits = 2, and
`moves_remaining ∈ {1, 2}`.

The default configuration has 1,272,868 parameters: 1,063,648 in the four
trunk blocks and 209,220 across input/final parameters and the three heads.

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

### 4.3 Slot and joint classes

A stone or cell occupies slot `s ∈ 0..5` of a window it belongs to. A
reflection reverses slot order, `s ↔ 5 − s`, so no encoding of a slot may
distinguish the two. There are two invariants of a pairing, and the model
uses each where it is the right one.

**Slot class** — `min(s, 5 − s) ∈ {0, 1, 2}` (end / near-end / centre). Used
for the stone↔window incidence of §5.1 and §5.2, where the pairing's other
half is a stone the window's own occupancy pattern already accounts for.

**Joint class** — for the legal-cell decoder of §6, the slot class discards
what the decoder most needs. The reflection acts on the pair
`(occupancy mask, candidate slot)` *jointly*, sending it to
`(reverse6(mask), 5 − s)`, so the orbits of that involution are the finest
reversal-invariant description of where a candidate sits among a window's
stones. The decoder class is the orbit's rank in ascending `(mask, slot)`
order, one of

```
DEC_CLASSES = 93
```

— the 186 pairs of a nonempty, nonfull mask with an empty slot, folded in
half: the involution has no fixed point, because no slot is its own mirror.

Keying the two halves separately, as `(canonical mask, slot class)`, is
coarser and realizes only 75 classes. The 18 it merges are exactly the pairs
of mirrored slots of a non-palindromic mask — for a lone stone at slot 0, a
candidate at slot 1 makes a contiguous pair and one at slot 4 a split pair,
and both are `(000001, near-end)`. Since a cell's decoder input is its window
rows summed plus its class counts, two cells with the same live windows and
the same counts get one row and therefore one logit and one action value
whatever the weights: the merge is an action alias, not an approximation.

Class embedding tables of width `H` appear in each place a pairing is
encoded; each site owns its own table. The policy and action-value decoders
each own a 93-row joint-class table. Replacing their former three-row tables
adds 23,040 parameters in the default model. Stone↔window incidence remains
keyed by the three-row slot class.

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

## 6. Legal-cell decoders

The policy and action-value heads are independently parameterized decoders over
one shared legal-cell incidence table. Each produces one result per legal cell
in **engine legal-move order** — the same lexicographic `(q, r)` order
`Position` exposes; result index `j` means `legal_moves[j]`, and this coupling
is versioned by `ACTION_ORDER_VERSION` (ENGINE_SPEC §9). Illegal cells are
never scored: masking is by construction, not by `−inf`.

For each legal cell `a`:

- **Window path** (cell lies in ≥ 1 live window):

  ```
  h_a^p          = Σ_{w ∋ a, live} ( P_W · W_w + E_pw[joint(a, w)] )
  h_a^q          = Σ_{w ∋ a, live} ( Q_W · W_w + E_qw[joint(a, w)] )  # ≤ 18 terms
  policy_logit   = MLP_P( [ h_a^p ; g ] )          # 2H → P_H → 1, ReLU
  [z_pos, z_neg] = MLP_Q( [ h_a^q ; g ] )          # 2H → P_H → 2, ReLU
  ```

  `joint(a, w)` is §4.3's joint class of `w`'s occupancy mask and `a`'s slot
  in it, so both `E_pw` and `E_qw` have `DEC_CLASSES = 93` rows. The builder
  emits, per legal cell, its list of (window index, joint class) pairs. One
  parameter-free gather-sum produces the shared window sum and joint-class
  counts; each head then applies its own projection and MLP. The decoder never
  searches for incidences in-forward.

- **Background path** (cell lies in no live window):

  ```
  h_a^p          = E_bg[ nearest-stone bucket(a) ]    # §4.2
  h_a^q          = E_qbg[ nearest-stone bucket(a) ]
  policy_logit   = MLP_P( [ h_a^p ; g ] )             # same policy MLP
  [z_pos, z_neg] = MLP_Q( [ h_a^q ; g ] )             # same critic MLP
  ```

The policy logit is exported raw. Appendix B defines how the critic's two
logits compose into `Q(s,a)` and how they are trained.

Both heads score **single placements**. The two-placements-per-turn
structure enters only through the token's `moves_remaining` input; pairing
the two placements of a turn is the search's job (it re-evaluates the
position between them). No softmax and no temperature exist anywhere in the
model; normalization is downstream.

## 7. State-value head

This board-level readout is distinct from the per-action critic of §6 and
appendix B. It uses multi-query attention over the windows plus the token,
then a binned output:

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
| decoder joint classes | orbits of a reversal acting on both halves (§4.3) |
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
(per-cell window/joint-class lists or background bucket, in engine order),
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
- Outputs: raw policy logits; action values composed in fp32 from the critic's
  two return-mass logits; the state-value bin distribution and its scalar
  decode. The model applies no policy softmax and does not clamp the decoded
  state value.

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

Two version constants govern compatibility. The current model has
`MODEL_REPR_VERSION = 2`:

- `ACTION_ORDER_VERSION` (engine-owned): a bump invalidates every
  checkpoint, as the policy indexes legal moves by position.
- `MODEL_REPR_VERSION` (model-owned): covers the builder and every feature
  encoding in §3–§4 (window liveness rule, pattern canonicalization, slot
  and joint classes, bucket tables, incidence layout). Any change to these
  bumps it and invalidates checkpoints. Formats are not backward compatible;
  there is one builder and one schema per version. Version 2 is the joint
  decoder class of §4.3; version 1 keyed the decoder by slot class alone.

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
8. **Decoder classes:** the 93 classes are exactly the orbits of the joint
   reversal — invariant on each orbit, distinct across orbits — with the
   ranking convention derived independently of the builder's table, since the
   Rust encoder must agree on it. Separation is asserted where it matters: a
   pair of legal moves that `(canonical mask, slot class)` gives one decoder
   row, given two.

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

The action-value head is §6's second decoder and emits one action value per
legal cell in engine legal-move order. It owns a window projection, 93-row
joint-class table, background-bucket table, and MLP distinct from the policy
decoder's parameters. The two decoders share only the parameter-free pass over
the decoder incidence table.

For each legal cell `a`, the head uses the same window/background routing as
§6:

```
h_a             = Σ_{w ∋ a, live} ( Q_W · W_w + E_qw[class(a, w)] )
[z_pos, z_neg]  = MLP_Q( [ h_a ; g ] )            # 2H → P_H → 2, ReLU
u_pos           = σ(z_pos),  u_neg = σ(z_neg)
Q(s, a)         = u_pos − u_neg
```

For a background cell, `h_a = E_qbg[nearest-stone bucket(a)]`. The two logits
decode to the **positive** and the **negative return mass** of the action:
`u_pos + u_neg` is the head's estimate of `E[|G|]`, and their difference is the
action value. Both masses lie in `(0, 1)`, so `Q ∈ (−1, 1)` and each mass moves
`Q` directly. The composition is fp32, as is every loss term over these values.

The raw pair is part of the interface: fitting reads `[z_pos, z_neg]` and
composes the taken action's `Q` from the same numbers, so one decoder pass
serves both. Every acting consumer sees only the composed value.

The KLENT operator and training contract are specified in
[`KLENT_FOR_HEXO.md`](KLENT_FOR_HEXO.md). The improvement step consumes the
policy logits and action values without an additional gain:

```
π′(a|s) ∝ exp[(Q(s,a) + τ·log π_θ(a|s)) / (τ+λ)]
v̂(s)    = E_{a~π′}[Q(s,a)]
```

Training selects the taken action and fits both the composed value and the two
masses. With `G_pos = max(G, 0)`, `G_neg = max(−G, 0)`, and
`η = mass_weight = 0.25`, the critic term per selected action is

```
L_Q = (Q − G)^2
      + (η/2) · [ BCEWithLogits(z_pos, G_pos)
                + BCEWithLogits(z_neg, G_neg) ]
```

The squared error has unit weight. The mass pair alone carries `η/2`, and all
three terms are evaluated in fp32. `KLENT_FOR_HEXO.md` §3 owns the complete
loss, including the unit-weight policy cross-entropy.

The parameterization obeys the exact identity
`sigmoid(2z) − sigmoid(−2z) = tanh(z)`. This is the function-preservation
identity used when converting compatible checkpoints; it is not a second
critic mode.

**Both the policy decoder's and action-value decoder's MLP output layers
initialize to zero**, overriding §10's framework default for those two
layers. Initial policy logits are therefore exactly zero, and so are the
initial action values: `z_pos = z_neg = 0` gives `u_pos = u_neg = 1/2`.

This head reads the trunk output and adds no inputs. Its two-logit readout
changes the checkpoint head shape but does not itself change
`MODEL_REPR_VERSION`; the current version is 2 because §4.3's decoder key is
part of the representation. The §7 state-value head is neither called nor
trained by the KLENT path.
