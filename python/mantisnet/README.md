# mantisnet

The MantisNet network of `docs/MODEL_SPEC.md`, real: the builder, the trunk,
both heads, and the losses, with the spec's §12 obligations as its test suite.

**Status: implemented and green.** 1.20 M parameters at the §2 defaults —
matching the spec's estimate — with every test passing on CPU and the bf16
smoke passing on CUDA. **Not yet a `ModelPackage`:** no encoder, no evaluator,
no sessions, no checkpoints. Search integration is deliberately deferred; this
package is the network the future package will own.

## Shape

```
python/mantisnet/
  pyproject.toml      # uv project; torch from the cu128 index, hexo-py by path
  mantisnet/
    __init__.py       # flat re-exports, MODEL_REPR_VERSION
    builder.py        # §3-§4, §9: graphs, index tables, collation
    model.py          # §5-§7, §10: trunk blocks, policy decoder, value head
    losses.py         # §6, §7, §10: targets, cross-entropies, decay grouping
  tests/              # the §12 obligations, one file per concern
  bench/
    bench_forward.py  # builder and forward throughput at spec defaults
```

Run everything from this directory:

```
uv sync                              # venv, hexo-py wheel via maturin, torch cu128
uv run pytest                        # the whole suite
uv run python bench/bench_forward.py # throughput on CPU and the local GPU
```

## Module map

| Module | Role |
| --- | --- |
| `builder` | `build` (raw §11 inputs to a `PositionGraph`), `from_position` (the `hexo_py` wrapper), `collate` (graphs to one `Batch` of index tensors). Owns `MODEL_REPR_VERSION` and every index convention. |
| `model` | `MantisConfig` (the §2 named parameters), `MantisNet`, `ModelOutput`. |
| `losses` | `value_target` (two-hot projection), `value_loss`, `policy_loss` (segmented CE over ragged engine-order logits), `param_groups` (§10 decay split). |

## Design notes

- **The builder re-derives windows; it never calls the engine's walk.** That is
  what keeps `windows_through` an *independent* oracle for §12.1 — a builder
  built on the engine's enumeration would agree with it by construction, which
  is the deleted-detector failure `CLAUDE.md` warns about. The enumeration is
  vectorised numpy end to end: candidate windows by broadcast, occupancy by one
  `searchsorted` against the packed stone keys, the decoder table by a second
  one against the live set.

- **34 canonical patterns, not the spec draft's 32.** The 62 nonempty, nonfull
  6-bit masks fold to `(62 + 6 palindromes) / 2 = 34` orbits under reversal.
  Raised and corrected in `MODEL_SPEC.md` §3.2; a test pins the count so the
  table cannot silently disagree with the doc again.

- **Two index conventions the spec left to the implementation**, fixed in
  `builder.py`'s docstring: attention buckets are `d-1` clamped, then `SELF`,
  then `TOKEN`, with `TOKEN` winning the token-token pair; the one stoneless
  position (ply 0) takes the background clamp bucket 7.

- **Batching is concatenation plus two padded layouts.** Message passing and
  both MLP heads run on concatenated entities with `index_add_`/`index_select`
  — no padding, no waste. Attention and the value readout run padded per
  position with the token at slot 0, masked block-diagonal by construction.
  Distance buckets are computed in-forward from padded coordinates —
  elementwise arithmetic, not index discovery — because shipping the
  `(P, T, T)` bucket tensor over PCIe would cost more than recomputing it.

- **`MLP([a; b])` is implemented as two linears** (`_PairMlp`): a linear over a
  concatenation is the sum of two linears, so the 2H-wide inputs of `MLP_W`,
  `MLP_S`, and `MLP_P` are never materialised. In the policy head the token
  half runs per *position* and is gathered to cells afterwards. Identical
  arithmetic and parameter count, ~10% off the batch forward; the batching-
  equivalence and D6 tests are what license calling it identical.

- **The forward allocates no default-dtype buffers.** Scatter targets derive
  their dtype from what is scattered into them, which is what lets the same
  code run fp32 and under bf16 autocast (§10); the value decode is fp32
  unconditionally so every consumer sees the same scalar.

- **`losses.policy_loss` refuses a target that does not sum to 1** per
  position, and `value_target` refuses outcomes outside `[-1, 1]` — both are
  silent-corruption inputs a training loop would otherwise absorb without a
  symptom.

## Deliberately absent

| Omitted | Why |
| --- | --- |
| Encoder / evaluator / sessions | Search integration is deferred by decision. The seam types live in `hexo-search`; wiring MantisNet to them is the Python-backed package of `CONTAINER_SPEC.md`, a change to be made there, not here. |
| The aux window head (spec appendix A) | Optional by spec, and adding it later touches no input — it reads trunk output. Deferred with the training loop. |
| A training loop | `losses.py` pins what outputs mean; when and on what to optimise is the container's `fit`. |
| Checkpoint I/O | The manifest and probe protocol are `hexo-model`'s, and arrive with the package. |
| `torch.compile` | Ragged shapes recompile per batch signature. Worth revisiting with bucketed batch shapes once a real self-play load exists. |

## Connections

- `python/hexo-py` supplies positions and the engine-order legal list — the
  builder's whole input, per `MODEL_SPEC.md` §11.
- `docs/MODEL_SPEC.md` is normative for everything here; where code and spec
  disagree, that is a finding to raise, and §3.2's pattern count was one.
- `hexo-model` / `crates/models/mock` show the package shape this model will
  eventually be wrapped in.
