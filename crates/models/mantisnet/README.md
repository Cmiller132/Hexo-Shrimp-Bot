# `hexo-model-mantisnet`

The first network-backed container package. It owns MantisNet's representation,
KLENT-improved opinion, sessions, checkpoint semantics, and PlyRecord
diagnostics while remaining entirely Python-free.

## One encoder, two consumers

`src/encoder.rs` is the one Rust implementation of `MODEL_SPEC.md`'s position
representation. The container calls it directly; `python/hexo-py` is a thin
PyO3 binding over the same functions. `MODEL_REPR_VERSION` is owned here and
exported to Python by that binding. The independent Python builder and its
golden-vector suite remain the parity oracle.

Worker threads encode one position as a strict little-endian item:

```text
magic[8], MODEL_REPR_VERSION:u32, moves_remaining:u8, reserved[3],
stones:u32, windows:u32, incidences:u32, legal:u32, decoder:u32, background:u32,
stone_own[i64], stone_qr[i32;2], window_feat[i64],
inc_stone[i64], inc_window[i64], inc_class[i64],
dec_cell[i64], dec_window[i64], dec_class[i64],
bg_cell[i64], bg_bucket[i64]
```

The batcher validates and collates those local graphs into the public
`RawBatch`. Counts are preflighted against the item length before allocation;
magic/version, features, coordinates, local indices, decoder coverage,
truncation, and trailing bytes are all refusals. This is the package encoder
version: any semantic or layout change moves `MODEL_REPR_VERSION`.

## Forward boundary and evaluator

`ForwardLoader` and `Forward` are injected Rust traits. They speak only
`Path`, `RawBatch`, and `Vec<f32>`:

```text
encoded bytes -> validated/collated RawBatch
              -> Forward::forward once for the whole batch
              -> flat (policy_logits, q_values), ragged by legal_offsets
              -> KLENT equation 3 per position
              -> Evaluation { priors: pi_prime, value: E_pi_prime[Q] }
```

The executable leaf implements the traits with PyO3 and live CPU Torch. No
Python, Torch, tensor, device, or GIL type enters this crate or another logic
crate. Runtime forward errors are fatal: an evaluator cannot substitute an
opinion after the network failed.

The equation-3 transcription uses the package's configured `tau` and `lambda`
in `f32`, matching `mantisnet.klent.improve.improved_policy`. Its committed
fixtures are regenerated, CPU-only, from `python/mantisnet`:

```text
python ../../crates/models/mantisnet/tests/fixtures/regenerate_improvement.py
```

The mathematically convex `v_hat` sum is projected onto `[-1, 1]` only when
sequential `f32` accumulation overshoots an endpoint by roundoff; the public
`Evaluation` convention is exact and sessions deliberately refuse values
outside it.

## Sessions and diagnostics

- self-play is `PolicySession`, sampling `pi_prime`;
- evaluation is the shared `hexo-search` `GumbelSession` at 32 simulations;
- variants are `policy`,
  `mcts:visits=N,inflight=N,cpuct=F`, and `gumbel:sims=N,m=N`.

Only self-play writes diagnostics. The nine-byte payload is:

```text
version:u8 (=1), v_hat:f32-le, entropy(pi_prime):f32-le
```

It describes the model opinion at the acted position without pretending to
contain a search tree or a training target that was never produced.

## Checkpoints and fitting

Python training writes the authoritative Torch `.pt` format. Container
initialisation seals that file as `weights.pt` beside `manifest.json`, after
loading it with the production version-refusing loader and computing the frozen
probe through the real evaluator. The manifest carries package, representation,
rules, action-order and protocol versions, `tau`, `lambda`, and the exact probe
hash. Every later load rebuilds a candidate live module, recomputes the probe,
and publishes it only after the hash agrees.

`fit` deliberately returns `PackageError::Unsupported`. Production MantisNet
training remains the KLENT loop in `mantisnet.klent.run`; moving that loop is an
owner decision. There is no partial container trainer.
