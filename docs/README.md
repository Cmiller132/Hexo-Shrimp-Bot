# Documentation index

## Specifications

- [`CONTAINER_SPEC.md`](CONTAINER_SPEC.md) defines how packages, evaluators,
  sessions, processes, checkpoints, and records compose. Read it before
  changing container execution or Rust/Python ownership boundaries.
- [`MANTIS_ACT_SPEC.md`](MANTIS_ACT_SPEC.md) defines the MantisNet-ACT v4
  architecture.
- [`MANTIS_ACT_DEVIATIONS.md`](MANTIS_ACT_DEVIATIONS.md) records deviations
  from the ACT spec encountered during implementation.

## KLENT references and evidence

- [`KLENT_PAPER.md`](KLENT_PAPER.md) states the KLENT method and its
  mathematical basis.
- [`KLENT_FOR_HEXO.md`](KLENT_FOR_HEXO.md) maps KLENT onto Hexo's model,
  self-play, targets, and evaluation pipeline.
- [`ABLATIONS.md`](ABLATIONS.md) records measured comparisons and outcomes.

## In-crate documentation

Engine rules and representation are in `crates/hexo-engine/README.md`. Model
architecture, lab harness, and deck are in `python/mantisnet/README.md`. Every
crate and Python package carries its own README.
