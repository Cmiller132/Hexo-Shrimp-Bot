# MantisNet-ACT v4 — deviations from the build specification

`docs/MANTIS_ACT_SPEC.md` is normative. Every point at which the implementation
departs from it is recorded here, with the section it departs from and the
reason. A reader of the spec must read this file too; nothing is deviated from
silently.

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
into `windows.py`, `cells.py`, `actions.py`, and `pairs.py`, with `builder.py`
retained as the orchestration entry point. §5's own instruction is to "add a new
package rather than expanding one monolithic file"; the split follows that
instruction further than the illustrative file list does.

## §6 — `axis_pool_mode` added to the config block

§12.5 requires a selectable invariant head pooling mode
(`"mean"` or `"learned_attention"`), but the §6 dataclass omits the field. It is
added with default `"learned_attention"`, the mode §12.5 recommends.

## §29 — the parameter-matched control preset is named

§16 and §29 both require a parameter-matched extra-FFN control alongside
`full_with_typed_window_attention`, but neither names it. It is added to
`PRESETS` as `full_extra_ffn_control`.

## §2, §25 — KLENT dispatch seam

§2 and §25 require the external `network_evaluate` interface to be unchanged.
It is. Internally, `mantisnet/klent/train.py::_policy_q` reaches into
`model.trunk(batch)` and `model.cell_head_logits(w, g, batch)` — a shape
contract specific to MantisNet's stone/window trunk that ACT's
cell/window/action/latent trunk does not have. Both architectures instead expose
`policy_q(batch) -> (policy_logits, critic_logits)` and `_policy_q` calls that.
The change is confined to one private function; every caller of
`network_evaluate` is unaffected.

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
