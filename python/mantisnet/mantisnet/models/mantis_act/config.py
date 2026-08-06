"""The MantisNet-ACT v4 configuration, its named presets, and its hash.

``MantisACTConfig`` is ``docs/MANTIS_ACT_SPEC.md`` §6 field for field. Every
enum-valued field is checked against an explicit vocabulary and every
cross-field invariant is checked at construction, so a configuration naming a
model nothing implements is refused where it is written rather than where a
tensor shape first disagrees.

``MANTIS_ACT_REPR_VERSION`` is a constant of this package and deliberately not
the ``hexo_py.MODEL_REPR_VERSION`` that §28 names. That one is a Rust constant
gating the legacy encoder's wire format, and every MantisNet checkpoint on disk
carries it: bumping it would fail their own version check and delete them,
which §32 and §37.13 forbid. ``4`` is the next repository value after that
constant's ``3``, and it names this representation only.

Invariants the constructor enforces past the per-field vocabularies. Each
rejected combination describes a model carrying parameters no forward would
read, which §29 forbids for the axis case and the house rule forbids in
general:

- axis channels exist exactly when ``d_axis > 0``, and axis latents need them;
- action-set latents exist exactly when ``num_action_latents > 0``, and
  ``global_mode="none"`` zeroes every latent count;
- pair messages are enabled exactly when ``pair_scope`` is not ``"none"``;
- ``occupied_radius <= d_max``: past ``d_max`` a displacement has no orbit
  class to carry, and the sparse paths emit no such edge (§11.2);
- ``pair_max_distance <= 5``: two cells further apart than a window's span
  share no six-cell window, so no pair evidence row could exist (§20.3).

``architecture_hash`` digests every field except ``dropout`` — under §28 that
is exactly the set changing a node set, a relation's meaning, a tensor shape,
or an output's meaning. It is a text digest of sorted ``field=repr(value)``
lines, so it is identical in every process: nothing here reads ``hash()``, set
iteration order, or dict insertion order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, replace

ARCHITECTURE_ID = "mantis_act_v4"
MANTIS_ACT_REPR_VERSION = 4

# A window spans five steps, so two cells more than five apart lie in no
# common window and can carry no pair evidence.
MAX_PAIR_DISTANCE = 5

# The allowed values of every enum-valued field. `architecture_id` is a
# one-value vocabulary: §28 fixes the id, and a config claiming another
# architecture would load into this model under a name it does not implement.
ENUM_VOCABULARIES: dict[str, frozenset[str]] = {
    "architecture_id": frozenset({ARCHITECTURE_ID}),
    "window_scope": frozenset({"live", "nonempty", "action_relevant"}),
    "cell_scope": frozenset({"occupied_only", "occupied_and_legal", "window_and_legal"}),
    "d6_relation_mode": frozenset({"orbit48", "coarse_distance_axis"}),
    "incidence_message": frozenset({"relation_gated", "additive"}),
    "incidence_reduce": frozenset({"sum", "mean", "attention"}),
    "global_mode": frozenset({"latents", "none"}),
    "window_window_mode": frozenset({"none", "typed_collinear_crossing"}),
    "pair_scope": frozenset(
        {
            "none",
            "current_legal_collinear",
            "post_action_collinear",
            "post_action_tactical",
        }
    ),
    "phase_conditioning": frozenset({"token_only", "film"}),
    "head_separation": frozenset(
        {"single_shared_head", "separate_output_mlps", "private_adapters"}
    ),
    "critic_type": frozenset({"categorical3", "scalar_tanh"}),
    "axis_pool_mode": frozenset({"mean", "learned_attention"}),
    "norm": frozenset({"layernorm"}),
    "activation": frozenset({"silu", "gelu", "relu"}),
}

# Widths and counts whose zero would remove a stream every path reads.
_POSITIVE_FIELDS = ("d_inv", "d_rel", "num_heads", "ffn_mult", "d_max")

# Widths, depths, and counts a valid ablation may zero. `d_axis` is the
# `full_no_axis` arm; the block counts collapse their stage.
_NON_NEGATIVE_FIELDS = (
    "d_axis",
    "state_blocks",
    "action_blocks",
    "policy_private_blocks",
    "critic_private_blocks",
    "num_inv_latents",
    "num_axis_latents",
    "num_action_latents",
    "occupied_radius",
)

# Dropout is the one field that changes no node set, relation, shape, or
# output meaning, so it is the one field outside the architecture hash (§28).
_UNHASHED_FIELDS = frozenset({"dropout"})


@dataclass(frozen=True)
class MantisACTConfig:
    """The resolved architecture of one MantisNet-ACT model (§6).

    The defaults are the ``full_act_v4`` model of §29: nonempty persistent
    windows, window-and-legal cells, axis channels, orbit48 geometry to
    radius 12, relation-gated incidence, four invariant and two axis state
    latents, two action latents, the 18-window counterfactual action encoder,
    post-action collinear partners, phase FiLM, private policy and critic
    adapters, the categorical three-class critic, no typed window attention,
    and no state-value head.
    """

    architecture_id: str = ARCHITECTURE_ID

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
    pair_max_distance: int = MAX_PAIR_DISTANCE
    use_action_set_latents: bool = True

    # Phase
    phase_conditioning: str = "film"
    use_three_way_phase: bool = True

    # Heads. `axis_pool_mode` is the §12.5 invariant head pool: the scores and
    # the channels permute together, so either mode is invariant.
    head_separation: str = "private_adapters"
    critic_type: str = "categorical3"
    axis_pool_mode: str = "learned_attention"
    enable_state_value_head: bool = False

    # Optional training-only heads
    enable_action_aux_heads: bool = False
    enable_window_fate_head: bool = False

    # Numerics
    norm: str = "layernorm"
    activation: str = "silu"
    dropout: float = 0.0
    layer_scale_init: float = 1e-2

    def __post_init__(self) -> None:
        for name, vocabulary in ENUM_VOCABULARIES.items():
            value = getattr(self, name)
            if value not in vocabulary:
                raise ValueError(
                    f"{name}={value!r} is not one of {sorted(vocabulary)}"
                )
        for name in _POSITIVE_FIELDS:
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name}={value} must be at least 1")
        for name in _NON_NEGATIVE_FIELDS:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name}={value} must not be negative")
        if self.d_inv % self.num_heads != 0:
            raise ValueError(
                f"d_inv={self.d_inv} must divide into num_heads={self.num_heads} heads"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout={self.dropout} must lie in [0, 1)")
        if self.layer_scale_init < 0.0:
            raise ValueError(
                f"layer_scale_init={self.layer_scale_init} must not be negative"
            )

        # Axis channels are the d_axis > 0 model. Either half alone names
        # parameters no forward reads, or a stream with no width.
        if self.use_axis_channels != (self.d_axis > 0):
            raise ValueError(
                f"use_axis_channels={self.use_axis_channels} disagrees with "
                f"d_axis={self.d_axis}: axis channels exist exactly when d_axis > 0"
            )
        if self.d_axis == 0 and self.num_axis_latents != 0:
            raise ValueError(
                f"num_axis_latents={self.num_axis_latents} needs axis channels, "
                f"but d_axis={self.d_axis}"
            )
        if self.global_mode == "none":
            for name in ("num_inv_latents", "num_axis_latents", "num_action_latents"):
                value = getattr(self, name)
                if value != 0:
                    raise ValueError(
                        f'global_mode="none" removes every latent, but {name}={value}'
                    )
        if self.use_action_set_latents != (self.num_action_latents > 0):
            raise ValueError(
                f"use_action_set_latents={self.use_action_set_latents} disagrees "
                f"with num_action_latents={self.num_action_latents}: the read and "
                f"broadcast exist exactly when there are action latents"
            )
        if self.use_action_pair_messages != (self.pair_scope != "none"):
            raise ValueError(
                f"use_action_pair_messages={self.use_action_pair_messages} "
                f"disagrees with pair_scope={self.pair_scope!r}: partner messages "
                f'are enabled exactly when the scope is not "none"'
            )

        # An edge past d_max has no orbit class to carry (§11.2).
        if self.occupied_radius > self.d_max:
            raise ValueError(
                f"occupied_radius={self.occupied_radius} exceeds d_max="
                f"{self.d_max}, past which a displacement has no relation class"
            )
        if self.use_occupied_radius_edges and self.occupied_radius < 1:
            raise ValueError(
                f"occupied_radius={self.occupied_radius} emits no edge; disable "
                f"use_occupied_radius_edges instead"
            )
        if not 1 <= self.pair_max_distance <= MAX_PAIR_DISTANCE:
            raise ValueError(
                f"pair_max_distance={self.pair_max_distance} must lie in "
                f"1..{MAX_PAIR_DISTANCE}: cells further apart share no window"
            )


# Every preset of §29 as a delta from `full_act_v4`, so each ablation's intent
# is its entry. A delta carries the partner fields the invariants demand —
# turning a component off means removing its parameters, not orphaning them.
PRESET_DELTAS: dict[str, dict[str, object]] = {
    # The full model: §6's defaults are exactly §29's full_act_v4 list.
    "full_act_v4": {},
    # Same-turn partner evidence off entirely (§20.5).
    "full_no_pair": {
        "use_action_pair_messages": False,
        "pair_scope": "none",
    },
    # Line direction carried by invariant features alone; no axis parameters
    # are retained (§29).
    "full_no_axis": {
        "d_axis": 0,
        "use_axis_channels": False,
        "num_axis_latents": 0,
    },
    # Current-style one-color persistent windows: mixed and dead windows stop
    # being nodes (§4).
    "full_live_windows": {"window_scope": "live"},
    # Also persist every empty window through a legal cell (§4).
    "full_action_relevant_windows": {"window_scope": "action_relevant"},
    # Local graph paths only; no global latent path at all (§17).
    "full_no_latents": {
        "global_mode": "none",
        "num_inv_latents": 0,
        "num_axis_latents": 0,
        "num_action_latents": 0,
        "use_action_set_latents": False,
    },
    # One invariant state latent, no axis or action latents (§29).
    "full_one_latent": {
        "num_inv_latents": 1,
        "num_axis_latents": 0,
        "num_action_latents": 0,
        "use_action_set_latents": False,
    },
    # The old distance plus on/off-axis scheme in place of the 48 orbits.
    "full_coarse_geometry": {"d6_relation_mode": "coarse_distance_axis"},
    # Exact orbit classes, but occupied edges only out to radius six (§29).
    "full_radius6": {"occupied_radius": 6},
    # The efficient control against persistent relevant empty cells (§8.1).
    "full_occupied_cells_only": {"cell_scope": "occupied_only"},
    # The legacy additive incidence message U h + E_r (§14).
    "full_additive_incidence": {"incidence_message": "additive"},
    # One shared action representation before the outputs (§23). Ablation only.
    "full_shared_head": {"head_separation": "single_shared_head"},
    # Direct typed collinear/crossing window attention (§16).
    "full_with_typed_window_attention": {
        "window_window_mode": "typed_collinear_crossing",
    },
    # The extra-FFN control §16 requires alongside typed window attention:
    # the same added parameter budget spent on width instead of on a new
    # communication path. §6 exposes no per-stream FFN width, so ffn_mult is
    # the lever, and §16 requires the parameter and time costs of both arms be
    # reported from the model summary rather than assumed equal.
    "full_extra_ffn_control": {"ffn_mult": 3},
    # Deterministic tactical action scalars off; post-placement pattern
    # encoding retained (§19).
    "full_no_tactical_inputs": {"use_action_tactical_features": False},
    # No action-set latent read or broadcast (§21).
    "full_no_action_latents": {
        "num_action_latents": 0,
        "use_action_set_latents": False,
    },
}

# Constructing here means every preset is validated at import: a delta that
# violates an invariant is an import error, not a training-launch error.
PRESETS: dict[str, MantisACTConfig] = {
    name: replace(MantisACTConfig(), **delta)
    for name, delta in PRESET_DELTAS.items()
}


def architecture_hash(cfg: MantisACTConfig) -> str:
    """A stable 16-hex-digit digest of every semantic field of ``cfg`` (§28).

    The digest is taken over the ``field=repr(value)`` lines in sorted field
    order, so it depends on nothing but the values: not on the declaration
    order of the dataclass, not on dict or set iteration, and not on the
    per-process seed of ``hash()``. Two processes reading the same checkpoint
    therefore agree, which is what makes strict load able to refuse a
    mismatch.
    """
    payload = "\n".join(
        f"{f.name}={getattr(cfg, f.name)!r}"
        for f in sorted(fields(cfg), key=lambda f: f.name)
        if f.name not in _UNHASHED_FIELDS
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# The summary layout: §6's own grouping, so a printed config reads against the
# spec block. The import check below refuses a field that appears in neither.
_SUMMARY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("widths", ("d_inv", "d_axis", "d_rel", "num_heads", "ffn_mult")),
    (
        "depth",
        (
            "state_blocks",
            "action_blocks",
            "policy_private_blocks",
            "critic_private_blocks",
        ),
    ),
    (
        "representation",
        (
            "window_scope",
            "cell_scope",
            "use_axis_channels",
            "use_global_numeric_features",
            "use_window_numeric_features",
            "use_action_tactical_features",
        ),
    ),
    (
        "geometry",
        (
            "d6_relation_mode",
            "d_max",
            "use_cell_adjacency",
            "use_occupied_radius_edges",
            "occupied_radius",
            "route_on_axis_radius_messages",
        ),
    ),
    (
        "messages",
        (
            "incidence_message",
            "incidence_reduce",
            "share_relation_embeddings_across_blocks",
        ),
    ),
    (
        "global",
        (
            "global_mode",
            "num_inv_latents",
            "num_axis_latents",
            "num_action_latents",
            "use_full_cell_attention",
            "window_window_mode",
        ),
    ),
    (
        "actions",
        (
            "use_counterfactual_action_windows",
            "use_action_pair_messages",
            "pair_scope",
            "pair_max_distance",
            "use_action_set_latents",
        ),
    ),
    ("phase", ("phase_conditioning", "use_three_way_phase")),
    (
        "heads",
        (
            "head_separation",
            "critic_type",
            "axis_pool_mode",
            "enable_state_value_head",
        ),
    ),
    ("aux heads", ("enable_action_aux_heads", "enable_window_fate_head")),
    ("numerics", ("norm", "activation", "dropout", "layer_scale_init")),
)

_SUMMARISED = {name for _group, names in _SUMMARY_GROUPS for name in names}
_MISSING = {f.name for f in fields(MantisACTConfig)} - _SUMMARISED - {"architecture_id"}
if _MISSING:
    raise RuntimeError(
        f"config fields absent from the summary layout: {sorted(_MISSING)}"
    )


def summarise(cfg: MantisACTConfig) -> str:
    """The resolved configuration as readable text, with its ablations named.

    The header carries the identity a checkpoint records — architecture id,
    representation version, architecture hash — and the trailing lines name
    every field that differs from ``full_act_v4``, so a run's log says which
    arm it is without a diff. ``window_scope="live"`` gets its own line: it is
    a legal configuration, but it drops the mixed and dead windows that §3 and
    §38 make part of the full model, and it is silent about it otherwise.
    """
    lines = [
        f"{cfg.architecture_id} repr_version={MANTIS_ACT_REPR_VERSION} "
        f"hash={architecture_hash(cfg)}",
    ]
    for group, names in _SUMMARY_GROUPS:
        body = "  ".join(f"{name}={getattr(cfg, name)!r}" for name in names)
        lines.append(f"  {group:<16}{body}")

    full = PRESETS["full_act_v4"]
    deltas = [
        f"{f.name}={getattr(cfg, f.name)!r}"
        for f in fields(cfg)
        if getattr(cfg, f.name) != getattr(full, f.name)
    ]
    lines.append(
        "  vs full_act_v4: " + (", ".join(deltas) if deltas else "identical")
    )
    if cfg.window_scope == "live":
        lines.append(
            '  ABLATION: window_scope="live" persists one-color nonempty '
            "windows only, so mixed and dead windows are not nodes at all. "
            "The full model represents them (§3, §38); this arm answers what "
            "they are worth."
        )
    return "\n".join(lines)
