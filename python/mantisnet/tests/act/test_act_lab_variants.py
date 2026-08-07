"""The lab's ablation arms: one variant per §29 preset, each with its own collator.

`VariantSpec.collate` takes only the games and their prefix lengths, so the
configuration a batch is built under is bound into the collator rather than
passed to it. For MantisNet that binding is empty — its Rust builder emits one
representation whatever `MantisConfig` says — and for MantisNet-ACT it is the
whole of the contract: `window_scope` and `cell_scope` decide the node set,
`d6_relation_mode` and `d_max` the relation vocabulary, and a model built under
one of those against graphs built under another has no shape disagreement to
show for it. The widths agree, the indices are in range, and the forward runs
on the wrong board.

That is why the arms are registered rather than expressed as cell overrides,
and it is what this file has to detect. The three detectors, and what each is
independent of:

- **Every arm is collated through its own `spec.collate`, never through a
  configuration the test passes in.** An arm that silently collated
  `full_act_v4` would satisfy every shape, dtype and index check in the
  package, so the arms are compared against `full_act_v4` on the *same* real
  prefixes and required to differ in the direction their preset names — fewer
  windows under `live`, fewer radius edges at radius six, no legal cell node
  under `occupied_only`.
- **The split between arms that must differ and arms that must not is derived
  from `PRESET_DELTAS`, not listed here.** A preset whose delta names a builder
  field must have a stated direction, and one whose delta does not must collate
  a batch structurally identical to the full model's. A new preset therefore
  cannot be added without landing on one side or the other.
- **The refused override set is read off the builder's source**, so a builder
  that starts reading another configuration field fails here rather than
  leaving that field silently overridable.

Positions are the real stack-939 self-play games `test_act_numerics.py`
embeds, replayed as stored prefixes because that is what a corpus sample is
and what `collate_prefixes` takes. Random playouts would misstate every one of
the counts compared below: `docs/MANTIS_ACT_DEVIATIONS.md` measures them at
five times the legal cells and a fifteenth of the mixed windows of a real
position of the same depth.

The per-preset forward and backward of §37.1 is checked at the model level in
`test_act_model.py`. What is checked here is the lab seam: the model comes from
`build_variant` and the batch from that same variant's bound collator, which is
the pair that can disagree.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import fields

import pytest
import torch

from mantisnet.lab.variants import (
    ACT_BUILDER_FIELDS,
    VARIANTS,
    build_variant,
    derived_cell_name,
    parse_model_kw,
    variant_spec,
)
from mantisnet.model import MantisConfig
from mantisnet.models.mantis_act import actions as actions_module
from mantisnet.models.mantis_act import builder as builder_module
from mantisnet.models.mantis_act import cells as cells_module
from mantisnet.models.mantis_act import windows as windows_module
from mantisnet.models.mantis_act.config import (
    PRESET_DELTAS,
    PRESETS,
    MantisACTConfig,
)
from mantisnet.models.mantis_act.heads import MASK_SUFFIX
from mantisnet.models.mantis_act.model import MantisACT
from mantisnet.models.mantis_act.pattern_classes import MIXED

from .test_act_numerics import moves

SEED = 20260806
MASS_FLOOR = 0.2

ACT_ARMS = tuple(PRESETS)

# Two real self-play games at two depths, as the (games, prefix lengths) pair a
# corpus hands the collator. Ply 21 is an opening-ish board and ply 61 the
# middle game, where the mixed windows and the radius family are already at
# trained-play density.
PLIES = (21, 61)
PREFIX_GAMES = [moves(game, max(PLIES)) for game in (0, 1)]
PREFIX_TS = list(PLIES)


# --------------------------------------------------------------------------
# Structure of a collated batch, in the families the presets move


def cells(batch) -> int:
    return int(batch.cell_offsets[-1])


def windows(batch) -> int:
    return int(batch.window_offsets[-1])


def legal(batch) -> int:
    return int(batch.legal_offsets[-1])


def radius_edges(batch) -> int:
    return int(batch.radius_offsets[-1])


def fingerprint(batch) -> tuple:
    """Everything about a batch that a §29 representation ablation can move.

    Two arms with equal fingerprints collated the same graph. This is what the
    complement test asserts of an arm whose delta touches no builder field, and
    what makes "differs somewhere" a real claim for one whose delta does.
    """
    return (
        cells(batch),
        windows(batch),
        legal(batch),
        int(batch.adjacency_offsets[-1]),
        radius_edges(batch),
        int(batch.radius_orbit_bound),
        int(batch.window_incidence_mask.sum()),
        int(batch.cell_is_legal.sum()),
        int((batch.legal_to_cell_index >= 0).sum()),
        int((batch.window_status == MIXED).sum()),
        batch.window_numeric.shape[1],
        batch.action_tactical_numeric.shape[1],
        batch.global_numeric.shape[1],
    )


@pytest.fixture(scope="module")
def collated() -> dict:
    """Every arm's own collator run over one set of real prefixes."""
    return {
        name: variant_spec(name).collate(PREFIX_GAMES, PREFIX_TS)
        for name in ACT_ARMS
    }


# --------------------------------------------------------------------------
# What each representation arm has to do to the graph


def live_windows(arm, full) -> None:
    """One-colour persistent windows only: mixed and dead windows stop being nodes."""
    assert windows(arm) < windows(full)
    assert int((full.window_status == MIXED).sum()) > 0
    assert int((arm.window_status == MIXED).sum()) == 0


def action_relevant_windows(arm, full) -> None:
    """Empty windows through a legal cell are persisted too, so there are more."""
    assert windows(arm) > windows(full)
    assert cells(arm) >= cells(full)


def occupied_cells_only(arm, full) -> None:
    """No legal cell is a node at all, so `legal_to_cell_index` is all sentinel."""
    assert cells(arm) < cells(full)
    assert int(arm.cell_is_legal.sum()) == 0
    assert int((arm.legal_to_cell_index >= 0).sum()) == 0
    assert int((full.legal_to_cell_index >= 0).sum()) == legal(full)
    assert legal(arm) == legal(full)


def coarse_geometry(arm, full) -> None:
    """The same edges, typed by 2·bucket + off-axis instead of by the 48 orbits."""
    assert radius_edges(arm) == radius_edges(full)
    assert int(arm.radius_orbit_bound) < int(full.radius_orbit_bound)
    assert int(arm.radius_orbit.max()) < int(full.radius_orbit.max())


def radius6(arm, full) -> None:
    """Exact orbit classes, but no occupied edge past radius six."""
    assert radius_edges(arm) < radius_edges(full)
    assert cells(arm) == cells(full)
    assert windows(arm) == windows(full)


def no_tactical_inputs(arm, full) -> None:
    """§19.3's deterministic scalars become a zero-width block, not zero columns."""
    assert arm.action_tactical_numeric.shape[1] == 0
    assert full.action_tactical_numeric.shape[1] > 0
    assert cells(arm) == cells(full)
    assert windows(arm) == windows(full)


# One entry per §29 preset whose delta names a field the builder reads. The
# test below requires this table to be exactly that set, so a preset that
# ablates the representation cannot be registered without stating what its
# graph must do.
REPRESENTATION_ARMS = {
    "full_live_windows": live_windows,
    "full_action_relevant_windows": action_relevant_windows,
    "full_occupied_cells_only": occupied_cells_only,
    "full_coarse_geometry": coarse_geometry,
    "full_radius6": radius6,
    "full_no_tactical_inputs": no_tactical_inputs,
}


# --------------------------------------------------------------------------
# The registry


def test_every_named_preset_is_a_lab_variant():
    """§37.1 over the registry: an arm exists for every preset and nothing else.

    Generated rather than listed — `VARIANTS` is built from `PRESETS` — so a
    preset added to §29's table is an arm the same day, and every test in this
    file that iterates the registry covers it without being edited.
    """
    assert set(VARIANTS) == {"mantis"} | set(PRESETS)
    assert VARIANTS["mantis"].config is MantisConfig
    assert VARIANTS["mantis"].rust_collate is True
    for name in ACT_ARMS:
        spec = variant_spec(name)
        assert spec.config is MantisACTConfig, name
        assert spec.rust_collate is False, name
        assert name in spec.description, name
    # A variant name becomes a sweep cell directory, and `derived_cell_name`
    # joins it to its overrides with `+`.
    for name in VARIANTS:
        assert name.isidentifier(), name
    assert derived_cell_name("full_radius6", {"d_inv": 32}) == "full_radius6+d_inv32"


def test_an_unknown_arm_is_refused_with_the_list_of_real_ones():
    """The refusal is the operator's index of the arms, so it has to carry them.

    `--variant` deliberately has no argparse `choices`: resolving one imports
    torch and the whole ACT package, which `--help` would then pay for.
    """
    with pytest.raises(ValueError, match="unknown lab variant 'full_radius_6'") as raised:
        variant_spec("full_radius_6")
    for name in VARIANTS:
        assert name in str(raised.value), name


def test_overrides_are_typed_against_the_arms_own_dataclass():
    """A `key=value` is checked against the architecture it is actually for."""
    assert parse_model_kw(["d_inv=32", "use_axis_channels=false"], "full_act_v4") == {
        "d_inv": 32,
        "use_axis_channels": False,
    }
    with pytest.raises(ValueError, match="unknown MantisACTConfig field 'h'"):
        parse_model_kw(["h=16"], "full_radius6")
    with pytest.raises(ValueError, match="unknown MantisConfig field 'd_inv'"):
        parse_model_kw(["d_inv=16"])


# --------------------------------------------------------------------------
# The binding: one board per arm, model and graph alike


def test_every_arm_builds_its_model_from_the_configuration_its_collator_uses():
    """The model is the preset, and an override cannot move it off that preset.

    Building from `MantisACTConfig()` and applying the preset's delta would
    agree only for as long as `full_act_v4` stayed the dataclass defaults, and
    a divergence between the two has no shape to show it — which is the failure
    the factory's own refusal message describes.
    """
    for name in ACT_ARMS:
        assert build_variant(name, {})[0].cfg == PRESETS[name], name
        # A width override lands on that preset's configuration, and leaves
        # every field the builder reads exactly where the preset put it.
        cfg = build_variant(name, {"d_inv": 32})[0].cfg
        assert cfg.d_inv == 32, name
        for field in ACT_BUILDER_FIELDS:
            assert getattr(cfg, field) == getattr(PRESETS[name], field), (name, field)


def test_every_arm_refuses_the_overrides_its_own_collator_cannot_honour():
    """The refusal is per variant, and it names the variant it is protecting.

    `full_live_windows` refuses `window_scope="live"` as well, though that is
    the scope it is already bound to: the field is what is refused, not the
    value, because a cell that carries it is asking the harness to decide which
    of the two configurations the collator should have used.
    """
    for name in ACT_ARMS:
        with pytest.raises(ValueError, match="window_scope") as raised:
            build_variant(name, {"window_scope": "live"})
        assert name in str(raised.value)
        assert "full_live_windows" in str(raised.value), name
        with pytest.raises(ValueError, match="use_cell_adjacency"):
            build_variant(name, {"use_cell_adjacency": False})


def test_the_refused_override_set_is_the_set_the_builder_reads():
    """The registry's list against the builder's source, not against a memory.

    A builder that starts reading another configuration field would otherwise
    leave that field overridable, and a cell whose graph and model describe
    different boards produces no shape disagreement at all.
    """
    read: set[str] = set()
    for module in (builder_module, windows_module, cells_module, actions_module):
        read |= set(re.findall(r"\bcfg\.([a-z_0-9]+)", inspect.getsource(module)))
    known = {f.name for f in fields(MantisACTConfig)}
    assert read <= known, sorted(read - known)
    assert read == set(ACT_BUILDER_FIELDS)


# --------------------------------------------------------------------------
# The graphs each arm actually collates


def test_the_stated_directions_cover_every_arm_that_moves_the_representation():
    """The split is `PRESET_DELTAS` against `ACT_BUILDER_FIELDS`, not a list."""
    moves_the_graph = {
        name
        for name, delta in PRESET_DELTAS.items()
        if set(delta) & ACT_BUILDER_FIELDS
    }
    assert set(REPRESENTATION_ARMS) == moves_the_graph


def test_each_representation_arm_collates_its_own_graph(collated):
    """The direction each preset names, measured through that arm's collator.

    This is the bug the design exists to prevent: an arm bound to the wrong
    configuration collates `full_act_v4`'s graph, and every width still agrees,
    every index is still in range, and the forward still runs.
    """
    full = collated["full_act_v4"]
    for name, direction in REPRESENTATION_ARMS.items():
        arm = collated[name]
        assert fingerprint(arm) != fingerprint(full), (
            f"{name} collated a graph structurally identical to full_act_v4's"
        )
        direction(arm, full)


def test_each_arm_that_moves_no_builder_field_collates_the_full_graph(collated):
    """The complement, which is what makes the test above a two-sided detector.

    An arm that changed only the model must collate the same graph. If it does
    not, either its collator is bound to something other than its own preset or
    the builder has started reading a field the registry does not refuse.
    """
    full = fingerprint(collated["full_act_v4"])
    for name in ACT_ARMS:
        if name in REPRESENTATION_ARMS:
            continue
        assert fingerprint(collated[name]) == full, name


def test_the_full_arm_carries_the_families_the_others_ablate(collated):
    """The comparisons above are only meaningful if the baseline has the parts."""
    full = collated["full_act_v4"]
    assert cells(full) > 0 and windows(full) > 0 and legal(full) > 0
    assert radius_edges(full) > 0
    assert int((full.window_status == MIXED).sum()) > 0
    assert full.action_tactical_numeric.shape[1] > 0
    assert int(full.position_count) == len(PREFIX_TS)


# --------------------------------------------------------------------------
# End to end through the lab seam


def scalar(out) -> torch.Tensor:
    """A loss every output stream reaches, so one backward covers all of them."""
    terms = [
        out.policy_logits.float().square().mean(),
        out.critic_logits.float().square().mean(),
    ]
    for name, value in sorted(out.aux.items()):
        if not name.endswith(MASK_SUFFIX):
            terms.append(value.float().square().mean())
    return sum(terms)


@pytest.mark.parametrize("name", ACT_ARMS)
def test_every_arm_runs_a_forward_and_a_backward_through_its_own_collator(name):
    """§37.1 through the lab seam, on real stored prefixes.

    `build_variant` and `spec.collate` are the pair that can disagree, so both
    ends come from the registry and neither is handed a configuration here.

    The weights are perturbed off their initialisation first. §23 zero-
    initialises both output layers, so a fresh model gives every action the
    same logit and a squared loss of exactly zero — a backward that would run
    just as happily over a graph the model cannot read. Per-parameter gradient
    completeness stays `test_act_model.py`'s check, over the same preset table
    and against §32's "disabled means absent".
    """
    torch.manual_seed(SEED)
    model, normalized, spec = build_variant(name, {})
    assert isinstance(model, MantisACT) and normalized == {}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter), alpha=0.02)

    batch = spec.collate(PREFIX_GAMES, PREFIX_TS)
    out = model(batch, MASS_FLOOR)
    assert tuple(out.policy_logits.shape) == (legal(batch),)
    assert torch.isfinite(out.policy_logits).all()
    assert torch.isfinite(out.q_value).all()

    loss = scalar(out)
    assert torch.isfinite(loss) and float(loss.detach()) > 0.0
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), f"{name}: {parameter_name}"
    assert any(float(gradient.abs().max()) > 0.0 for gradient in gradients)
