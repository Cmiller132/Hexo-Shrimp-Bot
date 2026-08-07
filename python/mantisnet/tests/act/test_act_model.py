"""The assembled model: §25's output, §28's checkpoint, and the KLENT seam.

Everything here runs the whole model — trunk, action encoder, heads — on
**real stack-939 self-play positions**, at both parities of the turn so all
three §13.1 phases are exercised, and at the empty board so OPENING is not a
phase the tests only assert about. The corpus is the one
`test_act_numerics.py` documents and embeds; sharing it is deliberate, because
a second transcription of two 161-ply games would be a second thing to keep
true rather than a second detector.

The detectors, and what each one is independent of:

- **Engine order is checked by permutation, not by reading the mapping.** The
  builder's own claim — that ``cell_qr[legal_to_cell_index[j]]`` is
  ``legal_moves()[j]`` — is asserted directly, and then a batch whose
  per-action rows are permuted is required to produce exactly the permuted
  output. The second check is the sharp one: it goes through the whole model
  and would catch a stage that sorted, grouped, or re-derived an action order
  of its own, which no assertion about the builder's tables can see.

- **Batched and single-position forwards must agree** (§31.10). A batch is the
  only place a segment boundary can be got wrong, and every quantity in this
  model is a segment reduction over one; a per-position forward has no
  boundaries to get wrong, so the two disagreeing is the detector.

- **Strict load is checked field by field.** Every semantic config field is
  perturbed in turn and the load required to refuse it by name, so the check
  is not one hash comparison that a future field would silently escape.
  ``dropout`` is required to be accepted, because §28 asks for agreement on
  shape and semantic fields and the architecture hash already excludes it.

- **The KLENT seam is checked with a stub.** `_policy_q` is required to reach
  the model through ``policy_q`` and nothing else, which a real model cannot
  demonstrate: it would answer either way.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, fields, replace

import pytest
import torch

import hexo_py

from mantisnet.builder import collate_prefixes as mantis_collate_prefixes
from mantisnet.klent import train as klent_train
from mantisnet.klent.train import KlentConfig, network_evaluate
from mantisnet.model import MantisConfig, MantisNet
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import (
    ARCHITECTURE_ID,
    MANTIS_ACT_REPR_VERSION,
    PRESETS,
    UNHASHED_FIELDS,
    MantisACTConfig,
    architecture_hash,
)
from mantisnet.models.mantis_act.heads import CATEGORICAL_CRITIC_LOGITS, MASK_SUFFIX
from mantisnet.models.mantis_act.model import (
    ACT_CHECKPOINT_FORMAT,
    FORMAT_KEY,
    ACTOutput,
    MantisACT,
    config_from_record,
)
from mantisnet.models.mantis_act.packed import (
    PHASE_FIRST,
    PHASE_OPENING,
    PHASE_SECOND,
    collate,
)
from mantisnet.models.mantis_act.summary import parameter_summary

from .test_act_numerics import position

SEED = 20260806
FULL = PRESETS["full_act_v4"]
MASS_FLOOR = 0.2

# §31.10's pinned tolerance. A batch and a single position reduce the same
# segments in a different order, so fp32 associativity is the whole of the
# expected difference; measured worst case over the ply ladder below is
# 5.8e-6 on the critic logits and 2.6e-6 on both Q roles.
BATCH_ATOL = 1e-5

RUNNABLE_PRESETS = tuple(PRESETS)

# Both parities of the turn, so a batch holds FIRST and SECOND positions
# together (§13.1); ply 0 is the OPENING board and is added separately, since
# it is the one position with no window and no stone at all.
PLIES = (21, 22, 61, 62, 121, 160, 161)

# The per-preset loops pay a build and a backward per arm, so they use one
# position of each phase rather than the whole ladder.
SHORT_PLIES = (21, 22)


# --------------------------------------------------------------------------
# Fixtures


def act_batch(cfg: MantisACTConfig, plies=PLIES, games=(0, 1)):
    """A packed batch of real self-play positions at each of ``plies``."""
    return collate([build(position(game, ply), cfg) for game in games for ply in plies])


@pytest.fixture(scope="module")
def batch():
    return act_batch(FULL)


@pytest.fixture(scope="module")
def model() -> MantisACT:
    torch.manual_seed(SEED)
    return MantisACT(FULL).eval()


@pytest.fixture(scope="module")
def trained(batch) -> MantisACT:
    """A model whose zero-initialised output layers have been perturbed.

    Every §23 output layer starts at zero, so a fresh model emits one constant
    policy logit and exactly zero Q — which is the §32 contract, and which
    makes a fresh model useless for any test about *which* action got which
    number. This is the same model with those two projections drawn from
    `N(0, 1)`, so its outputs actually vary across actions.
    """
    torch.manual_seed(SEED + 1)
    module = MantisACT(FULL).eval()
    for layer in (module.heads.policy.out, module.heads.critic.out):
        torch.nn.init.normal_(layer.weight, std=1.0)
        torch.nn.init.normal_(layer.bias, std=1.0)
    return module


def sum_of(out: ACTOutput) -> torch.Tensor:
    """A scalar every output stream reaches, so one backward covers all."""
    terms = [
        out.policy_logits.float().square().mean(),
        out.critic_logits.float().square().mean(),
    ]
    for name, value in sorted(out.aux.items()):
        if name.endswith(MASK_SUFFIX):
            continue
        terms.append(value.float().square().mean())
    return sum(terms)


# --------------------------------------------------------------------------
# §25's output


def test_the_output_carries_exactly_the_fields_the_spec_names(model, batch):
    """§25: six fields, flat over legal actions, with the composition in fp32."""
    assert tuple(f.name for f in fields(ACTOutput)) == (
        "policy_logits",
        "critic_logits",
        "q_value",
        "q_score",
        "legal_offsets",
        "aux",
    )
    out = model(batch, MASS_FLOOR)
    n_legal = int(batch.legal_offsets[-1])
    assert tuple(out.policy_logits.shape) == (n_legal,)
    assert tuple(out.critic_logits.shape) == (n_legal, CATEGORICAL_CRITIC_LOGITS)
    assert tuple(out.q_value.shape) == (n_legal,)
    assert tuple(out.q_score.shape) == (n_legal,)
    # §27: the critic composition is fp32 whatever the logits' precision.
    assert out.q_value.dtype is torch.float32
    assert out.q_score.dtype is torch.float32
    assert out.legal_offsets is batch.legal_offsets
    assert isinstance(out.aux, dict)
    for name, value in out.aux.items():
        assert isinstance(value, torch.Tensor), name


def test_a_disabled_head_contributes_no_aux_key(model, batch):
    """§32: a module the configuration removes makes exactly no contribution."""
    out = model(batch, MASS_FLOOR)
    # The default holds no auxiliary and no state-value head (§6, §38), so the
    # only aux entry is §23.2's committed mass, which the categorical critic
    # decomposes on the way to the acting score.
    assert sorted(out.aux) == ["committed_mass"]

    torch.manual_seed(SEED)
    with_value = MantisACT(replace(FULL, enable_state_value_head=True)).eval()
    value_out = with_value(batch, MASS_FLOOR)
    assert sorted(value_out.aux) == ["committed_mass", "state_value"]
    assert tuple(value_out.aux["state_value"].shape) == (batch.position_count,)
    assert value_out.aux["state_value"].dtype is torch.float32


def test_the_auxiliary_heads_appear_with_their_masks_when_enabled(batch):
    """§24: an enabled head returns logits and the mask of labelled rows."""
    cfg = replace(
        FULL, enable_action_aux_heads=True, use_action_tactical_features=False
    )
    torch.manual_seed(SEED)
    module = MantisACT(
        cfg, aux_weights={"win_now": 1.0, "winning_partner_exists": 0.5}
    ).eval()
    out = module(act_batch(cfg, plies=SHORT_PLIES, games=(0,)), MASS_FLOOR)
    assert sorted(out.aux) == [
        "committed_mass",
        "win_now",
        "win_now" + MASK_SUFFIX,
        "winning_partner_exists",
        "winning_partner_exists" + MASK_SUFFIX,
    ]
    # §24.1 labels the partner auxiliaries on first-placement states only.
    assert out.aux["win_now" + MASK_SUFFIX].all()
    assert not out.aux["winning_partner_exists" + MASK_SUFFIX].all()


def test_zero_initialised_outputs_give_a_uniform_policy_and_exactly_zero_q(
    model, batch
):
    """§23.1, §23.2, §27, §32: a fresh model ranks nothing and commits nothing."""
    out = model(batch, MASS_FLOOR)
    assert torch.equal(out.policy_logits, torch.zeros_like(out.policy_logits))
    assert torch.equal(out.q_value, torch.zeros_like(out.q_value))
    assert torch.equal(out.q_score, torch.zeros_like(out.q_score))
    # Three equal logits, so the committed mass is exactly two thirds.
    assert torch.allclose(
        out.aux["committed_mass"],
        torch.full_like(out.aux["committed_mass"], 2.0 / 3.0),
    )


# --------------------------------------------------------------------------
# Engine legal order (§8.3, §32, §37.5)


def test_every_legal_move_maps_to_its_own_cell_node_in_engine_order():
    """§32: the output order is asserted from `legal_to_cell_index` itself."""
    for game in (0, 1):
        for ply in PLIES:
            pos = position(game, ply)
            graph = build(pos, FULL)
            legal = pos.legal_moves()
            assert graph.n_legal == len(legal)
            for j, move in enumerate(legal):
                cell = int(graph.legal_to_cell_index[j])
                assert tuple(graph.cell_qr[cell]) == tuple(move), (game, ply, j)


def test_permuting_the_action_rows_permutes_the_output_the_same_way(trained):
    """Row ``j`` of every output is action ``j`` and nothing else (§37.5).

    The builder's mapping is one claim; that the whole model *carries* it is
    another, and a stage that sorted or regrouped actions would satisfy the
    first while failing this.
    """
    single = collate([build(position(0, 61), FULL)])
    n_legal = int(single.legal_offsets[-1])
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(n_legal, generator=generator)

    shuffled = replace(
        single,
        legal_to_cell_index=single.legal_to_cell_index[order],
        action_window_index=single.action_window_index[order],
        action_post1_class=single.action_post1_class[order],
        action_pre_status=single.action_pre_status[order],
        action_tactical_numeric=single.action_tactical_numeric[order],
    )
    with torch.no_grad():
        base = trained(single, MASS_FLOOR)
        moved = trained(shuffled, MASS_FLOOR)

    assert torch.allclose(moved.policy_logits, base.policy_logits[order], atol=1e-5)
    assert torch.allclose(moved.critic_logits, base.critic_logits[order], atol=1e-5)
    assert torch.allclose(moved.q_value, base.q_value[order], atol=1e-5)
    # A permutation that changed nothing would pass vacuously.
    assert not torch.allclose(moved.policy_logits, base.policy_logits, atol=1e-3)


# --------------------------------------------------------------------------
# Real positions, every ply and every phase


def test_the_model_runs_at_every_ply_and_every_phase(model):
    """All three §13.1 phases, including the opening board with no window."""
    seen = set()
    opening = collate([build(hexo_py.Position(), FULL)])
    assert int(opening.phase_id[0]) == PHASE_OPENING
    for one in [opening] + [
        collate([build(position(game, ply), FULL)])
        for game in (0, 1)
        for ply in PLIES
    ]:
        seen.add(int(one.phase_id[0]))
        with torch.no_grad():
            out = model(one, MASS_FLOOR)
        assert torch.isfinite(out.policy_logits).all()
        assert torch.isfinite(out.critic_logits).all()
        assert torch.isfinite(out.q_value).all()
        assert torch.isfinite(out.q_score).all()
        assert out.policy_logits.shape[0] == int(one.legal_offsets[-1]) >= 1
    assert seen == {PHASE_OPENING, PHASE_FIRST, PHASE_SECOND}


def test_a_batch_holds_both_placement_phases(batch):
    """The ply ladder is both parities, so a batch is not one phase throughout."""
    assert set(batch.phase_id.tolist()) == {PHASE_FIRST, PHASE_SECOND}


def test_batched_and_single_position_forwards_agree(trained):
    """§31.10: the only place a segment boundary can be wrong is a batch."""
    graphs = [build(position(game, ply), FULL) for game in (0, 1) for ply in PLIES]
    packed = collate(graphs)
    with torch.no_grad():
        together = trained(packed, MASS_FLOOR)
        apart = [trained(collate([graph]), MASS_FLOOR) for graph in graphs]

    offsets = packed.legal_offsets.tolist()
    for index, one in enumerate(apart):
        lo, hi = offsets[index], offsets[index + 1]
        assert torch.allclose(
            together.policy_logits[lo:hi], one.policy_logits, atol=BATCH_ATOL
        )
        assert torch.allclose(
            together.critic_logits[lo:hi], one.critic_logits, atol=BATCH_ATOL
        )
        assert torch.allclose(together.q_value[lo:hi], one.q_value, atol=BATCH_ATOL)
        # The acting score's divisor is the position's own maximum committed
        # mass, so a batched score that leaked across positions shows here.
        assert torch.allclose(together.q_score[lo:hi], one.q_score, atol=BATCH_ATOL)


# --------------------------------------------------------------------------
# Every preset (§29, §37.1)


@pytest.mark.parametrize("name", RUNNABLE_PRESETS)
def test_every_preset_constructs_and_runs_forward_and_backward(name):
    """§37.1, on real positions of both phases, with a real backward."""
    cfg = PRESETS[name]
    torch.manual_seed(SEED)
    module = MantisACT(cfg)
    packed = act_batch(cfg, plies=SHORT_PLIES, games=(0,))
    out = module(packed, MASS_FLOOR)

    n_legal = int(packed.legal_offsets[-1])
    assert tuple(out.policy_logits.shape) == (n_legal,)
    assert torch.isfinite(out.policy_logits).all()
    assert torch.isfinite(out.q_value).all()

    sum_of(out).backward()
    for parameter_name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"{name}: {parameter_name} got no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name}: {parameter_name}"


# --------------------------------------------------------------------------
# bf16 autocast (§27, §32)


def test_bf16_autocast_forward_and_backward_are_finite_on_real_positions(batch):
    """§32's smoke run at the density the model will actually be trained at."""
    torch.manual_seed(SEED)
    module = MantisACT(FULL)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(batch, MASS_FLOOR)

    for name, stream in (
        ("policy_logits", out.policy_logits),
        ("critic_logits", out.critic_logits),
        ("q_value", out.q_value),
        ("q_score", out.q_score),
    ):
        assert torch.isfinite(stream).all(), name
    # §27: the composition is fp32 and outside autocast, so both Q roles are
    # fp32 whatever precision autocast chose for the head projections.
    assert out.q_value.dtype is torch.float32
    assert out.q_score.dtype is torch.float32

    loss = sum_of(out)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in module.named_parameters():
        assert parameter.dtype is torch.float32, f"{name} is not stored in fp32"
        assert parameter.grad is not None, f"{name} received no gradient"
        assert parameter.grad.dtype is torch.float32, f"{name}'s gradient is not fp32"
        assert torch.isfinite(parameter.grad).all(), name

    optimiser = torch.optim.AdamW(module.parameters(), lr=1e-3)
    optimiser.step()
    for name, parameter in module.named_parameters():
        assert torch.isfinite(parameter).all(), f"{name} is not finite after a step"


# --------------------------------------------------------------------------
# The debug forward (§31)


def test_the_debug_forward_is_the_forward_plus_intermediates(trained, batch):
    """§31: selected intermediates, and numerically the same answer."""
    sites = trained.debug_sites()
    assert "action.state" in sites and "action.latent" in sites
    assert set(trained.trunk.debug_sites()) <= set(sites)

    with torch.no_grad():
        plain = trained(batch, MASS_FLOOR)
        traced, tensors = trained.debug_forward(batch, MASS_FLOOR)
    assert torch.equal(plain.policy_logits, traced.policy_logits)
    assert torch.equal(plain.q_value, traced.q_value)

    n_legal = int(batch.legal_offsets[-1])
    assert tuple(tensors["action.state.inv"].shape) == (n_legal, FULL.d_inv)
    assert tuple(tensors["action.state.axis"].shape) == (n_legal, 3, FULL.d_axis)
    assert tuple(tensors["action.latent.inv"].shape) == (
        batch.position_count,
        FULL.num_action_latents,
        FULL.d_inv,
    )

    with torch.no_grad():
        _out, narrow = trained.debug_forward(batch, MASS_FLOOR, ["action.state"])
    assert sorted(narrow) == ["action.state.axis", "action.state.inv"]
    with pytest.raises(ValueError, match="unknown debug site"):
        trained.debug_forward(batch, MASS_FLOOR, ["final.cell", "not.a.site"])


# --------------------------------------------------------------------------
# Batch validation the stages cannot make


def test_a_batch_whose_families_disagree_about_the_position_count_is_refused(
    model, batch
):
    """Every offset table still validates alone; only their agreement fails."""
    with pytest.raises(TypeError, match="PackedACTBatch"):
        model({"legal_offsets": batch.legal_offsets}, MASS_FLOOR)
    broken = replace(batch, position_count=batch.position_count + 1)
    with pytest.raises(ValueError, match="cell_offsets must be"):
        model(broken, MASS_FLOOR)
    broken = replace(batch, phase_id=batch.phase_id[:-1])
    with pytest.raises(ValueError, match="phase_id must be"):
        model(broken, MASS_FLOOR)
    broken = replace(batch, global_numeric=batch.global_numeric[:-1])
    with pytest.raises(ValueError, match="global_numeric must be"):
        model(broken, MASS_FLOOR)


# --------------------------------------------------------------------------
# §28 checkpointing


def test_the_checkpoint_records_the_identity_the_spec_names(model):
    """§28: architecture id, representation version, resolved config, hash."""
    identity = model.checkpoint_identity()
    assert identity[FORMAT_KEY] == ACT_CHECKPOINT_FORMAT
    assert identity["architecture_id"] == ARCHITECTURE_ID
    assert identity["repr_version"] == MANTIS_ACT_REPR_VERSION
    assert identity["architecture_hash"] == architecture_hash(FULL)
    assert identity["model_config"] == asdict(FULL)
    assert identity["head_weights"] == {
        "action_aux": {},
        "window_fate": 0.0,
        "instantiate_zero_weight": False,
    }
    payload = model.checkpoint()
    assert set(payload) == set(identity) | {"model"}
    assert set(payload["model"]) == set(model.state_dict())


def test_a_checkpoint_round_trips_to_identical_outputs(trained, batch):
    """§28: reconstruct from the record alone, then load strictly."""
    payload = trained.checkpoint()
    clone = MantisACT.from_checkpoint(payload)
    clone.eval()
    assert clone.cfg == trained.cfg
    with torch.no_grad():
        before = trained(batch, MASS_FLOOR)
        after = clone(batch, MASS_FLOOR)
    assert torch.equal(before.policy_logits, after.policy_logits)
    assert torch.equal(before.critic_logits, after.critic_logits)

    # And into an independently initialised model of the same architecture.
    torch.manual_seed(SEED + 99)
    fresh = MantisACT(FULL).eval()
    with torch.no_grad():
        assert not torch.equal(fresh(batch, MASS_FLOOR).critic_logits, after.critic_logits)
    fresh.load_checkpoint(payload)
    with torch.no_grad():
        assert torch.equal(fresh(batch, MASS_FLOOR).critic_logits, after.critic_logits)


def test_a_model_with_auxiliary_heads_round_trips(batch):
    """The §24 weights decide the parameter set, so the record carries them."""
    cfg = replace(
        FULL, enable_action_aux_heads=True, use_action_tactical_features=False
    )
    torch.manual_seed(SEED)
    module = MantisACT(cfg, aux_weights={"win_now": 0.25})
    payload = module.checkpoint()
    assert payload["head_weights"]["action_aux"] == {"win_now": 0.25}
    clone = MantisACT.from_checkpoint(payload)
    assert clone.aux_weights == {"win_now": 0.25}
    assert set(clone.state_dict()) == set(module.state_dict())

    other = MantisACT(cfg, aux_weights={"win_now": 0.5})
    with pytest.raises(ValueError, match="head weights"):
        other.load_checkpoint(payload)


@pytest.mark.parametrize(
    "field,delta",
    [
        ("d_inv", {"d_inv": 32}),
        ("state_blocks", {"state_blocks": 2}),
        ("window_scope", {"window_scope": "live"}),
        ("cell_scope", {"cell_scope": "occupied_only"}),
        ("critic_type", {"critic_type": "scalar_tanh"}),
        # The two fields the config ties together must move together, or the
        # arm is refused at construction instead of at load.
        (
            "num_action_latents",
            {"num_action_latents": 0, "use_action_set_latents": False},
        ),
        ("occupied_radius", {"occupied_radius": 6}),
        ("incidence_message", {"incidence_message": "additive"}),
    ],
)
def test_a_semantic_config_difference_is_refused_by_name(model, field, delta):
    """§28: strict load requires exact agreement on every semantic field."""
    payload = model.checkpoint()
    torch.manual_seed(SEED)
    into = MantisACT(replace(FULL, **delta))
    with pytest.raises(ValueError, match=field):
        into.load_checkpoint(payload)


def test_every_semantic_field_is_compared_and_dropout_is_not(model):
    """The comparison is the hash's complement, field for field.

    A field added to the dataclass and forgotten here would be compared by
    nothing, so the set is derived rather than listed.
    """
    payload = model.checkpoint()
    semantic = [f.name for f in fields(MantisACTConfig) if f.name not in UNHASHED_FIELDS]
    assert "dropout" not in semantic and len(semantic) == len(
        fields(MantisACTConfig)
    ) - len(UNHASHED_FIELDS)

    # Dropout changes no shape and no output meaning, and the architecture hash
    # already says so; a resume that only changed regularisation must load.
    torch.manual_seed(SEED)
    dropped = MantisACT(replace(FULL, dropout=0.1))
    dropped.load_checkpoint(payload)

    for name in semantic:
        recorded = dict(payload["model_config"])
        recorded[name] = "a value no vocabulary holds"
        broken = {**payload, "model_config": recorded}
        with pytest.raises((ValueError, TypeError)):
            model.load_checkpoint(broken)


def test_a_foreign_or_malformed_payload_is_refused_loudly(model):
    """§28: the refusals name what disagreed, before any tensor is read."""
    payload = model.checkpoint()

    with pytest.raises(TypeError, match="must be a mapping"):
        model.load_checkpoint(["not", "a", "payload"])
    with pytest.raises(ValueError, match="checkpoint format"):
        model.load_checkpoint({**payload, FORMAT_KEY: ACT_CHECKPOINT_FORMAT + 1})
    with pytest.raises(ValueError, match="architecture_id"):
        model.load_checkpoint({**payload, "architecture_id": "mantis_act_v5"})
    with pytest.raises(ValueError, match="representation version"):
        model.load_checkpoint({**payload, "repr_version": MANTIS_ACT_REPR_VERSION + 1})
    with pytest.raises(ValueError, match="architecture_hash"):
        model.load_checkpoint({**payload, "architecture_hash": "0" * 16})

    missing = dict(payload["model_config"])
    missing.pop("d_inv")
    with pytest.raises(ValueError, match="missing \\['d_inv'\\]"):
        model.load_checkpoint({**payload, "model_config": missing})
    extra = {**payload["model_config"], "pair_scope": "post_action_collinear"}
    with pytest.raises(ValueError, match="unexpected \\['pair_scope'\\]"):
        model.load_checkpoint({**payload, "model_config": extra})

    with pytest.raises(TypeError, match="state dict"):
        model.load_checkpoint({**payload, "model": None})
    truncated = dict(payload["model"])
    truncated.pop(next(iter(truncated)))
    with pytest.raises(RuntimeError, match="[Mm]issing key"):
        model.load_checkpoint({**payload, "model": truncated})


def test_a_mantisnet_checkpoint_is_never_loaded_into_this_model(model):
    """§28, §37.13: no conversion path exists, and the refusal says so."""
    legacy_cfg = MantisConfig(h=16, heads=2, blocks=1, value_bins=5)
    torch.manual_seed(SEED)
    legacy = {
        "model": MantisNet(legacy_cfg).state_dict(),
        "model_config": asdict(legacy_cfg),
        "iteration": 3,
    }
    with pytest.raises(ValueError, match="MantisNet checkpoints"):
        model.load_checkpoint(legacy)
    with pytest.raises(ValueError, match="MantisNet checkpoints"):
        MantisACT.from_checkpoint(legacy)


def test_a_recorded_config_is_revalidated_rather_than_trusted():
    """A record naming a model this build cannot express is refused (§28)."""
    recorded = asdict(FULL)
    recorded["window_scope"] = "everything"
    with pytest.raises(ValueError, match="window_scope"):
        config_from_record(recorded)
    with pytest.raises(TypeError, match="must be a mapping"):
        config_from_record(["d_inv", 64])


# --------------------------------------------------------------------------
# The KLENT seam (§2, §25, §37.10)


def test_klent_reaches_both_architectures_through_policy_q_alone():
    """`_policy_q` names no model's internals: a stub with the seam suffices."""
    sentinel = (object(), object())

    class OnlySeam:
        def __init__(self):
            self.seen = None

        def policy_q(self, batch):
            self.seen = batch
            return sentinel

    stub = OnlySeam()
    marker = object()
    assert klent_train._policy_q(stub, marker) is sentinel
    assert stub.seen is marker


def test_mantisnets_policy_q_is_its_existing_pass():
    """The one permitted edit to `mantisnet/model.py` is additive (§32, §37.13)."""
    torch.manual_seed(SEED)
    legacy = MantisNet(MantisConfig(h=16, heads=2, blocks=1, value_bins=5)).eval()
    legacy_batch = mantis_collate_prefixes(
        [[(0, 0), (1, 0), (0, 1)], [(0, 0), (2, 0), (0, 2)]], [3, 3]
    )
    with torch.no_grad():
        _s, w, g = legacy.trunk(legacy_batch)
        expected = legacy.cell_head_logits(w, g, legacy_batch)
        got = legacy.policy_q(legacy_batch)
    assert torch.equal(got[0], expected[0])
    assert torch.equal(got[1], expected[1])


def test_the_act_model_answers_the_same_seam(trained, batch):
    """`policy_q` is the uncomposed pair `forward` composes (§25)."""
    with torch.no_grad():
        policy, critic = trained.policy_q(batch)
        out = trained(batch, MASS_FLOOR)
    assert torch.equal(policy, out.policy_logits)
    assert torch.equal(critic, out.critic_logits)


def test_network_evaluate_is_unchanged_and_serves_the_act_model(trained, batch):
    """§2, §37.10: the external evaluator interface is untouched."""
    assert list(inspect.signature(network_evaluate).parameters) == ["model", "cfg"]
    cfg = KlentConfig(device="cpu", mass_floor=MASS_FLOOR)
    evaluate = network_evaluate(trained, cfg)
    policy, q_score, q_value = evaluate(batch)

    n_legal = int(batch.legal_offsets[-1])
    assert tuple(policy.shape) == (n_legal,)
    assert tuple(q_score.shape) == (n_legal,)
    assert tuple(q_value.shape) == (n_legal,)
    with torch.no_grad():
        out = trained(batch, cfg.mass_floor)
    # The evaluator composes with MantisNet's own operator; §23.2 forbids the
    # architecture altering it, so the two must agree exactly.
    assert torch.allclose(policy, out.policy_logits, atol=0.0)
    assert torch.allclose(q_value, out.q_value, atol=0.0)
    assert torch.allclose(q_score, out.q_score, atol=0.0)


# --------------------------------------------------------------------------
# Parameters (§6, §32, §34)


def test_the_summary_partitions_the_parameters_and_names_the_stages(model):
    """§32: the summary's total equals `sum(p.numel())`, exactly."""
    summary = parameter_summary(model)
    assert summary.total == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    assert sum(count for _name, count in summary.groups) == summary.total

    stages = parameter_summary(model, depth=1)
    by_stage = dict(stages.groups)
    assert set(by_stage) == {"trunk", "actions", "heads"}
    assert sum(by_stage.values()) == summary.total
    # The state trunk is Stage C's, unchanged by this stage.
    assert by_stage["trunk"] == parameter_summary(model.trunk).total
    assert by_stage["actions"] == parameter_summary(model.actions).total
    assert by_stage["heads"] == parameter_summary(model.heads).total


def test_the_full_model_is_below_the_specs_parameter_target(model):
    """§6 targets 2.5-4M; the model built at §6's own widths is under it.

    Left as a measurement rather than padded to the range: §6 fixes the widths
    and the depths, and adding parameters that nothing in the spec asks for to
    reach a round number would be tuning the model to the sentence rather than
    to the game. The number is asserted so the shortfall stays visible and a
    later width change has to come here and say so.
    """
    total = sum(p.numel() for p in model.parameters())
    assert total == 1_726_468
    assert total < 2_500_000
