"""§37's thirteen acceptance criteria, and §33's structural alias diagnostic.

This is the file that says whether MantisNet-ACT v4 is finished. Each §37
criterion gets one test named for it, so a failure names the criterion rather
than a module, and every criterion is re-checked here against the assembled
model even where a stage-level suite already covers a part of it: §37 is a
statement about the whole model, and a criterion answered only by the module
that implements it is answered by its own author.

Criterion 9 is **void**: §20's same-turn partner modeling was removed by owner
ruling, so there are no prospective partners to include newly legal cells in.
`test_37_9_...` asserts the absence itself rather than skipping, because a
criterion that is void because a subsystem is gone is only void while it stays
gone.

## §33, and why the old representation has to be measured too

§33's claim is not "ACT has few aliases". It is that **the full model should
not retain a systematic background alias caused solely by nearest-stone
distance**, and that is a claim about a specific failure the old architecture
has. MantisNet describes a legal cell by the live windows through it and, when
there are none, by its nearest-stone distance bucket and nothing else — so
every cell of the outer halo collapses onto one of eight buckets, and its
decoder row is then literally `e_bg[bucket]`, giving bitwise-equal policy and
critic logits for every cell in the bucket. Measuring ACT alone could not tell
whether it fixed that, so both representations are hashed here, on the same
real positions, with the same signature machinery.

The old representation's signature is built in this file rather than in
`mantis_act/diagnostics.py`: `docs/MANTIS_ACT_DEVIATIONS.md` records that
nothing in the ACT package imports from MantisNet, and the decisive check needs
the old *model* as well as its builder. A test module is already the place that
holds both.

## What the evidence in this file is, and what it is not

- An **equal signature is not proof of an alias** in ACT. An action's state is
  a product of four rounds of message passing over the whole position graph, so
  two actions with equal local bundles can still be separated by what lies
  further out. Every alias group this file finds in ACT is therefore also run
  through the model, and the measured output spread is asserted, not assumed.
- An equal signature **is** proof of an alias on the old background path, which
  reads one embedding row and nothing else. That is why the comparison is
  decisive rather than merely suggestive.
- The diagnostic is shown able to fail: a deliberately coarsened signature —
  ACT's, with the geometry line deleted, which is exactly what the old
  representation does to the halo — must collapse into the same kind of large
  background group.

Positions are the real stack-939 self-play games `test_act_numerics.py`
embeds, plus the contact-play generator of `test_act_d6.py` where a whole D6
orbit of one board is needed. Random playouts are not used for anything here:
`docs/MANTIS_ACT_DEVIATIONS.md` records that they carry five times the legal
cells and a fifteenth of the mixed windows of a real board, and every figure in
this file is a count over those two families.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

import hexo_py

from mantisnet.builder import build as legacy_build
from mantisnet.builder import collate as legacy_collate
from mantisnet.klent import train as klent_train
from mantisnet.klent.train import KlentConfig, network_evaluate
from mantisnet.lab.variants import VARIANTS, build_variant
from mantisnet.model import MantisConfig, MantisNet
from mantisnet.models.mantis_act import diagnostics
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import (
    ARCHITECTURE_ID,
    MANTIS_ACT_REPR_VERSION,
    PRESETS,
    MantisACTConfig,
)
from mantisnet.models.mantis_act.diagnostics import (
    ABSENT_LINE,
    KIND_IN_WINDOW,
    KIND_NO_WINDOW,
    SIGNATURE_LINES,
    ActionSignatures,
    act_signatures,
    alias_report,
    graph_geometry,
    profile,
)
from mantisnet.models.mantis_act.equivariant import permute_axis_channels
from mantisnet.models.mantis_act.heads import CATEGORICAL_CRITIC_LOGITS
from mantisnet.models.mantis_act.model import MantisACT
from mantisnet.models.mantis_act.packed import (
    NUM_AXES,
    POST_ACTION_ROWS,
    WINDOW_LEN,
    collate,
    telemetry,
)
from mantisnet.models.mantis_act.pattern_classes import (
    ALL_CELL_WINDOW_REL_CLASSES,
    ALL_WINDOW_PATTERN_CLASSES,
    MIXED,
    NONEMPTY_CELL_WINDOW_REL_CLASSES,
    NONEMPTY_WINDOW_PATTERN_CLASSES,
    POST1_REL_CLASSES,
)
from mantisnet.models.mantis_act.summary import parameter_summary
from mantisnet.models.mantis_act.symmetry import (
    D6_ORBITS_DMAX12,
    axis_permutation,
    orbit_table,
)
from mantisnet.models.mantis_act.windows import window_cells

from .test_act_d6 import (
    ATOL,
    RTOL,
    TRANSFORMS,
    contact_game,
    make_sample,
    randomise_,
)
from .test_act_numerics import position

SEED = 20260806
FULL = PRESETS["full_act_v4"]
MASS_FLOOR = 0.2

RUNNABLE_PRESETS = tuple(PRESETS)

# Real self-play plies for the alias sweep and the per-criterion checks: both
# parities of the turn, and depths where the halo, the mixed windows, and the
# radius family are all at their trained-play density.
ALIAS_PLIES = (21, 61, 121, 161)

# One D6 orbit of a contact board is twelve builds and twelve forwards, so the
# equivariance criteria run at one depth rather than the ladder.
ORBIT_PLIES = 61


# --------------------------------------------------------------------------
# The two representations, hashed the same way


def content_digest(values) -> np.ndarray:
    """One 64-bit digest per action of arbitrary Python line content.

    `diagnostics` hashes its own lines straight out of the builder's integer
    tables, which is what makes the ACT pass vectorised; the old
    representation's lines are Python tuples assembled here, so they are hashed
    here too. ``blake2b`` rather than ``hash()``: the digests are compared
    across processes and Python's string hash is seeded per process.
    """
    return np.array(
        [
            int.from_bytes(
                hashlib.blake2b(repr(value).encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            for value in values
        ],
        dtype=np.uint64,
    )


def legacy_signatures(graph) -> ActionSignatures:
    """The old MantisNet representation's per-legal-action signature.

    MantisNet's policy decoder reads exactly two things about a legal cell: the
    live windows through it, each contributing its window feature and the joint
    ``(occupancy, slot)`` class of the cell's own slot, and — when the cell lies
    in no live window at all — one nearest-stone bucket embedding. So this is
    the complete builder-side description, in the same four-line shape
    `diagnostics.combine` folds ACT's in. The lines the old representation
    simply does not have are `ABSENT_LINE`, which is what they are.

    The background line is *exact*: two cells sharing it produce a bitwise
    equal decoder row, hence bitwise equal policy and critic logits.
    `test_the_old_representation_aliases_the_halo_by_nearest_distance` asserts
    that rather than resting on the reading.
    """
    by_cell: dict[int, list[tuple[int, int]]] = {}
    for cell, window, cls in zip(graph.dec_cell, graph.dec_window, graph.dec_class):
        by_cell.setdefault(int(cell), []).append(
            (int(graph.window_feat[int(window)]), int(cls))
        )
    bucket = {int(cell): int(b) for cell, b in zip(graph.bg_cell, graph.bg_bucket)}

    cells, windows, kinds = [], [], []
    for action in range(graph.n_legal):
        described = by_cell.get(action)
        cells.append(() if described is not None else (bucket[action],))
        windows.append(tuple(sorted(described)) if described is not None else ())
        kinds.append(KIND_IN_WINDOW if described is not None else KIND_NO_WINDOW)
    absent = np.full(graph.n_legal, ABSENT_LINE, dtype=np.uint64)
    return ActionSignatures(
        label="mantisnet (live windows, else nearest-stone bucket)",
        kind=tuple(kinds),
        lines={
            "cell": content_digest(cells),
            "windows": content_digest(windows),
            "post1": absent,
            "orbit": absent,
        },
    )


def coarsened_signatures(signatures: ActionSignatures) -> ActionSignatures:
    """ACT's signature with the geometry line deleted — the negative control.

    Removing the ``orbit`` line is exactly what the old representation does to
    a halo cell: everything is gone but the nearest-stone bucket the ``cell``
    line carries. A deleted line is a constant column, which is what a
    representation that does not read it produces. If the diagnostic cannot see
    the collapse this produces, it could not have seen a real one either.
    """
    return ActionSignatures(
        label=f"{signatures.label} without the geometry line",
        kind=signatures.kind,
        lines={
            **signatures.lines,
            "orbit": np.full(len(signatures), ABSENT_LINE, dtype=np.uint64),
        },
    )


class Case:
    """One real position under both representations, with both models run."""

    def __init__(self, game: int, ply: int, act_model, old_model) -> None:
        self.game, self.ply = game, ply
        self.position = position(game, ply)
        stones = np.asarray(self.position.stones(), dtype=np.int64).reshape(-1, 3)
        legal = np.asarray(self.position.legal_moves(), dtype=np.int64).reshape(-1, 2)

        self.act_graph = build(self.position, FULL)
        self.legacy_graph = legacy_build(
            stones[:, :2],
            stones[:, 2],
            self.position.current_player,
            legal,
            self.position.moves_remaining,
        )
        self.geometry = graph_geometry(self.act_graph, d_max=FULL.d_max)
        self.act_sig = act_signatures(self.act_graph)
        self.legacy_sig = legacy_signatures(self.legacy_graph)
        self.act = alias_report(self.act_sig, self.describe, samples=3)
        self.legacy = alias_report(self.legacy_sig, self.describe, samples=3)
        with torch.no_grad():
            self.act_policy, self.act_critic = act_model.policy_q(
                collate([self.act_graph])
            )
            self.old_policy, self.old_critic = old_model.policy_q(
                legacy_collate([self.legacy_graph])
            )

    @property
    def name(self) -> str:
        return f"game {self.game} ply {self.ply}"

    def describe(self, rows):
        """The geometry of a sampled group, off the descriptor built above."""
        return [self.geometry[row] for row in rows]

    def halo(self, signatures: ActionSignatures) -> list[int]:
        """Every legal action that representation describes without a window."""
        return [
            action
            for action, kind in enumerate(signatures.kind)
            if kind == KIND_NO_WINDOW
        ]


@pytest.fixture(scope="module")
def act_model() -> MantisACT:
    """The full model, off its initialisation.

    §23 zero-initialises both output layers, so a fresh model gives every legal
    action the same logit and could not tell an alias from a separation.
    """
    torch.manual_seed(SEED)
    return randomise_(MantisACT(FULL), SEED)


@pytest.fixture(scope="module")
def old_model() -> MantisNet:
    torch.manual_seed(SEED)
    return randomise_(MantisNet(MantisConfig()), SEED)


@pytest.fixture(scope="module")
def cases(act_model, old_model) -> tuple[Case, ...]:
    return tuple(
        Case(game, ply, act_model, old_model)
        for game in (0, 1)
        for ply in ALIAS_PLIES
    )


@pytest.fixture(scope="module")
def orbit():
    """One contact board under all twelve transforms, as one batch."""
    return make_sample("contact", contact_game(SEED, ORBIT_PLIES))


@pytest.fixture(scope="module")
def orbit_out(orbit, act_model):
    with torch.no_grad():
        return act_model.debug_forward(orbit.batch, MASS_FLOOR)


def _slice(tensor, offsets, position_index: int):
    return tensor[int(offsets[position_index]) : int(offsets[position_index + 1])]


def budget(reference) -> float:
    """§31's pinned slack, in the ulp-aware form this suite already uses."""
    if reference.numel() == 0:
        return ATOL
    return ATOL + RTOL * float(reference.detach().abs().max())


def deviation(got, want) -> float:
    if got.shape != want.shape:
        raise AssertionError(f"shape {tuple(got.shape)} against {tuple(want.shape)}")
    if got.numel() == 0:
        return 0.0
    return float((got.detach().double() - want.detach().double()).abs().max())


# The reassociation floor of a GEMM over rows that are equal by construction:
# identical inputs, but the reduction may be split between tiles differently.
# Measured worst case over the halo comparisons below is 1.1e-8.
ULP_FLOOR = 1e-7


def spread(values) -> float:
    """The widest disagreement inside a set of outputs, over all columns."""
    flat = values.detach().double().reshape(len(values), -1)
    return float((flat.max(dim=0).values - flat.min(dim=0).values).max())


def distinct(values, tolerance: float = 1e-6) -> int:
    """How many genuinely different numbers a vector holds."""
    ordered = torch.sort(values.detach().double().flatten()).values
    if not ordered.numel():
        return 0
    return 1 + int(((ordered[1:] - ordered[:-1]) > tolerance).sum())


# --------------------------------------------------------------------------
# §33: the structural alias diagnostic


def test_the_diagnostic_reports_every_figure_section_33_asks_for(cases):
    """Legal actions, unique signatures, groups, the largest group, samples."""
    for case in cases:
        report = case.act
        assert report.legal_actions == case.act_graph.n_legal
        assert 1 <= report.unique_signatures <= report.legal_actions
        # The five figures have to agree with each other: the groups partition
        # the actions that are not unique.
        assert report.unique_signatures == (
            report.legal_actions - report.aliased_actions + report.alias_groups
        ), case.name
        assert report.max_group_size == max(
            (len(group) for group in report.groups), default=0
        )
        fields = set(case.geometry[0]) - {"coordinate"}
        for sample in report.samples:
            assert len(sample.coordinates) == len(sample.rows) >= 2
            assert len(set(sample.coordinates)) == len(sample.coordinates)
            # §33 asks for the differing omitted geometry, so the split has to
            # be a partition of the descriptor: an entry that appeared in
            # neither list would be geometry the report silently dropped.
            assert {field for field, _ in sample.differing}.isdisjoint(sample.identical)
            assert {field for field, _ in sample.differing} | set(sample.identical) == fields
            assert sample.text()


def test_the_signature_is_a_function_of_the_position_not_of_its_frame(orbit):
    """A signature that moved under the group would count aliases wrongly.

    The model is exactly equivariant, so two actions related by a symmetry of
    the board get equal outputs by construction and the signature must call
    them equal. Here the *board* is transformed, so the whole signature —
    incidence classes, orbit ids, the eighteen counterfactual rows — is
    rediscovered by the builder from a position the engine produced.
    """
    base = act_signatures(orbit.graphs[0])
    for t in TRANSFORMS:
        image = act_signatures(orbit.graphs[t])
        moved = image.value[orbit.legal[t]]
        wrong = int((moved != base.value).sum())
        assert not wrong, (
            f"transform {t}: {wrong} of {len(base)} legal actions change "
            "signature under a symmetry of the board"
        )


def test_the_old_representation_aliases_the_halo_by_nearest_distance(cases):
    """The failure §33 exists for, measured on real positions.

    Two claims, and the second is what makes a group an alias rather than a
    coincidence of the hash: about half of every real position's legal actions
    lie in no live window, and the old model's outputs over a whole bucket
    agree to the last bit of fp32 — 1.1e-8 at worst here, against a 0.15-0.21
    spread across the buckets themselves. Not literally bitwise, because a
    cuBLAS GEMM over identical rows may still split its reduction differently
    between tiles; the reassociation floor is what `ULP_FLOOR` is.
    """
    for case in cases:
        rows = case.halo(case.legacy_sig)
        assert len(rows) > 0.2 * case.legacy.legal_actions, (
            f"{case.name}: only {len(rows)} of {case.legacy.legal_actions} legal "
            "actions lie in no live window, so the halo is not the bulk case "
            "this criterion is about"
        )
        for group in case.legacy.groups_of_kind(KIND_NO_WINDOW):
            policy = case.old_policy[list(group.rows)]
            critic = case.old_critic[list(group.rows)]
            assert spread(policy) <= ULP_FLOOR, (
                f"{case.name}: a {len(group)}-action bucket spreads "
                f"{spread(policy):.3e} in the policy logit"
            )
            assert spread(critic) <= ULP_FLOOR
        # The whole halo carries one number per nearest-stone bucket, and the
        # old builder has eight of those.
        assert distinct(case.old_policy[rows]) <= 8, (
            f"{case.name}: {distinct(case.old_policy[rows])} distinct policy "
            f"logits over {len(rows)} halo actions"
        )


def test_the_full_model_does_not_retain_that_background_alias(cases):
    """§33's claim, on the same actions the old model cannot tell apart.

    Every legal action the old representation routes through its background
    path is required to receive its own ACT signature *and* its own ACT policy
    logit. The second requirement is the load-bearing one: a signature is a
    statement about the builder, and the criterion is about the model.

    The separation is not marginal. Over the eight positions here the halo is
    295-512 actions wide, the old model gives it 6-8 distinct policy logits,
    and ACT gives it 295-512 — spread across 0.32-0.51 where the old model's
    within-bucket spread is 1e-8.
    """
    for case in cases:
        halo = case.halo(case.legacy_sig)
        signatures = case.act_sig.value[halo]
        assert len(np.unique(signatures)) == len(halo), (
            f"{case.name}: {len(halo) - len(np.unique(signatures))} of {len(halo)} "
            "background actions still share an ACT signature"
        )
        logits = case.act_policy[halo]
        assert len(torch.unique(logits)) == len(halo), (
            f"{case.name}: {len(halo) - len(torch.unique(logits))} of {len(halo)} "
            "background actions still share an ACT policy logit"
        )
        assert spread(logits) > 1e-1
        assert spread(logits) > 1e6 * max(
            spread(case.old_policy[list(group.rows)])
            for group in case.legacy.groups_of_kind(KIND_NO_WINDOW)
        )


def test_what_alias_the_full_model_does_retain(cases):
    """The residual, bounded and characterised rather than left implicit.

    On these eight positions ACT retains none at all. The command's own sweep
    over 24 real positions of the ``mnorm-late-v1`` corpus — 17,461 legal
    actions — retains 24 groups holding 95 actions, largest eleven, of which
    one group of two lies inside a persistent window and the rest do not.
    Those are a different animal from a distance bucket: cells out at the legal
    radius whose *entire* radius-``d_max`` stone neighbourhood is one stone in
    one D6 orbit, so what would separate them lies outside the model's geometry
    range rather than inside a bucket it chose.

    The bound and the structural guard below hold on either corpus. The guard
    is the one that would notice a real regression: every member of a group
    must share the orbit multiset the geometry line hashes, so a signature that
    stopped reading it would show up here as a group whose members' geometry
    disagrees.
    """
    total_actions = sum(case.act.legal_actions for case in cases)
    aliased = sum(case.act.aliased_actions for case in cases)
    assert aliased < 0.01 * total_actions, (
        f"{aliased} of {total_actions} legal actions are structurally aliased"
    )
    for case in cases:
        for group in case.act.groups:
            entries = [case.geometry[row] for row in group.rows]
            orbits = {entry["stone_orbits"] for entry in entries}
            assert len(orbits) == 1, (
                f"{case.name}: an alias group holds {len(orbits)} distinct stone "
                "orbit multisets, so the geometry line is not being hashed"
            )


def test_the_diagnostic_can_see_an_alias_that_is_really_there(cases):
    """The negative control: delete the geometry line and the halo collapses.

    Without it, ACT describes a cell in no persistent window by its nearest
    bucket and eighteen identical empty-window rows — the old representation's
    situation exactly. The diagnostic must report the collapse, or it is not a
    detector.
    """
    for case in cases:
        coarse = alias_report(coarsened_signatures(act_signatures(case.act_graph)))
        halo = coarse.groups_of_kind(KIND_NO_WINDOW)
        assert halo, f"{case.name}: the coarsened signature found no halo group"
        assert coarse.max_group_size > 10 * max(case.act.max_group_size, 1), (
            f"{case.name}: coarsening the signature grew the largest group only "
            f"from {case.act.max_group_size} to {coarse.max_group_size}"
        )
        assert coarse.unique_signatures < case.act.unique_signatures


def test_the_command_runs_over_a_built_position():
    """§33 asks for a command; this is its library entry point end to end."""
    report = diagnostics.act_alias_report(position(0, 61), FULL, samples=1)
    assert report.legal_actions > 0
    assert set(SIGNATURE_LINES) == set(
        act_signatures(build(position(0, 61), FULL)).lines
    )
    assert report.text().splitlines()[0].strip() == ARCHITECTURE_ID


def test_the_diagnostic_refuses_malformed_input(cases):
    """Every input the diagnostic cannot honour is refused by name."""
    case = cases[0]
    # The full model retains no alias here, so the describing callable is only
    # reached through a signature that groups something — the coarsened one.
    with pytest.raises(ValueError, match="geometry entries"):
        alias_report(
            coarsened_signatures(case.act_sig),
            lambda rows: case.describe(rows)[:-1],
            samples=1,
        )
    with pytest.raises(ValueError, match="samples"):
        alias_report(case.act_sig, samples=-1)
    with pytest.raises(ValueError, match="signature lines"):
        ActionSignatures(label="two lines", kind=(), lines={"cell": [], "windows": []})
    with pytest.raises(ValueError, match="either a model or a config"):
        profile([case.position], FULL, model=MantisACT(FULL))
    with pytest.raises(ValueError, match="at least one position"):
        profile([], FULL)
    # `occupied_only` represents no legal cell, so no legal action has a
    # coordinate to describe; §33's sampled geometry says so rather than
    # reporting the wrong cell.
    with pytest.raises(ValueError, match="represents no legal cell"):
        graph_geometry(build(case.position, PRESETS["full_occupied_cells_only"]))


# --------------------------------------------------------------------------
# §37.1 — every named preset constructs and runs


def test_37_1_every_named_preset_constructs_and_runs(cases):
    """All sixteen, including §16's typed window attention arm."""
    graph_of = {}
    for name in RUNNABLE_PRESETS:
        cfg = PRESETS[name]
        graph_of[name] = build(cases[0].position, cfg)
        torch.manual_seed(SEED)
        model = MantisACT(cfg).eval()
        with torch.no_grad():
            out = model(collate([graph_of[name]]), mass_floor=MASS_FLOOR)
        assert out.policy_logits.shape == (graph_of[name].n_legal,)
        assert torch.isfinite(out.policy_logits).all(), name
        assert torch.isfinite(out.q_value).all(), name

    # §29 also names `full_no_pair`, which §20's removal leaves describing the
    # full model itself; the deviation register carries that.
    assert "full_no_pair" not in PRESETS


def test_37_2_the_class_counts_hold():
    """378/377 window patterns, 2187/2184 joint classes, 729 post1, 48 orbits."""
    assert (ALL_WINDOW_PATTERN_CLASSES, NONEMPTY_WINDOW_PATTERN_CLASSES) == (378, 377)
    assert (ALL_CELL_WINDOW_REL_CLASSES, NONEMPTY_CELL_WINDOW_REL_CLASSES) == (
        2187,
        2184,
    )
    assert POST1_REL_CLASSES == 729
    assert D6_ORBITS_DMAX12 == 48
    # Generated, not restated: the orbit count is the table's own, and the
    # nonempty counts are their totals less the all-empty pattern's classes.
    assert orbit_table(12).count == 48
    assert ALL_WINDOW_PATTERN_CLASSES - NONEMPTY_WINDOW_PATTERN_CLASSES == 1
    assert ALL_CELL_WINDOW_REL_CLASSES - NONEMPTY_CELL_WINDOW_REL_CLASSES == WINDOW_LEN // 2


def test_37_3_full_output_d6_equivariance(orbit, orbit_out):
    """Every §25 output tensor maps by the engine's own legal permutation."""
    out, _tensors = orbit_out
    offsets = out.legal_offsets
    failures = []
    for field in ("policy_logits", "critic_logits", "q_value", "q_score"):
        tensor = getattr(out, field)
        base = _slice(tensor, offsets, 0)
        for t in TRANSFORMS:
            image = _slice(tensor, offsets, t)
            got = image[torch.as_tensor(orbit.legal[t])]
            measured, allowed = deviation(got, base), budget(base)
            if measured > allowed:
                failures.append(f"{field} under transform {t}: {measured:.3e} > {allowed:.3e}")
    assert not failures, "\n".join(failures)


def test_37_4_intermediate_axis_channel_equivariance(orbit, orbit_out):
    """§12.1's law at every site the assembled model exposes, action rows too.

    The trunk's own sites are covered here as well as in `test_act_d6.py`
    because §37.4 is a statement about the whole model: the two action sites
    are new, and a comparison that ran only over them would not say that the
    two halves of the model agree with each other.
    """
    _out, tensors = orbit_out
    rows_of = {"cell": orbit.cells, "window": orbit.windows, "state": orbit.legal}
    offsets_of = {
        "cell": orbit.batch.cell_offsets,
        "window": orbit.batch.window_offsets,
        "state": orbit.batch.legal_offsets,
    }
    failures = []
    for key, tensor in tensors.items():
        entity, stream = key.split(".")[1], key.split(".")[-1]
        for t in TRANSFORMS:
            if entity == "latent":
                # Latents are per position and fixed in count, so slot `p` of
                # the batch is position `p`'s block and needs no gather.
                base, image = tensor[0], tensor[t]
            else:
                base = _slice(tensor, offsets_of[entity], 0)
                image = _slice(tensor, offsets_of[entity], t).index_select(
                    0, torch.from_numpy(rows_of[entity][t])
                )
            if stream == "axis":
                base = permute_axis_channels(base, axis_permutation(t))
            measured, allowed = deviation(image, base), budget(base)
            if measured > allowed:
                failures.append(
                    f"{key} under transform {t}: {measured:.3e} > {allowed:.3e}"
                )
    assert not failures, "\n".join(failures)
    # The two sites §37.4 gains over the trunk suite, and both streams present.
    assert {"action.state.inv", "action.state.axis", "action.latent.inv"} <= set(tensors)
    assert sum(key.endswith(".axis") for key in tensors) > 0


class _AxisWeightedActions(torch.nn.Module):
    """The action encoder, with a gain on absolute axis channel 0 (§12.2).

    "Learn different weights for absolute axes" is the second construction
    §12.2 forbids, in its smallest form: one channel scaled by 1.01. It changes
    no shape, no count, and no invariant, and it reads every action's own
    channel-0 value rather than adding a constant — a constant is a *uniform*
    shift of every action of every board, and a comparison between two boards
    cancels it exactly. Measured: an added constant moves the output by 4e-7
    at any magnitude up to 1.0, and this gain moves it by about 7e-3 times the
    gain.
    """

    MAGNITUDE = 1e-2

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, batch, trunk):
        out = self.inner(batch, trunk)
        axis = out.actions.axis.clone()
        axis[:, 0, :] = axis[:, 0, :] * (1.0 + self.MAGNITUDE)
        return replace(out, actions=replace(out.actions, axis=axis))


def test_the_equivariance_criteria_can_fail(orbit, act_model):
    """§37.3 and §37.4 are shown able to catch a §12.2 violation.

    A criterion that has never failed is a claim about the test rather than
    about the model. The same two comparisons are run over a model carrying an
    absolute-axis gain on the action state, and each is required to report it.

    Not every transform reports it, and that is the point of running the whole
    group rather than one element: a transform whose induced axis permutation
    fixes channel 0 moves the gain onto itself, and the comparison it makes is
    blind to it. §31 asks for all eleven for exactly this reason.

    The two criteria are also not equally sharp, which is why §37 states them
    separately. Against this gain the axis-state comparison sits about a
    hundredfold over its budget and the output comparison about sixfold, so a
    tenth of this gain would be caught by §37.4 and missed by §37.3.
    """
    broken = MantisACT(FULL)
    broken.load_state_dict(act_model.state_dict())
    broken.actions = _AxisWeightedActions(broken.actions)
    with torch.no_grad():
        out, tensors = broken.eval().debug_forward(orbit.batch, MASS_FLOOR)

    axis = tensors["action.state.axis"]
    caught = {"output": [], "axis state": []}
    for t in TRANSFORMS:
        rows = torch.from_numpy(orbit.legal[t])
        base = _slice(out.policy_logits, out.legal_offsets, 0)
        image = _slice(out.policy_logits, out.legal_offsets, t).index_select(0, rows)
        caught["output"].append(deviation(image, base) > budget(base))

        want = permute_axis_channels(
            _slice(axis, orbit.batch.legal_offsets, 0), axis_permutation(t)
        )
        got = _slice(axis, orbit.batch.legal_offsets, t).index_select(0, rows)
        caught["axis state"].append(deviation(got, want) > budget(want))

    missed = [name for name, fired in caught.items() if not any(fired)]
    assert not missed, (
        f"{missed} missed an absolute-axis gain of {_AxisWeightedActions.MAGNITUDE} "
        "on the action state under every one of the eleven transforms"
    )


def test_37_5_every_legal_output_aligns_with_engine_order(cases):
    """The builder's mapping, and then the whole model through a permutation."""
    for case in cases:
        graph = case.act_graph
        legal = np.asarray(case.position.legal_moves(), dtype=np.int64).reshape(-1, 2)
        assert np.array_equal(graph.cell_qr[graph.legal_to_cell_index], legal)
        assert len(case.act_policy) == len(legal)


def test_37_5_the_model_carries_a_permuted_action_order(act_model, cases):
    """A batch whose action rows are permuted must give the permuted output.

    Asserting the builder's own mapping restates the builder. Permuting the
    rows goes through every stage and would catch one that sorted, grouped, or
    re-derived an order of its own.
    """
    case = cases[0]
    graph = case.act_graph
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(graph.n_legal, generator=generator).numpy()
    permuted = replace(
        graph,
        legal_to_cell_index=graph.legal_to_cell_index[order],
        action_window_index=graph.action_window_index[order],
        action_post1_class=graph.action_post1_class[order],
        action_pre_status=graph.action_pre_status[order],
        action_tactical_numeric=graph.action_tactical_numeric[order],
    )
    with torch.no_grad():
        got, _critic = act_model.policy_q(collate([permuted]))
    assert deviation(got, case.act_policy[order]) <= budget(case.act_policy)


def test_37_6_mixed_windows_are_nodes_in_the_default_model(cases):
    """§38's first invariant, on boards where mixed windows are the majority."""
    for case in cases:
        mixed = int((case.act_graph.window_status == MIXED).sum())
        assert mixed > 0, f"{case.name}: no mixed window is a node"
        if case.ply >= 121:
            assert mixed > 0.5 * case.act_graph.n_windows, (
                f"{case.name}: {mixed} of {case.act_graph.n_windows} windows are "
                "mixed, against the >50% the deviation register measures"
            )
    # And the ablation that removes them really removes them.
    live = build(cases[-1].position, PRESETS["full_live_windows"])
    assert int((live.window_status == MIXED).sum()) == 0


def test_37_7_default_relevant_cells_include_empty_persistent_window_cells(cases):
    """Every slot of every persistent window is a node, empty ones included."""
    for case in cases:
        graph = case.act_graph
        assert graph.window_incidence_mask.all(), (
            f"{case.name}: a persistent window slot is outside the cell scope"
        )
        slots = window_cells(graph.window_id).reshape(-1, 2)
        indices = np.unique(graph.window_cell_index)
        empty = indices[graph.cell_is_occupied[indices] == 0]
        assert empty.size > 0, f"{case.name}: every window slot holds a stone"
        assert len(slots) == graph.n_windows * WINDOW_LEN
        # The narrow scope is what removes them, which is what makes this a
        # property of the default rather than of the game.
        occupied_only = build(case.position, PRESETS["full_occupied_cells_only"])
        assert int(occupied_only.cell_is_occupied.sum()) == occupied_only.n_cells


def test_37_8_every_legal_action_receives_all_18_post_placement_rows(cases):
    """Dense `[num_legal, 3, 6]`, in range, for every action including halo ones."""
    for case in cases:
        graph = case.act_graph
        shape = (graph.n_legal, NUM_AXES, WINDOW_LEN)
        assert graph.action_post1_class.shape == shape
        assert graph.action_pre_status.shape == shape
        assert graph.action_window_index.shape == shape
        assert NUM_AXES * WINDOW_LEN == POST_ACTION_ROWS
        assert graph.action_post1_class.min() >= 0
        assert graph.action_post1_class.max() < POST1_REL_CLASSES
        # The case §19.2 singles out: an action in no persistent window at all
        # still gets all eighteen rows, which is the path that replaces the old
        # nearest-distance background.
        outside = case.halo(case.act_sig)
        assert outside, f"{case.name}: every legal action lies in some window"
        assert (graph.action_window_index[outside] == -1).all()
        assert (graph.action_post1_class[outside] >= 0).all()
        # And the nonempty window scope never puts an action outside a window
        # that the old live-only scope kept inside one. The containment is an
        # equality at most depths: a legal cell whose only windows are mixed is
        # rare, because a cell touching both colours usually also touches a
        # one-colour window. What ACT gains on the halo is the counterfactual
        # rows and the geometry, not membership.
        assert set(outside) <= set(case.halo(case.legacy_sig))


def test_37_9_is_void_because_section_20_does_not_exist():
    """Criterion 9 names prospective partners; there are none to name.

    Asserted rather than skipped: the criterion is void only while the
    subsystem is gone, and a reintroduced `pair_*` field anywhere in the
    configuration, the packed batch, or the graph would make it live again.
    """
    from mantisnet.models.mantis_act.packed import ACTGraph, PackedACTBatch

    for container in (MantisACTConfig, ACTGraph, PackedACTBatch):
        named = [
            field
            for field in container.__dataclass_fields__
            if "pair" in field or "partner" in field
        ]
        assert not named, f"{container.__name__} carries {named}"
    assert not hasattr(MantisACT(FULL), "pairs")


def test_37_10_the_external_klent_seam_is_unchanged():
    """`network_evaluate`'s interface, and the one private function under it."""
    signature = inspect.signature(network_evaluate)
    assert list(signature.parameters) == ["model", "cfg"]
    # The dispatch reaches a model through `policy_q` and nothing else, so no
    # architecture's internals are named in the trainer (§2, §25). Read off the
    # compiled code rather than the source: the docstring explains what the
    # trunk is, and a text search would be answered by the explanation.
    assert klent_train._policy_q.__code__.co_names == ("policy_q",)
    for architecture in (MantisACT, MantisNet):
        assert callable(architecture.policy_q)
        assert list(inspect.signature(architecture.policy_q).parameters) == [
            "self",
            "batch",
        ]


def test_37_10_the_evaluator_runs_over_both_architectures(cases):
    """The seam, exercised rather than only inspected."""
    cfg = KlentConfig(device="cpu", autocast=False, compile=False)
    torch.manual_seed(SEED)
    for model, batch in (
        (MantisACT(FULL).eval(), collate([case.act_graph for case in cases[:2]])),
        (
            MantisNet(MantisConfig()).eval(),
            legacy_collate([case.legacy_graph for case in cases[:2]]),
        ),
    ):
        policy, score, value = network_evaluate(model, cfg)(batch)
        assert policy.device.type == "cpu"
        assert torch.isfinite(policy).all()
        assert torch.isfinite(score).all()
        assert torch.isfinite(value).all()


def test_37_11_a_bf16_smoke_training_step_is_finite(cases):
    """Forward, backward, and one optimiser step, all finite (§27, §32)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    model = MantisACT(FULL).to(device).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = collate([case.act_graph for case in cases[:3]]).to(torch.device(device))

    with torch.autocast(device, torch.bfloat16, enabled=True):
        out = model(batch, mass_floor=MASS_FLOOR)
    for name in ("policy_logits", "critic_logits", "q_value", "q_score"):
        assert torch.isfinite(getattr(out, name)).all(), name
    loss = out.policy_logits.float().pow(2).mean() + out.critic_logits.float().pow(2).mean()
    assert torch.isfinite(loss)
    loss.backward()

    without_gradient = [
        name for name, p in model.named_parameters() if p.grad is None
    ]
    assert not without_gradient, without_gradient
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32, name
        assert torch.isfinite(parameter.grad).all(), name
    optimiser.step()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name


def test_37_12_node_edge_time_and_memory_telemetry_is_available(cases):
    """§34's figures, from the two calls that produce them."""
    batch = collate([case.act_graph for case in cases])
    counts = telemetry(batch)
    for name in (
        "cells",
        "windows",
        "legal_actions",
        "window_incidences",
        "adjacency_edges",
        "radius_edges",
        "post_action_rows",
        "windows_mixed",
        "windows_own_live",
        "windows_opp_live",
        "windows_empty",
    ):
        assert counts[f"{name}_mean"] > 0 or name == "windows_empty", name
        assert counts[f"{name}_max"] >= counts[f"{name}_mean"], name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = profile(
        [case.position for case in cases[:3]], FULL, device=device, repeats=2
    )
    for stage in ("build", "collate", "forward", "backward"):
        assert report.seconds[stage] > 0.0, stage
    for rate in ("positions_per_second", "legal_actions_per_second"):
        assert report.rate[rate] > 0.0, rate
    assert report.parameters.total == sum(
        p.numel() for p in MantisACT(FULL).parameters()
    )
    assert set(report.graph) == set(counts)
    if device == "cuda":
        assert report.memory_mib["peak_allocated"] > 0.0
        assert (
            report.memory_mib["peak_reserved"] >= report.memory_mib["peak_allocated"]
        )
    assert report.text()


def test_37_13_the_old_mantisnet_is_untouched_and_selectable(cases):
    """It still builds, runs, checkpoints, and is one of two lab variants.

    "Untouched" is checked where it can bite: MantisNet's own representation
    version is unchanged, its checkpoint round-trips into a fresh model, and no
    ACT payload can be loaded into it or its into ACT's.
    """
    # One lab variant for MantisNet, and one per §29 ACT preset — each arm
    # carrying its own bound collator (`tests/act/test_act_lab_variants.py`).
    assert set(VARIANTS) == {"mantis"} | set(PRESETS)
    assert isinstance(build_variant("mantis", {})[0], MantisNet)
    assert isinstance(build_variant("full_act_v4", {})[0], MantisACT)

    # ACT took the next repository representation value as its own constant
    # rather than bumping MantisNet's, which every checkpoint on disk carries.
    assert hexo_py.MODEL_REPR_VERSION == 3
    assert MANTIS_ACT_REPR_VERSION == 4

    torch.manual_seed(SEED)
    old = MantisNet(MantisConfig()).eval()
    batch = legacy_collate([case.legacy_graph for case in cases[:2]])
    with torch.no_grad():
        before, _critic = old.policy_q(batch)
    fresh = MantisNet(MantisConfig()).eval()
    fresh.load_state_dict(old.state_dict(), strict=True)
    with torch.no_grad():
        after, _critic = fresh.policy_q(batch)
    assert torch.equal(before, after)

    with pytest.raises(ValueError, match="act_checkpoint_format"):
        MantisACT(FULL).load_checkpoint(
            {"model_config": {}, "model": old.state_dict()}
        )


def test_the_parameter_count_is_reported_rather_than_padded():
    """§6's target, and the shortfall the deviation register records.

    The register states 1,726,468 against §6's 2.5-4M floor. A number that
    drifted from the register's would make the register wrong, which is what
    this asserts — not that the model reached a target it was deliberately not
    padded to.
    """
    torch.manual_seed(SEED)
    summary = parameter_summary(MantisACT(FULL))
    assert summary.total == 1_726_468, (
        f"the model holds {summary.total:,} parameters; "
        "docs/MANTIS_ACT_DEVIATIONS.md records 1,726,468"
    )
    assert summary.total < 2_500_000
    assert CATEGORICAL_CRITIC_LOGITS == 3
