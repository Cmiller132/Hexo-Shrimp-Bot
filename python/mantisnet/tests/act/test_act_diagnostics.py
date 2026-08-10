"""§33's alias diagnostic, §34's phase split, and §24.1's labels.

Three claims this file is responsible for, each with the oracle that makes it
evidence rather than a restatement of the code under test.

**The vectorised signature is the signature.** `diagnostics.act_signatures`
hashes each of §33's four lines with one polynomial over a padded table, which
is what makes it cost a fifth of the build instead of four times it. Speed is
worth nothing if the canonicalisation moved, so :func:`exact_signature` builds
the same four lines as exact Python tuples — sorted multisets, reversal-folded
axes, groups ordered by content — and the *partition* the two induce over a
position's legal actions is required to be identical. Tuples and 64-bit hashes
share no arithmetic, so a wrong sort key, a lost group, or a padding slot that
leaked into a digest separates them.

**The diagnostic fires.** A diagnostic that has never reported anything is not
evidence that there is nothing to report. `PLANTED_*` is a board built so that
two structurally distinct legal cells reach the model as the same encoding: two
identical stone neighbourhoods forty steps apart, with a third stone nineteen
steps from one of them — outside `d_max = 12`, and so outside everything the
representation reads. The two cells must be reported as an alias group, the
group's sampled geometry must name the stone that separates them, and the same
diagnostic must stay silent on real self-play boards.

**The labels are the rules.** `aux_labels` derives §24.1's six labels from the
eighteen pre- and post-placement window codes `actions.py` gathers. The oracle
here is the *engine's* own window walk, `Position.windows_through`, which the
builder is forbidden to call — so the two agree only if both read the board
correctly. The two partner labels get a second oracle that plays the second
placement and asks the engine whether the game ended.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

import hexo_py

from mantisnet.models.mantis_act.aux_labels import (
    action_aux_labels,
    position_aux_labels,
)
from mantisnet.models.mantis_act.builder import build, build_from_arrays
from mantisnet.models.mantis_act.config import PRESETS
from mantisnet.models.mantis_act.diagnostics import (
    KIND_IN_WINDOW,
    PHASE_NAMES,
    TOP_POLICY_ACTIONS,
    act_alias_report,
    act_signatures,
    alias_report,
    graph_geometry,
    phase_diagnostics,
    profile,
    signature_groups,
)
from mantisnet.models.mantis_act.heads import AUX_COUNT_CAP, AUX_SPECS
from mantisnet.models.mantis_act.model import MantisACT
from mantisnet.models.mantis_act.packed import (
    NUM_AXES,
    PHASE_FIRST,
    WINDOW_LEN,
    collate,
)

from .test_act_d6 import randomise_
from .test_act_numerics import position

SEED = 20260807
FULL = PRESETS["full_act_v4"]
MASS_FLOOR = 0.2

# Plies at which the signature is checked against the exact oracle: shallow
# enough that the halo dominates, deep enough that the window and radius
# families are at trained-play density.
PLIES = (21, 61, 121, 161)

# The planted alias (see the module docstring). Two own stones forty steps
# apart carry identical neighbourhoods; the opponent stone nineteen steps from
# the second is outside `d_max` of every cell described here, so it reaches no
# line of the signature and separates the two cells in the board alone.
PLANTED_STONES = np.array([[0, 0], [40, 0], [60, 0]], dtype=np.int64)
PLANTED_OWNERS = np.array([0, 0, 1], dtype=np.int64)
PLANTED_LEGAL = np.array([[1, 0], [2, 0], [41, 0], [42, 0]], dtype=np.int64)
PLANTED_PAIRS = ((0, 2), (1, 3))

# An engine-legal game in which both sides hold a live four and neither has
# won: P0 builds `(0..3, 0)` while P1 builds `(0..3, 7)`, seven rows apart so
# no window holds cells of both. Nine placements leaves P1 to move with two
# placements; the tenth gives P1 a live five and one placement left, which is
# the only state in which `win_now` can fire.
THREAT_GAME = [
    (0, 0),
    (0, 7), (1, 7),
    (1, 0), (2, 0),
    (2, 7), (3, 7),
    (3, 0), (3, 3),
]
WIN_GAME = [*THREAT_GAME, (4, 7)]


@pytest.fixture(scope="module")
def planted():
    return build_from_arrays(PLANTED_STONES, PLANTED_OWNERS, 0, PLANTED_LEGAL, 1, FULL)


@pytest.fixture(scope="module")
def graphs():
    """One real graph per (game, ply), plus the planted board."""
    return [build(position(game, ply), FULL) for game in (0, 1) for ply in PLIES]


@pytest.fixture(scope="module")
def prefix_tables():
    """One real position's §19.2 action tables and its legal coordinates."""
    from mantisnet.models.mantis_act.actions import action_tables
    from mantisnet.models.mantis_act.windows import enumerate_windows

    pos = position(0, 21)
    stones = np.asarray(pos.stones(), dtype=np.int64).reshape(-1, 3)
    legal = np.asarray(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
    own = (stones[:, 2] != int(pos.current_player)).astype(np.int64)
    window_set = enumerate_windows(stones[:, :2], own, legal, FULL)
    return action_tables(window_set, stones[:, :2], own, legal, FULL), legal


# --------------------------------------------------------------------------
# §33: the exact oracle the vectorised signature is held against


def exact_signature(graph) -> tuple[tuple, ...]:
    """§33's four lines as exact Python content, one tuple per legal action.

    Written from §33's own list rather than from `diagnostics`: every group is
    a sorted tuple, each axis of the post-placement rows is folded against its
    own reversal, and the three axis groups are ordered by their contents. It
    shares no arithmetic with the hashed pass — no polynomial, no padding, no
    64-bit anything — so agreement between the two partitions is a statement
    about the canonicalisation and not about the hash.
    """
    incident: dict[int, list[tuple[int, int, int, int]]] = {}
    for window, slot in zip(*np.nonzero(graph.window_cell_index >= 0)):
        incident.setdefault(int(graph.window_cell_index[window, slot]), []).append(
            (
                int(graph.window_axis[window]),
                int(graph.window_pattern_class[window]),
                int(graph.window_status[window]),
                int(graph.window_incidence_class[window, slot]),
            )
        )
    radius: dict[int, list[tuple[int, int]]] = {}
    for edge in range(graph.n_radius):
        radius.setdefault(int(graph.radius_dst[edge]), []).append(
            (
                int(graph.radius_orbit[edge]),
                int(graph.cell_occupancy[graph.radius_src[edge]]),
            )
        )

    def by_axis(entries):
        groups = [[] for _ in range(NUM_AXES)]
        for axis, *rest in entries:
            groups[axis].append(tuple(rest))
        return tuple(sorted(tuple(sorted(group)) for group in groups))

    out = []
    for action in range(graph.n_legal):
        cell = int(graph.legal_to_cell_index[action])
        if cell < 0:
            cell_line: tuple = ()
            window_line: tuple = ()
            orbit_line: tuple = ()
        else:
            cell_line = (
                int(graph.cell_occupancy[cell]),
                int(graph.cell_is_legal[cell]),
                int(graph.cell_nearest_bucket[cell]),
            )
            window_line = by_axis(incident.get(cell, []))
            orbit_line = tuple(sorted(radius.get(cell, [])))
        post = []
        for axis in range(NUM_AXES):
            rowset = tuple(
                (
                    int(graph.action_post1_class[action, axis, slot]),
                    int(graph.action_pre_status[action, axis, slot]),
                    int(graph.action_window_index[action, axis, slot] >= 0),
                )
                for slot in range(WINDOW_LEN)
            )
            post.append(min(rowset, rowset[::-1]))
        out.append((cell_line, window_line, tuple(sorted(post)), orbit_line))
    return tuple(out)


def partition(values) -> np.ndarray:
    """A grouping in a canonical form two groupings can be compared in.

    Each row is labelled by the first row sharing its value, so two arrays
    induce the same partition exactly when these are equal — independently of
    what the values themselves are.
    """
    _unique, first, inverse = np.unique(
        np.asarray(values), return_index=True, return_inverse=True
    )
    return first[inverse.reshape(-1)]


def test_the_hash_signature_induces_the_same_partition_as_an_exact_one(
    graphs, planted
):
    """§33's grouping is the exact one, on real boards and on the plant."""
    for graph in [*graphs, planted]:
        hashed = partition(act_signatures(graph).value)
        exact = partition(np.array([hash(line) for line in exact_signature(graph)]))
        disagreements = int((hashed != exact).sum())
        assert not disagreements, (
            f"{graph.n_legal} legal actions, {disagreements} placed in a different "
            "group by the vectorised signature than by the exact one"
        )


def test_the_kind_agrees_with_the_incidence_the_exact_pass_reads(graphs):
    """`in_window` means the exact window line is nonempty, and nothing else."""
    for graph in graphs:
        signatures = act_signatures(graph)
        for action, (_cell, windows, _post, _orbit) in enumerate(
            exact_signature(graph)
        ):
            in_window = any(group for group in windows)
            assert (signatures.kind[action] == KIND_IN_WINDOW) == in_window


# --------------------------------------------------------------------------
# §33: the diagnostic fires, and stays silent


def test_a_planted_alias_is_reported(planted):
    """Two structurally distinct cells with one encoding must be found."""
    signatures = act_signatures(planted)
    groups = signature_groups(signatures)
    assert {group.rows for group in groups} == set(PLANTED_PAIRS), (
        f"the plant produced groups {[group.rows for group in groups]}"
    )
    for left, right in PLANTED_PAIRS:
        assert signatures.value[left] == signatures.value[right]
        for name, column in signatures.lines.items():
            assert column[left] == column[right], name
    # The two boards really are different: the plant is an alias rather than a
    # symmetry of the position, because one cell has a stone nineteen steps out
    # and the other has nothing there at all.
    described = graph_geometry(planted, [0, 2], d_max=FULL.d_max)
    assert described[0]["omitted_stones"] != described[1]["omitted_stones"]


def test_the_planted_group_reports_the_geometry_that_separates_it(planted):
    """§33 asks a sampled group for its differing *omitted* geometry."""
    report = alias_report(
        act_signatures(planted),
        lambda rows: graph_geometry(planted, rows, d_max=FULL.d_max),
        samples=2,
    )
    assert report.legal_actions == 4
    assert report.unique_signatures == 2
    assert report.alias_groups == 2
    assert report.max_group_size == 2
    for sample in report.samples:
        assert [name for name, _ in sample.differing] == ["omitted_stones"]
        # Everything inside the representation's own radius agrees, which is
        # what makes this an alias rather than a hash collision.
        assert "stone_orbits" in sample.identical
        assert "stone_displacements" in sample.identical
    assert "omitted_stones" in report.text()


def test_the_diagnostic_is_silent_on_a_real_board():
    """The same command, on boards where there is nothing to report."""
    for ply in PLIES:
        report = act_alias_report(position(0, ply), FULL, samples=2)
        assert report.alias_groups == 0, report.text()
        assert report.unique_signatures == report.legal_actions
        assert report.samples == ()


def test_the_omitted_geometry_is_exactly_the_stones_outside_the_radius():
    """`omitted_stones` is the field that reaches past what the model reads."""
    graph = build(position(0, 61), FULL)
    described = graph_geometry(graph, [0, 1, 2], d_max=FULL.d_max)
    stones = graph.cell_qr[np.flatnonzero(graph.cell_is_occupied)]
    for row, entry in zip((0, 1, 2), described):
        coordinate = np.array(entry["coordinate"], dtype=np.int64)
        delta = stones - coordinate
        distance = np.max(
            np.abs(np.stack([delta[:, 0], delta[:, 1], delta[:, 0] + delta[:, 1]])),
            axis=0,
        )
        assert len(entry["omitted_stones"]) == int((distance > FULL.d_max).sum())
        assert len(entry["stone_orbits"]) == int((distance <= FULL.d_max).sum())


def test_the_geometry_of_a_subset_is_the_whole_descriptor_restricted():
    graph = build(position(1, 21), FULL)
    whole = graph_geometry(graph, d_max=FULL.d_max)
    rows = (0, 5, 17, len(whole) - 1)
    assert graph_geometry(graph, rows, d_max=FULL.d_max) == tuple(
        whole[row] for row in rows
    )


# --------------------------------------------------------------------------
# §24.1: the labels, against the engine's own window walk


def engine_labels(pos) -> dict[str, np.ndarray]:
    """§24.1's six labels from `Position.windows_through` and nothing else.

    The engine's window walk returns, for each of the eighteen windows through
    a cell, the axis, the start, and one occupancy bitmask per seat. The
    builder is forbidden to call it (see `crates/hexo-engine/README.md`), so it is an
    independent reading of the same board.
    """
    mover = pos.current_player
    legal = [tuple(int(c) for c in move) for move in pos.legal_moves()]
    axes = ((1, 0), (0, 1), (1, -1))

    winning: set[tuple[int, int]] = set()
    per_action = []
    for q, r in legal:
        rows = []
        for axis, start_q, start_r, mask_p0, mask_p1 in pos.windows_through(q, r):
            own = mask_p0 if mover == 0 else mask_p1
            opp = mask_p1 if mover == 0 else mask_p0
            slot = (q - start_q) // axes[axis][0] if axes[axis][0] else (
                (r - start_r) // axes[axis][1]
            )
            rows.append((axis, start_q, start_r, own, opp, slot))
        per_action.append(rows)
        if any(bin(own).count("1") == 5 and opp == 0 for _a, _q, _r, own, opp, _s in rows):
            winning.add((q, r))

    labels = {name: np.zeros(len(legal), dtype=np.int64) for name in AUX_SPECS}
    for action, ((q, r), rows) in enumerate(zip(legal, per_action)):
        own_after = [bin(own).count("1") + 1 for _a, _q, _r, own, _o, _s in rows]
        labels["win_now"][action] = int(max(own_after) == WINDOW_LEN)
        labels["own_max_occupancy"][action] = max(own_after)
        labels["opponent_threats_hit"][action] = min(
            sum(
                1
                for _a, _q, _r, own, opp, _s in rows
                if own == 0 and bin(opp).count("1") in (4, 5)
            ),
            AUX_COUNT_CAP,
        )
        five = [
            (axis, start_q, start_r, own, slot)
            for axis, start_q, start_r, own, opp, slot in rows
            if bin(own).count("1") + 1 == 5 and opp == 0
        ]
        labels["own_five_windows_after"][action] = min(len(five), AUX_COUNT_CAP)

        partners = set(winning)
        for axis, start_q, start_r, own, slot in five:
            empty = [k for k in range(WINDOW_LEN) if k != slot and not own >> k & 1]
            assert len(empty) == 1
            step = axes[axis]
            partners.add(
                (start_q + step[0] * empty[0], start_r + step[1] * empty[0])
            )
        count = 0 if (q, r) in winning else len(partners)
        labels["winning_partner_count"][action] = min(count, AUX_COUNT_CAP)
        labels["winning_partner_exists"][action] = int(count > 0)
    return labels


@pytest.mark.parametrize("game,ply", [(0, 21), (0, 61), (1, 41), (1, 120)])
def test_the_labels_match_the_engines_own_window_walk(game, ply):
    """Every §24.1 label, every legal action, two readings of the board."""
    pos = position(game, ply)
    got = position_aux_labels(pos, FULL)
    want = engine_labels(pos)
    assert set(got) == set(want)
    for name in want:
        assert np.array_equal(got[name], want[name]), (
            f"game {game} ply {ply}: {name} disagrees on "
            f"{int((got[name] != want[name]).sum())} of {len(want[name])} actions"
        )


def test_every_label_fires_on_a_crafted_game():
    """A label family that never fires would agree with any implementation.

    Four of the six are structurally rare in trained self-play, and
    `docs/MANTIS_ACT_DEVIATIONS.md` records the census that says so: the mover
    never holds an unanswered live four, because the opponent has just blocked
    it. So the positions that exercise them are built rather than sampled —
    from an engine-legal game, so the same engine oracle still applies.
    """
    fired = set()
    for placements in (THREAT_GAME, WIN_GAME):
        pos = hexo_py.Position.replay(placements)
        assert not pos.is_terminal
        got = position_aux_labels(pos, FULL)
        want = engine_labels(pos)
        for name in AUX_SPECS:
            assert np.array_equal(got[name], want[name]), name
            if int(got[name].max()) > 0:
                fired.add(name)
    # `win_now` needs a live five, which only exists once the first placement
    # of the turn has made one, so the two states between them cover all six.
    assert fired == set(AUX_SPECS), sorted(set(AUX_SPECS) - fired)


def test_every_label_lies_in_the_vocabulary_its_head_emits():
    """A label past its head's logits would be an index error somewhere else."""
    for placements in (THREAT_GAME, WIN_GAME):
        labels = position_aux_labels(hexo_py.Position.replay(placements), FULL)
        for name, value in labels.items():
            spec = AUX_SPECS[name]
            top = 1 if spec.logits == 1 else spec.logits - 1
            assert int(value.min()) >= 0 and int(value.max()) <= top, name


def test_the_partner_labels_match_playing_the_second_placement():
    """The definition, checked by the engine rather than by a second reading.

    On a first-placement state the label is the number of distinct second
    placements that end the game, so the engine can answer it directly: play
    the first stone, then play every legal reply and ask whether the position
    became terminal. That is `O(legal**2)` engine calls, which is why it runs
    on one position rather than the ladder.
    """
    pos = position(0, 41)
    assert pos.moves_remaining == 2, "this oracle only defines a FIRST state"
    labels = position_aux_labels(pos, FULL)
    for action, move in enumerate(pos.legal_moves()):
        after = pos.copy()
        after.advance(*move)
        if after.is_terminal:
            assert labels["winning_partner_count"][action] == 0
            assert labels["win_now"][action] == 1
            continue
        count = 0
        for second in after.legal_moves():
            reply = after.copy()
            reply.advance(*second)
            count += reply.is_terminal
        assert labels["winning_partner_count"][action] == min(count, AUX_COUNT_CAP), (
            f"action {action} at {tuple(move)}: {count} winning replies"
        )
        assert labels["winning_partner_exists"][action] == int(count > 0)


def test_the_labels_refuse_coordinates_that_are_not_the_tables(prefix_tables):
    """The partner labels read the coordinates, so a mismatched list is a fault."""
    tables, legal = prefix_tables
    with pytest.raises(ValueError, match="legal coordinates"):
        action_aux_labels(tables, legal[:-1])


# --------------------------------------------------------------------------
# §34: the per-phase split


AUX_WEIGHTS = {"winning_partner_exists": 1.0, "winning_partner_count": 1.0}
AUX_CFG = replace(FULL, enable_action_aux_heads=True)


@pytest.fixture(scope="module")
def aux_model():
    """A randomised model carrying the two auxiliaries a full config allows.

    §23 zero-initialises every output layer, so a fresh model gives one policy
    logit to every action and one class to every auxiliary — a split of which
    nothing is a measurement.
    """
    torch.manual_seed(SEED)
    return randomise_(
        MantisACT(AUX_CFG, aux_weights=AUX_WEIGHTS),
        SEED,
    ).eval()


@pytest.fixture(scope="module")
def phase_batch():
    """One batch spanning all three phases, with its labels."""
    positions = [hexo_py.Position()]
    for ply in (20, 21, 60, 61):
        positions.append(position(0, ply))
    graphs = [build(pos, AUX_CFG) for pos in positions]
    labels_per_position = [position_aux_labels(pos, AUX_CFG) for pos in positions]
    labels = {
        name: np.concatenate([entry[name] for entry in labels_per_position])
        for name in labels_per_position[0]
    }
    return collate(graphs, AUX_CFG), labels


def test_the_split_covers_every_phase_the_batch_holds(aux_model, phase_batch):
    """§34: entropy, top-policy Q spread, and auxiliary accuracy per phase."""
    batch, labels = phase_batch
    with torch.no_grad():
        out = aux_model(batch, mass_floor=MASS_FLOOR)
    reports = phase_diagnostics(out, batch.phase_id, labels)

    assert {report.phase for report in reports} == set(PHASE_NAMES.values()), (
        "the batch was chosen to hold all three phases"
    )
    assert sum(report.positions for report in reports) == batch.position_count
    assert sum(report.legal_actions for report in reports) == int(
        batch.legal_offsets[-1]
    )
    for report in reports:
        assert report.policy_entropy >= 0.0
        assert report.top_q_std >= 0.0
        assert np.isfinite(report.policy_entropy) and np.isfinite(report.top_q_std)
        assert report.text()
        for name, value in report.aux_accuracy.items():
            assert name in AUX_WEIGHTS
            assert 0.0 <= value <= 1.0
    # §24.1 masks auxiliaries 5 and 6 to first-placement rows, so no other
    # phase may carry an accuracy for them.
    scored = {report.phase for report in reports if report.aux_accuracy}
    assert scored == {PHASE_NAMES[PHASE_FIRST]}


def test_the_entropy_is_the_per_position_policy_entropy(aux_model, phase_batch):
    """The split's own arithmetic, against a direct reading of the same rows."""
    batch, labels = phase_batch
    with torch.no_grad():
        out = aux_model(batch, mass_floor=MASS_FLOOR)
    reports = {report.phase: report for report in phase_diagnostics(out, batch.phase_id, labels)}
    offsets = batch.legal_offsets.tolist()
    for phase, name in PHASE_NAMES.items():
        rows = [p for p in range(batch.position_count) if int(batch.phase_id[p]) == phase]
        entropy, spread = [], []
        for p in rows:
            logits = out.policy_logits[offsets[p] : offsets[p + 1]].float()
            probabilities = torch.softmax(logits, dim=0)
            entropy.append(float(-(probabilities * torch.log(probabilities)).sum()))
            top = logits.topk(min(TOP_POLICY_ACTIONS, len(logits))).indices
            chosen = out.q_value[offsets[p] : offsets[p + 1]].float()[top]
            spread.append(float(chosen.std(correction=0)) if len(top) > 1 else 0.0)
        assert reports[name].policy_entropy == pytest.approx(np.mean(entropy), abs=1e-5)
        assert reports[name].top_q_std == pytest.approx(np.mean(spread), abs=1e-6)


def test_the_accuracy_is_the_fraction_of_labelled_rows_the_head_gets_right(
    aux_model, phase_batch
):
    """Feeding the head's own prediction back as the label must give 1.0."""
    batch, labels = phase_batch
    with torch.no_grad():
        out = aux_model(batch, mass_floor=MASS_FLOOR)
    perfect = {
        "winning_partner_exists": (out.aux["winning_partner_exists"] > 0)
        .long()
        .numpy(),
        "winning_partner_count": out.aux["winning_partner_count"]
        .argmax(dim=-1)
        .numpy(),
    }
    for report in phase_diagnostics(out, batch.phase_id, perfect):
        for value in report.aux_accuracy.values():
            assert value == 1.0
    wrong = {name: (value + 1) % 2 for name, value in perfect.items()}
    scored = [
        value
        for report in phase_diagnostics(out, batch.phase_id, wrong)
        for value in report.aux_accuracy.values()
    ]
    assert scored and all(value < 1.0 for value in scored)


def test_a_head_without_a_label_is_refused(aux_model, phase_batch):
    """§34's accuracy column may not silently omit a head that is training."""
    batch, labels = phase_batch
    with torch.no_grad():
        out = aux_model(batch, mass_floor=MASS_FLOOR)
    with pytest.raises(ValueError, match="no label was given"):
        phase_diagnostics(out, batch.phase_id, {})
    short = {name: value[:-1] for name, value in labels.items()}
    with pytest.raises(ValueError, match="rows against the batch"):
        phase_diagnostics(out, batch.phase_id, short)
    with pytest.raises(ValueError, match="phases against"):
        phase_diagnostics(out, batch.phase_id[:-1], labels)


def test_a_model_without_auxiliaries_reports_the_rest_of_the_split(phase_batch):
    """The two §34 lines that need no label are reported without one."""
    batch, _labels = phase_batch
    torch.manual_seed(SEED)
    model = randomise_(MantisACT(FULL), SEED).eval()
    with torch.no_grad():
        out = model(batch, mass_floor=MASS_FLOOR)
    reports = phase_diagnostics(out, batch.phase_id)
    assert len(reports) == len(PHASE_NAMES)
    assert all(report.aux_accuracy == {} for report in reports)


# --------------------------------------------------------------------------
# §34: the profile


def test_the_profile_reports_every_figure_section_34_asks_for(aux_model):
    """Node and edge counts, the four stage times, throughput, memory, phases."""
    report = profile(
        [position(0, 21), position(0, 20)],
        model=aux_model,
        device="cpu",
        repeats=1,
        mass_floor=MASS_FLOOR,
    )
    assert set(report.seconds) == {"build", "collate", "labels", "forward", "backward"}
    assert all(value > 0.0 for value in report.seconds.values())
    assert set(report.rate) == {
        "positions_per_second",
        "legal_actions_per_second",
        "positions_built_per_second",
    }
    assert report.positions == 2
    assert report.legal_actions > 0
    assert report.parameters.total == sum(
        p.numel() for p in aux_model.parameters() if p.requires_grad
    )
    for family in ("cells", "legal_actions", "radius_edges"):
        assert any(key.startswith(family) for key in report.graph), sorted(report.graph)
    assert {r.phase for r in report.phases} == {PHASE_NAMES[PHASE_FIRST], "SECOND"}
    assert "by phase" in report.text()
    # §34's `pair row counts` line is dead with §20 itself, and a zero column
    # would read as a measurement rather than as an absence.
    assert "pair" not in report.text()


def test_the_profile_needs_a_label_producer_for_the_heads_it_holds(aux_model):
    """A model with auxiliaries is profiled with their labels, or not at all."""
    assert aux_model.heads.auxiliaries is not None
    report = profile(
        [position(0, 41)], model=aux_model, device="cpu", repeats=1, mass_floor=None
    )
    assert report.seconds["labels"] > 0.0
    first = [r for r in report.phases if r.phase == PHASE_NAMES[PHASE_FIRST]]
    assert first and set(first[0].aux_accuracy) == set(AUX_WEIGHTS)
