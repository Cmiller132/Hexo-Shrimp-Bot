"""§31: the D6 equivariance suite, and the proof that it can fail.

This is the acceptance criterion for Stages B and C (§37.3, §37.4). The hazard
it exists for is stated in ``CLAUDE.md``: a module that *looks* equivariant and
is not applies and un-applies its asymmetry identically, so every round-trip,
shape, and invariant test in this package passes over it unchanged. Only an
explicit transform-and-compare sees it, and only if the comparison is built
from something other than the thing it is checking.

The suite is three detectors, because no one of them covers §12.2 alone.

1. `law_deviations` — the position law of §31. The transform is applied to the
   *game*: a sample replays the same move list through each of the twelve D6
   elements and builds twelve positions from the engine, so the node sets, the
   window identities, the pattern classes, the orbit ids, the incidence
   classes, the legal-move order and the edge order are all *rediscovered* by
   the builder from a board the engine produced. It is the only detector that
   can see a fault living between the engine and the model — a wrong axis
   route, a wrong slot canonicalisation, a node correspondence the builder does
   not honour — and it is the one bounded by fp32 arithmetic.
2. `structural_violations` — every parameter in the assembled trunk whose shape
   is indexed by an absolute axis. Two entitled exceptions, both three-valued
   vocabularies that are not axes. Holds at any magnitude.
3. `module_violations` — every module of the assembled trunk that takes an
   `EquivariantState`, exercised alone in fp64 and required to either permute
   with a channel permutation or be invariant under it. Dispatch is by
   signature, not by class name, so a substituted or wrapped module cannot
   escape by being unfamiliar. Its floor is a few ulps rather than 1e-5.

What makes detector 1 independent of the code it watches.

- **The node correspondence comes from the transformed position.** Cells are
  matched by transformed coordinate, windows by their transformed six-cell set,
  and legal actions by the transformed engine legal move. Every lookup is a
  dictionary hit into the *image* graph's own tables, so a missing image is a
  ``KeyError`` rather than a silently shortened comparison. Permuting the base
  graph's rows instead would restate the builder rather than check it. That the
  matching is load-bearing is itself asserted, by exchanging two of its entries
  and requiring the comparison to fail.
- **Every intermediate is checked, not only the output.** §37.4 asks for
  intermediate axis-channel equivariance, so the comparison runs at every site
  `StateTrunk.debug_forward` exposes — the input embeddings, all four blocks,
  and the final norms — for cells, windows, and latents, in both streams.
- **The model is given weights that make every path contribute.** At
  initialisation LayerScale is 1e-2, FiLM is exactly the identity, and every
  embedding is N(0, 0.02); a break inside FiLM or inside a relation table would
  move the output by less than the floating-point budget and pass. `randomise_`
  moves every parameter off its initialisation, so each of the eleven stages of
  a block is a live contributor to the comparison.

What the suite is *known* to catch. Six forbidden constructions are introduced
one at a time into an otherwise identical trunk and run through all three
detectors — §12.2's three sentences (a per-absolute-axis bias, a fixed-order
concatenation of channels 0/1/2 into an unconstrained MLP, an absolute-axis
embedding lookup), §17.1's per-channel latent base, §12.5's per-channel pooling
score, and a line message routed into a fixed channel rather than the
structural one. Each declares which detectors must fire and which must not, so
the mutation runs are a coverage matrix rather than a smoke test. Two facts
that matrix records and that no single detector would have shown:

- the pooling-score bias moves an fp32 forward by about 1e-5 — under the
  position law's own noise floor — and is caught only by the shape and module
  detectors. A suite that ran the board alone would miss it;
- the fixed-channel route holds no parameter and has no module of its own, and
  is caught only by the position law. A suite that inspected the model alone
  would miss it.

Where each detector runs. Detectors 2 and 3 read parameters and modules, so a
device would tell them nothing. Detector 1 reads arithmetic, and this package's
   arithmetic is not the same code on the two devices: the registered message,
   equivariant, latent, segment, post-row, and class-embedding families dispatch
   to their optimized paths only for supported CUDA inputs. A host-only run of
   the position law has therefore never put §31 to those kernels at all — and a
   kernel that read a fixed absolute axis channel instead of the row's own would
   pass it. The law runs twice where a device exists, once per device, with every
   fused acceptance counted so a silent fallback to the torch reference cannot
   pass as a device run, and with the same forbidden constructions introduced
   under the kernels.

Positions come from two generators, both engine play.

- **Randomised**: uniformly random legal playouts. Sparse, wide legal halos, few
  mixed windows.
- **Real**: contact play — placement restricted to the ring of cells touching a
  stone, blocking an opponent five when one exists and never completing a six,
  so a game runs long instead of ending in eleven plies. It reproduces the
  self-play density ``docs/MANTIS_ACT_DEVIATIONS.md`` measures on stack-939:
  at ply 161 it gives 874 windows, 63% of them mixed, and 71,420 radius edges
  against that document's 71,700. That is a claim about board *shape*, not about
  playing strength; strength is irrelevant to a symmetry law, and density is not
  — the doc records that random playouts carry a fifteenth of the mixed windows
  a real board does, and mixed windows are more than half of what the model
  reads at depth.

Tolerance. `ATOL` and `RTOL` below are the ulp-aware pair
``tests/test_symmetry.py`` and the lab check battery already use in this
repository, applied per tensor rather than per element — see their definition
for why the scale of a segment sum, not the size of its result, is what sets
the error. Two assertions keep them honest rather than convenient: the measured
worst case over the whole suite must stay inside half the budget
(`test_the_measured_drift_stays_well_inside_the_budget` — it sits at 22%), and
the residual must be the same size as the residual of a comparison with no
symmetry in it at all (`test_the_residual_is_the_size_of_pure_reassociation_noise`).
The second is the argument that what remains is arithmetic: a typed asymmetry
survives reassociation, and noise does not.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, replace

import hexo_py
import numpy as np
import pytest
import torch
from torch import Tensor, nn

from mantisnet.models.mantis_act import (
    class_embedding,
    latent_attention,
    post_rows,
    segment_message,
)
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.equivariant import (
    AXIS_CHANNELS,
    PHASE_IDS,
    AxisPool,
    EquivariantState,
    permute_axis_channels,
)
from mantisnet.models.mantis_act.latents import LatentState
from mantisnet.models.mantis_act.messages import TypedEdges
from mantisnet.models.mantis_act.model import MantisACT
from mantisnet.models.mantis_act.packed import (
    PHASE_FIRST,
    PHASE_SECOND,
    ACTGraph,
    PackedACTBatch,
    collate,
)
from mantisnet.models.mantis_act.pattern_classes import MIXED
from mantisnet.models.mantis_act.state_trunk import StateTrunk
from mantisnet.models.mantis_act.symmetry import (
    D6_TRANSFORMS,
    axis_permutation,
    transform_coords,
)
from mantisnet.models.mantis_act.windows import window_cells

SEED = 20260806

FULL = PRESETS["full_act_v4"]

# The eleven nonidentity elements of §31. Index 0 is the identity and is the
# base every other one is compared against.
TRANSFORMS = tuple(range(1, len(D6_TRANSFORMS)))

# §31's pinned tolerance, ulp-aware in the repository's established form: a
# fixed floor plus a relative term. The relative term is taken against the
# reference tensor's *scale* rather than against each element, because the
# quantity that sets the error of a scatter-order-dependent sum is the
# magnitude of the terms being summed, not the magnitude of the result — an
# element that cancels to near zero still carries the absolute error of the
# hundreds of fp32 additions that produced it. Every comparison here is
# downstream of at least one segment reduction whose row order the transform
# genuinely changes, so a per-element rtol would be a floor of `ATOL` alone and
# would say nothing.
ATOL = 1e-5
RTOL = 5e-6

# Randomised sample depths: two placements in (both movers, FIRST and SECOND),
# and boards sparse-but-populated enough that every edge family is nonempty.
RANDOM_PLIES = (2, 21, 60)

# Real contact-play depths, at the plies docs/MANTIS_ACT_DEVIATIONS.md
# measures its self-play density table on.
CONTACT_PLIES = (21, 61, 121)

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))

# The magnitude every §12.2 mutation is introduced at. Small on purpose: it is
# four orders below the weight scale the mutation sits among, so a suite that
# catches it is not merely noticing that the model changed.
MUTATION_MAGNITUDE = 1e-3


# --------------------------------------------------------------------------
# Position generators (engine play, both of them)


def random_playout(plies: int, seed: int) -> list[tuple[int, int]]:
    """A seeded uniformly random legal playout of exactly ``plies`` placements."""
    for attempt in range(100):
        rng = random.Random(seed * 7919 + attempt * 31 + plies)
        position = hexo_py.Position()
        moves: list[tuple[int, int]] = []
        for _ in range(plies):
            move = rng.choice(position.legal_moves())
            position.advance(*move)
            moves.append(move)
        if not position.is_terminal:
            return moves
    raise AssertionError(f"no nonterminal {plies}-ply random playout in 100 seeds")


def contact_move(position, rng: random.Random) -> tuple[int, int]:
    """One contact placement: on the ring, blocking a five, never winning.

    The engine is the only authority consulted — its legal list and its
    ``windows_through`` — so the boards this produces are boards the engine
    itself reaches. The three preferences exist to keep the game going and the
    stones together: a greedy line extender wins in eleven plies and a uniform
    random player scatters, and neither produces the mixed-window majority a
    real board carries.
    """
    occupied = {(q, r) for q, r, _ in position.stones()}
    legal = {tuple(move) for move in position.legal_moves()}
    ring = sorted(
        {(q + dq, r + dr) for q, r in occupied for dq, dr in NEIGHBOURS} & legal
    )
    mover = position.current_player
    quiet: list[tuple[int, int]] = []
    blocking: list[tuple[int, int]] = []
    winning: list[tuple[int, int]] = []
    for q, r in ring or sorted(legal):
        wins = blocks = False
        for _axis, _sq, _sr, mask0, mask1 in position.windows_through(q, r):
            own, opponent = (mask0, mask1) if mover == 0 else (mask1, mask0)
            wins |= opponent == 0 and bin(own).count("1") == 5
            blocks |= own == 0 and bin(opponent).count("1") == 5
        (winning if wins else blocking if blocks else quiet).append((q, r))
    return rng.choice(blocking or quiet or winning)


def contact_game(seed: int, plies: int) -> list[tuple[int, int]]:
    """A seeded contact-play game of exactly ``plies`` nonterminal placements."""
    rng = random.Random(seed)
    position = hexo_py.Position()
    moves: list[tuple[int, int]] = []
    while len(moves) < plies:
        move = contact_move(position, rng)
        position.advance(*move)
        moves.append(move)
        if position.is_terminal:
            raise AssertionError(f"contact game ended at ply {len(moves)}")
    return moves


# --------------------------------------------------------------------------
# Node correspondence, built from the transformed position's own tables


def _rows_by_coordinate(qr: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(q), int(r)): row for row, (q, r) in enumerate(qr)}


def cell_correspondence(base: ACTGraph, image: ACTGraph, t: int) -> np.ndarray:
    """``out[i]``: the row of ``image`` holding the image of ``base`` cell ``i``.

    A base cell whose image is not a node of the transformed position raises,
    which is the check that the node *set* is carried by the group rather than
    merely the same size.
    """
    rows = _rows_by_coordinate(image.cell_qr)
    moved = transform_coords(t, base.cell_qr)
    return np.array(
        [rows[(int(q), int(r))] for q, r in moved], dtype=np.int64
    )


def window_correspondence(base: ACTGraph, image: ACTGraph, t: int) -> np.ndarray:
    """``out[i]``: the row of ``image`` holding the image of ``base`` window ``i``.

    Matched by the six-cell *set*, not by the stored identity: a transform may
    reverse a window's slot order, so ``(native_axis, start_q, start_r)`` is not
    carried by the group while the set of covered coordinates is.
    """

    def shapes(window_id: np.ndarray) -> list[tuple]:
        return [
            tuple(sorted((int(q), int(r)) for q, r in cells))
            for cells in window_cells(window_id)
        ]

    rows = {shape: row for row, shape in enumerate(shapes(image.window_id))}
    moved = transform_coords(
        t, window_cells(base.window_id).reshape(-1, 2)
    ).reshape(-1, 6, 2)
    return np.array(
        [
            rows[tuple(sorted((int(q), int(r)) for q, r in cells))]
            for cells in moved
        ],
        dtype=np.int64,
    )


def legal_correspondence(
    base_legal: np.ndarray, image_legal: np.ndarray, t: int
) -> np.ndarray:
    """``out[j]``: the engine legal-order row of the image of base action ``j``.

    Both lists are the engine's own ``legal_moves()`` in its own order, so this
    is the permutation §31.1 requires policy rows to be read through. It is not
    the identity: the engine's ordering is a coordinate ordering, and a rotation
    scrambles it.
    """
    rows = _rows_by_coordinate(image_legal)
    moved = transform_coords(t, base_legal)
    return np.array([rows[(int(q), int(r))] for q, r in moved], dtype=np.int64)


@dataclass(frozen=True, eq=False)
class Sample:
    """One position under all twelve transforms, packed as one batch.

    ``graphs[t]`` is the position built from the move list replayed through
    transform ``t``, and slot ``t`` of ``batch`` is that graph, so one forward
    covers the whole orbit and the comparison is between slices of a single
    batch. ``cells[t]``, ``windows[t]``, and ``legal[t]`` carry base row ``i``
    to its image's row *inside position ``t``*.
    """

    name: str
    graphs: tuple[ACTGraph, ...]
    legal_qr: tuple[np.ndarray, ...]
    batch: PackedACTBatch
    cells: tuple[np.ndarray, ...]
    windows: tuple[np.ndarray, ...]
    legal: tuple[np.ndarray, ...]

    @property
    def base(self) -> ACTGraph:
        return self.graphs[0]


def make_sample(name: str, moves: list[tuple[int, int]]) -> Sample:
    """Replay ``moves`` through every transform and build the correspondences."""
    positions = [
        hexo_py.Position.replay([D6_TRANSFORMS[t](move) for move in moves])
        for t in range(len(D6_TRANSFORMS))
    ]
    for t, position in enumerate(positions):
        assert not position.is_terminal, f"{name}: transform {t} reached a terminal state"
    graphs = tuple(build(position, FULL) for position in positions)
    legal_qr = tuple(
        np.asarray(position.legal_moves(), dtype=np.int64).reshape(-1, 2)
        for position in positions
    )
    base = graphs[0]
    return Sample(
        name=name,
        graphs=graphs,
        legal_qr=legal_qr,
        batch=collate(list(graphs), FULL),
        cells=tuple(
            cell_correspondence(base, graphs[t], t) for t in range(len(graphs))
        ),
        windows=tuple(
            window_correspondence(base, graphs[t], t) for t in range(len(graphs))
        ),
        legal=tuple(
            legal_correspondence(legal_qr[0], legal_qr[t], t) for t in range(len(graphs))
        ),
    )


# --------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="module")
def samples() -> tuple[Sample, ...]:
    made = [
        make_sample(f"random{plies}", random_playout(plies, SEED))
        for plies in RANDOM_PLIES
    ]
    game = contact_game(SEED, max(CONTACT_PLIES))
    made += [
        make_sample(f"contact{plies}", game[:plies]) for plies in CONTACT_PLIES
    ]
    return tuple(made)


@pytest.fixture(scope="module")
def dense(samples) -> Sample:
    """The densest *real* sample: the one the mutation runs are measured on.

    A random playout of the same depth carries more cells and more windows, but
    the board this model will see is the contact one — 60% mixed windows and a
    mean nearest-stone distance near one — so that is the board a mutation is
    asked to be visible on.
    """
    return max(
        (sample for sample in samples if sample.name.startswith("contact")),
        key=lambda sample: sample.base.n_windows,
    )


def randomise_(module: nn.Module, seed: int) -> nn.Module:
    """Move every parameter off its initialisation, keeping the model sane.

    A trunk at §27's initialisation is a poor detector: LayerScale is 1e-2 so
    every branch is a whisper, FiLM's projections are exactly zero so phase
    conditioning is a literal no-op, and the relation, pattern, and latent
    tables are N(0, 0.02). A break inside any of those would move the output by
    less than the floating-point budget. Norms keep unit-ish gain and
    LayerScale a fraction of one, so the residual stream stays bounded — the
    point is to make every stage contribute, not to make the model diverge.
    """
    generator = torch.Generator().manual_seed(seed)

    def fill(tensor: Tensor, mean: float, std: float) -> None:
        with torch.no_grad():
            tensor.copy_(torch.empty_like(tensor).normal_(mean, std, generator=generator))

    seen: set[int] = set()
    for sub in module.modules():
        if isinstance(sub, nn.Embedding):
            fill(sub.weight, 0.0, 0.3)
            seen.add(id(sub.weight))
        elif isinstance(sub, nn.LayerNorm):
            for parameter in (sub.weight, sub.bias):
                if parameter is not None:
                    fill(parameter, 1.0 if parameter is sub.weight else 0.0, 0.1)
                    seen.add(id(parameter))
        elif isinstance(sub, nn.Linear):
            for parameter in (sub.weight, sub.bias):
                if parameter is None:
                    continue
                # Zero-initialised projections are FiLM's (§27); left alone they
                # would make phase conditioning untested.
                if bool((parameter == 0).all()):
                    fill(parameter, 0.0, 0.05)
                seen.add(id(parameter))
    for name, parameter in module.named_parameters():
        if id(parameter) in seen:
            continue
        fill(parameter, 0.5 if name.endswith("gamma") else 0.0, 0.2)
    return module.eval()


def fresh_trunk(seed: int = SEED) -> StateTrunk:
    torch.manual_seed(seed)
    return randomise_(StateTrunk(FULL), seed)


@pytest.fixture(scope="module")
def trunk() -> StateTrunk:
    return fresh_trunk()


# --------------------------------------------------------------------------
# The law itself


def budget(reference: Tensor) -> float:
    """The pinned slack for one comparison: ``ATOL + RTOL * scale``."""
    if reference.numel() == 0:
        return ATOL
    return ATOL + RTOL * float(reference.detach().abs().max())


def _slice(tensor: Tensor, offsets: Tensor, position: int) -> Tensor:
    return tensor[int(offsets[position]) : int(offsets[position + 1])]


def _deviation(got: Tensor, want: Tensor) -> float:
    """The worst elementwise disagreement, measured in fp64 whatever came in."""
    if got.shape != want.shape:
        raise AssertionError(f"shape {tuple(got.shape)} against {tuple(want.shape)}")
    if got.numel() == 0:
        return 0.0
    return float((got.detach().double() - want.detach().double()).abs().max())


def _record(
    into: dict[str, tuple[float, float]], name: str, got: Tensor, want: Tensor
) -> None:
    """Keep the worst deviation seen for ``name`` and the budget it must meet."""
    deviation, allowed = _deviation(got, want), budget(want)
    previous = into.get(name)
    if previous is None or deviation > previous[0]:
        into[name] = (deviation, allowed)


def law_deviations(trunk: StateTrunk, sample: Sample) -> dict[str, tuple[float, float]]:
    """Every §31 comparison for one sample: name -> (worst deviation, budget).

    One `debug_forward` over the whole orbit, then, for each of the eleven
    nonidentity transforms:

    - §31.4 and §31.6: invariant cell, window, and latent states, at every site,
      unchanged under the node correspondence;
    - §31.5 and §31.7: their axis states carried by the induced axis permutation;
    - §31.1: the same two, gathered in engine legal order through the legal-move
      correspondence — the rows a policy head will be read off;
    - §31.3: the symmetric scalar readouts of every family, unchanged.
    """
    with torch.no_grad():
        _out, tensors = trunk.debug_forward(sample.batch)
    batch = sample.batch
    # The correspondences are built on the host and follow the batch, so one
    # law runs wherever the batch is — which is what puts the fused kernels
    # under it (§36) instead of only the torch reference.
    device = batch.cell_offsets.device
    into: dict[str, tuple[float, float]] = {}

    def index(array: np.ndarray) -> Tensor:
        return torch.from_numpy(array).to(device)

    for t in TRANSFORMS:
        permutation = axis_permutation(t)
        cells = index(sample.cells[t])
        windows = index(sample.windows[t])
        legal = index(sample.legal[t])
        base_legal_cells = index(sample.graphs[0].legal_to_cell_index)
        image_legal_cells = index(sample.graphs[t].legal_to_cell_index)

        for site in trunk.debug_sites():
            family = site.split(".")[1]
            for stream in ("inv", "axis"):
                key = f"{site}.{stream}"
                if key not in tensors:
                    continue
                tensor = tensors[key]
                if family == "latent":
                    want, got = tensor[0], tensor[t]
                elif family == "cell":
                    want = _slice(tensor, batch.cell_offsets, 0)
                    got = _slice(tensor, batch.cell_offsets, t).index_select(0, cells)
                else:
                    want = _slice(tensor, batch.window_offsets, 0)
                    got = _slice(tensor, batch.window_offsets, t).index_select(0, windows)
                if stream == "axis":
                    want = permute_axis_channels(want, permutation)
                _record(into, key, got, want)

                # §31.3's scalar readouts, off the same tensors. Both means are
                # over dimensions the transform permutes — the node set, and for
                # the axis stream the three channels — so both are invariants,
                # and the channel mean is unaffected by the permutation applied
                # to `want` above.
                pooled = (0,) if stream == "inv" else (0, 1)
                _record(
                    into,
                    f"scalar.{site}.{stream}",
                    got.mean(dim=pooled),
                    want.mean(dim=pooled),
                )

        # §31.1: the action rows a policy head reads, in engine legal order.
        # `legal[j]` is the image position's policy row for base row `j`, and
        # `legal_to_cell_index` is the cell node that row is read from, so the
        # gather composes the legal-order correspondence with the cell one.
        for stream in ("inv", "axis"):
            tensor = tensors[f"final.cell.{stream}"]
            want = _slice(tensor, batch.cell_offsets, 0).index_select(0, base_legal_cells)
            got = (
                _slice(tensor, batch.cell_offsets, t)
                .index_select(0, image_legal_cells)
                .index_select(0, legal)
            )
            if stream == "axis":
                want = permute_axis_channels(want, permutation)
            _record(into, f"policy.{stream}", got, want)

    return into


@pytest.fixture(scope="module")
def deviations(trunk, samples) -> dict[str, dict[str, tuple[float, float]]]:
    return {sample.name: law_deviations(trunk, sample) for sample in samples}


def over_budget(measured: dict[str, tuple[float, float]], prefix: str) -> list[str]:
    """The checks under ``prefix`` whose deviation exceeded its budget."""
    return [
        f"{name}: {deviation:.3e} > {allowed:.3e}"
        for name, (deviation, allowed) in sorted(measured.items())
        if name.startswith(prefix) and deviation > allowed
    ]


# --------------------------------------------------------------------------
# The premises: the transform set, and what the builder does with it


def test_the_transform_set_is_the_whole_group_and_realises_every_axis_permutation():
    """§30.9, and the premise the mutation runs depend on.

    A per-absolute-axis parameter is invisible to a transform whose induced
    axis permutation is the identity — the 180-degree rotation is one — so a
    suite drawing on fewer than all six permutations of the three axes could
    miss §12.2 entirely. All six are realised, twice each.
    """
    assert len(D6_TRANSFORMS) == 12 and len(TRANSFORMS) == 11
    realised = [axis_permutation(t) for t in range(len(D6_TRANSFORMS))]
    assert len(set(realised)) == 6
    assert sorted(realised.count(p) for p in set(realised)) == [2] * 6
    assert realised[0] == (0, 1, 2)
    # And at least one nonidentity transform fixes every axis, which is why the
    # count above is the thing being asserted rather than "eleven transforms".
    assert any(realised[t] == (0, 1, 2) for t in TRANSFORMS)


def test_the_samples_cover_both_generators_and_the_density_they_claim(samples):
    """The position set is what the docstring says it is."""
    names = {sample.name for sample in samples}
    assert any(name.startswith("random") for name in names)
    assert any(name.startswith("contact") for name in names)

    phases = {int(sample.base.phase_id) for sample in samples}
    assert phases == {PHASE_FIRST, PHASE_SECOND}

    deepest = max(
        (s for s in samples if s.name.startswith("contact")),
        key=lambda s: s.base.n_windows,
    )
    graph = deepest.base
    mixed = int((graph.window_status == MIXED).sum())
    # docs/MANTIS_ACT_DEVIATIONS.md measures the mixed share past 50% by ply
    # 60 of real self-play; a suite run on random playouts alone would see 4%.
    assert mixed / graph.n_windows > 0.5
    # Every edge family and both radius routes are populated, so no path in the
    # trunk is dead code during the comparison.
    for sample in samples:
        assert sample.base.n_windows > 0
        assert sample.base.n_adjacency > 0
        assert sample.base.n_radius > 0
        assert bool((sample.base.radius_axis_or_neg1 >= 0).any())
        assert bool((sample.base.radius_axis_or_neg1 < 0).any())


def test_the_transformed_game_is_the_same_position_to_the_engine(samples):
    """The premise of every comparison below, checked against the engine."""
    for sample in samples:
        base = sample.legal_qr[0]
        for t in TRANSFORMS:
            moved = {tuple(qr) for qr in transform_coords(t, base).tolist()}
            assert moved == {tuple(qr) for qr in sample.legal_qr[t].tolist()}
            assert sample.graphs[t].n_cells == sample.base.n_cells
            assert sample.graphs[t].n_windows == sample.base.n_windows
            assert sample.graphs[t].n_legal == sample.base.n_legal
            assert sample.graphs[t].n_adjacency == sample.base.n_adjacency
            assert sample.graphs[t].n_radius == sample.base.n_radius


def test_the_legal_order_really_is_permuted_by_the_transform(samples):
    """§31.1 would be vacuous if the engine's legal order were D6-stable."""
    scrambled = 0
    for sample in samples:
        for t in TRANSFORMS:
            permutation = sample.legal[t]
            # A permutation, exactly once onto each row.
            assert np.array_equal(np.sort(permutation), np.arange(len(permutation)))
            if not np.array_equal(permutation, np.arange(len(permutation))):
                scrambled += 1
    assert scrambled > 0, "the legal-move correspondence is the identity everywhere"


def test_the_builder_carries_its_own_tables_by_the_correspondence(samples):
    """§31.4's premise on the input side: what the model reads is invariant.

    Everything the trunk is given about a cell or a window is either a D6
    invariant — occupancy, legality, the nearest-stone bucket, the
    reversal-canonical pattern class, the status, the numeric block, the joint
    incidence class — or the one structural axis label, which must permute. This
    is an exact integer check, and it is the check that separates "the builder
    and the model are wrong together" from "the model is wrong".
    """
    for sample in samples:
        base = sample.base
        for t in TRANSFORMS:
            image, cells, windows = sample.graphs[t], sample.cells[t], sample.windows[t]
            permutation = np.array(axis_permutation(t), dtype=np.int64)

            for field in ("cell_occupancy", "cell_is_legal", "cell_nearest_bucket"):
                assert np.array_equal(
                    getattr(image, field)[cells], getattr(base, field)
                ), f"{sample.name} t={t}: {field}"
            for field in ("window_pattern_class", "window_status"):
                assert np.array_equal(
                    getattr(image, field)[windows], getattr(base, field)
                ), f"{sample.name} t={t}: {field}"
            assert np.allclose(image.window_numeric[windows], base.window_numeric)
            # The one label that moves, and it moves by the induced permutation.
            assert np.array_equal(
                image.window_axis[windows], permutation[base.window_axis]
            )
            # §13.3's position scalars are invariants of the board (§31.3).
            assert np.array_equal(image.global_numeric, base.global_numeric)
            assert image.phase_id == base.phase_id
            assert image.moves_remaining == base.moves_remaining

            # The incidence class is a joint (pattern, slot) orbit, so it
            # survives a transform that reverses the window's slot order; the
            # slot itself does not, so the row is compared as a multiset.
            for row_base, row_image in zip(
                base.window_incidence_class, image.window_incidence_class[windows]
            ):
                assert sorted(row_base.tolist()) == sorted(row_image.tolist())
            # And every slot's cell is the image of the base slot's cell, up to
            # that same reversal.
            for row_base, row_image in zip(
                cells[base.window_cell_index], image.window_cell_index[windows]
            ):
                assert sorted(row_base.tolist()) == sorted(row_image.tolist())


def test_the_legal_correspondence_agrees_with_the_cell_correspondence(samples):
    """§31.1: a policy row, its action's cell node, and the transform agree."""
    for sample in samples:
        base = sample.base
        assert (base.legal_to_cell_index >= 0).all()
        # The engine's legal order is what the builder stored, not a re-sort.
        assert np.array_equal(base.cell_qr[base.legal_to_cell_index], sample.legal_qr[0])
        for t in TRANSFORMS:
            image = sample.graphs[t]
            # Row `legal[j]` of the image's policy output is base row `j`, and
            # the cell it names is the image of base row `j`'s cell.
            assert np.array_equal(
                image.legal_to_cell_index[sample.legal[t]],
                sample.cells[t][base.legal_to_cell_index],
            )


# --------------------------------------------------------------------------
# §31.3 to §31.7: the model


def _failures(
    deviations: dict[str, dict[str, tuple[float, float]]],
    *,
    families: tuple[str, ...],
    stream: str,
) -> list[str]:
    """Every node-state check of these families and this stream that is over."""
    return [
        f"{name}: {check}: {deviation:.3e} > {allowed:.3e}"
        for name, measured in deviations.items()
        for check, (deviation, allowed) in sorted(measured.items())
        if not check.startswith(("scalar.", "policy."))
        and check.split(".")[1] in families
        and check.endswith(f".{stream}")
        and deviation > allowed
    ]


def test_invariant_cell_and_window_states_map_unchanged(deviations):
    """§31.4, at every site the debug forward exposes."""
    failures = _failures(deviations, families=("cell", "window"), stream="inv")
    assert not failures, "\n".join(failures)


def test_cell_and_window_axis_states_permute_by_the_induced_permutation(deviations):
    """§31.5, at every site."""
    failures = _failures(deviations, families=("cell", "window"), stream="axis")
    assert not failures, "\n".join(failures)


def test_invariant_latents_remain_invariant(deviations):
    """§31.6."""
    failures = _failures(deviations, families=("latent",), stream="inv")
    assert not failures, "\n".join(failures)


def test_axis_latents_permute_correctly(deviations):
    """§31.7."""
    failures = _failures(deviations, families=("latent",), stream="axis")
    assert not failures, "\n".join(failures)


def test_scalar_state_outputs_are_unchanged(deviations):
    """§31.3: every symmetric readout of every family, at every site."""
    failures = [
        f"{name}: {failure}"
        for name, measured in deviations.items()
        for failure in over_budget(measured, "scalar.")
    ]
    assert not failures, "\n".join(failures)


def test_policy_order_rows_map_through_the_legal_correspondence(deviations):
    """§31.1, as far as a trunk with no policy head can carry it.

    The per-action state a policy head will read is the final cell state
    gathered at ``legal_to_cell_index``. Read that way on both boards, the two
    agree only if the legal-order permutation, the cell node correspondence and
    the equivariance of the trunk all hold together.
    """
    failures = [
        f"{name}: {failure}"
        for name, measured in deviations.items()
        for failure in over_budget(measured, "policy.")
    ]
    assert not failures, "\n".join(failures)


def test_the_measured_drift_stays_well_inside_the_budget(deviations):
    """The tolerance is a floating-point budget, not a place to hide a fault.

    Scatter-order noise is what remains after the law holds exactly, and it is
    asserted to occupy less than half the pinned slack. A change that pushes the
    residual toward the tolerance fails here — before the tolerance quietly
    starts absorbing something typed.
    """
    worst = max(
        (
            (deviation / allowed, f"{name}/{check}", deviation, allowed)
            for name, measured in deviations.items()
            for check, (deviation, allowed) in measured.items()
        ),
    )
    ratio, where, deviation, allowed = worst
    assert ratio < 0.5, (
        f"worst drift {deviation:.3e} at {where} is {ratio:.0%} of its "
        f"{allowed:.3e} budget"
    )


def reassociation_deviations(trunk: StateTrunk, sample: Sample) -> dict[str, float]:
    """The same trunk, the same position, a different tensor layout.

    A position evaluated alone and the same position evaluated as one slot of a
    batch are the same function of the same inputs. Nothing symmetric is
    involved and no correspondence is applied; all that differs is the order in
    which the segment reductions accumulate their rows. Whatever this
    disagreement is, it is arithmetic by construction.
    """
    with torch.no_grad():
        _out, batched = trunk.debug_forward(sample.batch)
    worst: dict[str, float] = {}
    for t in (0, 1, 5):
        with torch.no_grad():
            _out, alone = trunk.debug_forward(collate([sample.graphs[t]], FULL))
        for key, got in alone.items():
            family = key.split(".")[1]
            if family == "latent":
                want = batched[key][t : t + 1]
            elif family == "cell":
                want = _slice(batched[key], sample.batch.cell_offsets, t)
            else:
                want = _slice(batched[key], sample.batch.window_offsets, t)
            worst[key] = max(worst.get(key, 0.0), _deviation(got, want))
    return worst


def test_the_residual_is_the_size_of_pure_reassociation_noise(trunk, samples):
    """The evidence that what remains under the law is arithmetic.

    The residual left by the D6 comparison is measured against the residual
    left by `reassociation_deviations`, which involves no symmetry at all. They
    come out the same size — within a factor of a few, at every depth. A typed
    asymmetry would not do that: a per-absolute-axis parameter is a fixed offset
    that survives any reassociation and shows up in the first comparison and not
    the second, which is exactly what the mutation runs below demonstrate.

    This is what entitles `ATOL`/`RTOL` to be called a floating-point budget
    rather than a tolerance chosen to make something pass.
    """
    for sample in samples:
        law = law_deviations(trunk, sample)
        noise = reassociation_deviations(trunk, sample)
        worst_law = max(deviation for deviation, _ in law.values())
        worst_noise = max(noise.values())
        assert worst_noise > 0.0, f"{sample.name}: reassociation is exact?"
        assert worst_law <= 10 * worst_noise, (
            f"{sample.name}: the D6 residual {worst_law:.3e} is far above the "
            f"{worst_noise:.3e} that pure reassociation of the same forward "
            f"produces — the remainder is not arithmetic"
        )


# --------------------------------------------------------------------------
# §31.10: batched and single-position forwards


def test_batched_and_single_position_forwards_agree(trunk, samples):
    """§31.10: no segment reduction and no softmax leaks across a position."""
    for sample in samples:
        batch = sample.batch
        with torch.no_grad():
            packed = trunk(batch)
        for t in (0,) + TRANSFORMS[:3]:
            with torch.no_grad():
                single = trunk(collate([sample.graphs[t]], FULL))
            for got, want in (
                (single.cells.inv, _slice(packed.cells.inv, batch.cell_offsets, t)),
                (single.cells.axis, _slice(packed.cells.axis, batch.cell_offsets, t)),
                (
                    single.windows.inv,
                    _slice(packed.windows.inv, batch.window_offsets, t),
                ),
                (
                    single.windows.axis,
                    _slice(packed.windows.axis, batch.window_offsets, t),
                ),
                (single.latents.inv, packed.latents.inv[t : t + 1]),
                (single.latents.axis, packed.latents.axis[t : t + 1]),
            ):
                deviation = _deviation(got, want)
                assert deviation <= budget(want), (
                    f"{sample.name} t={t}: {deviation:.3e} > {budget(want):.3e}"
                )




# --------------------------------------------------------------------------
# The comparison's own load-bearing parts


def _swap_two(rows: np.ndarray) -> np.ndarray:
    """The same correspondence with two of its entries exchanged."""
    corrupted = rows.copy()
    if len(corrupted) >= 2:
        first, second = 0, len(corrupted) // 2
        corrupted[[first, second]] = corrupted[[second, first]]
    return corrupted


@pytest.mark.parametrize(
    ("field", "expect"),
    (("cells", ".cell."), ("windows", ".window."), ("legal", "policy.")),
)
def test_the_node_correspondence_is_load_bearing(trunk, dense, field, expect):
    """A wrong correspondence must fail, or a right one proves nothing.

    Every §31.4-§31.7 comparison is "these two tensors agree once the rows are
    matched up". That is only a test if the rows are *distinguishable*: if the
    trunk gave every cell nearly the same state, any correspondence at all
    would satisfy it, and the suite would pass over a model that had thrown the
    board away. Exchanging two entries of each correspondence in turn puts the
    comparison over budget, which is the evidence that the states carry the
    position and that the matching is doing work.
    """
    corrupted = replace(
        dense, **{field: tuple(_swap_two(rows) for rows in getattr(dense, field))}
    )
    measured = law_deviations(trunk, corrupted)
    caught = [
        check for check, (deviation, allowed) in measured.items() if deviation > allowed
    ]
    assert any(expect in check for check in caught), (
        f"exchanging two {field} entries changed no {expect} comparison: the "
        f"states being compared do not distinguish those two nodes"
    )


def test_the_latent_slots_are_distinguishable(trunk, dense):
    """§31.6 and §31.7 compare latents slot for slot, so slots must differ.

    `K_inv` identical invariant latents would make the latent comparison
    invariant under any slot permutation and therefore no test of anything.
    """
    with torch.no_grad():
        out = trunk(dense.batch)
    for slots in (out.latents.inv[0], out.latents.axis[0]):
        for k in range(1, slots.shape[0]):
            assert float((slots[k] - slots[0]).abs().max()) > 1e-3


def test_the_randomised_weights_leave_no_dead_parameter(trunk):
    """`randomise_` must actually reach every parameter (§27's zero inits).

    FiLM's projections and the policy/critic finals are zero at initialisation
    by design; a suite that ran with them still zero would be silent about
    every fault inside them.
    """
    dead = [
        name
        for name, parameter in trunk.named_parameters()
        for values in (parameter.detach(),)
        if float(values.abs().max()) == 0.0 or float(values.float().std()) < 1e-6
    ]
    assert not dead, f"parameters left at a constant: {dead}"


# --------------------------------------------------------------------------
# Two detectors that do not go through the board, and are not limited by fp32
#
# The position law above is the integration detector: it is the only thing that
# can see a wrong axis route, a wrong orbit id, a wrong slot canonicalisation,
# or a node correspondence the builder does not honour, because all of those
# live between the engine and the model rather than inside a module. What it
# cannot do is beat its own arithmetic. Its floor is the fp32 scatter noise of
# four blocks of segment reductions over tens of thousands of edges, which the
# tolerance measures at ~1e-5, so a forbidden construction whose effect on the
# trunk's output is smaller than that is invisible to it however exhaustively
# it is applied. That is a property of floating point, not of the comparison.
#
# The two detectors here have no such floor. Neither runs the board: one reads
# parameter shapes and one exercises single modules on synthetic states in
# fp64. Between them they cover the half of §12.2 that is a *parameter* — which
# is the half the position law is weakest on, since a per-axis parameter can be
# arbitrarily small and still be exactly wrong.


# The only parameters in the trunk entitled to a dimension of size three, keyed
# by the class that owns them and the attribute it holds them under. There are
# exactly two, and neither is an axis: one is an occupancy and one is a phase.
# Anything else with a dimension of AXIS_CHANNELS is §12.2's "different
# weights, biases, norms, or base embeddings for absolute axes" written down.
#
# Keyed by (owner class, attribute) rather than by parameter path, because a
# path is not a property of the thing: wrapping `CellEmbedding` in anything at
# all renames `cell_embedding.occupancy.weight`, and an allowlist that lost its
# match there would report the innocent table and hide the guilty one behind
# the noise.
AXIS_SHAPED_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("CellEmbedding", "occupancy"): "§8.2's EMPTY / OWN / OPP vocabulary",
    ("PhaseFiLM", "embed"): "§13.1's OPENING / FIRST / SECOND phase vocabulary",
}

# `module_violations` dispatches on *shape of signature*, not on class name: a
# module is exercised if it accepts one `EquivariantState`, whoever wrote it.
# These three take something else and are given their arguments by hand.
_TAKES_PHASE = frozenset({"PhaseFiLM"})
_TAKES_DELTAS = frozenset({"EquivariantResidual"})
_TAKES_BARE_TENSOR = frozenset({"LayerScale"})

# The module classes the sweep is expected to reach on a pristine trunk, so its
# reach is pinned rather than whatever the try/except happens to admit.
_EXERCISED = frozenset(
    {
        "AxisMix",
        "AxisPool",
        "EquivariantFFN",
        "EquivariantNorm",
        "EquivariantResidual",
        "LayerScale",
        "PhaseFiLM",
    }
)

# Every other module class the trunk contains, with the reason it is not
# exercised here. The classification is asserted to be complete, so a module
# class added to this package forces a decision rather than slipping past.
_NOT_EXERCISED: dict[str, str] = {
    "StateTrunk": "a container; it is what the position law runs",
    "StateTrunkBlock": "a container; covered by the position law per block",
    "CellEmbedding": "reads the batch, not a state; covered by input.cell",
    "WindowEmbedding": "reads the batch, not a state; covered by input.window",
    "StateLatents": "holds the bases; covered by input.latent and the shape check",
    "LatentPass": "needs ragged streams; covered by the per-block latent sites",
    "CellWindowIncidence": "a container over two RelationGatedMessages",
    "AdjacencyMessage": "a container over one RelationGatedMessage",
    "RadiusMessage": "a container over one RelationGatedMessage",
    "RelationGatedMessage": "needs a typed edge family; the route is the thing "
    "under test and only a real board carries it",
    "_RelationTables": "three embedding tables, no axis behaviour",
    "_PairMLP": "one linear over a concatenation of two *streams*, not channels",
    "Linear": "acts on the trailing width; a leaf",
    "LayerNorm": "acts on the trailing width; a leaf",
    "Embedding": "a leaf; its shape is what the structural check reads",
    "Dropout": "elementwise, and off in eval",
    "Sequential": "a container",
    "ModuleList": "a container",
    "SiLU": "elementwise",
    "GELU": "elementwise",
    "ReLU": "elementwise",
}

# fp64 dense arithmetic on seven rows: the residual here is a few ulps, so the
# module sweep sees a break five orders of magnitude below what the position
# law can.
MODULE_TOLERANCE = 1e-10


def structural_violations(trunk: nn.Module) -> list[str]:
    """Parameters whose shape is indexed by an absolute axis (§12.2).

    Exhaustive by construction: it reads `named_parameters` rather than a list
    of places to look, so a per-axis table added anywhere in the tree — a new
    module, a graft, a wrapper — is found without the check knowing about it.
    """
    entitled: set[int] = set()
    for _path, module in trunk.named_modules():
        owner = type(module).__name__
        for attribute, child in module.named_children():
            if (owner, attribute) in AXIS_SHAPED_EXCEPTIONS:
                entitled.update(id(p) for p in child.parameters(recurse=False))
    return [
        f"{name} {tuple(parameter.shape)}"
        for name, parameter in trunk.named_parameters()
        if AXIS_CHANNELS in tuple(parameter.shape) and id(parameter) not in entitled
    ]


_MODULE_ROWS = 7
_MODULE_PERMUTATION = (1, 2, 0)


def _exercise_module(module: nn.Module, generator) -> tuple[object, object] | None:
    """``(output on permuted input, output on the input)``, or ``None``.

    ``None`` means the module does not take an `EquivariantState` and this
    sweep has nothing to say about it — a container, a bare ``nn.Linear``, or a
    message module whose subject is a typed edge family rather than a state.
    Dispatch is on the *signature*, not on the class name, so a module the
    trunk substituted in cannot escape the sweep by being unfamiliar.
    """
    import copy

    kind = type(module).__name__
    probe = copy.deepcopy(module).double().eval()

    def state() -> EquivariantState:
        return EquivariantState(
            torch.randn(
                _MODULE_ROWS, FULL.d_inv, generator=generator, dtype=torch.float64
            ),
            torch.randn(
                _MODULE_ROWS,
                AXIS_CHANNELS,
                FULL.d_axis,
                generator=generator,
                dtype=torch.float64,
            ),
        )

    x = state()
    with torch.no_grad():
        if kind in _TAKES_PHASE:
            phase = torch.arange(_MODULE_ROWS, dtype=torch.long) % len(PHASE_IDS)
            return probe(x.permute_axes(_MODULE_PERMUTATION), phase), probe(x, phase)
        if kind in _TAKES_DELTAS:
            delta = state()
            return (
                probe(
                    x.permute_axes(_MODULE_PERMUTATION),
                    delta.inv,
                    permute_axis_channels(delta.axis, _MODULE_PERMUTATION),
                ),
                probe(x, delta.inv, delta.axis),
            )
        if kind in _TAKES_BARE_TENSOR:
            stream = torch.randn(
                _MODULE_ROWS,
                AXIS_CHANNELS,
                int(probe.gamma.shape[0]),
                generator=generator,
                dtype=torch.float64,
            )
            return (
                probe(permute_axis_channels(stream, _MODULE_PERMUTATION)),
                probe(stream),
            )
        try:
            return probe(x.permute_axes(_MODULE_PERMUTATION)), probe(x)
        except Exception:  # noqa: BLE001 — "does not take a state" is the answer
            return None


def _matches(got, want) -> bool:
    pairs = (
        ((got.inv, want.inv), (got.axis, want.axis))
        if isinstance(got, EquivariantState)
        else ((got, want),)
    )
    return all(_deviation(a, b) <= MODULE_TOLERANCE for a, b in pairs)


def _permuted(value):
    """``value`` with its channels carried, or ``None`` when it has no channels.

    A pooled readout is a ``(..., d_axis)`` tensor with no channel dimension at
    all, so "permutes correctly" is not a statement that can be made about it
    and only invariance is left to ask for.
    """
    if isinstance(value, EquivariantState):
        return value.permute_axes(_MODULE_PERMUTATION)
    if isinstance(value, Tensor) and value.ndim >= 2 and value.shape[-2] == AXIS_CHANNELS:
        return permute_axis_channels(value, _MODULE_PERMUTATION)
    return None


def module_violations(trunk: nn.Module) -> list[str]:
    """Modules of the assembled trunk that neither permute nor stay invariant.

    Each module that accepts an `EquivariantState` is deep-copied to fp64,
    handed a synthetic state, and asked for §12.1 directly. The requirement is
    the disjunction the law actually allows: permuting the three channels of
    the input must either permute the output's channels the same way (§12.1 for
    a state-valued module) or leave the output alone (§12.3's symmetric pooling
    for a readout). A module that does *neither* has an absolute axis in it,
    whatever it is called and whoever wrote it.

    Why this is not the same test as `tests/act/test_act_equivariant.py`. That
    file builds a fresh instance of each class and checks it. This walks the
    module tree of the trunk the model actually assembles, so it also sees a
    module the *trunk* got wrong — a substituted variant, a wrapper, a class
    with no unit test yet — and it is not limited by fp32, because nothing here
    touches a segment reduction.
    """
    generator = torch.Generator().manual_seed(20260806)
    failures: list[str] = []
    for name, module in trunk.named_modules():
        kind = type(module).__name__
        if kind in _NOT_EXERCISED:
            continue
        exercised = _exercise_module(module, generator)
        if exercised is None:
            continue
        got, want = exercised
        references = [
            reference for reference in (_permuted(want), want) if reference is not None
        ]
        if any(_matches(got, reference) for reference in references):
            continue
        deviation = min(
            max(_deviation(a, b) for a, b in _pairs(got, reference))
            for reference in references
        )
        failures.append(
            f"{name or '<trunk>'} [{kind}] is neither equivariant nor invariant "
            f"under a channel permutation: {deviation:.3e} > {MODULE_TOLERANCE:.0e}"
        )
    return failures


def _pairs(got, want):
    if isinstance(got, EquivariantState):
        return ((got.inv, want.inv), (got.axis, want.axis))
    return ((got, want),)


def exercised_module_classes(trunk: nn.Module) -> set[str]:
    """Which module classes `module_violations` actually reached."""
    generator = torch.Generator().manual_seed(1)
    return {
        type(module).__name__
        for module in trunk.modules()
        if type(module).__name__ not in _NOT_EXERCISED
        and _exercise_module(module, generator) is not None
    }


def test_no_parameter_of_the_trunk_is_shaped_by_an_absolute_axis(trunk):
    """§12.2's parameter half, read straight off the module tree.

    This is the check the position law cannot beat: it holds at every
    magnitude, including one too small to move an fp32 forward at all.
    """
    assert not structural_violations(trunk), "\n".join(structural_violations(trunk))
    # The check is only meaningful if `AXIS_CHANNELS` is not also a width the
    # model happens to use, or every parameter of that width would be reported.
    widths = {FULL.d_inv, FULL.d_axis, FULL.d_rel, FULL.num_heads, FULL.ffn_mult}
    assert AXIS_CHANNELS not in widths
    # And every declared exception must still name something, or it is a stale
    # hole standing open for whatever grows into it.
    present = {
        (type(module).__name__, attribute)
        for _path, module in trunk.named_modules()
        for attribute, _child in module.named_children()
    }
    stale = set(AXIS_SHAPED_EXCEPTIONS) - present
    assert not stale, f"declared exceptions matching nothing: {sorted(stale)}"


def test_every_module_class_in_the_trunk_is_classified(trunk):
    """`module_violations` skips nothing by omission, and reaches what it claims.

    The sweep decides what to exercise from the signature, so its reach is a
    fact rather than a declaration; ``_EXERCISED`` records that fact and this
    test pins it. A module class that stops being reachable — because its
    signature changed, or because it was replaced — shows up here as a shrunken
    reach rather than as a quietly narrower audit.
    """
    kinds = {type(module).__name__ for module in trunk.modules()}
    unclassified = kinds - _EXERCISED - set(_NOT_EXERCISED)
    assert not unclassified, (
        f"module class(es) {sorted(unclassified)} are neither exercised by "
        f"module_violations nor declared unexercised with a reason"
    )
    assert exercised_module_classes(trunk) == _EXERCISED


def test_every_exercised_module_commutes_with_a_channel_permutation(trunk):
    """§12.1 module by module, at fp64, over the trunk the model assembles."""
    failures = module_violations(trunk)
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------
# §12.2's forbidden constructions, and the proof that the suite catches them


class _PerAxisBias(nn.Module):
    """§12.2, second bullet: a learned bias per absolute axis channel.

    Wraps a stage of the trunk and adds a ``(3, d_axis)`` parameter to its axis
    output. It is the cheapest way to write the mistake — one broadcast of the
    wrong shape — and the hardest to see, because the model still trains, still
    round-trips, and still passes every shape and invariant test in this
    package.
    """

    def __init__(self, inner: nn.Module, d_axis: int, magnitude: float, generator) -> None:
        super().__init__()
        self.inner = inner
        self.bias = nn.Parameter(
            torch.randn(AXIS_CHANNELS, d_axis, generator=generator) * magnitude
        )

    def forward(self, state: EquivariantState) -> EquivariantState:
        out = self.inner(state)
        return EquivariantState(out.inv, out.axis + self.bias.to(out.axis.dtype))


class _FixedOrderChannelMLP(nn.Module):
    """§12.2, first bullet: channels 0/1/2 concatenated into an unconstrained MLP.

    Replaces an `AxisMix` — the module whose whole job is to let the channels
    talk without naming them — with the construction the spec forbids: one flat
    ``3 * d_axis`` vector through a dense layer and back.
    """

    def __init__(self, cfg: MantisACTConfig, magnitude: float, generator) -> None:
        super().__init__()
        self.d_axis = cfg.d_axis
        width = AXIS_CHANNELS * cfg.d_axis
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        with torch.no_grad():
            for layer in (self.mlp[0], self.mlp[2]):
                layer.weight.copy_(
                    torch.randn(layer.weight.shape, generator=generator) * magnitude
                )
                layer.bias.copy_(
                    torch.randn(layer.bias.shape, generator=generator) * magnitude
                )

    def forward(self, state: EquivariantState) -> EquivariantState:
        axis = state.require_axis("the fixed-order channel mutation")
        flat = axis.reshape(*axis.shape[:-2], AXIS_CHANNELS * self.d_axis)
        delta = self.mlp(flat).reshape(axis.shape)
        return EquivariantState(state.inv, axis + delta)


class _AbsoluteAxisEmbedding(nn.Module):
    """§12.2, third bullet: absolute axis identity as an embedding lookup.

    Wraps an initial embedding and adds ``E[channel]`` to the axis stream. §8.2
    gives a cell's three channels one shared base precisely because a bare cell
    has no direction of its own; this gives each channel an identity the board
    cannot move.
    """

    def __init__(self, inner: nn.Module, d_axis: int, magnitude: float, generator) -> None:
        super().__init__()
        self.inner = inner
        self.table = nn.Embedding(AXIS_CHANNELS, d_axis)
        with torch.no_grad():
            self.table.weight.copy_(
                torch.randn(self.table.weight.shape, generator=generator) * magnitude
            )

    def forward(self, batch: PackedACTBatch) -> EquivariantState:
        state = self.inner(batch)
        rows = self.table(torch.arange(AXIS_CHANNELS, device=state.inv.device))
        return EquivariantState(state.inv, state.axis + rows.to(state.inv.dtype))


class _PerChannelScoredPool(nn.Module):
    """§12.5's stated negative control: a per-channel bias on the pooling score.

    `AxisPool`'s learned mode is invariant *because the scores and the channels
    permute together*. One bias per absolute channel breaks the pairing while
    every weight in the module stays shared, so the axis stream itself remains
    perfectly equivariant and only the invariant quantity read off it moves.
    That is the shape of break most likely to survive a suite that watches the
    axis channels and nothing else. The arithmetic is `AxisPool`'s own, plus the
    bias.
    """

    def __init__(self, inner: AxisPool, magnitude: float, generator) -> None:
        super().__init__()
        if inner.mode != "learned_attention":
            raise ValueError("this control only applies to the learned pool (§12.5)")
        self.inner = inner
        self.bias = nn.Parameter(
            torch.randn(AXIS_CHANNELS, 1, generator=generator) * magnitude
        )

    def forward(self, state: EquivariantState) -> Tensor:
        axis = state.require_axis("the per-channel score-bias mutation")
        scores = self.inner.score(
            torch.tanh(
                self.inner.from_axis(axis) + self.inner.from_inv(state.inv).unsqueeze(-2)
            )
        )
        weight = (scores + self.bias.to(scores.dtype)).float().softmax(dim=-2)
        return (weight * axis.float()).sum(dim=-2).to(axis.dtype)


class _ConstantAxisRoute(nn.Module):
    """Not §12.2, but the break this architecture is most exposed to.

    §12.3 permits "route line messages into the structural native axis" — and
    everything rests on the route being *structural*. A family routed into a
    fixed channel instead holds no per-axis parameter, concatenates nothing,
    and looks in every code review like the allowed construction. Only a
    transform-and-compare separates the two.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self, edges: TypedEdges, source: EquivariantState, destination: EquivariantState
    ) -> EquivariantState:
        rerouted = TypedEdges(
            src=edges.src,
            dst=edges.dst,
            relation=edges.relation,
            axis=torch.zeros_like(edges.axis),
            n_src=edges.n_src,
            n_dst=edges.n_dst,
            num_relations=edges.num_relations,
            dst_sorted=edges.dst_sorted,
            fully_routed=True,  # every row now routes into channel 0
            name=f"{edges.name} (rerouted)",
        )
        return self.inner(rerouted, source, destination)


@dataclass(frozen=True)
class Mutation:
    """One forbidden construction, where it is planted, and what must catch it.

    ``detectors`` names the checks required to fire, out of the suite's three:
    ``structural`` (parameter shapes), ``module`` (the fp64 per-module sweep),
    and ``law`` (the end-to-end position comparison). Declaring it per mutation
    rather than asking for "something fired" is what turns the mutation runs
    into a coverage matrix: a detector that stops catching what it used to is a
    failure here, and so is one that starts catching something it should not
    be able to see.

    ``first_site`` is the earliest site `StateTrunk.debug_forward` exposes past
    the mutation's own position, and applies to the ``law`` detector only.
    Requiring the catch *there* rather than merely somewhere makes the suite a
    localiser: a break in block 2 that surfaced only in the final norms would
    be nearly as hard to find as one nobody detected.
    """

    clause: str
    where: str
    detectors: frozenset[str]
    apply: object
    first_site: str = ""
    scaled: bool = True


def _mutate_per_axis_bias(trunk: StateTrunk, magnitude: float, generator) -> None:
    block = trunk.blocks[2]
    block.cell_ffn = _PerAxisBias(block.cell_ffn, FULL.d_axis, magnitude, generator)


def _mutate_fixed_order_concat(trunk: StateTrunk, magnitude: float, generator) -> None:
    trunk.blocks[1].window_mix = _FixedOrderChannelMLP(FULL, magnitude, generator)


def _mutate_absolute_axis_embedding(
    trunk: StateTrunk, magnitude: float, generator
) -> None:
    trunk.cell_embedding = _AbsoluteAxisEmbedding(
        trunk.cell_embedding, FULL.d_axis, magnitude, generator
    )


def _mutate_per_channel_latent_base(
    trunk: StateTrunk, magnitude: float, generator
) -> None:
    stack = trunk.latents
    extra = nn.Parameter(
        torch.randn(
            FULL.num_axis_latents, AXIS_CHANNELS, FULL.d_axis, generator=generator
        )
        * magnitude
    )
    stack.register_parameter("mutation_per_channel_base", extra)
    original = stack.initial

    def initial(global_numeric: Tensor) -> LatentState:
        state = original(global_numeric)
        return LatentState(inv=state.inv, axis=state.axis + extra.to(state.axis.dtype))

    stack.initial = initial


def _mutate_per_channel_pool_score(
    trunk: StateTrunk, magnitude: float, generator
) -> None:
    latent_pass = trunk.latents[0]
    latent_pass.pool_src_axis = _PerChannelScoredPool(
        latent_pass.pool_src_axis, magnitude, generator
    )


def _mutate_constant_axis_route(trunk: StateTrunk, magnitude: float, generator) -> None:
    incidence = trunk.blocks[0].incidence
    incidence.to_windows = _ConstantAxisRoute(incidence.to_windows)


MUTATIONS: dict[str, Mutation] = {
    "fixed_order_concat": Mutation(
        clause="§12.2: concatenate channels 0/1/2 in fixed order into an MLP",
        where="block 1's window AxisMix",
        # Its weights are (3 d_axis, 3 d_axis): no dimension names an axis, so
        # the shape check is blind to it and the other two must carry it.
        detectors=frozenset({"module", "law"}),
        first_site="block1.window.axis",
        apply=_mutate_fixed_order_concat,
    ),
    "per_axis_bias": Mutation(
        clause="§12.2: different weights, biases, norms, or bases for absolute axes",
        where="block 2's cell FFN",
        detectors=frozenset({"structural", "module", "law"}),
        first_site="block2.cell.axis",
        apply=_mutate_per_axis_bias,
    ),
    "absolute_axis_embedding": Mutation(
        clause="§12.2: absolute axis identity as an embedding lookup",
        where="the cell embedding",
        detectors=frozenset({"structural", "law"}),
        first_site="input.cell.axis",
        apply=_mutate_absolute_axis_embedding,
    ),
    "per_channel_latent_base": Mutation(
        clause="§17.1: one axis-latent base per channel instead of one replicated",
        where="the state latent bases",
        detectors=frozenset({"structural", "law"}),
        first_site="block0.latent.axis",
        apply=_mutate_per_channel_latent_base,
    ),
    "per_channel_pool_score": Mutation(
        clause="§12.5: a per-channel bias on the invariant pool's score",
        where="block 0's latent read pool",
        # The one construction the position law does *not* catch at this
        # magnitude, and the reason the other two detectors exist. It enters
        # the model only through one key of an attention over a thousand rows,
        # so a 1e-3 bias moves the trunk's output by about 1e-5 — under the
        # fp32 floor the law comparison sits on. Its shape and its module both
        # give it away immediately.
        detectors=frozenset({"structural", "module"}),
        apply=_mutate_per_channel_pool_score,
    ),
    "constant_axis_route": Mutation(
        clause="§12.3: a line message routed into a fixed channel, not the native one",
        where="block 0's cells-into-windows incidence",
        # No parameter and no module of its own: routing is a property of the
        # edge family, so only a real board can show it. This is the mutation
        # that justifies the position law's existence beside the other two.
        detectors=frozenset({"law"}),
        first_site="block0.window.axis",
        apply=_mutate_constant_axis_route,
        scaled=False,
    ),
}


def mutated_trunk(name: str, magnitude: float) -> StateTrunk:
    """A trunk identical to the fixture's, with one forbidden construction in it."""
    trunk = fresh_trunk()
    generator = torch.Generator().manual_seed(sum(ord(c) for c in name))
    MUTATIONS[name].apply(trunk, magnitude, generator)
    return trunk.eval()


def fired(
    trunk: StateTrunk, sample: Sample
) -> tuple[dict[str, list[str]], dict[str, tuple[float, float]]]:
    """Which of the suite's three detectors fire on ``trunk``, and on what."""
    measured = law_deviations(trunk, sample)
    return {
        "structural": structural_violations(trunk),
        "module": module_violations(trunk),
        "law": [
            f"{check}: {deviation:.3e} > {allowed:.3e}"
            for check, (deviation, allowed) in sorted(measured.items())
            if deviation > allowed
        ],
    }, measured


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_the_suite_catches_each_forbidden_construction(dense, name):
    """The detector is known to be able to fail (CLAUDE.md, §12.2).

    Each construction is introduced into an otherwise identical trunk and run
    through every detector the suite has. The assertion is the mutation's own
    declared coverage, exactly — not "something fired":

    1. every detector it declares must fire, so a detector silently losing its
       reach is a failure rather than a shrug;
    2. no detector it does not declare may fire, so the record stays an honest
       statement of what each detector can and cannot see;
    3. for the position law, the catch must land at the mutation's own site and
       with a wide margin, since a catch that only just cleared the tolerance
       would say the suite is at the edge of missing it.
    """
    mutation = MUTATIONS[name]
    detectors, measured = fired(mutated_trunk(name, MUTATION_MAGNITUDE), dense)
    for detector, findings in detectors.items():
        expected = detector in mutation.detectors
        assert bool(findings) == expected, (
            f"{name} ({mutation.clause}) at {mutation.where}: the {detector!r} "
            f"detector {'found nothing' if expected else 'fired unexpectedly'}."
            f" Findings: {findings[:3]}"
        )
    if "law" not in mutation.detectors:
        return
    deviation, allowed = measured[mutation.first_site]
    assert deviation > allowed, (
        f"{name} was caught, but not at {mutation.first_site} where it is "
        f"planted: {deviation:.3e} <= {allowed:.3e}"
    )
    ratio = max(
        deviation / allowed
        for deviation, allowed in measured.values()
        if deviation > allowed
    )
    assert ratio > 100, f"{name} is only {ratio:.1f}x over the tolerance"


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_a_hundredfold_smaller_mutation_is_still_caught(dense, name):
    """How much margin the suite has: the smallest break it is asked to see.

    At a hundredth of `MUTATION_MAGNITUDE` — 1e-5, the size of the arithmetic
    noise itself — the parameter-shape and per-module detectors are unaffected,
    because neither compares floats against a board. Whether the position law
    still sees it is measured rather than asserted: what it can see is bounded
    by fp32, and the point of the other two detectors is that the bound does
    not matter.
    """
    mutation = MUTATIONS[name]
    if not mutation.scaled:
        pytest.skip(f"{name} is structural and has no magnitude to shrink")
    tiny = MUTATION_MAGNITUDE / 100
    detectors, _measured = fired(mutated_trunk(name, tiny), dense)
    assert any(detectors.values()), (
        f"{name} at {mutation.where} slipped through every detector at "
        f"magnitude {tiny}"
    )
    # The two magnitude-independent detectors keep exactly their reach.
    for detector in ("structural", "module"):
        assert bool(detectors[detector]) == (detector in mutation.detectors)


def test_an_equivariant_change_of_the_same_size_is_not_caught(dense):
    """The control for the mutations: the suite fires on asymmetry, not change.

    One *shared* bias of the same magnitude, broadcast identically to the three
    channels, is a legal construction (§12.3) and moves the model's outputs by
    about as much as the per-axis mutation does. It must pass. Without this the
    mutation runs would be consistent with a suite that fails on any
    perturbation at all.
    """
    trunk = fresh_trunk()
    generator = torch.Generator().manual_seed(4242)
    shared = torch.randn(FULL.d_axis, generator=generator) * MUTATION_MAGNITUDE

    class _SharedBias(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner
            self.bias = nn.Parameter(shared.clone())

        def forward(self, state: EquivariantState) -> EquivariantState:
            out = self.inner(state)
            return EquivariantState(out.inv, out.axis + self.bias.to(out.axis.dtype))

    block = trunk.blocks[2]
    block.cell_ffn = _SharedBias(block.cell_ffn)
    detectors, _measured = fired(trunk.eval(), dense)
    assert not any(detectors.values()), (
        "a channel-shared bias is §12.3-legal but was reported: "
        + repr({k: v[:3] for k, v in detectors.items() if v})
    )

    # And it really did move the model, so the pass above is not vacuous.
    with torch.no_grad():
        before = fresh_trunk()(dense.batch).cells.axis
        after = trunk(dense.batch).cells.axis
    assert float((before - after).abs().max()) > 100 * budget(before)


def test_the_mutations_are_small_beside_the_weights_they_sit_among(trunk):
    """The breaks are not detected by being large: they are not large."""
    scales = [
        float(parameter.detach().abs().mean())
        for name, parameter in trunk.named_parameters()
        if "axis" in name
    ]
    assert scales
    assert MUTATION_MAGNITUDE < min(scales) / 10
    # And still well clear of the noise floor the tolerance describes.
    assert MUTATION_MAGNITUDE > ATOL * 10


def test_the_suite_reports_its_own_coverage(deviations):
    """Every family, stream, and site is actually compared, not silently skipped.

    A check that raised no comparison would leave no entry, and an empty
    dictionary satisfies every "no failures" assertion above. This is the
    assertion that the suite ran.
    """
    for name, measured in deviations.items():
        sites = {
            check for check in measured if not check.startswith(("scalar.", "policy."))
        }
        for family in ("cell", "window", "latent"):
            for stream in ("inv", "axis"):
                for site in ("input", "block0", "block3", "final"):
                    assert f"{site}.{family}.{stream}" in sites, (
                        f"{name}: no comparison for {site}.{family}.{stream}"
                    )
        assert {"policy.inv", "policy.axis"} <= set(measured)
        assert any(check.startswith("scalar.") for check in measured)
        # And no comparison was made against an empty tensor.
        assert all(math.isfinite(deviation) for deviation, _ in measured.values())


# --------------------------------------------------------------------------
# The same law on the device, where the fused kernels live (§31, §36)
#
# The segment-message, latent-attention, post-row, and class-embedding-backward
# families dispatch to hand-written implementations only on supported CUDA
# inputs, so a host-only suite has never put §31 to those kernels at all — and
# a kernel that read a fixed absolute axis channel instead of the row's own
# would pass it. The device arm uses the same detectors and samples, with every
# eligible forward/backward launch counted so a silent reference fallback
# cannot pass as the device run the CPU arm already covers.


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused kernels need a CUDA device"
)


@contextmanager
def counted_fused_dispatch(*, include_post_rows: bool = False):
    """Count eligible calls and successful fused forward/backward launches.

    Eligibility alone is not acceptance: each dispatcher consults a failure
    cache after ``_supported`` and may silently use its torch reference. Count a
    launch only after the launch helper returns, so ``eligible == launched`` is
    the assertion that no eligible call fell back.
    """
    families = {
        "segment_message": (
            segment_message,
            ("_launch_forward", "_launch_backward"),
        ),
        "latent_attention": (
            latent_attention,
            (
                "_launch_read",
                "_launch_read_backward",
                "_launch_broadcast",
                "_launch_broadcast_backward",
            ),
        ),
    }
    if include_post_rows:
        families["post_rows"] = (
            post_rows,
            (
                "_launch_gather",
                "_launch_gather_backward",
                "_launch_row_gate",
                "_launch_row_gate_backward",
            ),
        )
        families["class_embedding"] = (
            class_embedding,
            ("_launch_backward",),
        )

    counts = {
        name: {
            "eligible": 0,
            "launched": 0,
            **{launch: 0 for launch in family_launches},
        }
        for name, (_module, family_launches) in families.items()
    }
    supported = {
        name: module._supported for name, (module, _launches) in families.items()
    }
    launches = {
        (name, launch): getattr(module, launch)
        for name, (module, family_launches) in families.items()
        for launch in family_launches
    }

    def counting_supported(name, original):
        def wrapped(*args, **kwargs):
            accepted = original(*args, **kwargs)
            counts[name]["eligible"] += int(bool(accepted))
            return accepted

        return wrapped

    def counting_launch(name, launch, original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            counts[name]["launched"] += 1
            counts[name][launch] += 1
            return result

        return wrapped

    for name, (module, family_launches) in families.items():
        module._supported = counting_supported(name, supported[name])
        for launch in family_launches:
            setattr(
                module,
                launch,
                counting_launch(name, launch, launches[(name, launch)]),
            )
    try:
        yield counts
    finally:
        for name, (module, family_launches) in families.items():
            module._supported = supported[name]
            for launch in family_launches:
                setattr(module, launch, launches[(name, launch)])


def assert_successful_fused_dispatch(
    counts: dict[str, dict[str, int]],
    *families: str,
    require_backward: bool = False,
) -> None:
    """Require every eligible call to have returned from its fused launch.

    The D6 comparisons run under ``no_grad`` and therefore require every
    forward helper.  The full-model device fixture additionally performs one
    bounded training pass and sets ``require_backward=True`` so every ordered
    segment/sentinel backward is load-bearing.
    """
    modules = {
        "segment_message": segment_message,
        "latent_attention": latent_attention,
        "post_rows": post_rows,
        "class_embedding": class_embedding,
    }
    for name in families:
        eligible = counts[name]["eligible"]
        launched = counts[name]["launched"]
        assert eligible == launched and launched > 0, counts
        launch_counts = {
            launch: count
            for launch, count in counts[name].items()
            if launch.startswith("_launch_")
        }
        required_launches = {
            launch: count
            for launch, count in launch_counts.items()
            if require_backward or "backward" not in launch
        }
        assert required_launches and all(
            count > 0 for count in required_launches.values()
        ), counts
        module = modules[name]
        for cache_name in (
            "_FAILED_SHAPES",
            "_FAILED_FORWARD_SHAPES",
            "_FAILED_BACKWARD_SHAPES",
        ):
            cache = getattr(module, cache_name, None)
            if cache is not None:
                assert not cache, {
                    "family": name,
                    "cache": cache_name,
                    "failures": dict(cache),
                }


def on_device(sample: Sample, device: str = "cuda") -> Sample:
    return replace(sample, batch=sample.batch.to(device))


def action_law_deviations(
    model: MantisACT, sample: Sample
) -> dict[str, tuple[float, float]]:
    """§31 over the action stack and its full-model policy/critic outputs."""
    with torch.no_grad():
        output, tensors = model.debug_forward(
            sample.batch,
            mass_floor=None,
            capture=("action.state", "action.latent"),
        )
    batch = sample.batch
    device = batch.legal_offsets.device
    into: dict[str, tuple[float, float]] = {}

    def index(array: np.ndarray) -> Tensor:
        return torch.from_numpy(array).to(device)

    for t in TRANSFORMS:
        permutation = axis_permutation(t)
        legal = index(sample.legal[t])

        # Action states are already reduced from the eighteen post-placement
        # rows to one row per engine-ordered legal action. The independently
        # built legal correspondence is therefore their complete node map.
        for stream in ("inv", "axis"):
            tensor = tensors[f"action.state.{stream}"]
            want = _slice(tensor, batch.legal_offsets, 0)
            got = _slice(tensor, batch.legal_offsets, t).index_select(0, legal)
            if stream == "axis":
                want = permute_axis_channels(want, permutation)
            _record(into, f"action.state.{stream}", got, want)

        # §21 action latents are invariant and position-indexed, so no node
        # correspondence or channel permutation is entitled here.
        latent = tensors["action.latent.inv"]
        _record(into, "action.latent.inv", latent[t], latent[0])

        # The public full-model rows stay in engine legal order. Checking both
        # raw heads and their fp32 critic compositions carries the same legal
        # correspondence through the complete action path.
        for name in ("policy_logits", "critic_logits", "q_value", "q_score"):
            tensor = getattr(output, name)
            want = _slice(tensor, batch.legal_offsets, 0)
            got = _slice(tensor, batch.legal_offsets, t).index_select(0, legal)
            _record(into, f"output.{name}", got, want)

    return into


@pytest.fixture(scope="module")
def cuda_law(
    samples,
) -> tuple[
    dict[str, dict[str, tuple[float, float]]], dict[str, dict[str, int]]
]:
    if not torch.cuda.is_available():
        pytest.skip("the fused kernels need a CUDA device")
    trunk = fresh_trunk().cuda()
    with counted_fused_dispatch() as counts:
        measured = {
            sample.name: law_deviations(trunk, on_device(sample)) for sample in samples
        }
        torch.cuda.synchronize()
    return measured, {name: dict(family) for name, family in counts.items()}


@pytest.fixture(scope="module")
def cuda_action_law(
    dense,
) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, int]]]:
    if not torch.cuda.is_available():
        pytest.skip("the fused kernels need a CUDA device")
    torch.manual_seed(SEED)
    model = randomise_(MantisACT(FULL), SEED).cuda()
    with counted_fused_dispatch(include_post_rows=True) as counts:
        measured = action_law_deviations(model, on_device(dense))
        # The law itself is intentionally no-grad: retaining the twelve-image
        # dense orbit for backward would turn a symmetry test into a memory
        # stress test. One independently collated base position still traverses
        # the identical default full model and makes every recompute backward,
        # ordered class reduction, and planned sentinel gather load-bearing.
        training_batch = collate([dense.base], FULL).to("cuda")
        model.zero_grad(set_to_none=True)
        output = model(training_batch, mass_floor=None)
        loss = (
            output.policy_logits.float().square().mean()
            + output.critic_logits.float().square().mean()
        )
        assert torch.isfinite(loss)
        loss.backward()
        torch.cuda.synchronize()
    return measured, {name: dict(family) for name, family in counts.items()}


@requires_cuda
def test_the_position_law_holds_where_the_fused_kernels_run(cuda_law, deviations):
    """§31 on the device, over every comparison the host arm makes.

    The assertion that this is not the host arm again under another name is the
    acceptance count: every trunk kernel family must have taken the fused path,
    and the set of comparisons must be exactly the host arm's, so neither a
    silent fallback nor a shortened comparison can pass as a device run.
    """
    measured, counts = cuda_law
    assert_successful_fused_dispatch(
        counts,
        "segment_message",
        "latent_attention",
    )
    assert {name: set(checks) for name, checks in measured.items()} == {
        name: set(checks) for name, checks in deviations.items()
    }
    failures = [
        f"{name}: {failure}"
        for name, checks in measured.items()
        for failure in over_budget(checks, "")
    ]
    assert not failures, "\n".join(failures)


@requires_cuda
def test_the_device_drift_stays_well_inside_the_budget(cuda_law):
    """The fused kernels reassociate differently; the slack must still hold."""
    measured, _counts = cuda_law
    ratio, where, deviation, allowed = max(
        (deviation / allowed, f"{name}/{check}", deviation, allowed)
        for name, checks in measured.items()
        for check, (deviation, allowed) in checks.items()
    )
    assert ratio < 0.5, (
        f"worst device drift {deviation:.3e} at {where} is {ratio:.0%} of its "
        f"{allowed:.3e} budget"
    )


@requires_cuda
def test_the_action_law_holds_through_the_full_model_on_the_device(cuda_action_law):
    """§31 and a training pass reach every default fused forward/backward."""
    measured, counts = cuda_action_law
    assert_successful_fused_dispatch(
        counts,
        "segment_message",
        "latent_attention",
        "post_rows",
        "class_embedding",
        require_backward=True,
    )
    expected = {
        "action.state.inv",
        "action.state.axis",
        "action.latent.inv",
        "output.policy_logits",
        "output.critic_logits",
        "output.q_value",
        "output.q_score",
    }
    assert set(measured) == expected
    failures = over_budget(measured, "")
    assert not failures, "\n".join(failures)


@requires_cuda
@pytest.mark.parametrize(
    "name", sorted(n for n, m in MUTATIONS.items() if "law" in m.detectors)
)
def test_the_position_law_still_catches_each_construction_on_the_device(dense, name):
    """The device arm's negative control: it is known to be able to fail.

    Every construction the position law catches on the host is introduced into
    a trunk that then runs on the device, under the fused kernels. A device
    test that could not fail would be the same defect as the missing device arm
    itself, one level up.
    """
    mutation = MUTATIONS[name]
    trunk = mutated_trunk(name, MUTATION_MAGNITUDE).cuda()
    with counted_fused_dispatch() as counts:
        measured = law_deviations(trunk, on_device(dense))
    assert_successful_fused_dispatch(
        counts,
        "segment_message",
        "latent_attention",
    )

    over = {
        check: (deviation, allowed)
        for check, (deviation, allowed) in measured.items()
        if deviation > allowed
    }
    assert over, f"{name} ({mutation.clause}) at {mutation.where} was not caught"
    deviation, allowed = measured[mutation.first_site]
    assert deviation > allowed, (
        f"{name} was caught, but not at {mutation.first_site} where it is "
        f"planted: {deviation:.3e} <= {allowed:.3e}"
    )
    ratio = max(deviation / allowed for deviation, allowed in over.values())
    assert ratio > 100, f"{name} is only {ratio:.1f}x over the tolerance"
