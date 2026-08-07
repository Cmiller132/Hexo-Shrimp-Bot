"""§26's packer limit for MantisNet-ACT, and the fitting seam that reads it.

A budget is a claim about cost, so what has to be tested is the claim, not the
arithmetic. Three detectors, none of which the others cover:

- **The identity the cost is built on.** ``ACTChunkCost`` charges a sample
  ``2 * stones + legal``, which is the graph's cell count plus its occupied
  cells only because ``cells == stones + legal`` holds exactly on this game.
  That is a property of Hexo and of ``cell_scope``, not of the cost class, so
  it is asserted against real builds at every ply — if a scope change ever
  breaks it, the budget silently under- or over-charges and nothing else would
  notice.

- **The packer honours the limit and loses nothing.** Every sample lands in
  exactly one chunk, no chunk exceeds the limit or the position cap, and a
  sample too large for any chunk is still fitted as a singleton rather than
  dropped.

- **The seam is real.** A KLENT fitting epoch is run end to end on the ACT
  model over real self-play prefixes: the model builds the batch, the model's
  own law packs it, ``policy_q`` answers, and the optimizer moves parameters
  with finite gradients throughout. MantisNet's ``fit`` is unchanged by any of
  it, which the same test asserts by running the same epoch through it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import hexo_py

from mantisnet.builder import PaddedPairChunkCost
from mantisnet.fitloop import ChunkCost, FitBudgets, pack_chunks
from mantisnet.klent.train import KlentConfig, _budgets, _chunk_cost, _pack, fit
from mantisnet.model import MantisConfig, MantisNet
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.model import MantisACT
from mantisnet.models.mantis_act.packed import (
    ACT_GRAPH_CELL_BUDGET,
    ACTChunkCost,
    collate,
    telemetry,
)

from .test_act_numerics import PLIES, moves, position

SEED = 20260806
FULL = PRESETS["full_act_v4"]

# Small enough to fit a whole fitting epoch into a test, and structurally
# complete: every stage the seam runs is present at these widths.
TINY = MantisACTConfig(
    d_inv=16,
    d_axis=8,
    d_rel=8,
    num_heads=2,
    state_blocks=1,
    action_blocks=1,
    policy_private_blocks=1,
    critic_private_blocks=1,
)


# --------------------------------------------------------------------------
# The identity the cost is built on


@pytest.mark.parametrize("game", (0, 1))
@pytest.mark.parametrize("ply", PLIES)
def test_a_graphs_cells_are_its_stones_plus_its_legal_moves(game, ply):
    """``cells == stones + legal``, on real boards, at every ply.

    ``ACTChunkCost`` charges ``2 * stones + legal`` because that is the cell
    count plus the occupied cells. The second term of that sum is not stored
    anywhere; it is this identity. Every empty cell of a persistent window lies
    within five steps of one of its stones and the legal radius is eight, so a
    window's empty cells are already legal cells and the scopes coincide.
    """
    pos = position(game, ply)
    graph = build(pos, FULL)
    stones = len(pos.stones())
    legal = len(pos.legal_moves())

    assert stones == ply
    assert graph.n_cells == stones + legal
    assert ACTChunkCost([stones], [legal], 1).units(0) == graph.n_cells + stones


def test_the_costs_unit_is_the_quantity_telemetry_reports():
    """The §26 counts the packer limits and the ones a batch reports agree."""
    plies = (21, 61, 121, 161)
    batch = collate([build(position(0, ply), FULL) for ply in plies])
    stats = telemetry(batch)

    cost = ACTChunkCost(
        list(plies),
        [len(position(0, ply).legal_moves()) for ply in plies],
        ACT_GRAPH_CELL_BUDGET,
    )
    charged = sum(cost.units(i) for i in range(len(plies)))
    cells = int(batch.cell_offsets[-1])
    assert charged == cells + sum(plies)
    assert stats["cells_mean"] * len(plies) == pytest.approx(cells)


# --------------------------------------------------------------------------
# The packer under this cost


def test_packing_honours_the_graph_cell_limit_and_loses_nothing():
    rng = np.random.default_rng(0)
    stones = [int(t) for t in rng.integers(1, 400, 200)]
    legal = [int(c) for c in rng.integers(1, 4_000, 200)]
    # Below the fixture's largest sample, so the singleton rule is exercised.
    budget, batch_size = 4_000, 32
    cost = ACTChunkCost(stones, legal, budget)

    chunks = pack_chunks(rng.permutation(len(stones)), batch_size, cost)

    assert isinstance(cost, ChunkCost)
    assert sorted(i for chunk in chunks for i in chunk) == list(range(len(stones)))
    oversized = [i for i in range(len(stones)) if 2 * stones[i] + legal[i] > budget]
    assert oversized, "the fixture must contain samples no chunk can hold"
    for chunk in chunks:
        assert len(chunk) <= batch_size
        units = sum(2 * stones[i] + legal[i] for i in chunk)
        if len(chunk) > 1:
            assert units <= budget
        # Descending pack order: the widest sample opens its chunk.
        assert 2 * stones[chunk[0]] + legal[chunk[0]] == max(
            2 * stones[i] + legal[i] for i in chunk
        )
    for i in oversized:
        assert [i] in chunks


def test_the_act_law_has_no_padded_pair_term():
    """The two architectures' laws are different laws, not one parameterised.

    MantisNet pads its attention, so eight one-cell positions of ply 400 cost
    it a chunk it must split; ACT pads nothing and takes all eight. A packer
    that had kept a quadratic term of its own would split both.
    """
    stones = [400] * 8
    legal = [1] * 8
    act = pack_chunks(range(8), 8, ACTChunkCost(stones, legal, 10_000))
    mantis = pack_chunks(
        range(8),
        8,
        PaddedPairChunkCost([s + 1 for s in stones], legal, 500_000, 10_000),
    )
    assert act == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert len(mantis) > 1
    assert all(len(chunk) * 401 * 401 <= 500_000 for chunk in mantis)


# --------------------------------------------------------------------------
# The model's half of the seam


def test_the_model_supplies_its_own_law_and_its_own_collation():
    torch.manual_seed(SEED)
    act = MantisACT(TINY)
    mantis = MantisNet(MantisConfig(h=16, heads=2, blocks=1, value_bins=5))
    budgets = FitBudgets(pair_budget=7, cell_budget=11, graph_cell_budget=13)

    act_cost = act.chunk_cost([3, 4], [5, 6], budgets)
    assert isinstance(act_cost, ACTChunkCost)
    assert act_cost.units(0) == 11 and act_cost.units(1) == 14
    assert act_cost.accepts(0, 0) is True  # 11 <= 13
    act_cost.take(0)
    assert act_cost.accepts(1, 1) is False  # 11 + 14 > 13

    assert isinstance(mantis.chunk_cost([3, 4], [5, 6], budgets), PaddedPairChunkCost)

    games = [moves(0, 40), moves(1, 40)]
    packed = act.collate_prefixes(games, [21, 31])
    expected = collate(
        [
            build(hexo_py.Position.replay(game[:ply]), TINY)
            for game, ply in zip(games, (21, 31))
        ]
    )
    assert int(packed.position_count) == 2
    assert torch.equal(packed.cell_offsets, expected.cell_offsets)
    assert torch.equal(packed.legal_offsets, expected.legal_offsets)
    assert torch.equal(packed.radius_orbit, expected.radius_orbit)

    # MantisNet's collation is its Rust builder's, over the same prefixes.
    from mantisnet.builder import collate_prefixes as mantis_collate_prefixes

    assert torch.equal(
        mantis.collate_prefixes(games, [21, 31]).legal_offsets,
        mantis_collate_prefixes(games, [21, 31]).legal_offsets,
    )


def test_the_default_budget_is_the_one_klent_offers():
    cfg = KlentConfig()
    assert cfg.graph_cell_budget == ACT_GRAPH_CELL_BUDGET
    assert _budgets(cfg).graph_cell_budget == ACT_GRAPH_CELL_BUDGET


# --------------------------------------------------------------------------
# A real fitting epoch through the seam


def _samples(count: int = 8):
    """Real self-play prefixes as KLENT fitting samples."""
    from types import SimpleNamespace

    rng = np.random.default_rng(SEED)
    out = []
    for k in range(count):
        game, ply = k % 2, 5 + 4 * k
        pos = position(game, ply)
        legal = len(pos.legal_moves())
        improved = rng.random(legal).astype(np.float32)
        improved /= improved.sum()
        out.append(
            SimpleNamespace(
                moves=moves(game, 60),
                t=ply,
                improved=improved,
                rank=int(rng.integers(0, legal)),
                g=float(rng.uniform(-1.0, 1.0)),
            )
        )
    return out


@pytest.mark.parametrize("which", ("act", "mantis"))
def test_a_klent_epoch_fits_either_architecture_through_the_seam(which):
    """Finite gradients and a moved parameter, from the model's own law.

    This is the whole seam at once: `fit` names neither representation, the
    model builds the batch, the model's `chunk_cost` packs it, `policy_q`
    answers, and `fit_epoch`'s post-step check passes on real gradients.
    """
    torch.manual_seed(SEED)
    if which == "act":
        model = MantisACT(TINY)
    else:
        model = MantisNet(MantisConfig(h=16, heads=2, blocks=1, value_bins=5))
    samples = _samples()
    # Small enough that the buffer packs into several chunks, so accumulation
    # and the optimizer group are exercised rather than one whole-buffer step.
    cfg = KlentConfig(
        device="cpu",
        batch_size=4,
        pair_budget=200_000,
        cell_budget=400,
        graph_cell_budget=900,
    )
    chunks = _pack(model, samples, range(len(samples)), cfg)
    assert len(chunks) > 1
    assert sorted(i for chunk in chunks for i in chunk) == list(range(len(samples)))

    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    metrics = fit(model, samples, optimizer, cfg, np.random.default_rng(0))

    assert metrics["fit_steps"] >= 1
    assert np.isfinite(metrics["policy_loss"]) and np.isfinite(metrics["critic_ce"])
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "the epoch left no gradient at all"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(
        not torch.equal(p.detach(), was) for p, was in zip(model.parameters(), before)
    )


def test_the_act_chunk_cost_klent_builds_is_the_models_own():
    torch.manual_seed(SEED)
    model = MantisACT(TINY)
    samples = _samples(4)
    cfg = KlentConfig(graph_cell_budget=7_777)
    cost = _chunk_cost(model, samples, cfg)
    assert isinstance(cost, ACTChunkCost)
    for i, sample in enumerate(samples):
        assert cost.units(i) == 2 * sample.t + len(sample.improved)
