"""S2 global magnitude: the whole-position channel on the latent init.

The knob is model-only — no builder change, since the 377-class canonical
pattern the histogram counts is already the window embedding's own feature —
so the tests hold it to the knob conventions: byte-identical when off, the
incumbent function at init when on, config recoverable from the state dict,
and pinned parameter counts. The histogram itself is checked against a plain
Python count and against the D6 images of the same position.
"""

from __future__ import annotations

import copy
from collections import Counter

import hexo_py
import pytest
import torch

from mantisnet.builder import TERN_PATTERNS, collate, from_position
from mantisnet.lab.families import infer_config
from mantisnet.lab.variants import normalize_model_kw, parse_model_kw
from mantisnet.model import MantisConfig, MantisNet, magnitude_features

from .conftest import d6_transforms

# Empty board, one stone, and boards of growing window counts; the last two
# repeat one game so a batch holds two positions of identical magnitude.
_GAMES = [
    [],
    [(0, 0)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (0, 3), (0, 4), (3, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, -1), (0, 2)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, -1), (0, 2)],
]

_D6_MOVES = [(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]


def _tiny(**overrides) -> MantisConfig:
    return MantisConfig(
        h=32, blocks=1, heads=2, policy_hidden=16, value_hidden=16, **overrides
    )


def _batch(games=_GAMES):
    return collate([from_position(hexo_py.Position.replay(m)) for m in games])


def _window_pos(batch):
    return batch.window_slot // batch.max_w


def test_the_knob_is_a_typed_config_field_defaulting_off():
    assert MantisConfig().global_magnitude is False
    assert MantisConfig(global_magnitude=True).global_magnitude is True
    # The lab override paths type the field off the dataclass, so a non-bool
    # is refused rather than silently coerced.
    assert parse_model_kw(["global_magnitude=true"]) == {"global_magnitude": True}
    with pytest.raises(ValueError) as error:
        parse_model_kw(["global_magnitude=1"])
    assert "must be true or false" in str(error.value)
    with pytest.raises(ValueError) as error:
        normalize_model_kw({"global_magnitude": 1})
    assert "must have type bool" in str(error.value)


def test_knob_off_is_byte_identical_to_the_incumbent():
    torch.manual_seed(2026)
    incumbent = MantisNet(_tiny())
    torch.manual_seed(2026)
    explicit_off = MantisNet(_tiny(global_magnitude=False))
    left, right = incumbent.state_dict(), explicit_off.state_dict()
    assert list(left) == list(right)
    for a, b in zip(left.values(), right.values()):
        assert torch.equal(a, b)
    assert not any("magnitude" in name for name in left)

    batch = _batch()
    incumbent.eval(), explicit_off.eval()
    with torch.no_grad():
        out_a = incumbent(batch, 0.2)
        out_b = explicit_off(batch, 0.2)
    for name in vars(out_a):
        va, vb = getattr(out_a, name), getattr(out_b, name)
        assert va.cpu().numpy().tobytes() == vb.cpu().numpy().tobytes(), name


def test_knob_on_starts_at_the_incumbent_function():
    """The zero-init table and linear make the added row exactly zero: a
    knob-on model whose shared parameters equal a knob-off model's computes
    the identical function at initialization."""
    torch.manual_seed(11)
    off = MantisNet(_tiny())
    aligned = MantisNet(_tiny(global_magnitude=True))
    assert torch.equal(
        aligned.magnitude_pattern, torch.zeros_like(aligned.magnitude_pattern)
    )
    assert torch.equal(
        aligned.magnitude_counts.weight,
        torch.zeros_like(aligned.magnitude_counts.weight),
    )
    assert torch.equal(
        aligned.magnitude_counts.bias, torch.zeros_like(aligned.magnitude_counts.bias)
    )
    aligned.load_state_dict({**aligned.state_dict(), **off.state_dict()}, strict=True)

    batch = _batch()
    off.eval(), aligned.eval()
    with torch.no_grad():
        out_off = off(batch, 0.2)
        out_aligned = aligned(batch, 0.2)
    for name in vars(out_off):
        assert torch.equal(getattr(out_off, name), getattr(out_aligned, name)), name


def test_histogram_and_counts_match_a_plain_python_count():
    graphs = [from_position(hexo_py.Position.replay(m)) for m in _GAMES]
    batch = collate(graphs)
    hist, counts = magnitude_features(batch, _window_pos(batch))

    assert hist.dtype == torch.int64 and counts.dtype == torch.int64
    assert hist.shape == (len(graphs), TERN_PATTERNS)
    assert counts.shape == (len(graphs), 2)

    for position, graph in enumerate(graphs):
        want = Counter(int(feature) for feature in graph.window_feat)
        got = {
            int(cls): int(hist[position, cls])
            for cls in hist[position].nonzero().flatten().tolist()
        }
        assert got == dict(want), _GAMES[position]
        assert int(hist[position].sum()) == graph.n_windows
        assert counts[position].tolist() == [graph.n_stones, graph.n_legal]

    # The set must actually span the interesting shapes: an empty board with
    # no windows at all, a one-stone board whose 18 windows fall in the three
    # slot orbits of a lone own stone, and crowded boards with many classes.
    live = (hist > 0).sum(dim=1).tolist()
    assert live[0] == 0 and int(hist[0].sum()) == 0
    assert live[1] == 3 and hist[1][hist[1] > 0].tolist() == [6, 6, 6]
    assert max(live) > 20
    assert len(set(counts[:, 0].tolist())) > 3


def test_a_single_pattern_batch_reduces_to_the_window_counts():
    """No legal position puts every window in one class (a lone stone already
    spans three slot orbits), so the degenerate bincount case is forced here.

    The class used is the last of the vocabulary — the top slot of each
    position's block — so a wrong segment stride would spill the count into
    the next position rather than land silently.
    """
    batch = _batch()
    window_pos = _window_pos(batch)
    forced = copy.copy(batch)
    forced.window_feat = torch.full_like(batch.window_feat, TERN_PATTERNS - 1)
    hist, _counts = magnitude_features(forced, window_pos)

    per_position = torch.bincount(window_pos, minlength=batch.n_pos)
    assert hist[:, TERN_PATTERNS - 1].tolist() == per_position.tolist()
    assert int(hist[:, : TERN_PATTERNS - 1].sum()) == 0


def test_features_are_d6_invariant():
    """A position and each of its twelve images give identical rows: the
    window patterns are reversal-canonical and the counts are of sets the
    symmetries merely permute."""
    base = _batch([_D6_MOVES])
    want_hist, want_counts = magnitude_features(base, _window_pos(base))
    for transform in d6_transforms():
        image = _batch([[transform(move) for move in _D6_MOVES]])
        hist, counts = magnitude_features(image, _window_pos(image))
        assert torch.equal(hist, want_hist)
        assert torch.equal(counts, want_counts)


@torch.no_grad()
def test_the_model_is_d6_invariant_with_the_knob_on():
    torch.manual_seed(11)
    model = MantisNet(_tiny(global_magnitude=True)).eval()
    model.mlp_p.out.weight.normal_(std=0.05)
    model.mlp_q.out.weight.normal_(std=0.05)
    model.magnitude_pattern.normal_(std=0.05)
    model.magnitude_counts.weight.normal_(std=0.05)

    base_position = hexo_py.Position.replay(_D6_MOVES)
    base = model(collate([from_position(base_position)]), 0.2)
    policy = dict(zip(base_position.legal_moves(), base.policy_logits.tolist()))
    values = dict(zip(base_position.legal_moves(), base.q_values.tolist()))
    for transform in d6_transforms():
        position = hexo_py.Position.replay([transform(move) for move in _D6_MOVES])
        output = model(collate([from_position(position)]), 0.2)
        moved_policy = dict(zip(position.legal_moves(), output.policy_logits.tolist()))
        moved_values = dict(zip(position.legal_moves(), output.q_values.tolist()))
        for move in policy:
            assert abs(moved_policy[transform(move)] - policy[move]) <= 1e-5
            assert abs(moved_values[transform(move)] - values[move]) <= 1e-5
        assert torch.allclose(output.value, base.value, atol=1e-5)


@torch.no_grad()
def test_the_row_is_one_shared_vector_per_position():
    torch.manual_seed(5)
    model = MantisNet(_tiny(global_magnitude=True)).eval()
    model.magnitude_pattern.normal_(std=0.05)
    model.magnitude_counts.weight.normal_(std=0.05)
    batch = _batch()
    row = model._magnitude_row(batch, _window_pos(batch))
    assert row.shape == (batch.n_pos, model.cfg.h)
    # The last two games are the same position, so their rows coincide; every
    # other pair differs in magnitude and so in its row.
    assert torch.equal(row[-1], row[-2])
    for position in range(batch.n_pos - 2):
        assert not torch.equal(row[position], row[-1])


def test_gradients_reach_every_magnitude_parameter():
    torch.manual_seed(3)
    model = MantisNet(_tiny(global_magnitude=True))
    # The zero-init decoder outputs block upstream flow; perturb them first.
    for head in (model.mlp_p, model.mlp_q):
        torch.nn.init.normal_(head.out.weight, std=0.05)
    out = model(_batch(), 0.2)
    (out.policy_logits.sum() + out.q_values.sum()).backward()
    for name in (
        "magnitude_pattern",
        "magnitude_counts.weight",
        "magnitude_counts.bias",
    ):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


@pytest.mark.parametrize("on", [False, True])
def test_the_config_is_inferable_from_the_state_dict(on):
    cfg = _tiny(global_magnitude=on)
    assert infer_config(MantisNet(cfg).state_dict()) == cfg


def test_parameter_counts_pin_the_knob():
    def count(**overrides) -> int:
        return sum(
            parameter.numel()
            for parameter in MantisNet(MantisConfig(**overrides)).parameters()
        )

    production = dict(
        cell_latents=True,
        cell_nodes=True,
        cell_node_scope="all",
        window_attention=False,
    )
    arm_b = dict(production, action_tactical=True)

    assert count() == 4_804_581
    assert count(**production) == 5_197_093
    assert count(**arm_b) == 5_215_141

    h = MantisConfig().h
    added = TERN_PATTERNS * h + (h * 2 + h)
    assert count(global_magnitude=True) - count() == added
    assert count(global_magnitude=True) == 4_853_221
    assert count(**arm_b, global_magnitude=True) == 5_263_781
