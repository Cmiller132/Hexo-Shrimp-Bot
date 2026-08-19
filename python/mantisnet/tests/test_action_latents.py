"""Step 6 action latents: knob wiring, gradients, D6, and the read contract.

The cycle is model-only — no builder change — so the tests hold it to the
knob conventions: byte-identical when off, incumbent function at init when
on, bias-free keys, every parameter trained, and a permutation-invariant
read (the latents see the legal set as a set).
"""

from __future__ import annotations

import torch

import hexo_py
from mantisnet.builder import collate, from_position
from mantisnet.model import ACTION_LATENTS, MantisConfig, MantisNet

_GAMES = [
    [],
    [(0, 0)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (0, 3), (0, 4), (3, 0)],
]

_LATENT_MARKS = (".act_latent_base", ".act_ln_", ".act_wq_", ".act_wk_", ".act_wv_", ".act_wo_")


def _tiny(**overrides) -> MantisConfig:
    return MantisConfig(
        h=32, blocks=1, heads=2, policy_hidden=16, value_hidden=16, **overrides
    )


def _batch():
    return collate([from_position(hexo_py.Position.replay(m)) for m in _GAMES])


def test_knob_off_is_byte_identical_to_the_incumbent():
    torch.manual_seed(2026)
    incumbent = MantisNet(_tiny())
    torch.manual_seed(2026)
    explicit_off = MantisNet(_tiny(action_latents=False))
    left, right = incumbent.state_dict(), explicit_off.state_dict()
    assert list(left) == list(right)
    for a, b in zip(left.values(), right.values()):
        assert torch.equal(a, b)
    assert not any(
        mark in "." + name for name in left for mark in _LATENT_MARKS
    )

    batch = _batch()
    incumbent.eval(), explicit_off.eval()
    with torch.no_grad():
        out_a = incumbent(batch, 0.2)
        out_b = explicit_off(batch, 0.2)
    for name in vars(out_a):
        va, vb = getattr(out_a, name), getattr(out_b, name)
        assert va.cpu().numpy().tobytes() == vb.cpu().numpy().tobytes(), name


def test_knob_on_starts_at_the_incumbent_function():
    """The zero-init broadcast output makes the cycle an exact identity."""
    torch.manual_seed(11)
    off = MantisNet(_tiny())
    aligned = MantisNet(_tiny(action_latents=True))
    aligned.load_state_dict(
        {**aligned.state_dict(), **off.state_dict()}, strict=True
    )
    batch = _batch()
    off.eval(), aligned.eval()
    with torch.no_grad():
        out_off = off(batch, 0.2)
        out_aligned = aligned(batch, 0.2)
    for name in vars(out_off):
        assert torch.equal(getattr(out_off, name), getattr(out_aligned, name)), name


def test_keys_are_bias_free():
    model = MantisNet(_tiny(action_latents=True))
    parameters = dict(model.named_parameters())
    for name in ("act_wk_read", "act_wk_mix", "act_wk_bcast"):
        assert name + ".weight" in parameters
        assert name + ".bias" not in parameters
    for name in (
        "act_wq_read", "act_wv_read", "act_wo_read",
        "act_wq_mix", "act_wv_mix", "act_wo_mix",
        "act_wq_bcast", "act_wv_bcast", "act_wo_bcast",
    ):
        assert name + ".bias" in parameters


def test_gradients_reach_every_latent_parameter():
    torch.manual_seed(3)
    model = MantisNet(_tiny(action_latents=True))
    # The zero-init decoder outputs block upstream flow; perturb them first,
    # and the zero-init broadcast output likewise gates the read/mix path.
    for head in (model.mlp_p, model.mlp_q):
        torch.nn.init.normal_(head.out.weight, std=0.05)
    torch.nn.init.normal_(model.act_wo_bcast.weight, std=0.05)
    out = model(_batch(), 0.2)
    (out.policy_logits.sum() + out.q_values.sum()).backward()
    for name, parameter in model.named_parameters():
        if not name.startswith(("act_latent", "act_ln_", "act_w")):
            continue
        grad = parameter.grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


def test_the_read_is_permutation_invariant():
    """Reversing a position's legal-cell order permutes the outputs and
    changes nothing else: the latents treat the legal set as a set."""
    torch.manual_seed(5)
    model = MantisNet(_tiny(action_latents=True))
    torch.nn.init.normal_(model.act_wo_bcast.weight, std=0.05)
    for head in (model.mlp_p, model.mlp_q):
        torch.nn.init.normal_(head.out.weight, std=0.05)
    model.eval()

    batch = _batch()
    with torch.no_grad():
        base = model(batch, 0.2)

    # Reverse every position's run of cells in the cell-indexed tables.
    permutation = torch.cat(
        [
            torch.arange(int(a), int(b)).flip(0)
            for a, b in zip(batch.legal_offsets[:-1], batch.legal_offsets[1:])
        ]
    )
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    reordered = {name: value for name, value in vars(batch).items()}
    for name in ("cell_pos", "cell_occupancy", "cell_is_legal", "cell_nearest",
                 "act_empty", "act_tactical"):
        reordered[name] = reordered[name][permutation]
    reordered["dec_cell"] = inverse[batch.dec_cell]
    reordered["radius_dst"] = inverse[batch.radius_dst]
    reordered["adjacency_src"] = inverse[batch.adjacency_src]
    reordered["adjacency_dst"] = inverse[batch.adjacency_dst]

    from mantisnet.builder import batch_from_arrays

    fields = {
        name: value
        for name, value in reordered.items()
        if name in vars(batch)
        and name
        not in ("n_pos", "max_t", "max_w", "n_cells",
                "relay_cell_ptr", "relay_window", "relay_class", "relay_win_ptr",
                "relay_wcell", "relay_cls_ptr", "relay_ccell")
    }
    permuted = batch_from_arrays(**fields)
    with torch.no_grad():
        moved = model(permuted, 0.2)
    assert torch.allclose(
        base.policy_logits[permutation], moved.policy_logits, atol=1e-5
    )
    assert torch.allclose(
        base.q_values[permutation], moved.q_values, atol=1e-5
    )


def test_parameter_counts_pin_the_knob():
    base = sum(p.numel() for p in MantisNet(MantisConfig()).parameters())
    on = sum(
        p.numel()
        for p in MantisNet(MantisConfig(action_latents=True)).parameters()
    )
    h = MantisConfig().h
    expected = (
        ACTION_LATENTS * h  # latent base
        + 5 * 2 * h  # five LayerNorms
        + 9 * (h * h + h)  # biased projections
        + 3 * h * h  # bias-free keys
    )
    assert base == 4_804_213
    assert on - base == expected
    assert on == 5_003_509
