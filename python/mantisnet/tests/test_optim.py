"""Recorded Adam execution policies preserve the optimizer recipe."""

from __future__ import annotations

import pytest
import torch

from mantisnet.optim import configure_adam, make_adam, resolve_adam_implementation


def _parameter(value=1.0, *, device="cpu"):
    return torch.nn.Parameter(torch.tensor([value, -value], device=device))


def test_auto_is_scalar_on_cpu_and_explicit_fused_cpu_is_refused():
    parameter = _parameter()
    optimizer, resolved = make_adam(
        [parameter], lr=1e-3, device="cpu", implementation="auto"
    )
    assert resolved == "scalar"
    assert optimizer.param_groups[0]["fused"] is False
    assert optimizer.param_groups[0]["foreach"] is False
    assert optimizer.param_groups[0]["capturable"] is False

    with pytest.raises(ValueError, match="CUDA execution policy"):
        resolve_adam_implementation("fused", "cpu")
    with pytest.raises(ValueError, match="adam_impl"):
        resolve_adam_implementation("surprise", "cpu")
    with pytest.raises(ValueError, match="resolved Adam"):
        configure_adam(optimizer, "auto", "cpu")


def test_foreach_and_scalar_adam_agree_on_cpu_and_policy_survives_state_load():
    scalar_parameter = _parameter()
    foreach_parameter = _parameter()
    scalar, _ = make_adam(
        [scalar_parameter], lr=1e-3, device="cpu", implementation="scalar"
    )
    foreach, resolved = make_adam(
        [foreach_parameter], lr=1e-3, device="cpu", implementation="foreach"
    )
    for parameter in (scalar_parameter, foreach_parameter):
        parameter.grad = torch.tensor([0.25, -0.5])
    scalar.step()
    foreach.step()
    assert resolved == "foreach"
    torch.testing.assert_close(foreach_parameter, scalar_parameter, atol=0.0, rtol=0.0)

    # Loading a scalar checkpoint restores its saved execution flags. The
    # current run's explicit policy must be re-applied afterwards.
    foreach.load_state_dict(scalar.state_dict())
    assert foreach.param_groups[0]["foreach"] is False
    configure_adam(foreach, "foreach", "cpu")
    assert foreach.param_groups[0]["foreach"] is True
    assert foreach.param_groups[0]["fused"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused Adam needs CUDA")
def test_fused_cuda_adam_is_bitwise_repeatable_and_close_to_scalar_recipe():
    gradients = [torch.linspace(-1.0, 1.0, 17, device="cuda") for _ in range(32)]

    def run(implementation):
        parameters = [
            torch.nn.Parameter(torch.linspace(-0.5, 0.5, 17, device="cuda"))
            for _ in gradients
        ]
        optimizer, resolved = make_adam(
            parameters, lr=1e-3, device="cuda", implementation=implementation
        )
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient.clone()
        optimizer.step()
        return resolved, torch.stack([parameter.detach() for parameter in parameters])

    resolved, first = run("fused")
    _resolved, second = run("fused")
    _scalar, anchor = run("scalar")
    assert resolved == "fused"
    assert torch.equal(first, second)
    torch.testing.assert_close(first, anchor, atol=2e-7, rtol=2e-6)
