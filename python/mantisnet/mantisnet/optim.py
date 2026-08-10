"""Explicit Adam execution policies for production and benchmark fitting.

The mathematical Adam recipe is unchanged.  ``fused`` and ``foreach`` choose
how PyTorch evaluates the same per-parameter update; their different reduction
association can move the last few floating-point bits, so the requested and
resolved implementation are part of every recorded training configuration.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


ADAM_IMPLEMENTATIONS = ("auto", "fused", "foreach", "scalar")


def resolve_adam_implementation(requested: str, device) -> str:
    """Resolve one recorded Adam policy for ``device`` or fail loudly."""
    if requested not in ADAM_IMPLEMENTATIONS:
        raise ValueError(
            f"adam_impl must be one of {list(ADAM_IMPLEMENTATIONS)}, got "
            f"{requested!r}"
        )
    device_type = torch.device(device).type
    if requested == "auto":
        return "fused" if device_type == "cuda" else "scalar"
    if requested == "fused" and device_type != "cuda":
        raise ValueError(
            f"fused Adam is a CUDA execution policy, got device={device!r}"
        )
    return requested


def _options(resolved: str, device) -> dict[str, bool]:
    if resolved not in ADAM_IMPLEMENTATIONS[1:]:
        raise ValueError(
            "resolved Adam implementation must be fused, foreach, or scalar; "
            f"got {resolved!r}"
        )
    cuda = torch.device(device).type == "cuda"
    if resolved == "fused" and not cuda:
        raise ValueError(
            f"fused Adam is a CUDA execution policy, got device={device!r}"
        )
    return {
        "fused": resolved == "fused",
        "foreach": resolved == "foreach",
        "capturable": cuda and resolved in ("fused", "foreach"),
    }


def configure_adam(optimizer: torch.optim.Adam, resolved: str, device) -> None:
    """Restore an explicit execution policy after loading an Adam state dict.

    ``Optimizer.load_state_dict`` restores the saved parameter-group execution
    flags too.  Reapplying the current run's recorded policy prevents an old
    scalar checkpoint from silently disabling fused execution (or vice versa).
    Step counters move to the location required by the selected implementation.
    """
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError(f"configure_adam needs torch.optim.Adam, got {type(optimizer)}")
    options = _options(resolved, device)
    optimizer.defaults.update(options)
    for group in optimizer.param_groups:
        group.update(options)
    on_parameter_device = options["capturable"] or options["fused"]
    for parameter, state in optimizer.state.items():
        step = state.get("step")
        if isinstance(step, Tensor):
            target = parameter.device if on_parameter_device else torch.device("cpu")
            state["step"] = step.to(target)


def make_adam(
    parameters: Iterable[Tensor],
    *,
    lr: float,
    device,
    implementation: str = "auto",
) -> tuple[torch.optim.Adam, str]:
    """Construct Adam and return the resolved, recordable implementation."""
    resolved = resolve_adam_implementation(implementation, device)
    optimizer = torch.optim.Adam(
        parameters,
        lr=lr,
        **_options(resolved, device),
    )
    return optimizer, resolved


__all__ = [
    "ADAM_IMPLEMENTATIONS",
    "configure_adam",
    "make_adam",
    "resolve_adam_implementation",
]
