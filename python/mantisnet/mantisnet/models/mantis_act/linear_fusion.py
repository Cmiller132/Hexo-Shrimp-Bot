"""Traceable horizontal fusion for projections sharing one input.

The individual ``nn.Linear`` modules remain the owners of their parameters so
checkpoint names and optimizer state do not change. Concatenating output rows
at execution time lets the outer whole-model compiler emit one GEMM, then split
the result into the literal projections' outputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def horizontal_linears(
    x: Tensor, projections: Sequence[nn.Linear]
) -> tuple[Tensor, ...]:
    """Evaluate same-input linears as one output-wide linear operation."""
    if len(projections) < 2:
        raise ValueError(
            f"horizontal_linears needs at least two projections, got {len(projections)}"
        )
    in_features = projections[0].in_features
    has_bias = projections[0].bias is not None
    for index, projection in enumerate(projections):
        if projection.in_features != in_features:
            raise ValueError(
                f"projection {index} has in_features={projection.in_features}, "
                f"expected {in_features}"
            )
        if (projection.bias is not None) != has_bias:
            raise ValueError("horizontal projections must agree on bias presence")

    weight = torch.cat([projection.weight for projection in projections], dim=0)
    bias = (
        torch.cat([projection.bias for projection in projections], dim=0)
        if has_bias
        else None
    )
    fused = F.linear(x, weight, bias)
    return fused.split([projection.out_features for projection in projections], dim=-1)


__all__ = ["horizontal_linears"]
