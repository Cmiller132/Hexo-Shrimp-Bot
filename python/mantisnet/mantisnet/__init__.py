"""MantisNet: the stone/window graph network of ``docs/MODEL_SPEC.md``.

The builder owns the representation (`MODEL_REPR_VERSION`), the model owns the
weights, and the losses pin what the outputs mean. Search integration is
elsewhere and later.
"""

from .builder import (
    MODEL_REPR_VERSION,
    NUM_PATTERNS,
    Batch,
    PositionGraph,
    build,
    collate,
    collate_positions,
    collate_prefixes,
    from_position,
)
from .losses import param_groups, policy_loss, value_loss, value_target
from .model import MantisConfig, MantisNet, ModelOutput

__all__ = [
    "MODEL_REPR_VERSION",
    "NUM_PATTERNS",
    "Batch",
    "PositionGraph",
    "build",
    "collate",
    "collate_positions",
    "collate_prefixes",
    "from_position",
    "MantisConfig",
    "MantisNet",
    "ModelOutput",
    "param_groups",
    "policy_loss",
    "value_loss",
    "value_target",
]
