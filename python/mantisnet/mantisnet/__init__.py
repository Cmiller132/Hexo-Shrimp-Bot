"""MantisNet: the stone/window graph network of ``docs/MODEL_SPEC.md``.

The shared Rust encoder owns the representation and ``MODEL_REPR_VERSION``;
this package's Python builder is its independent parity reference. The model
owns the weights, the losses define raw-head targets, and the Python-free
container package owns improved evaluation and sessions.
"""

from .builder import (
    DEC_CLASSES,
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
    "DEC_CLASSES",
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
