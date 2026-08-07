"""Parameter summary by subsystem (§6, §32).

Subsystem is derived from the parameter's module-tree path: drop the
parameter name and numeric indices, keep the first ``depth`` segments.
The grouped total is asserted against ``sum(p.numel())``.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

# Two components is the depth at which the state trunk's parts separate:
# `blocks.incidence`, `blocks.radius`, `latents.passes`, `final_cell.inv`.
DEFAULT_DEPTH = 2


def subsystem_of(name: str, depth: int = DEFAULT_DEPTH) -> str:
    """The subsystem label of a parameter path (see the module docstring)."""
    if depth < 1:
        raise ValueError(f"depth={depth} must be at least 1")
    parts = [part for part in name.split(".")[:-1] if not part.isdigit()]
    if not parts:
        # A parameter held directly on the summarised module itself.
        return name.split(".")[-1]
    return ".".join(parts[:depth])


@dataclass(frozen=True)
class ParameterSummary:
    """Trainable parameters by subsystem, largest first, plus the total.

    ``groups`` is a partition: every trainable parameter of the summarised
    module is counted in exactly one entry, and ``total`` is their sum.
    """

    groups: tuple[tuple[str, int], ...]
    total: int
    trainable_tensors: int

    def __str__(self) -> str:
        return self.text()

    def text(self, width: int = 28) -> str:
        """The summary as a table, one line per subsystem (§34)."""
        lines = []
        for name, count in self.groups:
            share = 100.0 * count / self.total if self.total else 0.0
            lines.append(f"  {name:<{width}}{count:>12,}  {share:5.1f}%")
        lines.append(f"  {'total':<{width}}{self.total:>12,}")
        return "\n".join(lines)


def parameter_summary(
    module: nn.Module, *, depth: int = DEFAULT_DEPTH
) -> ParameterSummary:
    """``module``'s trainable parameters grouped by subsystem (§6, §32).

    Frozen parameters are excluded from both the groups and the total; nothing
    else is filtered. The total is checked against ``sum(p.numel())`` over the
    same set before the result is returned, so a grouping rule that lost or
    duplicated a parameter raises here instead of reporting a wrong number.
    """
    counts: dict[str, int] = {}
    tensors = 0
    total = 0
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        label = subsystem_of(name, depth)
        counts[label] = counts.get(label, 0) + parameter.numel()
        tensors += 1
        total += parameter.numel()

    expected = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if total != expected:
        raise RuntimeError(
            f"the subsystem grouping counted {total} parameters against "
            f"sum(p.numel()) = {expected}"
        )
    grouped = sum(counts.values())
    if grouped != expected:
        raise RuntimeError(
            f"the subsystem groups sum to {grouped} against sum(p.numel()) = "
            f"{expected}"
        )
    groups = tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return ParameterSummary(groups=groups, total=total, trainable_tensors=tensors)


__all__ = ["DEFAULT_DEPTH", "ParameterSummary", "parameter_summary", "subsystem_of"]
