"""The model summary §6 and §32 require: parameters by subsystem, and a total.

§6 asks for "a model-summary function reporting parameters by subsystem"; §32
requires its total to match ``sum(p.numel())``; §16 needs the parameter cost of
two arms compared before either is called a matched control; and §34 lists
"parameters by subsystem" among the figures a run reports. All four are this
module.

A subsystem is read off the parameter's own path in the module tree rather than
looked up in a registry of module names. A registry has to be maintained beside
the model and goes stale silently — a renamed or newly added module quietly
falls into "other", or worse, into whichever bucket a prefix still matches. Here
the label is derived:

```text
blocks.2.incidence.to_windows.wv_inv.weight  ->  blocks.incidence
latents.passes.0.q_read_inv.weight           ->  latents.passes
cell_embedding.axis_base                     ->  cell_embedding
```

that is: drop the parameter's own name, drop the numeric components — a block
index names a copy of a subsystem, not a subsystem — and keep the first
``depth`` of what remains. A module that is renamed changes its label; a module
that is added appears as its own line; no parameter is ever dropped or
double-counted, and :func:`parameter_summary` asserts exactly that against
``sum(p.numel())`` before it returns.

Sharing is counted once. ``named_parameters`` deduplicates by identity, so the
relation tables the state trunk shares across its four blocks are one line at
their owner's path, and the summary's total is the number of trainable scalars
the optimiser sees rather than the number of places they are read from.
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

    Frozen parameters are excluded from both the groups and the total: the
    summary reports what is being trained. Nothing else is filtered, and the
    total is checked against ``sum(p.numel())`` over the same set before the
    result is returned, so a grouping rule that lost or duplicated a parameter
    raises here instead of reporting a plausible wrong number.
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
