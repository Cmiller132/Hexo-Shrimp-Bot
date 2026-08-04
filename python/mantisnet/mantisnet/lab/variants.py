"""Model variants available to the supervised lab harness.

The registry is intentionally small.  A representation only belongs here once
its model and collation path exist; ordinary MantisNet width/depth/head
ablations are typed ``MantisConfig`` overrides, not separate variants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable, Mapping, Sequence, get_type_hints

from torch import nn

from ..builder import Batch, collate_prefixes
from ..model import MantisConfig, MantisNet


Collate = Callable[[Sequence[Sequence[tuple[int, int]]], Sequence[int]], Batch]
Factory = Callable[[Mapping[str, object]], nn.Module]


@dataclass(frozen=True)
class VariantSpec:
    """Construction and corpus-collation contract for one model family."""

    factory: Factory
    collate: Collate
    description: str
    rust_collate: bool


def _mantis_factory(overrides: Mapping[str, object]) -> MantisNet:
    return MantisNet(MantisConfig(**dict(overrides)))


# Exactly one implemented representation ships with format version 1.
VARIANTS: dict[str, VariantSpec] = {
    "mantis": VariantSpec(
        factory=_mantis_factory,
        collate=collate_prefixes,
        description="Production MantisNet with the Rust prefix builder",
        rust_collate=True,
    )
}


def _parse_typed_value(key: str, value: str, expected: type) -> object:
    if expected is bool:
        normalized = value.casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise ValueError(f"model override {key!r} must be true or false, got {value!r}")
    if expected is int:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"model override {key!r} must be an int, got {value!r}") from exc
    if expected is float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"model override {key!r} must be a float, got {value!r}") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"model override {key!r} must be finite, got {value!r}")
        return parsed
    if expected is str:
        return value
    raise TypeError(f"unsupported MantisConfig field type for {key!r}: {expected!r}")


def parse_model_kw(items: Sequence[str] | None) -> dict[str, object]:
    """Parse CLI ``key=value`` entries against the ``MantisConfig`` dataclass.

    Unknown and duplicate keys are errors.  Constructing the model remains the
    final validation step for relationships such as ``h % heads == 0``.
    """

    if not items:
        return {}
    field_names = {field.name for field in fields(MantisConfig)}
    hints = get_type_hints(MantisConfig)
    parsed: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"model override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        if not key or not raw:
            raise ValueError(f"model override must be key=value, got {item!r}")
        if key not in field_names:
            raise ValueError(f"unknown MantisConfig field {key!r}")
        if key in parsed:
            raise ValueError(f"duplicate model override {key!r}")
        parsed[key] = _parse_typed_value(key, raw, hints[key])
    return parsed


def normalize_model_kw(overrides: Mapping[str, object] | None) -> dict[str, object]:
    """Validate programmatic overrides with the same field types as the CLI."""

    if not overrides:
        return {}
    field_names = {field.name for field in fields(MantisConfig)}
    hints = get_type_hints(MantisConfig)
    normalized: dict[str, object] = {}
    for key, value in overrides.items():
        if key not in field_names:
            raise ValueError(f"unknown MantisConfig field {key!r}")
        expected = hints[key]
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if type(value) is not expected:
            raise ValueError(
                f"model override {key!r} must have type {expected.__name__}, "
                f"got {type(value).__name__}"
            )
        if expected is float and not math.isfinite(value):
            raise ValueError(f"model override {key!r} must be finite, got {value!r}")
        normalized[key] = value
    return normalized


def variant_spec(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(VARIANTS))
        raise ValueError(f"unknown lab variant {name!r}; available: {available}") from exc


def build_variant(
    name: str, overrides: Mapping[str, object] | None = None
) -> tuple[nn.Module, dict[str, object], VariantSpec]:
    """Build a registered variant and return its normalized identity."""

    model_kw = normalize_model_kw(overrides)
    spec = variant_spec(name)
    return spec.factory(model_kw), model_kw, spec


def count_parameters(model: nn.Module) -> int:
    """Count every model parameter, whether or not it currently needs grad."""

    return sum(parameter.numel() for parameter in model.parameters())


def refuse_param_budget(
    count: int, budget: int | None, tolerance: float = 0.02
) -> None:
    """Refuse a parameter count outside the inclusive requested interval."""

    if budget is None:
        return
    if budget <= 0:
        raise ValueError(f"parameter budget must be positive, got {budget}")
    if not math.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError(f"parameter tolerance must be in [0, 1), got {tolerance}")
    lower = math.ceil(budget * (1.0 - tolerance))
    upper = math.floor(budget * (1.0 + tolerance))
    if count < lower or count > upper:
        raise ValueError(
            f"parameter count {count} is outside budget {budget} bounds "
            f"[{lower}, {upper}] at tolerance {tolerance:g}"
        )


def derived_cell_name(name: str, overrides: Mapping[str, object] | None = None) -> str:
    """Return the stable default cell name (for example ``mantis+h96``)."""

    model_kw = normalize_model_kw(overrides)
    suffixes = []
    for key, value in sorted(model_kw.items()):
        if isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        suffixes.append(f"{key}{rendered}")
    return "+".join([name, *suffixes])
