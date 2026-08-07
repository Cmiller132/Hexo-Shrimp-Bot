"""Model variants for the supervised lab harness.

A variant pairs a model family with the collator that builds its batches and
the configuration dataclass its overrides are validated against: MantisNet is
one variant, and each §29 MantisNet-ACT preset in ``PRESETS`` is another, with
its own collator bound to its own configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Callable, Mapping, Sequence, get_type_hints

from torch import nn

from ..builder import Batch, collate_prefixes
from ..model import MantisConfig, MantisNet
from ..models.mantis_act import PRESETS as ACT_PRESETS
from ..models.mantis_act import MantisACT, MantisACTConfig, PackedACTBatch
from ..models.mantis_act.builder import collate_prefixes as act_collate_prefixes
from ..models.mantis_act.config import PRESET_DELTAS as ACT_PRESET_DELTAS


Collate = Callable[
    [Sequence[Sequence[tuple[int, int]]], Sequence[int]], Batch | PackedACTBatch
]
Factory = Callable[[Mapping[str, object]], nn.Module]


@dataclass(frozen=True)
class VariantSpec:
    """Construction and corpus-collation contract for one model family."""

    factory: Factory
    collate: Collate
    config: type
    description: str
    rust_collate: bool


def _mantis_factory(overrides: Mapping[str, object]) -> MantisNet:
    return MantisNet(MantisConfig(**dict(overrides)))


# MantisACTConfig fields the ACT builder reads. An override changing one of
# these would build a graph under a different node set than the collator
# (bound to one configuration) expects, with no shape mismatch to catch it.
ACT_BUILDER_FIELDS = frozenset(
    {
        "window_scope",
        "cell_scope",
        "d6_relation_mode",
        "d_max",
        "occupied_radius",
        "use_cell_adjacency",
        "use_occupied_radius_edges",
        "use_global_numeric_features",
        "use_window_numeric_features",
        "use_action_tactical_features",
    }
)


def _arms_changing(field: str) -> list[str]:
    """The §29 presets whose delta names ``field``, for the refusal message."""
    return sorted(name for name, delta in ACT_PRESET_DELTAS.items() if field in delta)


def _act_variant(name: str, cfg: MantisACTConfig) -> VariantSpec:
    """One §29 preset as a lab variant, collator and model bound to ``cfg``."""

    def factory(overrides: Mapping[str, object]) -> MantisACT:
        refused = sorted(set(overrides) & ACT_BUILDER_FIELDS)
        if refused:
            arms = sorted({arm for field in refused for arm in _arms_changing(field)})
            raise ValueError(
                f"model override(s) {refused} change what the ACT builder emits, "
                f"but the {name!r} collator is bound to that preset's own "
                "representation: the graph and the model would describe "
                "different boards with no shape disagreement to catch it. Those "
                "arms are named presets, not cell overrides"
                + (f"; try --variant {' or '.join(arms)}" if arms else "")
            )
        return MantisACT(replace(cfg, **dict(overrides)))

    def collate(
        games: Sequence[Sequence[tuple[int, int]]], ts: Sequence[int]
    ) -> PackedACTBatch:
        return act_collate_prefixes(games, ts, cfg)

    delta = ACT_PRESET_DELTAS[name]
    rendered = (
        ", ".join(f"{key}={value!r}" for key, value in sorted(delta.items()))
        if delta
        else "the full model"
    )
    return VariantSpec(
        factory=factory,
        collate=collate,
        config=MantisACTConfig,
        description=f"MantisNet-ACT v4 §29 preset {name} ({rendered})",
        rust_collate=False,
    )


VARIANTS: dict[str, VariantSpec] = {
    "mantis": VariantSpec(
        factory=_mantis_factory,
        collate=collate_prefixes,
        config=MantisConfig,
        description="Production MantisNet with the Rust prefix builder",
        rust_collate=True,
    ),
}
for _name, _cfg in ACT_PRESETS.items():
    if _name in VARIANTS:
        raise RuntimeError(f"§29 preset {_name!r} collides with a registered variant")
    VARIANTS[_name] = _act_variant(_name, _cfg)


def _config_of(variant: str) -> type:
    """The configuration dataclass ``variant``'s overrides are validated against."""
    return variant_spec(variant).config


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
    raise TypeError(f"unsupported configuration field type for {key!r}: {expected!r}")


def parse_model_kw(
    items: Sequence[str] | None, variant: str = "mantis"
) -> dict[str, object]:
    """Parse CLI ``key=value`` entries against ``variant``'s configuration.

    Unknown and duplicate keys are errors.  Constructing the model remains the
    final validation step for relationships such as ``h % heads == 0``.
    """

    if not items:
        return {}
    config = _config_of(variant)
    field_names = {field.name for field in fields(config)}
    hints = get_type_hints(config)
    parsed: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"model override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        if not key or not raw:
            raise ValueError(f"model override must be key=value, got {item!r}")
        if key not in field_names:
            raise ValueError(f"unknown {config.__name__} field {key!r}")
        if key in parsed:
            raise ValueError(f"duplicate model override {key!r}")
        parsed[key] = _parse_typed_value(key, raw, hints[key])
    return parsed


def normalize_model_kw(
    overrides: Mapping[str, object] | None, variant: str = "mantis"
) -> dict[str, object]:
    """Validate programmatic overrides with the same field types as the CLI."""

    if not overrides:
        return {}
    config = _config_of(variant)
    field_names = {field.name for field in fields(config)}
    hints = get_type_hints(config)
    normalized: dict[str, object] = {}
    for key, value in overrides.items():
        if key not in field_names:
            raise ValueError(f"unknown {config.__name__} field {key!r}")
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

    model_kw = normalize_model_kw(overrides, name)
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

    model_kw = normalize_model_kw(overrides, name)
    suffixes = []
    for key, value in sorted(model_kw.items()):
        if isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        suffixes.append(f"{key}{rendered}")
    return "+".join([name, *suffixes])
