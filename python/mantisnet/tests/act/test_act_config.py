"""§6/§28/§29 configuration: vocabularies, invariants, presets, and the hash.

The hash tests mutate fields with ``object.__setattr__`` rather than
``dataclasses.replace``. That deliberately bypasses ``__post_init__``: the
question is whether ``architecture_hash`` reads a field at all, and several
fields cannot be varied alone through a valid configuration — ``pair_scope``
drags ``use_action_pair_messages`` with it, and ``architecture_id`` has a
one-value vocabulary. Bypassing validation is the only way to ask about every
field the same way, and coverage of every field is the property §28 wants.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import subprocess
import sys

import pytest

from mantisnet.models.mantis_act import config as config_module
from mantisnet.models.mantis_act.config import (
    ARCHITECTURE_ID,
    ENUM_VOCABULARIES,
    MANTIS_ACT_REPR_VERSION,
    PRESET_DELTAS,
    PRESETS,
    MantisACTConfig,
    architecture_hash,
    summarise,
)

# §29 names these fifteen; §16 and §29 require the parameter-matched extra-FFN
# control alongside typed window attention.
SPEC_PRESETS = (
    "full_act_v4",
    "full_no_pair",
    "full_no_axis",
    "full_live_windows",
    "full_action_relevant_windows",
    "full_no_latents",
    "full_one_latent",
    "full_coarse_geometry",
    "full_radius6",
    "full_occupied_cells_only",
    "full_additive_incidence",
    "full_shared_head",
    "full_with_typed_window_attention",
    "full_no_tactical_inputs",
    "full_no_action_latents",
    "full_extra_ffn_control",
)


def _mutated(cfg: MantisACTConfig, name: str) -> MantisACTConfig:
    """A copy of ``cfg`` with one field changed, validation bypassed."""
    value = getattr(cfg, name)
    if isinstance(value, bool):
        other: object = not value
    elif isinstance(value, int):
        other = value + 1
    elif isinstance(value, float):
        other = value + 1.0
    else:
        other = value + "!"
    out = MantisACTConfig()
    object.__setattr__(out, name, other)
    return out


def _reimport():
    """Execute the module again under its own name, leaving importers alone.

    A second execution is the re-import the stability claim is about. It runs
    under a throwaway module name rather than replacing the real entry, which
    would hand every later importer a different ``MantisACTConfig`` class —
    and dataclass equality is class-identity checked. The entry exists only
    while the module body runs, because ``@dataclass`` resolves annotations
    through ``sys.modules[cls.__module__]``.
    """
    name = "act_config_reimport"
    spec = importlib.util.spec_from_file_location(name, config_module.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


def test_repr_version_is_four():
    assert MANTIS_ACT_REPR_VERSION == 4


def test_defaults_are_the_full_model():
    assert PRESETS["full_act_v4"] == MantisACTConfig()
    assert MantisACTConfig().architecture_id == ARCHITECTURE_ID


def test_every_named_preset_exists_and_constructs():
    assert set(PRESETS) == set(SPEC_PRESETS)
    for name in SPEC_PRESETS:
        cfg = PRESETS[name]
        assert isinstance(cfg, MantisACTConfig)
        # Reconstructing from the same delta must validate again, so that a
        # preset cannot be an instance that __post_init__ would refuse.
        assert dataclasses.replace(MantisACTConfig(), **PRESET_DELTAS[name]) == cfg


def test_every_ablation_actually_differs():
    full = PRESETS["full_act_v4"]
    for name in SPEC_PRESETS:
        if name == "full_act_v4":
            continue
        assert PRESETS[name] != full, name
        assert architecture_hash(PRESETS[name]) != architecture_hash(full), name


def test_preset_intents_match_the_spec():
    assert PRESETS["full_no_axis"].d_axis == 0
    assert PRESETS["full_no_axis"].use_axis_channels is False
    assert PRESETS["full_no_axis"].num_axis_latents == 0
    assert PRESETS["full_live_windows"].window_scope == "live"
    assert PRESETS["full_action_relevant_windows"].window_scope == "action_relevant"
    assert PRESETS["full_no_latents"].global_mode == "none"
    assert PRESETS["full_one_latent"].num_inv_latents == 1
    assert PRESETS["full_coarse_geometry"].d6_relation_mode == "coarse_distance_axis"
    assert PRESETS["full_radius6"].occupied_radius == 6
    assert PRESETS["full_occupied_cells_only"].cell_scope == "occupied_only"
    assert PRESETS["full_additive_incidence"].incidence_message == "additive"
    assert PRESETS["full_shared_head"].head_separation == "single_shared_head"
    typed = PRESETS["full_with_typed_window_attention"]
    assert typed.window_window_mode == "typed_collinear_crossing"
    control = PRESETS["full_extra_ffn_control"]
    assert control.window_window_mode == "none"
    assert control.ffn_mult > PRESETS["full_act_v4"].ffn_mult
    assert PRESETS["full_no_tactical_inputs"].use_action_tactical_features is False
    assert PRESETS["full_no_action_latents"].num_action_latents == 0


@pytest.mark.parametrize("name", sorted(ENUM_VOCABULARIES))
def test_enum_field_rejects_an_unknown_value(name):
    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(MantisACTConfig(), **{name: "no_such_value"})
    message = str(excinfo.value)
    assert name in message
    assert "no_such_value" in message


@pytest.mark.parametrize("name", sorted(ENUM_VOCABULARIES))
def test_every_listed_value_passes_its_own_vocabulary(name):
    """No vocabulary entry is refused by the field it is listed under.

    Some entries still need partner fields moved with them — ``global_mode``
    of ``"none"`` zeroes every latent count — so the construction may raise.
    It must never raise this field's own "is not one of".
    """
    for value in sorted(ENUM_VOCABULARIES[name]):
        try:
            dataclasses.replace(MantisACTConfig(), **{name: value})
        except ValueError as err:
            assert f"{name}={value!r} is not one of" not in str(err)


def test_cross_field_invariants():
    # d_axis == 0 needs the axis channels and axis latents gone with it.
    with pytest.raises(ValueError, match="use_axis_channels"):
        dataclasses.replace(MantisACTConfig(), d_axis=0, num_axis_latents=0)
    with pytest.raises(ValueError, match="num_axis_latents"):
        dataclasses.replace(
            MantisACTConfig(), d_axis=0, use_axis_channels=False
        )
    # global_mode "none" leaves no latent of any kind.
    with pytest.raises(ValueError, match="num_inv_latents"):
        dataclasses.replace(MantisACTConfig(), global_mode="none")
    # Action-set latents and their count move together.
    with pytest.raises(ValueError, match="use_action_set_latents"):
        dataclasses.replace(MantisACTConfig(), num_action_latents=0)
    # Pair messages and pair scope move together.
    with pytest.raises(ValueError, match="use_action_pair_messages"):
        dataclasses.replace(MantisACTConfig(), pair_scope="none")
    # No edge past d_max has a relation class.
    with pytest.raises(ValueError, match="occupied_radius"):
        dataclasses.replace(MantisACTConfig(), occupied_radius=13)
    # No two cells more than five apart share a six-cell window.
    with pytest.raises(ValueError, match="pair_max_distance"):
        dataclasses.replace(MantisACTConfig(), pair_max_distance=6)


def test_numeric_validation():
    with pytest.raises(ValueError, match="num_heads=5"):
        dataclasses.replace(MantisACTConfig(), num_heads=5)
    with pytest.raises(ValueError, match="d_inv=0"):
        dataclasses.replace(MantisACTConfig(), d_inv=0)
    with pytest.raises(ValueError, match="state_blocks=-1"):
        dataclasses.replace(MantisACTConfig(), state_blocks=-1)
    with pytest.raises(ValueError, match="d_axis=-8"):
        dataclasses.replace(MantisACTConfig(), d_axis=-8)
    with pytest.raises(ValueError, match="dropout=1.0"):
        dataclasses.replace(MantisACTConfig(), dropout=1.0)
    with pytest.raises(ValueError, match="layer_scale_init=-1.0"):
        dataclasses.replace(MantisACTConfig(), layer_scale_init=-1.0)


def test_hash_is_sixteen_hex_digits():
    digest = architecture_hash(MantisACTConfig())
    assert len(digest) == 16
    assert set(digest) <= set("0123456789abcdef")


def test_hash_covers_every_semantic_field():
    base = MantisACTConfig()
    reference = architecture_hash(base)
    for field in dataclasses.fields(base):
        if field.name == "dropout":
            continue
        assert architecture_hash(_mutated(base, field.name)) != reference, field.name


def test_dropout_alone_does_not_change_the_hash():
    base = MantisACTConfig()
    assert architecture_hash(dataclasses.replace(base, dropout=0.25)) == (
        architecture_hash(base)
    )


def test_hash_is_stable_across_a_reimport():
    fresh = _reimport()
    for name, cfg in PRESETS.items():
        assert fresh.architecture_hash(fresh.PRESETS[name]) == architecture_hash(cfg)


@pytest.mark.parametrize("seed", ["0", "12345"])
def test_hash_is_stable_across_processes(seed):
    """A fresh interpreter under a different string-hash seed agrees."""
    script = (
        "from mantisnet.models.mantis_act.config import "
        "PRESETS, architecture_hash;"
        "print(' '.join(architecture_hash(c) for c in PRESETS.values()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    expected = " ".join(architecture_hash(c) for c in PRESETS.values())
    assert out.stdout.strip() == expected


def test_summarise_names_every_field_and_the_identity():
    text = summarise(MantisACTConfig())
    for field in dataclasses.fields(MantisACTConfig):
        # The architecture id is the header's subject, carried by value.
        if field.name == "architecture_id":
            continue
        assert field.name in text, field.name
    assert ARCHITECTURE_ID in text
    assert architecture_hash(MantisACTConfig()) in text
    assert "identical" in text


def test_summarise_lists_the_delta_of_an_ablation():
    text = summarise(PRESETS["full_radius6"])
    assert "occupied_radius=6" in text.split("vs full_act_v4:")[1]


def test_summarise_flags_live_windows_as_an_ablation():
    assert "ABLATION" in summarise(PRESETS["full_live_windows"])
    for name in SPEC_PRESETS:
        if name != "full_live_windows":
            assert "ABLATION" not in summarise(PRESETS[name]), name
