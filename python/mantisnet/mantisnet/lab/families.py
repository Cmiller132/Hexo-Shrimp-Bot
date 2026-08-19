"""Checkpoint-family compatibility for the MantisNet laboratory.

Identifies a checkpoint's historical critic format and loads it with its
native readout, without guessing how an experimental head behaved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

import torch
from torch import Tensor, nn

from ..builder import (
    TACTICAL_FEATURES,
    TERN_DEC_CLASSES,
    TERN_OCC_CLASSES,
    TERN_PATTERNS,
    TERN_POST1_CLASSES,
)
from ..attention import AXIS_ROWS, ORBIT_CLASSES
from ..cell_latents import LINE_CLASSES
from ..cell_nodes import ADJACENCY_CLASSES, NEAREST_BUCKETS, RADIUS_CLASSES
from ..klent.train import KlentConfig, _gpu_lock, _policy_q_fn
from ..model import (
    ACTION_LATENTS,
    MAGNITUDE_COUNTS,
    MantisConfig,
    MantisNet,
    ModelOutput,
    strip_legacy_knobs,
)
from ..segments import segment_ids, segment_max
from ..window_pairs import WA_CLASSES


_HISTORIC_OCC_CLASSES = 93
_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.")
_DEAD_KEY_BIAS = re.compile(r"^blocks\.\d+\.(?:wk|wk_wa)\.bias$")


def _drop_dead_key_biases(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Remove only the historical softmax-dead key projection biases."""
    return {
        key: value
        for key, value in state_dict.items()
        if _DEAD_KEY_BIAS.fullmatch(key) is None
    }


@dataclass(frozen=True, slots=True)
class Composition:
    """The fp32 interpretation of one family's native critic logits."""

    name: str
    width: int
    q_formula: str
    mass_formula: str
    _decode: Callable[[Tensor], tuple[Tensor, Tensor]]

    def _q_mass(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        if logits.ndim < 1 or logits.shape[-1] != self.width:
            raise ValueError(
                f"{self.name} composition requires logits ending in width "
                f"{self.width}, got {tuple(logits.shape)}"
            )
        q, mass = self._decode(logits.float())
        return q.float(), mass.float()

    def q_value(self, logits: Tensor) -> Tensor:
        return self._q_mass(logits)[0]

    def mass(self, logits: Tensor) -> Tensor:
        return self._q_mass(logits)[1]

    def q_score(self, logits: Tensor, offsets: Tensor, mass_floor: float) -> Tensor:
        if not 0 < mass_floor <= 1:
            raise ValueError(f"mass_floor must be in (0, 1], got {mass_floor}")
        q, mass = self._q_mass(logits)
        segments = segment_ids(offsets)
        scale = segment_max(mass, segments, offsets.shape[0] - 1).clamp(
            min=mass_floor
        )
        return q / scale.index_select(0, segments)

    @property
    def flags(self) -> dict[str, object]:
        return {
            "name": self.name,
            "native_logits": self.width,
            "q_value": self.q_formula,
            "mass": self.mass_formula,
            "acting_score": "Q / max(max_b M(s,b), mass_floor)",
            "fp32": True,
        }


def _trinomial(logits: Tensor) -> tuple[Tensor, Tensor]:
    positive, negative, zero = logits.softmax(dim=-1).unbind(dim=-1)
    return positive - negative, 1.0 - zero


def _bipolar(logits: Tensor) -> tuple[Tensor, Tensor]:
    positive, negative = logits.sigmoid().unbind(dim=-1)
    return positive - negative, positive + negative


def _scalar(logits: Tensor) -> tuple[Tensor, Tensor]:
    q = logits.squeeze(-1).tanh()
    return q, q.abs()


TRINOMIAL = Composition(
    "trinomial", 3, "softmax(z)[+] - softmax(z)[-]", "1 - softmax(z)[0]", _trinomial
)
BIPOLAR = Composition(
    "bipolar", 2, "sigmoid(z[+]) - sigmoid(z[-])", "sigmoid(z[+]) + sigmoid(z[-])", _bipolar
)
SCALAR = Composition("scalar", 1, "tanh(z)", "abs(tanh(z))", _scalar)


class FamilyMantisNet(MantisNet):
    """A current trunk/decoder with a family's native critic readout."""

    def __init__(self, cfg: MantisConfig, composition: Composition) -> None:
        super().__init__(cfg)
        self.mlp_q.out = nn.Linear(cfg.policy_hidden, composition.width)
        self.family_composition = composition

    def cell_heads(self, w, g, cells, batch, mass_floor):
        policy, critic = self.cell_head_logits(w, g, cells, batch)
        composition = self.family_composition
        return (
            policy,
            composition.q_score(critic, batch.legal_offsets, mass_floor),
            composition.q_value(critic),
        )

    def forward(self, batch, mass_floor: float) -> ModelOutput:
        _stones, windows, token, cells = self.trunk(batch)
        value, value_dist, value_logits = self.value_head(windows, token, batch)
        policy, q_score, q_values = self.cell_heads(
            windows, token, cells, batch, mass_floor
        )
        return ModelOutput(
            policy_logits=policy,
            q_score=q_score,
            q_values=q_values,
            value=value,
            value_dist=value_dist,
            value_logits=value_logits,
        )


def _block_count(state_dict: Mapping[str, Tensor]) -> int | None:
    indices = {
        int(match.group(1))
        for key in state_dict
        if (match := _BLOCK_KEY.match(key)) is not None
    }
    if not indices or indices != set(range(max(indices) + 1)):
        return None
    return max(indices) + 1


_COMMON_KEYS = {
    "stone_table.weight",
    "window_table.weight",
    "token_moves.weight",
    "ln_out.weight",
    "ln_out.bias",
    "p.weight",
    "e_pw.weight",
    "mlp_p.lin_a.weight",
    "mlp_p.lin_a.bias",
    "mlp_p.lin_b.weight",
    "mlp_p.out.weight",
    "mlp_p.out.bias",
    "q.weight",
    "e_qw.weight",
    "mlp_q.lin_a.weight",
    "mlp_q.lin_a.bias",
    "mlp_q.lin_b.weight",
    "mlp_q.out.weight",
    "mlp_q.out.bias",
    "value_queries",
    "ln_value.weight",
    "ln_value.bias",
    "mlp_v.0.weight",
    "mlp_v.0.bias",
    "mlp_v.2.weight",
    "mlp_v.2.bias",
}

_ACT_KEYS = {
    "act_proj.weight", "act_proj.bias", "act_table.weight", "act_empty_base",
    "p_act.weight", "q_act.weight",
}

_BLOCK_SUFFIXES = {
    "ln_ws_s.weight", "ln_ws_s.bias", "ln_ws_w.weight", "ln_ws_w.bias",
    "u.weight", "e_ws.weight", "mlp_w.lin_a.weight", "mlp_w.lin_a.bias",
    "mlp_w.lin_b.weight", "mlp_w.out.weight", "mlp_w.out.bias",
    "ln_sw_w.weight", "ln_sw_w.bias", "ln_sw_s.weight", "ln_sw_s.bias",
    "v.weight", "e_sw.weight", "mlp_s.lin_a.weight", "mlp_s.lin_a.bias",
    "mlp_s.lin_b.weight", "mlp_s.out.weight", "mlp_s.out.bias",
    "ln_attn.weight", "ln_attn.bias", "wq.weight", "wq.bias", "wk.weight",
    "wv.weight", "wv.bias", "wo.weight", "wo.bias", "axis_bias", "orbit_bias",
    "ln_ffn.weight", "ln_ffn.bias", "ffn.0.weight", "ffn.0.bias",
    "ffn.2.weight", "ffn.2.bias",
}

# Trunk-stage parameter groups: the relay (§5.1b), typed window attention
# (§5.1c), and joint incidence classes are baked into this build. The profile is still read off the state dict so an
# unexpected checkpoint shape refuses with its actual profile named, rather
# than a key-set mismatch.
_CP_SUFFIXES = {
    "ln_cp_in.weight", "ln_cp_in.bias", "u_cp.weight", "e_cp.weight",
    "ln_cp_w.weight", "ln_cp_w.bias", "mlp_cp.lin_a.weight",
    "mlp_cp.lin_a.bias", "mlp_cp.lin_b.weight", "mlp_cp.out.weight",
    "mlp_cp.out.bias",
}
_WA_SUFFIXES = {
    "ln_wa.weight", "ln_wa.bias", "wq_wa.weight", "wq_wa.bias",
    "wk_wa.weight", "wv_wa.weight", "wv_wa.bias",
    "wo_wa.weight", "wo_wa.bias", "wa_bias",
}
_LATENT_SUFFIXES = {
    "latent_ln_read_q.weight", "latent_ln_read_q.bias",
    "latent_ln_read_w.weight", "latent_ln_read_w.bias",
    "latent_wq_read.weight", "latent_wq_read.bias", "latent_wk_read.weight",
    "latent_wv_read.weight", "latent_wv_read.bias",
    "latent_wo_read.weight", "latent_wo_read.bias",
    "latent_ln_mix.weight", "latent_ln_mix.bias",
    "latent_wq_mix.weight", "latent_wq_mix.bias", "latent_wk_mix.weight",
    "latent_wv_mix.weight", "latent_wv_mix.bias",
    "latent_wo_mix.weight", "latent_wo_mix.bias",
    "latent_ln_bcast_q.weight", "latent_ln_bcast_q.bias",
    "latent_ln_bcast_l.weight", "latent_ln_bcast_l.bias",
    "latent_wq_bcast.weight", "latent_wq_bcast.bias",
    "latent_wk_bcast.weight", "latent_wv_bcast.weight", "latent_wv_bcast.bias",
    "latent_wo_bcast.weight", "latent_wo_bcast.bias",
}
_CELL_SUFFIXES = {
    "ln_cr_c.weight", "ln_cr_c.bias", "ln_cr_w.weight", "ln_cr_w.bias",
    "cr_wq.weight", "cr_wq.bias", "cr_wk.weight",
    "cr_wv.weight", "cr_wv.bias", "cr_wo.weight", "cr_wo.bias",
    "cr_bias", "cr_vclass.weight",
    "ln_wr_w.weight", "ln_wr_w.bias", "ln_wr_c.weight", "ln_wr_c.bias",
    "wr_wq.weight", "wr_wq.bias", "wr_wk.weight",
    "wr_wv.weight", "wr_wv.bias", "wr_wo.weight", "wr_wo.bias",
    "wr_bias",
}
_RADIUS_SUFFIXES = {
    "ln_radius_c.weight", "ln_radius_c.bias", "ln_radius_s.weight", "ln_radius_s.bias",
    "radius_wq.weight", "radius_wq.bias", "radius_wk.weight",
    "radius_wv.weight", "radius_wv.bias", "radius_wo.weight", "radius_wo.bias",
    "radius_bias", "radius_vclass.weight",
}
_ADJACENCY_SUFFIXES = {
    "ln_adj_q.weight", "ln_adj_q.bias", "ln_adj_k.weight", "ln_adj_k.bias",
    "adj_wq.weight", "adj_wq.bias", "adj_wk.weight",
    "adj_wv.weight", "adj_wv.bias", "adj_wo.weight", "adj_wo.bias",
    "adj_bias", "adj_vclass.weight",
}
_LP_SUFFIXES = {
    "ln_lp.weight", "ln_lp.bias",
    "lp_wq.weight", "lp_wq.bias", "lp_wk.weight",
    "lp_wv.weight", "lp_wv.bias", "lp_wo.weight", "lp_wo.bias",
    "lp_bias",
}
# Step 5/6 knob keys live on the model, not the blocks.
_TACTICAL_KEYS = {
    "tactical_a.weight", "tactical_a.bias",
    "tactical_out.weight", "tactical_out.bias",
}
_ACT_LATENT_KEYS = {
    "act_latent_base",
    "act_ln_read_q.weight", "act_ln_read_q.bias",
    "act_ln_read_rows.weight", "act_ln_read_rows.bias",
    "act_wq_read.weight", "act_wq_read.bias", "act_wk_read.weight",
    "act_wv_read.weight", "act_wv_read.bias",
    "act_wo_read.weight", "act_wo_read.bias",
    "act_ln_mix.weight", "act_ln_mix.bias",
    "act_wq_mix.weight", "act_wq_mix.bias", "act_wk_mix.weight",
    "act_wv_mix.weight", "act_wv_mix.bias",
    "act_wo_mix.weight", "act_wo_mix.bias",
    "act_ln_bcast_l.weight", "act_ln_bcast_l.bias",
    "act_ln_bcast_q.weight", "act_ln_bcast_q.bias",
    "act_wq_bcast.weight", "act_wq_bcast.bias", "act_wk_bcast.weight",
    "act_wv_bcast.weight", "act_wv_bcast.bias",
    "act_wo_bcast.weight", "act_wo_bcast.bias",
}
_MAGNITUDE_KEYS = {
    "magnitude_pattern",
    "magnitude_counts.weight", "magnitude_counts.bias",
}


@dataclass(frozen=True, slots=True)
class _Knobs:
    cell_pass: bool
    cell_pass_from: int
    joint_incidence: bool
    window_attention: bool
    mixed_windows: bool
    state_latents: int
    cell_latents: bool
    line_pass: bool
    cell_nodes: bool
    cell_adjacency: bool
    action_tactical: bool
    action_latents: bool
    global_magnitude: bool


def _knob_profile(state_dict: Mapping[str, Tensor], blocks: int) -> _Knobs:
    """Read the trunk-knob profile off block tensors.

    Detection uses block 0 (block ``cell_pass_from`` for the relay); the exact
    key-set comparison in :func:`_claims` is what rejects a dict whose blocks
    disagree with the detected profile.
    """
    relayed = [
        index
        for index in range(blocks)
        if f"blocks.{index}.u_cp.weight" in state_dict
    ]
    window_table = state_dict.get("window_table.weight")
    mixed = (
        isinstance(window_table, Tensor)
        and window_table.ndim == 2
        and window_table.shape[0] == TERN_PATTERNS
    )
    e_ws = state_dict.get("blocks.0.e_ws.weight")
    latent_base = state_dict.get("latent_base")
    return _Knobs(
        cell_pass=bool(relayed),
        cell_pass_from=relayed[0] if relayed else 0,
        joint_incidence=(
            isinstance(e_ws, Tensor)
            and e_ws.ndim == 2
            and e_ws.shape[0]
            == (TERN_OCC_CLASSES if mixed else _HISTORIC_OCC_CLASSES)
        ),
        window_attention="blocks.0.wa_bias" in state_dict,
        mixed_windows=mixed,
        state_latents=(
            int(latent_base.shape[0])
            if isinstance(latent_base, Tensor) and latent_base.ndim == 2
            else 0
        ),
        cell_latents=(
            "cell_base" in state_dict and "cell_nearest_table.weight" not in state_dict
        ),
        line_pass="blocks.0.lp_bias" in state_dict,
        cell_nodes="cell_nearest_table.weight" in state_dict,
        cell_adjacency="blocks.0.adj_bias" in state_dict,
        action_tactical="tactical_a.weight" in state_dict,
        action_latents="act_latent_base" in state_dict,
        global_magnitude="magnitude_pattern" in state_dict,
    )


# The profile this build instantiates. Window attention remains live, so its
# value is normalized during profile comparison.
_BAKED = _Knobs(
    cell_pass=True,
    cell_pass_from=0,
    joint_incidence=True,
    window_attention=True,
    mixed_windows=True,
    state_latents=4,
    cell_latents=False,
    line_pass=False,
    cell_nodes=False,
    cell_adjacency=False,
    action_tactical=False,
    action_latents=False,
    global_magnitude=False,
)


def _base_keys(blocks: int, knobs: _Knobs) -> set[str]:
    keys = _COMMON_KEYS | {
        f"blocks.{index}.{suffix}"
        for index in range(blocks)
        for suffix in _BLOCK_SUFFIXES
    }
    keys.add("latent_base")
    if knobs.cell_latents or knobs.cell_nodes:
        keys.add("cell_base")
    if knobs.cell_nodes:
        keys.add("cell_nearest_table.weight")
    keys |= _ACT_KEYS
    if knobs.action_tactical:
        keys |= _TACTICAL_KEYS
    if knobs.action_latents:
        keys |= _ACT_LATENT_KEYS
    if knobs.global_magnitude:
        keys |= _MAGNITUDE_KEYS
    for index in range(blocks):
        prefix = f"blocks.{index}."
        if knobs.window_attention:
            keys |= {prefix + suffix for suffix in _WA_SUFFIXES}
        if knobs.line_pass:
            keys |= {prefix + suffix for suffix in _LP_SUFFIXES}
        keys |= {prefix + suffix for suffix in _LATENT_SUFFIXES}
        if knobs.cell_latents or knobs.cell_nodes:
            keys |= {prefix + suffix for suffix in _CELL_SUFFIXES}
        if knobs.cell_nodes:
            keys |= {prefix + suffix for suffix in _RADIUS_SUFFIXES}
        if knobs.cell_adjacency:
            keys |= {prefix + suffix for suffix in _ADJACENCY_SUFFIXES}
        if knobs.cell_pass and index >= knobs.cell_pass_from:
            keys |= {prefix + suffix for suffix in _CP_SUFFIXES}
    return keys


def _shape(state_dict: Mapping[str, Tensor], key: str) -> tuple[int, ...] | None:
    value = state_dict.get(key)
    return tuple(value.shape) if isinstance(value, Tensor) else None


def _claims(
    state_dict: Mapping[str, Tensor],
    *,
    width: int,
    extras: set[str] | None = None,
) -> bool:
    blocks = _block_count(state_dict)
    if blocks is None:
        return False
    knobs = _knob_profile(state_dict, blocks)
    if set(state_dict) != _base_keys(blocks, knobs) | (extras or set()):
        return False
    if not knobs.mixed_windows:
        return False
    stone = _shape(state_dict, "stone_table.weight")
    readout = _shape(state_dict, "mlp_q.out.weight")
    return bool(
        stone is not None
        and len(stone) == 2
        and stone[0] == 2
        and _shape(state_dict, "e_pw.weight") == (TERN_DEC_CLASSES, stone[1])
        and _shape(state_dict, "e_qw.weight") == (TERN_DEC_CLASSES, stone[1])
        and readout is not None
        and len(readout) == 2
        and readout[0] == width
        and _shape(state_dict, "mlp_q.out.bias") == (width,)
    )


def _require_tensor(state_dict: Mapping[str, Tensor], key: str) -> Tensor:
    value = state_dict.get(key)
    if not isinstance(value, Tensor):
        raise ValueError(f"tensor {key!r} is missing or is not a tensor")
    return value


def _expect_shape(state_dict: Mapping[str, Tensor], key: str, shape: tuple[int, ...]) -> None:
    tensor = _require_tensor(state_dict, key)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"tensor {key!r} has shape {tuple(tensor.shape)}, expected {shape}")


def _expected_shapes(
    cfg: MantisConfig, *, table_rows: int, critic_width: int
) -> dict[str, tuple[int, ...]]:
    h, ph, vh, q, k = (
        cfg.h, cfg.policy_hidden, cfg.value_hidden, cfg.value_queries, cfg.value_bins
    )
    shapes = {
        "stone_table.weight": (2, h),
        "window_table.weight": (cfg.window_vocab, h),
        "token_moves.weight": (2, h),
        "ln_out.weight": (h,), "ln_out.bias": (h,),
        "p.weight": (h, h), "e_pw.weight": (table_rows, h),
        "mlp_p.lin_a.weight": (ph, h), "mlp_p.lin_a.bias": (ph,),
        "mlp_p.lin_b.weight": (ph, h), "mlp_p.out.weight": (1, ph),
        "mlp_p.out.bias": (1,),
        "q.weight": (h, h), "e_qw.weight": (table_rows, h),
        "mlp_q.lin_a.weight": (ph, h), "mlp_q.lin_a.bias": (ph,),
        "mlp_q.lin_b.weight": (ph, h),
        "mlp_q.out.weight": (critic_width, ph),
        "mlp_q.out.bias": (critic_width,),
        "value_queries": (q, h), "ln_value.weight": (h,), "ln_value.bias": (h,),
        "mlp_v.0.weight": (vh, q * h), "mlp_v.0.bias": (vh,),
        "mlp_v.2.weight": (k, vh), "mlp_v.2.bias": (k,),
    }
    shapes["latent_base"] = (4, h)
    shapes.update({
        "act_proj.weight": (h, h), "act_proj.bias": (h,),
        "act_table.weight": (TERN_POST1_CLASSES, h),
        "act_empty_base": (h,),
        "p_act.weight": (h, h), "q_act.weight": (h, h),
    })
    if cfg.uses_cell_state:
        shapes["cell_base"] = (h,)
    if cfg.cell_nodes:
        shapes["cell_nearest_table.weight"] = (NEAREST_BUCKETS, h)
    if cfg.action_tactical:
        shapes.update({
            "tactical_a.weight": (h, TACTICAL_FEATURES),
            "tactical_a.bias": (h,),
            "tactical_out.weight": (h, h),
            "tactical_out.bias": (h,),
        })
    if cfg.action_latents:
        shapes["act_latent_base"] = (ACTION_LATENTS, h)
        for name in (
            "act_ln_read_q", "act_ln_read_rows", "act_ln_mix",
            "act_ln_bcast_l", "act_ln_bcast_q",
        ):
            shapes[name + ".weight"] = (h,)
            shapes[name + ".bias"] = (h,)
        for name in (
            "act_wq_read", "act_wv_read", "act_wo_read",
            "act_wq_mix", "act_wv_mix", "act_wo_mix",
            "act_wq_bcast", "act_wv_bcast", "act_wo_bcast",
        ):
            shapes[name + ".weight"] = (h, h)
            shapes[name + ".bias"] = (h,)
        for name in ("act_wk_read", "act_wk_mix", "act_wk_bcast"):
            shapes[name + ".weight"] = (h, h)
    if cfg.global_magnitude:
        shapes["magnitude_pattern"] = (cfg.window_vocab, h)
        shapes["magnitude_counts.weight"] = (h, MAGNITUDE_COUNTS)
        shapes["magnitude_counts.bias"] = (h,)
    fh = cfg.ffn_factor * h
    ln_names = ["ln_ws_s", "ln_ws_w", "ln_sw_w",
                "ln_sw_s", "ln_attn", "ln_ffn"]
    biased = ["wq", "wv", "wo"]
    bias_free = ["wk"]
    if not cfg.uses_cell_state:
        ln_names += ["ln_cp_in", "ln_cp_w"]
    else:
        ln_names += ["ln_cr_c", "ln_cr_w", "ln_wr_w", "ln_wr_c"]
        biased += ["cr_wq", "cr_wv", "cr_wo", "wr_wq", "wr_wv", "wr_wo"]
        bias_free += ["cr_wk", "wr_wk"]
    if cfg.cell_nodes:
        ln_names += ["ln_radius_c", "ln_radius_s"]
        biased += ["radius_wq", "radius_wv", "radius_wo"]
        bias_free += ["radius_wk"]
    if cfg.cell_adjacency:
        ln_names += ["ln_adj_q", "ln_adj_k"]
        biased += ["adj_wq", "adj_wv", "adj_wo"]
        bias_free += ["adj_wk"]
    if cfg.window_attention:
        ln_names.append("ln_wa")
        biased += ["wq_wa", "wv_wa", "wo_wa"]
        bias_free.append("wk_wa")
    if cfg.line_pass:
        ln_names.append("ln_lp")
        biased += ["lp_wq", "lp_wv", "lp_wo"]
        bias_free.append("lp_wk")
    ln_names += [
        "latent_ln_read_q", "latent_ln_read_w", "latent_ln_mix",
        "latent_ln_bcast_q", "latent_ln_bcast_l",
    ]
    biased += [
        "latent_wq_read", "latent_wv_read", "latent_wo_read",
        "latent_wq_mix", "latent_wv_mix", "latent_wo_mix",
        "latent_wq_bcast", "latent_wv_bcast", "latent_wo_bcast",
    ]
    bias_free += [
        "latent_wk_read", "latent_wk_mix", "latent_wk_bcast",
    ]
    for index in range(cfg.blocks):
        prefix = f"blocks.{index}."
        for name in ln_names:
            shapes[prefix + name + ".weight"] = (h,)
            shapes[prefix + name + ".bias"] = (h,)
        mlps = ["mlp_w", "mlp_s"]
        for name in ("u", "v"):
            shapes[prefix + name + ".weight"] = (h, h)
        for name in ("e_ws", "e_sw"):
            shapes[prefix + name + ".weight"] = (cfg.occ_classes, h)
        if cfg.uses_cell_state:
            shapes[prefix + "cr_bias"] = (cfg.heads, cfg.dec_classes)
            shapes[prefix + "cr_vclass.weight"] = (cfg.dec_classes, h)
            shapes[prefix + "wr_bias"] = (cfg.heads, cfg.dec_classes)
        else:
            shapes[prefix + "u_cp.weight"] = (h, h)
            shapes[prefix + "e_cp.weight"] = (cfg.dec_classes, h)
            mlps.append("mlp_cp")
        if cfg.cell_nodes:
            shapes[prefix + "radius_bias"] = (cfg.heads, RADIUS_CLASSES)
            shapes[prefix + "radius_vclass.weight"] = (RADIUS_CLASSES, h)
        if cfg.cell_adjacency:
            shapes[prefix + "adj_bias"] = (cfg.heads, ADJACENCY_CLASSES)
            shapes[prefix + "adj_vclass.weight"] = (ADJACENCY_CLASSES, h)
        for name in mlps:
            shapes[prefix + name + ".lin_a.weight"] = (h, h)
            shapes[prefix + name + ".lin_a.bias"] = (h,)
            shapes[prefix + name + ".lin_b.weight"] = (h, h)
            shapes[prefix + name + ".out.weight"] = (h, h)
            shapes[prefix + name + ".out.bias"] = (h,)
        for name in biased:
            shapes[prefix + name + ".weight"] = (h, h)
            shapes[prefix + name + ".bias"] = (h,)
        for name in bias_free:
            shapes[prefix + name + ".weight"] = (h, h)
        if cfg.window_attention:
            shapes[prefix + "wa_bias"] = (cfg.heads, WA_CLASSES)
        if cfg.line_pass:
            shapes[prefix + "lp_bias"] = (cfg.heads, LINE_CLASSES)
        shapes[prefix + "axis_bias"] = (cfg.heads, AXIS_ROWS)
        shapes[prefix + "orbit_bias"] = (cfg.heads, ORBIT_CLASSES)
        shapes[prefix + "ffn.0.weight"] = (fh, h)
        shapes[prefix + "ffn.0.bias"] = (fh,)
        shapes[prefix + "ffn.2.weight"] = (h, fh)
        shapes[prefix + "ffn.2.bias"] = (h,)
    return shapes


def _baked_profile_error(knobs: _Knobs) -> ValueError:
    return ValueError(
        f"state dict trunk profile {knobs} is not the baked architecture "
        f"{_BAKED}; checkpoints predating a baked stage are not "
        "instantiable in this build and must be measured from a build of "
        "their era (see python/mantisnet/README.md)"
    )


def infer_config(state_dict: Mapping[str, Tensor]) -> MantisConfig:
    """Recover every tensor-backed MantisConfig field from a state dict.

    The dict must carry every baked trunk stage; a knob-era or pre-knob
    checkpoint (missing relay, §5.1c, or axis tensors, or three-row incidence
    tables) names its actual profile in the refusal.
    """

    stone = _require_tensor(state_dict, "stone_table.weight")
    if stone.ndim != 2 or stone.shape[0] != 2 or stone.shape[1] <= 0:
        raise ValueError(
            f"tensor 'stone_table.weight' has shape {tuple(stone.shape)}, expected (2, H)"
        )
    h = int(stone.shape[1])
    blocks = _block_count(state_dict)
    if blocks is None:
        raise ValueError("tensor block indices are missing or not contiguous from blocks.0")

    bias = _require_tensor(state_dict, "blocks.0.orbit_bias")
    if bias.ndim != 2 or bias.shape[0] <= 0 or bias.shape[1] != ORBIT_CLASSES:
        raise ValueError(
            f"tensor 'blocks.0.orbit_bias' has shape {tuple(bias.shape)}, "
            f"expected (heads, {ORBIT_CLASSES})"
        )
    heads = int(bias.shape[0])
    if h % heads:
        raise ValueError(
            "tensors 'stone_table.weight' and 'blocks.0.orbit_bias' imply "
            f"H={h}, heads={heads}, which do not divide evenly"
        )

    ffn = _require_tensor(state_dict, "blocks.0.ffn.0.weight")
    if ffn.ndim != 2 or ffn.shape[0] <= 0 or ffn.shape[1] != h or ffn.shape[0] % h:
        raise ValueError(
            f"tensor 'blocks.0.ffn.0.weight' has shape {tuple(ffn.shape)}, expected (ffn_factor*{h}, {h})"
        )
    ffn_factor = int(ffn.shape[0] // h)

    policy = _require_tensor(state_dict, "mlp_p.lin_a.weight")
    if policy.ndim != 2 or policy.shape[1] != h or policy.shape[0] <= 0:
        raise ValueError(
            f"tensor 'mlp_p.lin_a.weight' has shape {tuple(policy.shape)}, expected (policy_hidden, {h})"
        )
    policy_hidden = int(policy.shape[0])

    queries = _require_tensor(state_dict, "value_queries")
    if queries.ndim != 2 or queries.shape[1] != h or queries.shape[0] <= 0:
        raise ValueError(
            f"tensor 'value_queries' has shape {tuple(queries.shape)}, expected (value_queries, {h})"
        )
    value_queries = int(queries.shape[0])

    value_first = _require_tensor(state_dict, "mlp_v.0.weight")
    if (
        value_first.ndim != 2
        or value_first.shape[0] <= 0
        or value_first.shape[1] != value_queries * h
    ):
        raise ValueError(
            f"tensor 'mlp_v.0.weight' has shape {tuple(value_first.shape)}, expected "
            f"(value_hidden, {value_queries * h})"
        )
    value_hidden = int(value_first.shape[0])

    value_out = _require_tensor(state_dict, "mlp_v.2.weight")
    if value_out.ndim != 2 or value_out.shape[1] != value_hidden or value_out.shape[0] <= 0:
        raise ValueError(
            f"tensor 'mlp_v.2.weight' has shape {tuple(value_out.shape)}, expected "
            f"(value_bins, {value_hidden})"
        )
    value_bins = int(value_out.shape[0])
    if value_bins % 2 == 0:
        raise ValueError(
            f"tensor 'mlp_v.2.weight' implies even value_bins={value_bins}; an odd width is required"
        )

    knobs = _knob_profile(state_dict, blocks)
    # The live knobs read off the dict are accepted as-is; everything else —
    # including the baked latents, and the relay, which is present exactly
    # when the cell-latent stage has not replaced it — must be the baked
    # profile.
    expected = replace(
        _BAKED,
        window_attention=knobs.window_attention,
        cell_latents=knobs.cell_latents,
        line_pass=knobs.line_pass,
        cell_nodes=knobs.cell_nodes,
        cell_adjacency=knobs.cell_adjacency,
        cell_pass=not (knobs.cell_latents or knobs.cell_nodes),
        action_tactical=knobs.action_tactical,
        action_latents=knobs.action_latents,
        global_magnitude=knobs.global_magnitude,
    )
    if knobs != expected:
        raise _baked_profile_error(knobs)
    cfg = MantisConfig(
        h=h,
        blocks=blocks,
        heads=heads,
        ffn_factor=ffn_factor,
        value_queries=value_queries,
        value_bins=value_bins,
        policy_hidden=policy_hidden,
        value_hidden=value_hidden,
        dropout=0.0,
        window_attention=knobs.window_attention,
        cell_latents=knobs.cell_latents,
        line_pass=knobs.line_pass,
        cell_nodes=knobs.cell_nodes,
        cell_adjacency=knobs.cell_adjacency,
        action_tactical=knobs.action_tactical,
        action_latents=knobs.action_latents,
        global_magnitude=knobs.global_magnitude,
    )

    table = _require_tensor(state_dict, "e_pw.weight")
    critic = _require_tensor(state_dict, "mlp_q.out.weight")
    allowed_rows = {TERN_DEC_CLASSES}
    if table.ndim != 2 or table.shape[0] not in allowed_rows:
        raise ValueError(
            f"tensor 'e_pw.weight' has shape {tuple(table.shape)}, expected rows "
            f"in {sorted(allowed_rows)} for this build"
        )
    if critic.ndim != 2 or critic.shape[0] not in {1, 2, 3}:
        raise ValueError(
            f"tensor 'mlp_q.out.weight' has shape {tuple(critic.shape)}, expected native width 1, 2, or 3"
        )
    for key, shape in _expected_shapes(
        cfg, table_rows=int(table.shape[0]), critic_width=int(critic.shape[0])
    ).items():
        _expect_shape(state_dict, key, shape)
    return cfg


@dataclass(frozen=True, slots=True)
class FamilyEntry:
    name: str
    claims: Callable[[Mapping[str, Tensor]], bool]
    scoreable: bool
    composition: Composition | None
    reason: str | None = None

    def load(
        self,
        checkpoint_model_dict: Mapping[str, Tensor],
        cfg: MantisConfig,
        device: str | torch.device,
    ) -> nn.Module:
        if not self.scoreable or self.composition is None:
            raise ValueError(_unscoreable_message(self))
        model = FamilyMantisNet(cfg, self.composition)
        model.load_state_dict(checkpoint_model_dict, strict=True)
        return model.to(device).eval()


def _entry(
    name: str,
    width: int,
    composition: Composition | None,
    *,
    extras: set[str] | None = None,
    reason: str | None = None,
) -> FamilyEntry:
    return FamilyEntry(
        name=name,
        claims=lambda state, w=width, e=extras: _claims(state, width=w, extras=e),
        scoreable=composition is not None,
        composition=composition,
        reason=reason,
    )


_COMPAT_REQUIREMENT = (
    "the lab contract (python/mantisnet/README.md) requires a family entry and a composition-parity test "
    "before checkpoints with a new critic parameterization, decoder key, or head format are scoreable"
)

FAMILIES: tuple[FamilyEntry, ...] = (
    _entry("trinomial-joint", 3, TRINOMIAL),
    _entry("bipolar-joint", 2, BIPOLAR),
    _entry("scalar-joint", 1, SCALAR),
)


def _names(entries=FAMILIES) -> str:
    return ", ".join(entry.name for entry in entries)


def _unscoreable_message(entry: FamilyEntry) -> str:
    return f"checkpoint family {entry.name!r} is not scoreable: {entry.reason}; {_COMPAT_REQUIREMENT}"


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    model: nn.Module
    family: FamilyEntry
    config: MantisConfig
    composition: Composition
    versions: Mapping[str, object]
    iteration: int | None

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "family": self.family.name,
            "versions": dict(self.versions),
            "iteration": self.iteration,
            "composition": self.composition.flags,
        }


def load_checkpoint(
    path: str | Path,
    *,
    family: str | None = None,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Identify, version-check, and load a production checkpoint for the lab."""

    checkpoint_path = Path(path)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("model"), Mapping):
        raise ValueError(f"{checkpoint_path} is not a production checkpoint with a model state dict")
    # Historical family checkpoints may carry these two exact per-block keys.
    # They never affected a softmax output, so the lab can score the checkpoint
    # exactly after dropping them. Ordinary KLENT loaders remain strict.
    state = _drop_dead_key_biases(raw["model"])
    blocks = _block_count(state)
    if blocks is not None:
        knobs = _knob_profile(state, blocks)
        if knobs.state_latents != _BAKED.state_latents:
            raise _baked_profile_error(knobs)
    versions = raw.get("versions")
    if not isinstance(versions, Mapping):
        raise ValueError(f"checkpoint {checkpoint_path} has no versions mapping")

    import hexo_py

    for key, running in (
        ("RULES_VERSION", hexo_py.RULES_VERSION),
        ("ACTION_ORDER_VERSION", hexo_py.ACTION_ORDER_VERSION),
    ):
        stored = versions.get(key)
        if stored != running:
            raise ValueError(
                f"checkpoint {key}={stored!r} does not match running hexo_py {key}={running!r}"
            )

    if family is not None:
        matches = [entry for entry in FAMILIES if entry.name == family]
        if not matches:
            raise ValueError(f"unknown checkpoint family {family!r}; registered families: {_names()}")
        entry = matches[0]
        if not entry.claims(state):
            candidates = [candidate for candidate in FAMILIES if candidate.claims(state)]
            found = _names(candidates) if candidates else "none"
            raise ValueError(
                f"checkpoint does not structurally claim named family {family!r}; "
                f"structural candidates: {found}"
            )
    else:
        candidates = [entry for entry in FAMILIES if entry.claims(state)]
        if not candidates:
            raise ValueError(
                f"checkpoint is not identifiable by the family registry ({_names()}); "
                "see python/mantisnet/README.md for the compatibility contract"
            )
        if len(candidates) != 1:
            raise ValueError(
                f"checkpoint family is ambiguous among {_names(candidates)}; "
                "pass --family with one of those candidates"
            )
        entry = candidates[0]

    if not entry.scoreable:
        raise ValueError(_unscoreable_message(entry))
    config = infer_config(state)
    recorded = raw.get("model_config")
    if recorded is not None:
        if not isinstance(recorded, Mapping):
            raise ValueError(
                f"checkpoint {checkpoint_path} model_config is not a mapping"
            )
        recorded_cfg = MantisConfig(**strip_legacy_knobs(dict(recorded)))
        # claim_reach leaves no tensor trace, so shape inference cannot see
        # it; it is compared out here and adopted from the record below.
        if replace(recorded_cfg, dropout=0.0, claim_reach=config.claim_reach) != config:
            raise ValueError(
                f"checkpoint model_config {recorded_cfg} does not match the "
                f"configuration inferred from its tensors {config}"
            )
        config = replace(config, claim_reach=recorded_cfg.claim_reach)
    model = entry.load(state, config, device)
    assert entry.composition is not None
    return LoadedCheckpoint(
        model=model,
        family=entry,
        config=config,
        composition=entry.composition,
        versions=dict(versions),
        iteration=raw.get("iteration"),
    )


def composition_evaluate(model, composition: Composition, cfg: KlentConfig):
    """Build the production evaluator seam with an explicit composition."""

    policy_q = _policy_q_fn(cfg)

    def evaluate(batch):
        with _gpu_lock:
            moved = batch.to(cfg.device)
            with torch.no_grad(), torch.autocast(
                cfg.device, torch.bfloat16, enabled=cfg.autocast
            ):
                policy, critic = policy_q(model, moved)
            return (
                policy.float().cpu(),
                composition.q_score(
                    critic, moved.legal_offsets, cfg.mass_floor
                ).cpu(),
                composition.q_value(critic).cpu(),
            )

    return evaluate


def family_evaluate(loaded: LoadedCheckpoint, cfg: KlentConfig):
    return composition_evaluate(loaded.model, loaded.composition, cfg)
