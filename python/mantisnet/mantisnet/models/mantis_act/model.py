"""Assembled model: §25's forward, §28's checkpoint, and the KLENT seam.

Composes ``StateTrunk``, ``ActionEncoder``, and ``ActionHeads`` — holds no
layers of its own. ``ACTOutput`` carries §25's six named fields; optional
heads live in ``aux`` (absent for heads the config omits, §32).

``policy_q`` is the KLENT interface: the same method MantisNet exposes, so
``klent/train.py`` names neither architecture's internals (§37.10).

Loading is strict (§28): ``load_checkpoint`` refuses mismatched format,
architecture id, repr version, config fields, head weights, or state-dict
keys. ``dropout`` is the one config field allowed to differ.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Sequence

from torch import Tensor, nn

from .action_encoder import ActionEncoder, ActionOutput
from .builder import collate_prefixes
from .config import (
    ARCHITECTURE_ID,
    MANTIS_ACT_REPR_VERSION,
    UNHASHED_FIELDS,
    MantisACTConfig,
    architecture_hash,
)
from .heads import ActionHeads
from .packed import ACTChunkCost, PackedACTBatch
from .state_trunk import StateTrunk, TrunkOutput, refuse_unimplemented_paths

# The on-disk shape of an ACT checkpoint. Formats here are not backward
# compatible: a change bumps this and the affected checkpoints are regenerated,
# rather than the loader learning to read two shapes.
ACT_CHECKPOINT_FORMAT = 1

# The key whose presence identifies a payload as this architecture's. A payload
# without it is refused before anything else is read, which is what keeps a
# MantisNet checkpoint from reaching a state-dict comparison.
FORMAT_KEY = "act_checkpoint_format"

_CONFIG_FIELDS = frozenset(f.name for f in fields(MantisACTConfig))

# The two sites this module exposes past the trunk's own (§31): the action
# state after the last action block and the action-set latents beside it.
_ACTION_SITES = ("action.state", "action.latent")

# The five ragged families of a packed batch, each with its own CSR offsets.
_OFFSET_FIELDS = (
    "cell_offsets",
    "window_offsets",
    "legal_offsets",
    "adjacency_offsets",
    "radius_offsets",
)


@dataclass(frozen=True, eq=False)
class ACTOutput:
    """§25's model output, flat over every legal action of every position.

    ``policy_logits`` is ``(N_legal,)`` and ``critic_logits`` is
    ``(N_legal, classes)``, both in engine legal order within each position;
    ``q_value`` and ``q_score`` are their fp32 composition (§23.2, §27).
    ``legal_offsets`` gives each position's slice, so a consumer never
    rediscovers a segment boundary.

    ``aux`` is everything past those four tensors: each optional head's logits
    and its ``.mask`` of labelled rows (§24), §23.2's ``committed_mass``, and
    §23.3's ``state_value`` when the configuration holds that head. A head the
    configuration does not hold contributes no key.

    Equality is identity: the fields are tensors, and an elementwise ``==``
    masquerading as an output comparison is a trap rather than a convenience.
    """

    policy_logits: Tensor
    critic_logits: Tensor
    q_value: Tensor
    q_score: Tensor
    legal_offsets: Tensor
    aux: dict[str, Tensor]


def _require_mapping(what: str, value: object) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _head_record(
    aux_weights: Mapping[str, float],
    window_fate_weight: float,
    instantiate_zero_weight: bool,
) -> dict[str, object]:
    """The §24 head weights as a checkpoint records them.

    Sorted by name and cast to plain floats, so two processes that built the
    same model write the same record and a comparison is a value comparison
    rather than a dict-order one.
    """
    return {
        "action_aux": {
            str(name): float(weight) for name, weight in sorted(aux_weights.items())
        },
        "window_fate": float(window_fate_weight),
        "instantiate_zero_weight": bool(instantiate_zero_weight),
    }


def _normalised_head_record(recorded: object) -> dict[str, object]:
    """A recorded head block in the shape :func:`_head_record` writes."""
    block = _require_mapping("the recorded head_weights", recorded)
    unknown = sorted(set(block) - {"action_aux", "window_fate", "instantiate_zero_weight"})
    missing = sorted({"action_aux", "window_fate", "instantiate_zero_weight"} - set(block))
    if unknown or missing:
        raise ValueError(
            f"the recorded head_weights block is not this build's: missing "
            f"{missing}, unexpected {unknown}"
        )
    return _head_record(
        _require_mapping("the recorded action_aux weights", block["action_aux"]),
        block["window_fate"],
        block["instantiate_zero_weight"],
    )


def _require_act_payload(payload: object) -> Mapping:
    """Refuse anything that is not this architecture's checkpoint (§28).

    The format key identifies a payload as ACT's, checked before any other
    field: a MantisNet checkpoint has a ``model_config`` and a ``model`` too,
    so without this check a later comparison would report a confusing symptom
    of the real problem.
    """
    block = _require_mapping("an ACT checkpoint", payload)
    if FORMAT_KEY not in block:
        raise ValueError(
            f"this payload carries no {FORMAT_KEY!r} and is not a "
            f"{ARCHITECTURE_ID} checkpoint (its keys are "
            f"{sorted(str(key) for key in block)}). MantisNet checkpoints are "
            "never converted into this model: §28 makes loading strict and "
            "conversion-free, and the two architectures share no "
            "representation, no parameter, and no layout"
        )
    return block


def config_from_record(recorded: object) -> MantisACTConfig:
    """The recorded ``model_config`` as a `MantisACTConfig` (§28).

    The field set must be this build's exactly: a missing field would be
    filled by a default the checkpoint never chose, and an extra one names a
    field this build no longer has. Loading is conversion-free (§28).
    """
    block = _require_mapping("the recorded model_config", recorded)
    names = set(block)
    missing = sorted(_CONFIG_FIELDS - names)
    unexpected = sorted(names - _CONFIG_FIELDS)
    if missing or unexpected:
        raise ValueError(
            f"the recorded model_config is not a {ARCHITECTURE_ID} configuration: "
            f"missing {missing}, unexpected {unexpected}. Loading is "
            "conversion-free (§28), so a differing field set names an "
            "architecture this build does not implement"
        )
    # Constructing revalidates every enum and cross-field invariant, so a
    # recorded configuration this build cannot express is refused by name.
    return MantisACTConfig(**dict(block))


class MantisACT(nn.Module):
    """§25: a packed batch of positions to one policy logit and value per action.

    ```python
    model = MantisACT()                       # the full_act_v4 defaults of §6
    out = model(batch, mass_floor=0.2)        # §25's ACTOutput
    policy, critic = model.policy_q(batch)    # the KLENT pass (§2, §37.10)
    ```

    ``aux_weights`` and ``window_fate_weight`` are §24's training-only head
    weights. They are constructor arguments rather than config fields because
    they are loss weights, but they decide which auxiliary heads exist, so they
    are recorded in the checkpoint and checked on load. A zero weight means the
    head is absent (§24); ``instantiate_zero_weight`` is the explicit debug
    option §24 allows for inspecting a disabled head's shapes.

    The forward is linear in every node, edge, and action family (§3.14, §26):
    the trunk is one gather and one segment reduction per edge family per
    block, the action encoder is a constant eighteen rows per action, and every
    attention in either stage has a configured constant for its key count.
    """

    def __init__(
        self,
        cfg: MantisACTConfig | None = None,
        *,
        aux_weights: Mapping[str, float] | None = None,
        window_fate_weight: float = 0.0,
        instantiate_zero_weight: bool = False,
    ) -> None:
        super().__init__()
        cfg = cfg or MantisACTConfig()
        # Before any submodule, so a configuration this build cannot honour is
        # refused with the missing input named rather than half-constructed.
        refuse_unimplemented_paths(cfg)
        self.cfg = cfg

        self.aux_weights = {
            str(name): float(weight) for name, weight in dict(aux_weights or {}).items()
        }
        self.window_fate_weight = float(window_fate_weight)
        self.instantiate_zero_weight = bool(instantiate_zero_weight)

        self.trunk = StateTrunk(cfg)
        self.actions = ActionEncoder(cfg)
        self.heads = ActionHeads(
            cfg,
            aux_weights=self.aux_weights,
            window_fate_weight=self.window_fate_weight,
            instantiate_zero_weight=self.instantiate_zero_weight,
        )

    # --- the batch check no single stage can make ---------------------------

    def _require_batch(self, batch: PackedACTBatch) -> None:
        """Refuse a batch whose families disagree about the position count.

        Each stage checks only the tables it reads; a batch whose five CSR
        families were built for different position counts still passes every
        per-stage check and runs over the wrong segments. Checked once here,
        before an embedding is indexed.
        """
        if not isinstance(batch, PackedACTBatch):
            raise TypeError(
                f"MantisACT consumes a PackedACTBatch, got "
                f"{type(batch).__name__}; `builder.collate_positions` and "
                "`packed.collate` are what produce one"
            )
        positions = int(batch.position_count)
        if positions < 1:
            raise ValueError(f"a batch must hold at least one position, got {positions}")
        for name in _OFFSET_FIELDS:
            offsets = getattr(batch, name)
            if offsets.ndim != 1 or int(offsets.shape[0]) != positions + 1:
                raise ValueError(
                    f"{name} must be ({positions + 1},) for a {positions}-position "
                    f"batch, got {tuple(offsets.shape)}"
                )
        for name in ("phase_id", "moves_remaining"):
            values = getattr(batch, name)
            if values.ndim != 1 or int(values.shape[0]) != positions:
                raise ValueError(
                    f"{name} must be ({positions},) for a {positions}-position "
                    f"batch, got {tuple(values.shape)}"
                )
        if batch.global_numeric.ndim != 2 or int(batch.global_numeric.shape[0]) != positions:
            raise ValueError(
                f"global_numeric must be ({positions}, G) for a {positions}-position "
                f"batch, got {tuple(batch.global_numeric.shape)}"
            )

    # --- the shared stages --------------------------------------------------

    def _encode(self, batch: PackedACTBatch) -> tuple[TrunkOutput, ActionOutput]:
        """The state trunk and the action encoder, which every entry point runs."""
        self._require_batch(batch)
        trunk = self.trunk(batch)
        return trunk, self.actions(batch, trunk)

    def _output(
        self,
        batch: PackedACTBatch,
        trunk: TrunkOutput,
        actions: ActionOutput,
        mass_floor: float | None,
    ) -> ACTOutput:
        """§23 and §24's heads over an encoded batch, as §25's output."""
        head = self.heads(
            actions.actions,
            legal_offsets=batch.legal_offsets,
            mass_floor=mass_floor,
            latents=trunk.latents,
            phase_id=batch.phase_id,
            windows=trunk.windows,
            window_status=batch.window_status,
        )
        aux = dict(head.aux)
        if head.committed_mass is not None:
            aux["committed_mass"] = head.committed_mass
        if head.state_value is not None:
            aux["state_value"] = head.state_value
        return ACTOutput(
            policy_logits=head.policy_logits,
            critic_logits=head.critic_logits,
            q_value=head.q_value,
            q_score=head.q_score,
            legal_offsets=batch.legal_offsets,
            aux=aux,
        )

    # --- the KLENT seam (§2, §25, §37.10) -----------------------------------

    def policy_q(self, batch: PackedACTBatch) -> tuple[Tensor, Tensor]:
        """The raw ``(policy_logits, critic_logits)`` KLENT trains on.

        The fitter scores the taken row and the evaluator composes both Q roles
        outside autocast, off these same logits. MantisNet answers the same
        method over its own trunk, so ``klent/train.py`` names neither
        architecture's internals.
        """
        trunk, actions = self._encode(batch)
        return self.heads.logits(
            actions.actions,
            legal_offsets=batch.legal_offsets,
            latents=trunk.latents,
        )

    # §29 gives this architecture no binned state-value head (§23.3's optional
    # head is a different quantity; see `supervised_heads` below).
    has_state_value_head = False

    def supervised_heads(
        self, batch: PackedACTBatch
    ) -> tuple[Tensor, Tensor, None, None]:
        """The lab's supervised pass: ``policy_q``'s pair and no state value.

        §29 gives this architecture no binned state-value head; §23.3's
        optional one is a different auxiliary scalar that the supervised
        recipe's third term and the lab's ``state_value`` channel do not read.
        ``critic_logits`` is the action-value categorical KLENT trains and
        every corpus comparison scores, so the critic is not missing — only
        the binned state-value term is absent.

        An enabled §23.3 head is refused rather than silently unscored: its
        parameters would be counted, trained by nothing, and read by no
        channel.
        """
        if self.heads.state_value is not None:
            raise ValueError(
                "enable_state_value_head=True instantiates §23.3's auxiliary "
                "scalar state value, which the supervised recipe neither trains "
                "nor scores: its third loss term and the lab's state_value "
                "channel read binned state-value logits. Remove the override"
            )
        policy_logits, critic_logits = self.policy_q(batch)
        return policy_logits, critic_logits, None, None

    def collate_prefixes(self, games, ts) -> PackedACTBatch:
        """Stored move prefixes as one packed batch of this representation.

        What a batch of ``(game, ply)`` pairs is depends on the configuration —
        the window and cell scopes decide the node set, ``d_max`` the relation
        vocabulary — so the model that will consume it is what builds it.
        MantisNet answers the same method over its own builder.
        """
        return collate_prefixes(games, ts, self.cfg)

    def chunk_cost(self, stones, legal, budgets) -> ACTChunkCost:
        """The packing law of ``fitloop``, over this model's own limits (§26).

        ``stones[i]`` is sample ``i``'s stone count — its ply — and
        ``legal[i]`` its legal-move count. Nothing here is padded, so unlike
        MantisNet there is no term quadratic in a chunk's longest position;
        :class:`ACTChunkCost` documents the quantity that binds.
        """
        return ACTChunkCost(stones, legal, budgets.graph_cell_budget)

    # --- §25's full forward -------------------------------------------------

    def forward(self, batch: PackedACTBatch, mass_floor: float | None) -> ACTOutput:
        """Every head this configuration holds, for one packed batch (§25).

        ``mass_floor`` is §23.2's acting-score scaling, and ``None`` means no
        scaling — the acting score is the value. It is required rather than
        defaulted because it decides the quantity π′ ranks by, and a caller that
        has not said which it wants has not decided.
        """
        trunk, actions = self._encode(batch)
        return self._output(batch, trunk, actions, mass_floor)

    # --- the debug surface (§31) --------------------------------------------

    def debug_sites(self) -> tuple[str, ...]:
        """Every site :meth:`debug_forward` can expose, in forward order."""
        return (*self.trunk.debug_sites(), *_ACTION_SITES)

    def debug_forward(
        self,
        batch: PackedACTBatch,
        mass_floor: float | None,
        capture: Sequence[str] | None = None,
    ) -> tuple[ACTOutput, dict[str, Tensor]]:
        """The same forward, plus the selected intermediate tensors (§31).

        ``capture`` names sites from :meth:`debug_sites` and defaults to all of
        them; the returned dict is keyed ``<site>.inv`` and ``<site>.axis``, and
        a stream the configuration removes contributes no key. The result is
        numerically identical to :meth:`forward` — the collector reads the same
        tensors rather than recomputing anything.
        """
        available = self.debug_sites()
        requested = tuple(available if capture is None else capture)
        unknown = [name for name in requested if name not in available]
        if unknown:
            raise ValueError(
                f"unknown debug site(s) {unknown}; this model exposes {list(available)}"
            )

        self._require_batch(batch)
        trunk_sites = [name for name in requested if name in self.trunk.debug_sites()]
        trunk, tensors = self.trunk.debug_forward(batch, trunk_sites)
        actions = self.actions(batch, trunk)

        wanted = frozenset(requested)
        if "action.state" in wanted:
            tensors["action.state.inv"] = actions.actions.inv
            if actions.actions.axis is not None:
                tensors["action.state.axis"] = actions.actions.axis
        if "action.latent" in wanted and actions.latents.inv is not None:
            tensors["action.latent.inv"] = actions.latents.inv

        return self._output(batch, trunk, actions, mass_floor), tensors

    # --- §28 checkpointing --------------------------------------------------

    def checkpoint_identity(self) -> dict[str, object]:
        """What a checkpoint records about this model's architecture (§28).

        The architecture id, the representation version, the complete resolved
        config, its stable hash, and the §24 head weights. The hash is stored
        as well as derivable so a load can check the record against itself: a
        payload whose hash disagrees with its own config was written by a
        different build, or truncated.
        """
        return {
            FORMAT_KEY: ACT_CHECKPOINT_FORMAT,
            "architecture_id": ARCHITECTURE_ID,
            "repr_version": MANTIS_ACT_REPR_VERSION,
            "architecture_hash": architecture_hash(self.cfg),
            "model_config": asdict(self.cfg),
            "head_weights": _head_record(
                self.aux_weights, self.window_fate_weight, self.instantiate_zero_weight
            ),
        }

    def checkpoint(self) -> dict[str, object]:
        """The full payload: this model's identity and its state dict (§28)."""
        return {**self.checkpoint_identity(), "model": self.state_dict()}

    def load_checkpoint(self, payload: Mapping[str, object]) -> None:
        """Strict-load ``payload`` into this model, refusing any mismatch (§28).

        Every check names what disagreed, and identity and config are compared
        before the state dict: a key mismatch is what a config mismatch looks
        like from the bottom, and it is far harder to read.
        """
        block = _require_act_payload(payload)
        recorded_format = block[FORMAT_KEY]
        if recorded_format != ACT_CHECKPOINT_FORMAT:
            raise ValueError(
                f"checkpoint format {recorded_format!r} against this build's "
                f"{ACT_CHECKPOINT_FORMAT}; formats are not backward compatible, "
                "so the checkpoint is regenerated rather than converted"
            )
        if block.get("architecture_id") != ARCHITECTURE_ID:
            raise ValueError(
                f"checkpoint architecture_id {block.get('architecture_id')!r} "
                f"against this model's {ARCHITECTURE_ID!r}"
            )
        if block.get("repr_version") != MANTIS_ACT_REPR_VERSION:
            raise ValueError(
                f"checkpoint representation version {block.get('repr_version')!r} "
                f"against this build's {MANTIS_ACT_REPR_VERSION}"
            )

        recorded_cfg = config_from_record(block.get("model_config"))
        recorded_hash = architecture_hash(recorded_cfg)
        if block.get("architecture_hash") != recorded_hash:
            raise ValueError(
                f"checkpoint architecture_hash {block.get('architecture_hash')!r} "
                f"does not describe its own recorded config, whose hash is "
                f"{recorded_hash!r}"
            )
        differing = sorted(
            f.name
            for f in fields(MantisACTConfig)
            if f.name not in UNHASHED_FIELDS
            and getattr(recorded_cfg, f.name) != getattr(self.cfg, f.name)
        )
        if differing:
            detail = ", ".join(
                f"{name}: checkpoint {getattr(recorded_cfg, name)!r} against model "
                f"{getattr(self.cfg, name)!r}"
                for name in differing
            )
            raise ValueError(
                f"checkpoint architecture {recorded_hash} does not match this "
                f"model's {architecture_hash(self.cfg)}; {detail}"
            )

        recorded_heads = _normalised_head_record(block.get("head_weights"))
        own_heads = _head_record(
            self.aux_weights, self.window_fate_weight, self.instantiate_zero_weight
        )
        if recorded_heads != own_heads:
            raise ValueError(
                f"checkpoint §24 head weights {recorded_heads} against this "
                f"model's {own_heads}: they decide which auxiliary heads exist, "
                "so a difference is a different parameter set"
            )

        state = block.get("model")
        if not isinstance(state, Mapping):
            raise TypeError(
                f"the checkpoint's 'model' entry must be a state dict, got "
                f"{type(state).__name__}"
            )
        # Strict and conversion-free (§28): no remap, no prefix strip, no
        # tolerated extra or missing tensor.
        self.load_state_dict(dict(state), strict=True)

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, object]) -> "MantisACT":
        """Rebuild the recorded model and strict-load it (§28).

        The configuration and the §24 head weights come from the payload, so a
        caller loading a run does not restate them. :meth:`load_checkpoint`
        still runs every check afterwards, so a payload whose hash and config
        disagree is refused rather than silently trusted.
        """
        block = _require_act_payload(payload)
        heads = _normalised_head_record(block.get("head_weights"))
        model = cls(
            config_from_record(block.get("model_config")),
            aux_weights=heads["action_aux"],
            window_fate_weight=heads["window_fate"],
            instantiate_zero_weight=heads["instantiate_zero_weight"],
        )
        model.load_checkpoint(block)
        return model


__all__ = [
    "ACT_CHECKPOINT_FORMAT",
    "FORMAT_KEY",
    "ACTOutput",
    "MantisACT",
    "config_from_record",
]
