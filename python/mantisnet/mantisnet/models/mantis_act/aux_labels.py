"""§24.1's six auxiliary labels, computed from the board without search.

Each label is a deterministic function of the current board and one
hypothetical own placement, read off pre- and post-placement window codes
from ``actions.py``. ``heads.py`` defines what each head predicts and how
wide it is.

``winning_partner_count`` counts distinct cells, not windows; an action that
wins outright gets count ``0`` (the turn ends on a win, §2).
"""

from __future__ import annotations

import numpy as np

from .actions import ActionTables
from .heads import AUX_COUNT_CAP, AUX_SPECS
from .packed import WINDOW_LEN
from .pattern_classes import OPP_COUNT, OWN_COUNT
from .symmetry import AXES

# The ternary code of a window holding an own stone in every slot, and the map
# from a power of three back to the slot that carries it. A five-own, no-
# opponent window differs from the all-own code exactly at its one empty slot,
# so `ALL_OWN - code` is `3 ** slot` and this table inverts it.
_ALL_OWN_CODE = int((3 ** np.arange(WINDOW_LEN, dtype=np.int64)).sum())
_POWER_TO_SLOT = np.full(_ALL_OWN_CODE + 1, -1, dtype=np.int64)
_POWER_TO_SLOT[3 ** np.arange(WINDOW_LEN, dtype=np.int64)] = np.arange(
    WINDOW_LEN, dtype=np.int64
)
_POWER_TO_SLOT.setflags(write=False)

# The axes a coordinate is packed on to make a set membership test one integer
# comparison. The legal radius is 8 steps per placement, so a coordinate is
# bounded by the ply, and this bias covers any game the engine can produce.
_COORD_BIAS = 1 << 20
_COORD_STRIDE = 1 << 21

# How many opponent stones make a window a threat the placement kills: a
# window holding fewer cannot be completed on the opponent's next turn, and
# one holding six has already ended the game.
_THREAT_STONES = (4, 5)

# The number of classes each label takes, derived from the rules rather than
# from the head that consumes it. A window holds six cells, so an own
# post-placement occupancy is one of 0..6; the three counted labels saturate at
# §24.1's cap; the two binary ones are 0 or 1.
_LABEL_CLASSES: dict[str, int] = {
    "win_now": 1,
    "own_max_occupancy": WINDOW_LEN + 1,
    "opponent_threats_hit": AUX_COUNT_CAP + 1,
    "own_five_windows_after": AUX_COUNT_CAP + 1,
    "winning_partner_exists": 1,
    "winning_partner_count": AUX_COUNT_CAP + 1,
}


def _check_vocabularies() -> None:
    """Refuse at import a §24.1 head/label width mismatch.

    The head's width is declared in `heads.py`; the label's range is fixed
    here by the rules. The two can drift since they live in different files.
    """
    if set(_LABEL_CLASSES) != set(AUX_SPECS):
        raise RuntimeError(
            f"this module labels {sorted(_LABEL_CLASSES)} against §24.1's "
            f"auxiliaries {sorted(AUX_SPECS)}"
        )
    wrong = {
        name: (width, AUX_SPECS[name].logits)
        for name, width in _LABEL_CLASSES.items()
        if AUX_SPECS[name].logits != width
    }
    if wrong:
        raise RuntimeError(
            "these §24.1 heads emit a different number of classes than their "
            f"label takes (label, head): {wrong}"
        )


_check_vocabularies()


def _coordinate_key(qr: np.ndarray) -> np.ndarray:
    """Pack ``(..., 2)`` axial coordinates into collision-free int64 keys."""
    if qr.size and int(np.abs(qr).max()) >= _COORD_BIAS:
        raise ValueError(
            f"a coordinate of magnitude {int(np.abs(qr).max())} does not fit the "
            f"±{_COORD_BIAS} key space; the legal radius bounds a real game far "
            "below it, so this input is not a position this engine produced"
        )
    return (qr[..., 0] + _COORD_BIAS) * _COORD_STRIDE + (qr[..., 1] + _COORD_BIAS)


def _capped(count: np.ndarray) -> np.ndarray:
    """§24.1's capped categorical: ``0..CAP``, the top class meaning "or more"."""
    return np.minimum(count, AUX_COUNT_CAP).astype(np.int64)


def _distinct_per_action(action: np.ndarray, key: np.ndarray, n_legal: int) -> np.ndarray:
    """How many distinct ``key`` values each action contributes.

    One lexicographic sort and a first-of-run flag, so the count is exact
    rather than a hash of the pairs, and the cost is one sort over the whole
    table instead of a set per action.
    """
    if not len(action):
        return np.zeros(n_legal, dtype=np.int64)
    order = np.lexsort((key, action))
    action, key = action[order], key[order]
    first = np.empty(len(action), dtype=bool)
    first[0] = True
    first[1:] = (action[1:] != action[:-1]) | (key[1:] != key[:-1])
    return np.bincount(action[first], minlength=n_legal).astype(np.int64)


def action_aux_labels(tables: ActionTables, legal_qr) -> dict[str, np.ndarray]:
    """§24.1's six labels for every legal action, each ``(n_legal,)`` int64.

    ``legal_qr`` is the ``(n_legal, 2)`` engine-order legal move list the tables
    were built from; the two partner labels need the coordinates, the other four
    do not. Every returned value is a class index into the width
    `heads.AUX_SPECS` gives that auxiliary, and the caller decides which rows
    carry a label — `heads.ActionAuxiliaryHeads` emits that mask beside its
    logits, and for auxiliaries 5 and 6 it is the first-placement rows.
    """
    legal_qr = np.asarray(legal_qr, dtype=np.int64).reshape(-1, 2)
    if len(legal_qr) != tables.n_legal:
        raise ValueError(
            f"{len(legal_qr)} legal coordinates against {tables.n_legal} action rows"
        )
    pre_code, post_code = tables.pre_code, tables.post_code
    n_legal = tables.n_legal
    rows = (1, 2)

    own_after = OWN_COUNT[post_code]
    opp_after = OPP_COUNT[post_code]
    opp_before = OPP_COUNT[pre_code]
    own_before = OWN_COUNT[pre_code]

    win_now = own_after.max(axis=rows) == WINDOW_LEN
    threat_hit = (own_before == 0) & np.isin(opp_before, _THREAT_STONES)
    own_five_after = (own_after == 5) & (opp_after == 0)

    # --- the two partner labels (see the module docstring) ------------------
    winning_key = _coordinate_key(legal_qr[win_now])
    # Every cell that wins immediately also wins as a second placement, whatever
    # the first placement was, so this set is one number shared by every action.
    always = len(winning_key)

    action_of, axis_of, slot_of = np.nonzero(own_five_after)
    empty_slot = _POWER_TO_SLOT[_ALL_OWN_CODE - post_code[own_five_after]]
    if empty_slot.size and int(empty_slot.min()) < 0:
        bad = int(np.argmin(empty_slot))
        raise ValueError(
            f"post-placement code {int(post_code[own_five_after][bad])} holds five "
            "own stones and no opponent stone but does not differ from the all-own "
            "code by one power of three; the slot-to-power mapping is wrong"
        )
    partner_qr = (
        legal_qr[action_of] + AXES[axis_of] * (empty_slot - slot_of)[:, None]
    )
    partner_key = _coordinate_key(partner_qr)
    fresh = ~np.isin(partner_key, winning_key)
    partner_count = always + _distinct_per_action(
        action_of[fresh], partner_key[fresh], n_legal
    )
    # A placement that wins ends the turn, so no second placement follows it.
    partner_count = np.where(win_now, 0, partner_count)

    return {
        "win_now": win_now.astype(np.int64),
        "own_max_occupancy": own_after.max(axis=rows).astype(np.int64),
        "opponent_threats_hit": _capped(threat_hit.sum(axis=rows)),
        "own_five_windows_after": _capped(own_five_after.sum(axis=rows)),
        "winning_partner_exists": (partner_count > 0).astype(np.int64),
        "winning_partner_count": _capped(partner_count),
    }


def position_aux_labels(position, cfg) -> dict[str, np.ndarray]:
    """§24.1's labels for a position, enumerating the windows it needs.

    The convenience entry point for a caller holding a position rather than an
    `ActionTables` — `diagnostics.py`'s §34 accuracy split is the one in this
    package. It repeats the window enumeration `builder.build` also does, which
    is why it is not what the builder calls: a label is a training-stage
    quantity, and the builder emits the representation.
    """
    from .actions import action_tables
    from .windows import enumerate_windows

    stones = np.asarray(position.stones(), dtype=np.int64).reshape(-1, 3)
    legal_qr = np.asarray(position.legal_moves(), dtype=np.int64).reshape(-1, 2)
    stone_qr = stones[:, :2]
    stone_own = (stones[:, 2] != int(position.current_player)).astype(np.int64)
    window_set = enumerate_windows(stone_qr, stone_own, legal_qr, cfg)
    tables = action_tables(window_set, stone_qr, stone_own, legal_qr, cfg)
    return action_aux_labels(tables, legal_qr)


__all__ = ["action_aux_labels", "position_aux_labels"]
