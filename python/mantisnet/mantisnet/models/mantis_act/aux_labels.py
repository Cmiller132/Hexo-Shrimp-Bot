"""§24.1 auxiliary labels produced by the Rust MantisNet-ACT builder.

The labels and graph action rows are two readings of the same eighteen
counterfactual windows. Rust owns that enumeration; this module preserves the
Python label-container contract and checks it against the head vocabulary.
"""

from __future__ import annotations

import numpy as np

from .builder import _rust_config
from .config import MantisACTConfig
from .heads import AUX_COUNT_CAP, AUX_SPECS
from .packed import WINDOW_LEN

_LABEL_CLASSES: dict[str, int] = {
    "win_now": 1,
    "own_max_occupancy": WINDOW_LEN + 1,
    "opponent_threats_hit": AUX_COUNT_CAP + 1,
    "own_five_windows_after": AUX_COUNT_CAP + 1,
    "winning_partner_exists": 1,
    "winning_partner_count": AUX_COUNT_CAP + 1,
}


def _check_vocabularies() -> None:
    """Refuse a §24.1 head/label width mismatch at import."""
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


def position_aux_labels(
    position, cfg: MantisACTConfig
) -> dict[str, np.ndarray]:
    """Return the six §24.1 labels for ``position`` in engine legal order."""
    import hexo_py

    labels = dict(hexo_py.build_act_aux_labels(position, _rust_config(cfg)))
    if set(labels) != set(_LABEL_CLASSES):
        raise RuntimeError(
            f"Rust returned auxiliary labels {sorted(labels)}; expected "
            f"{sorted(_LABEL_CLASSES)}"
        )

    legal = int(position.legal_count)
    for name, value in labels.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"auxiliary label {name!r} must be a numpy array")
        if value.dtype != np.int64 or value.shape != (legal,):
            raise ValueError(
                f"auxiliary label {name!r} must be int64 shape ({legal},), "
                f"got {value.dtype} {value.shape}"
            )
        if value.size:
            top = 1 if _LABEL_CLASSES[name] == 1 else _LABEL_CLASSES[name] - 1
            if int(value.min()) < 0 or int(value.max()) > top:
                raise ValueError(
                    f"auxiliary label {name!r} lies outside its 0..{top} vocabulary"
                )
    return labels


__all__ = ["position_aux_labels"]
