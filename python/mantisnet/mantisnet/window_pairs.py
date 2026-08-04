"""Window-pair relation tables for §5.1c window attention.

Like the cell-pass relay tables, these are derived once at collation and the
forward performs no index discovery. Unlike them, the derivation is opt-in
(``pairs=True`` on the collators): only a ``window_attention`` model consumes
the tables, and the join is heavy enough that arms which never read it should
not pay for it at every chunk.

Two windows relate in exactly one of two game-mechanical ways, and both are
functions of the identity triples ``(axis, start_q, start_r)`` alone:

- **Colinear** (classes ``0..10``): the same line, signed start offset ``o``.
  A reflection reverses the line, flipping the sign, so the class is
  ``|o| - 1`` for ``|o| <= 11`` — overlap at 1..5, a gap of at most five
  cells at 6..11. Beyond that the spans cannot participate in one forcing
  sequence, and there is no edge.
- **Crossing** (classes ``11..46``): two non-parallel hex-axis lines always
  meet at exactly one lattice cell (every axis-pair determinant is ±1). The
  relation is where that cell sits in each window's span parameter — slot
  ``t`` in the destination, ``u`` in the source — including up to five cells
  beyond an end, mirroring the colinear reach. Each side folds to
  ``{in0, in1, in2, out1, out2, out3+}``; the class is
  ``11 + fold(t) * 6 + fold(u)``.
- **SELF** (class ``47``): one loop per window, so every softmax segment is
  nonempty.

D6-invariance of the crossing fold: a board symmetry permutes the three axes
and may reverse the two lines' slot parameterizations *independently* (a 60°
rotation maps the axes with mixed direction flips), so an invariant class may
not couple the two sides' orientations. Reversal maps ``t`` to ``5 - t``,
which preserves the in-span fold ``min(t, 5 - t)`` and the out-of-span
distance ``max(-t, t - 5)`` — each side's fold is invariant on its own, and
the product therefore under every group element.

Edge enumeration is a sorted join, cell-key style: each window claims its 16
line cells ``t ∈ -5..10``; two windows cross with mutual reach exactly when
they claim one common cell with different axes, and they do so at exactly one
cell, so the join yields each directed edge once. Colinear edges join on the
line key instead. Both joins are per-position by construction (the keys pack
the position index).
"""

from __future__ import annotations

import torch
from torch import Tensor

# 11 colinear + 36 crossing + SELF.
WA_CLASSES = 48
_SELF = 47
_REACH = 5  # cells beyond a span end, matching the colinear gap of <= 5
_MAX_OFFSET = 11

# fold(t) for t in -_REACH..5+_REACH, indexed by t + _REACH: in-span slots to
# min(t, 5 - t), out-of-span to 2 + min(distance, 3).
_FOLD = torch.tensor(
    [2 + min(d, 3) for d in range(_REACH, 0, -1)]
    + [min(t, 5 - t) for t in range(6)]
    + [2 + min(d, 3) for d in range(1, _REACH + 1)],
    dtype=torch.long,
)

# Unit steps of the engine's axes, canonical order Q, R, QR (builder.AXES).
_AXES = torch.tensor([[1, 0], [0, 1], [1, -1]], dtype=torch.long)

# Key packing: coordinates are i16-bounded, so 17 bits per component after an
# offset is collision-free, and the position index rides above them.
_KOFF = 1 << 16
_KSPAN = 1 << 17


def pair_tables(window_id: Tensor, window_pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The (ptr, src, class) CSR of directed §5.1c edges, destination-major.

    ``window_id`` is the batch's ``(N_w, 3)`` identity table and
    ``window_pos`` the ``(N_w,)`` position of each window. Returns
    ``wa_ptr (N_w + 1,)``, ``wa_src (E,)``, ``wa_class (E,)`` with each
    destination window's in-edges contiguous.
    """
    if window_id.ndim != 2 or window_id.shape[1] != 3:
        raise ValueError("window_id must have shape (N_w, 3)")
    if window_pos.shape != window_id.shape[:1]:
        raise ValueError("window_pos must have one entry per window")
    n_w = window_id.shape[0]
    axis, sq, sr = window_id.unbind(1)

    dsts, srcs, classes = [], [], []

    # Colinear: sort by (position, axis, line, position-on-line); partners sit
    # within 11 slots of one another, and starts on a line are distinct, so
    # eleven shifted comparisons enumerate every pair once.
    line = torch.where(axis == 0, sr, torch.where(axis == 1, sq, sq + sr))
    pos_on = torch.where(axis == 1, sr, sq)
    key = ((window_pos * 4 + axis) * _KSPAN + (line + _KOFF)) * _KSPAN + (
        pos_on + _KOFF
    )
    order = torch.argsort(key)
    skey = key[order]
    group = skey // _KSPAN
    spos = skey % _KSPAN
    for shift in range(1, _MAX_OFFSET + 1):
        if shift >= n_w:
            break
        near, far = order[:-shift], order[shift:]
        delta = spos[shift:] - spos[:-shift]
        ok = (group[:-shift] == group[shift:]) & (delta <= _MAX_OFFSET)
        near, far, cls = near[ok], far[ok], delta[ok] - 1
        dsts.append(torch.cat([near, far]))
        srcs.append(torch.cat([far, near]))
        classes.append(torch.cat([cls, cls]))

    # Crossing: join the claimed line cells. Runs share a (position, cell)
    # key; a pair with different axes in one run is a crossing within reach.
    t_ext = torch.arange(-_REACH, 6 + _REACH, device=window_id.device)
    vec = _AXES.to(window_id.device)[axis]  # (N_w, 2)
    cq = sq[:, None] + t_ext[None, :] * vec[:, 0:1]
    cr = sr[:, None] + t_ext[None, :] * vec[:, 1:2]
    span = t_ext.shape[0]
    ckey = (
        (window_pos[:, None] * _KSPAN + (cq + _KOFF)) * _KSPAN + (cr + _KOFF)
    ).reshape(-1)
    cwin = (
        torch.arange(n_w, device=window_id.device)[:, None].expand(-1, span).reshape(-1)
    )
    ct = t_ext[None, :].expand(n_w, -1).reshape(-1)
    corder = torch.argsort(ckey)
    rkey = ckey[corder]
    rwin = cwin[corder]
    rt = ct[corder]
    raxis = axis[rwin]
    fold = _FOLD.to(window_id.device)
    if rkey.numel():
        run_lengths = torch.unique_consecutive(rkey, return_counts=True)[1]
        for shift in range(1, int(run_lengths.max())):
            ok = (rkey[:-shift] == rkey[shift:]) & (raxis[:-shift] != raxis[shift:])
            wi, wj = rwin[:-shift][ok], rwin[shift:][ok]
            fi, fj = fold[rt[:-shift][ok] + _REACH], fold[rt[shift:][ok] + _REACH]
            dsts.append(torch.cat([wi, wj]))
            srcs.append(torch.cat([wj, wi]))
            classes.append(torch.cat([11 + fi * 6 + fj, 11 + fj * 6 + fi]))

    # SELF.
    loop = torch.arange(n_w, device=window_id.device)
    dsts.append(loop)
    srcs.append(loop)
    classes.append(torch.full((n_w,), _SELF, dtype=torch.long, device=window_id.device))

    dst = torch.cat(dsts)
    src = torch.cat(srcs)
    cls = torch.cat(classes)
    order = torch.argsort(dst, stable=True)
    dst, src, cls = dst[order], src[order], cls[order]
    ptr = torch.searchsorted(
        dst, torch.arange(n_w + 1, device=window_id.device)
    )
    return ptr, src, cls
