"""The twelve symmetries of the board, as maps on axial ``(q, r)``.

The group belongs to the game, not to any model: it preserves hex distance,
legality, the three window axes, and the winner, so every consumer — the
opening atlas, a representation's relation classes, an equivariance test —
is describing the same twelve maps. They are defined once here.

Generators are the 60-degree rotation ``(q, r) -> (-r, q + r)`` and the
reflection ``(q, r) -> (r, q)``. Index ``0`` is the identity.

This module imports nothing, so it is safe to import from anywhere.
"""

from __future__ import annotations

from typing import Callable

Transform = Callable[[tuple[int, int]], tuple[int, int]]


def _d6_transforms() -> tuple[Transform, ...]:
    def rot(m):
        return (-m[1], m[0] + m[1])

    def ref(m):
        return (m[1], m[0])

    out = []
    for base in (lambda m: m, ref):
        f = base
        for _ in range(6):
            out.append(f)
            f = (lambda g: lambda m: rot(g(m)))(f)
    return tuple(out)


D6_TRANSFORMS: tuple[Transform, ...] = _d6_transforms()
