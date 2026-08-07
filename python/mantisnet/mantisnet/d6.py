"""The twelve D6 symmetries of the hex board, as maps on axial ``(q, r)``.

Generators: 60-degree rotation ``(q, r) -> (-r, q + r)`` and reflection
``(q, r) -> (r, q)``.  Index ``0`` is the identity.
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
