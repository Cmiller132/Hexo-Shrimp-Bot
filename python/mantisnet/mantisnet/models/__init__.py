"""Selectable model architectures.

Each subpackage is one architecture with its own representation, builder, and
checkpoint version. Architectures do not share modules; a representation that
two of them would need is lifted to ``mantisnet`` proper rather than imported
across the boundary.
"""
