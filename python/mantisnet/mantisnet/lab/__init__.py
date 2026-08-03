"""The frozen-corpus supervised benchmark and model-measurement harness.

``python -m mantisnet.lab`` is the public command surface.  The modules also
expose their artifact-oriented functions for tests and programmatic runs.
"""

from .corpus import FORMAT_VERSION as CORPUS_FORMAT_VERSION
from .evaluate import DISTANCE_BUCKETS
from .variants import VARIANTS, VariantSpec

__all__ = [
    "CORPUS_FORMAT_VERSION",
    "DISTANCE_BUCKETS",
    "VARIANTS",
    "VariantSpec",
]

