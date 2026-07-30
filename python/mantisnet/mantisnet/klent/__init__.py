"""KLENT training for Hexo as specified by ``docs/KLENT_FOR_HEXO.md``.

The package trains MantisNet's trunk, policy head, and return-mass
action-value head. Its loss does not include the state-value head; see §3
and §9. ``graft`` is a one-off command-line conversion, not part of the
training surface.
"""

from .improve import ImprovedPolicy, improved_policy
from .returns import lambda_returns, signs_from_moves_remaining
from .selfplay import Collector, Episode, Sample, collection_stats, episode_samples
from .train import KlentConfig, collect_episodes, fit
from .evaluate import argmax_choose, play_match
from .run import run_training
from .telemetry import Telemetry, connect, open_telemetry
from .inspect import inspect_position

__all__ = [
    "ImprovedPolicy",
    "improved_policy",
    "lambda_returns",
    "signs_from_moves_remaining",
    "Episode",
    "Sample",
    "collection_stats",
    "episode_samples",
    "Collector",
    "KlentConfig",
    "collect_episodes",
    "fit",
    "argmax_choose",
    "play_match",
    "run_training",
    "Telemetry",
    "connect",
    "open_telemetry",
    "inspect_position",
]
