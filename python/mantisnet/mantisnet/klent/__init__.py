"""KLENT for Hexo: the closed-form policy improvement of ``docs/KLENT_FOR_HEXO.md``.

Faithful first: the paper's algorithm unchanged wherever Hexo permits it, and
every deviation traceable to ``KLENT_FOR_HEXO.md`` §9. This package is the
training path; the model it trains is MantisNet's trunk with the policy and
action-value heads, and the state-value head is outside its loss by the
paper's own ablation (no V head, ``KLENT_FOR_HEXO.md`` §3).
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
