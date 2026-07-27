"""KLENT for Hexo: the closed-form policy improvement of ``docs/KLENT_DESIGN.md``.

Faithful first: the paper's algorithm unchanged wherever Hexo permits it, and
every deviation traceable to §3 of the design doc. This package is the
training path; the model it trains is MantisNet's trunk with the policy and
action-value heads, and the state-value head is outside its loss by the
paper's own ablation (no V head, design doc fidelity ledger).
"""

from .improve import ImprovedPolicy, improved_policy
from .returns import lambda_returns, signs_from_moves_remaining
from .seeds import line_builder_choose, line_builder_game
from .selfplay import Episode, Sample, collection_stats, episode_samples, play_episodes
from .train import KlentConfig, fit, iterate
from .evaluate import play_match
from .run import run_training

__all__ = [
    "ImprovedPolicy",
    "improved_policy",
    "lambda_returns",
    "signs_from_moves_remaining",
    "line_builder_choose",
    "line_builder_game",
    "Episode",
    "Sample",
    "collection_stats",
    "episode_samples",
    "play_episodes",
    "KlentConfig",
    "fit",
    "iterate",
    "play_match",
    "run_training",
]
