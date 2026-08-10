"""Supervised lab-cell fitting through the production model and fit engine."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..builder import MODEL_REPR_VERSION
from ..fitloop import FitBudgets, fit_epoch, pack_chunks
from ..klent.train import KlentConfig, _gpu_lock
from ..losses import policy_loss, value_loss
from ..optim import make_adam, resolve_adam_implementation
from .corpus import FrozenCorpus, SampleSplit, load_corpus
from .variants import (
    build_variant,
    count_parameters,
    refuse_param_budget,
    variant_spec,
)


@dataclass(frozen=True)
class TrainConfig:
    """The supervised recipe and the production batching limits it uses."""

    epochs: int = 8
    batch_size: int = KlentConfig.batch_size
    pair_budget: int = KlentConfig.pair_budget
    cell_budget: int = KlentConfig.cell_budget
    graph_cell_budget: int = KlentConfig.graph_cell_budget
    collect_pair_budget: int = KlentConfig.collect_pair_budget
    collect_cell_budget: int = KlentConfig.collect_cell_budget
    collect_graph_cell_budget: int = KlentConfig.collect_graph_cell_budget
    lr: float = KlentConfig.lr
    adam_impl: str = KlentConfig.adam_impl
    lr_schedule: str = "constant"
    ema_decay: float = 0.0
    device: str = "cpu"
    autocast: bool | None = None
    compile: bool = False

    def __post_init__(self) -> None:
        resolve_adam_implementation(self.adam_impl, self.device)
        expected_autocast = torch.device(self.device).type == "cuda"
        if self.autocast is None:
            object.__setattr__(self, "autocast", expected_autocast)
        elif self.autocast is not expected_autocast:
            raise ValueError(
                "supervised fitting uses bf16 autocast exactly on CUDA; "
                f"device={self.device!r}, autocast={self.autocast}"
            )
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        for name in (
            "batch_size",
            "pair_budget",
            "cell_budget",
            "graph_cell_budget",
            "collect_pair_budget",
            "collect_cell_budget",
            "collect_graph_cell_budget",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError(f"lr must be finite and positive, got {self.lr}")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}"
            )
        if not math.isfinite(self.ema_decay) or not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(
                f"ema_decay must be finite and in [0.0, 1.0), got {self.ema_decay}"
            )

    def budgets(self) -> FitBudgets:
        """The fitting limits this recipe offers a model's ``chunk_cost``."""

        return FitBudgets(
            pair_budget=self.pair_budget,
            cell_budget=self.cell_budget,
            graph_cell_budget=self.graph_cell_budget,
        )

    def collect_budgets(self) -> FitBudgets:
        """The no-grad limits validation and evaluation pack under."""

        return FitBudgets(
            pair_budget=self.collect_pair_budget,
            cell_budget=self.collect_cell_budget,
            graph_cell_budget=self.collect_graph_cell_budget,
        )

    def epoch_lr(self, epoch: int) -> float:
        """The learning rate for a 1-based epoch under this recipe."""

        if self.lr_schedule == "constant":
            return self.lr
        # Cosine annealing: full lr at epoch 1, approaching zero past the last.
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / self.epochs))


def current_versions() -> dict[str, object]:
    """Compatibility pins carried by every lab artifact."""

    import hexo_py

    return {
        "MODEL_REPR_VERSION": MODEL_REPR_VERSION,
        "RULES_VERSION": hexo_py.RULES_VERSION,
        "ACTION_ORDER_VERSION": hexo_py.ACTION_ORDER_VERSION,
        "torch": torch.__version__,
    }


def _as_corpus(corpus: str | os.PathLike[str] | FrozenCorpus) -> FrozenCorpus:
    return corpus if isinstance(corpus, FrozenCorpus) else load_corpus(Path(corpus))


def sample_sizes(corpus: FrozenCorpus, samples: SampleSplit) -> tuple[np.ndarray, np.ndarray]:
    """Return the stone and legal-cell counts of every sample in a split.

    ``chunk_cost`` is written over these two quantities, so packing needs no
    built graph. The corpus stores targets, not the legal count, so each
    selected game is replayed once here to recover it.
    """

    stones = np.asarray(samples.t, dtype=np.int64)
    cells = np.empty(len(samples), dtype=np.int64)
    by_game: dict[int, list[tuple[int, int]]] = {}
    for index, (game, t) in enumerate(zip(samples.game, samples.t, strict=True)):
        by_game.setdefault(int(game), []).append((int(t), index))

    import hexo_py

    for game, requested in by_game.items():
        requested.sort()
        moves = corpus.moves_for(game)
        pos = hexo_py.Position()
        cursor = 0
        for t, index in requested:
            if t < 0 or t >= len(moves):
                raise ValueError(
                    f"corpus sample {index} names ply {t} outside game {game} "
                    f"length {len(moves)}"
                )
            while cursor < t:
                pos.advance(*moves[cursor])
                cursor += 1
            if pos.is_terminal:
                raise ValueError(f"corpus sample {index} names terminal game {game} ply {t}")
            cells[index] = len(pos.legal_moves())
    return stones, cells


# A no-grad chunk is capped at this many positions however small they are: the
# per-position Python in every scoring loop is what the cap bounds, not memory.
INFERENCE_POSITION_CAP = 256


def pack_inference_chunks(
    model,
    stones: np.ndarray,
    cells: np.ndarray,
    budgets: FitBudgets,
    *,
    position_cap: int = INFERENCE_POSITION_CAP,
) -> list[list[int]]:
    """Pack a no-grad pass under ``model``'s own law and the collect budgets.

    Uses the same packing engine and cost law as fitting; only the limits
    differ, since a no-grad chunk holds no backward graph.
    """

    if len(stones) != len(cells):
        raise ValueError("stone and legal-cell arrays must have equal length")
    if position_cap <= 0:
        raise ValueError(f"inference position cap must be positive, got {position_cap}")
    return pack_chunks(
        range(len(stones)), position_cap, model.chunk_cost(stones, cells, budgets)
    )


def collate_samples(
    corpus: FrozenCorpus,
    samples: SampleSplit,
    indices: Sequence[int],
    collate,
) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize one supervised chunk on the CPU prefetch worker."""

    games = [corpus.moves_for(int(samples.game[index])) for index in indices]
    ts = [int(samples.t[index]) for index in indices]
    batch = collate(games, ts)
    ranks = torch.tensor([int(samples.rank[index]) for index in indices], dtype=torch.long)
    z = torch.tensor([int(samples.z[index]) for index in indices], dtype=torch.float32)
    counts = batch.legal_offsets[1:] - batch.legal_offsets[:-1]
    bad = (ranks < 0) | (ranks >= counts)
    if torch.any(bad):
        where = int(torch.nonzero(bad, as_tuple=False)[0])
        raise ValueError(
            f"corpus sample {indices[where]} has rank {int(ranks[where])} "
            f"for {int(counts[where])} legal moves"
        )
    # One target row per legal cell of the chunk, which every representation
    # states the same way: the last legal offset. It is the only tensor width
    # the recipe needs from a batch it otherwise never looks inside.
    target = torch.zeros(int(batch.legal_offsets[-1]), dtype=torch.float32)
    target[batch.legal_offsets[:-1] + ranks] = 1.0
    return batch, target, ranks, z


def _supervised_heads(model, batch):
    """The supervised pass, whichever architecture is being fitted.

    ``supervised_heads`` is the seam analogous to KLENT's ``policy_q``: the
    recipe holds a corpus, not a representation. The last two return values
    are ``None`` for an architecture with no state-value head.
    """
    return model.supervised_heads(batch)


_supervised_heads_compiled = None
_compile_lock = threading.Lock()


def _supervised_fn(compile_model: bool):
    global _supervised_heads_compiled
    if not compile_model:
        return _supervised_heads
    with _compile_lock:
        if _supervised_heads_compiled is None:
            _supervised_heads_compiled = torch.compile(_supervised_heads, dynamic=True)
    return _supervised_heads_compiled


def fit_supervised_epoch(
    model,
    optimizer,
    corpus: str | os.PathLike[str] | FrozenCorpus,
    split: str,
    cfg: TrainConfig,
    rng: np.random.Generator,
    *,
    variant: str = "mantis",
    progress=None,
    sizes: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Fit one corpus epoch without creating artifacts.

    Bench mode uses this entry point so its throughput measures exactly the
    same collation, forward, loss, accumulation, and optimizer engine as a lab
    cell.
    """

    frozen = _as_corpus(corpus)
    samples = frozen.split_samples(split)
    if not len(samples):
        raise ValueError(f"corpus split {split!r} is empty")
    stones, cells = sizes if sizes is not None else sample_sizes(frozen, samples)
    collate = variant_spec(variant).collate
    forward = _supervised_fn(cfg.compile)
    device_type = torch.device(cfg.device).type

    def prepare(indices: Sequence[int]):
        batch, target, ranks, z = collate_samples(frozen, samples, indices, collate)
        if device_type == "cuda":
            # Pinning on the prefetch worker makes step's transfers truly
            # async; from pageable memory non_blocking degrades to a staged
            # copy on the compute thread.
            return (
                batch.pin_memory(),
                target.pin_memory(),
                ranks.pin_memory(),
                z.pin_memory(),
            )
        return batch, target, ranks, z

    def step(payload):
        batch, target, ranks, z = payload
        batch = batch.to(cfg.device)
        target = target.to(cfg.device, non_blocking=True)
        ranks = ranks.to(cfg.device, non_blocking=True)
        z = z.to(cfg.device, non_blocking=True)
        with torch.autocast(
            device_type, dtype=torch.bfloat16, enabled=cfg.autocast
        ):
            policy_logits, critic_logits, _value, value_logits = forward(model, batch)
        policy_ce = policy_loss(policy_logits.float(), batch.legal_offsets, target)
        taken = critic_logits.index_select(
            0, batch.legal_offsets[:-1] + ranks
        ).float()
        critic_target = torch.stack(
            (z.clamp(min=0.0), (-z).clamp(min=0.0), 1.0 - z.abs()), dim=-1
        )
        critic_ce = -(
            critic_target * F.log_softmax(taken, dim=-1)
        ).sum(dim=-1).mean()
        loss = policy_ce + critic_ce
        stats = {
            "policy_loss": policy_ce.detach(),
            "critic_ce": critic_ce.detach(),
        }
        # The state-value term is added only when the architecture has that
        # head; `critic_ce` above is the action-value critic every
        # architecture trains.
        if value_logits is not None:
            state_value_ce = value_loss(value_logits, z)
            loss = loss + state_value_ce
            stats["value_loss"] = state_value_ce.detach()
        return loss, stats

    model.train()
    return fit_epoch(
        model,
        optimizer,
        rng,
        sample_count=len(samples),
        batch_size=cfg.batch_size,
        cost=model.chunk_cost(stones, cells, cfg.budgets()),
        prepare=prepare,
        step=step,
        lock=_gpu_lock,
        progress=progress,
    )


@torch.no_grad()
def validate_supervised(
    model,
    corpus: str | os.PathLike[str] | FrozenCorpus,
    split: str,
    cfg: TrainConfig,
    *,
    variant: str = "mantis",
    sizes: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Run the per-epoch imitation/value validation pass."""

    frozen = _as_corpus(corpus)
    samples = frozen.split_samples(split)
    if not len(samples):
        raise ValueError(f"corpus split {split!r} is empty")
    stones, cells = sizes if sizes is not None else sample_sizes(frozen, samples)
    chunks = pack_inference_chunks(model, stones, cells, cfg.collect_budgets())
    collate = variant_spec(variant).collate
    forward = _supervised_fn(cfg.compile)
    device_type = torch.device(cfg.device).type
    correct = 0
    sign_correct = 0
    absolute_error = 0.0
    model.eval()
    for indices in chunks:
        batch, _target, ranks, z = collate_samples(frozen, samples, indices, collate)
        batch = batch.to(cfg.device)
        ranks = ranks.to(cfg.device)
        z = z.to(cfg.device)
        with _gpu_lock, torch.autocast(
            device_type, dtype=torch.bfloat16, enabled=cfg.autocast
        ):
            policy, _critic, value, _value_logits = forward(model, batch)
        # One transfer per tensor keeps validation packed on CUDA instead of
        # synchronizing separately for every position's scalar conversions.
        offsets = batch.legal_offsets.cpu().tolist()
        ranks_cpu = ranks.cpu().tolist()
        policy_cpu = policy.float().cpu()
        for row in range(len(indices)):
            lo, hi = offsets[row], offsets[row + 1]
            correct += int(int(policy_cpu[lo:hi].argmax()) == ranks_cpu[row])
        if value is not None:
            value_cpu = value.float().cpu()
            z_cpu = z.cpu()
            sign_correct += int((torch.sign(value_cpu) == z_cpu).sum())
            absolute_error += float((value_cpu - z_cpu).abs().sum())
    n = len(samples)
    metrics: dict[str, float | int] = {"samples": n, "imitation_top1": correct / n}
    if model.has_state_value_head:
        metrics["value_sign_accuracy"] = sign_correct / n
        metrics["value_mae"] = absolute_error / n
    return metrics


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class _EMAOptimizer:
    """Update a named fp32 EMA immediately after each optimizer step."""

    def __init__(self, optimizer, parameters, ema, decay: float) -> None:
        self.optimizer = optimizer
        self.parameters = parameters
        self.ema = ema
        self.decay = decay
        self.param_groups = optimizer.param_groups

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    @torch.no_grad()
    def step(self, *args, **kwargs):
        result = self.optimizer.step(*args, **kwargs)
        for name, parameter in self.parameters:
            self.ema[name].mul_(self.decay).add_(
                parameter, alpha=1.0 - self.decay
            )
        return result


def train_cell(
    corpus: str | os.PathLike[str] | FrozenCorpus,
    cell_dir: str | os.PathLike[str],
    *,
    variant: str = "mantis",
    model_kw: Mapping[str, object] | None = None,
    seed: int = 0,
    config: TrainConfig | None = None,
    epochs: int | None = None,
    lr_schedule: str | None = None,
    ema_decay: float | None = None,
    adam_impl: str | None = None,
    device: str | None = None,
    compile: bool | None = None,
    batch_size: int | None = None,
    cell_budget: int | None = None,
    graph_cell_budget: int | None = None,
    param_budget: int | None = None,
    param_tol: float = 0.02,
    progress=None,
):
    """Train one fresh variant/seed cell and write its complete artifacts.

    ``cell_budget`` and ``graph_cell_budget`` cap accumulation micro-chunks
    (an execution/memory knob, not the optimizer batch), one per
    architecture's binding quantity. ``batch_size`` is the effective
    optimizer batch and does change the recipe.
    """

    cfg = config or TrainConfig()
    updates = asdict(cfg)
    if epochs is not None:
        updates["epochs"] = epochs
    if lr_schedule is not None:
        updates["lr_schedule"] = lr_schedule
    if ema_decay is not None:
        updates["ema_decay"] = ema_decay
    if adam_impl is not None:
        updates["adam_impl"] = adam_impl
    if device is not None:
        updates["device"] = device
        updates["autocast"] = torch.device(device).type == "cuda"
    if compile is not None:
        updates["compile"] = compile
    if batch_size is not None:
        updates["batch_size"] = batch_size
    if cell_budget is not None:
        updates["cell_budget"] = cell_budget
    if graph_cell_budget is not None:
        updates["graph_cell_budget"] = graph_cell_budget
    cfg = TrainConfig(**updates)
    frozen = _as_corpus(corpus)

    torch.manual_seed(seed)
    model, normalized_kw, _spec = build_variant(variant, model_kw)
    parameter_count = count_parameters(model)
    refuse_param_budget(parameter_count, param_budget, param_tol)

    destination = Path(cell_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"lab cell directory is nonempty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    model = model.to(cfg.device)
    optimizer, adam_resolved = make_adam(
        model.parameters(),
        lr=cfg.lr,
        device=cfg.device,
        implementation=cfg.adam_impl,
    )
    ema_parameters = None
    if cfg.ema_decay > 0:
        named_parameters = list(model.named_parameters())
        with torch.no_grad():
            ema_parameters = {
                name: parameter.clone().detach().float()
                for name, parameter in named_parameters
            }
        optimizer = _EMAOptimizer(
            optimizer, named_parameters, ema_parameters, cfg.ema_decay
        )
    versions = current_versions()
    cell_config = {
        "lab_cell_format": 1,
        "variant": variant,
        "model_kw": normalized_kw,
        "corpus": {"name": frozen.name, "sha256": frozen.sha256},
        "recipe": {
            **asdict(cfg),
            "optimizer": "Adam",
            "adam_resolved": adam_resolved,
            # Every term this architecture trains, at equal weight; state_value
            # present only when the model has that head.
            "loss_weights": {
                "policy": 1.0,
                "critic": 1.0,
                **({"state_value": 1.0} if model.has_state_value_head else {}),
            },
        },
        "seed": seed,
        "versions": versions,
        "param_count": parameter_count,
    }
    _write_json(destination / "config.json", cell_config)

    train_samples = frozen.split_samples("train")
    val_samples = frozen.split_samples("val")
    train_sizes = sample_sizes(frozen, train_samples)
    val_sizes = sample_sizes(frozen, val_samples)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    with (destination / "metrics.jsonl").open("x", encoding="utf-8") as metrics_file:
        for epoch in range(1, cfg.epochs + 1):
            lr = cfg.epoch_lr(epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            started = time.perf_counter()
            fit_metrics = fit_supervised_epoch(
                model,
                optimizer,
                frozen,
                "train",
                cfg,
                rng,
                variant=variant,
                progress=progress,
                sizes=train_sizes,
            )
            seconds = time.perf_counter() - started
            validation = validate_supervised(
                model,
                frozen,
                "val",
                cfg,
                variant=variant,
                sizes=val_sizes,
            )
            row = {
                "epoch": epoch,
                "lr": lr,
                **fit_metrics,
                "seconds": seconds,
                "samples_per_second": len(train_samples) / seconds,
                "val": validation,
            }
            metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
            metrics_file.flush()
            os.fsync(metrics_file.fileno())
            rows.append(row)

    checkpoint = {
        "lab_cell_format": 1,
        "model": model.state_dict(),
        "variant": variant,
        "model_kw": normalized_kw,
        "corpus_sha256": frozen.sha256,
        "versions": versions,
        "param_count": parameter_count,
    }
    checkpoint_path = destination / "checkpoint_final.pt"
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    if ema_parameters is not None:
        ema_state = model.state_dict()
        for name, parameter in ema_parameters.items():
            ema_state[name] = parameter.to(dtype=ema_state[name].dtype)
        ema_checkpoint = {**checkpoint, "model": ema_state}
        ema_checkpoint_path = destination / "checkpoint_ema.pt"
        ema_temporary = ema_checkpoint_path.with_suffix(".tmp")
        torch.save(ema_checkpoint, ema_temporary)
        ema_temporary.replace(ema_checkpoint_path)
    return {"model": model, "config": cell_config, "metrics": rows}
