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

from ..builder import MODEL_REPR_VERSION, Batch
from ..fitloop import FitBudgets, fit_epoch
from ..klent.train import KlentConfig, _gpu_lock
from ..losses import policy_loss, value_loss
from ..optim import make_adam, resolve_adam_implementation
from .corpus import FrozenCorpus, SampleSplit, load_corpus
from .variants import (
    build_variant,
    count_parameters,
    refuse_param_budget,
    scoped_collate,
)


@dataclass(frozen=True)
class TrainConfig:
    """The supervised recipe and the production batching limits it uses."""

    epochs: int = 8
    batch_size: int = KlentConfig.batch_size
    pair_budget: int = KlentConfig.pair_budget
    cell_budget: int = KlentConfig.cell_budget
    collect_pair_budget: int = KlentConfig.collect_pair_budget
    collect_cell_budget: int = KlentConfig.collect_cell_budget
    lr: float = KlentConfig.lr
    adam_impl: str = KlentConfig.adam_impl
    lr_schedule: str = "constant"
    ema_decay: float = 0.0
    device: str = "cpu"
    autocast: bool | None = None
    compile: bool = False
    # Fitted-split cap: 0 fits the whole split; a positive count selects one
    # deterministic subset (by ``train_subset_seed``, independent of the cell
    # seed) so every arm and seed of a paired screen fits identical samples.
    train_subset: int = 0
    train_subset_seed: int = 0
    # Ablation-by-freezing control: parameter-name suffixes held at their
    # initial value (left out of the optimizer). Empty = train everything.
    freeze: tuple[str, ...] = ()

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
            "collect_pair_budget",
            "collect_cell_budget",
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
        if self.train_subset < 0:
            raise ValueError(
                f"train_subset must be 0 (whole split) or positive, got {self.train_subset}"
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


def subset_samples(samples: SampleSplit, count: int, seed: int) -> SampleSplit:
    """One deterministic ``count``-sample subset of a split, in corpus order.

    The selection is a function of ``(len(samples), count, seed)`` alone, so
    every arm and seed of a paired screen draws exactly the same samples.
    """
    if not 0 < count < len(samples):
        raise ValueError(
            f"subset count must be in 1..{len(samples) - 1}, got {count}"
        )
    picks = np.sort(
        np.random.default_rng(seed).choice(len(samples), size=count, replace=False)
    )
    return SampleSplit(
        game=samples.game[picks],
        t=samples.t[picks],
        rank=samples.rank[picks],
        mover=samples.mover[picks],
        z=samples.z[picks],
        dist=samples.dist[picks],
    )


def _fit_samples(frozen: FrozenCorpus, split: str, cfg: TrainConfig) -> SampleSplit:
    """The samples one fitting epoch runs over, after the recipe's cap."""
    samples = frozen.split_samples(split)
    if not len(samples):
        raise ValueError(f"corpus split {split!r} is empty")
    if cfg.train_subset:
        samples = subset_samples(samples, cfg.train_subset, cfg.train_subset_seed)
    return samples


def sample_sizes(corpus: FrozenCorpus, samples: SampleSplit) -> tuple[np.ndarray, np.ndarray]:
    """Return exact attention-row and legal-cell sizes for a frozen split.

    The corpus format intentionally stores targets rather than representation
    products.  Packing still needs exact legal widths, so each selected game is
    replayed once here.  Rust ``collate_prefixes`` remains deferred to each
    prefetch callback and no batch/tensor is materialized up front.
    """

    lengths = np.asarray(samples.t, dtype=np.int64) + 1
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
    return lengths, cells


def scoped_attention_lengths(lengths: np.ndarray) -> np.ndarray:
    """Grow ``sample_sizes``' one global row to the four-row layout."""
    return lengths + 3


def pack_inference_indices(
    lengths: np.ndarray,
    cells: np.ndarray,
    *,
    pair_budget: int,
    cell_budget: int,
    position_cap: int = 256,
) -> list[list[int]]:
    """Pack a no-grad pass under the production collect budgets."""

    if len(lengths) != len(cells):
        raise ValueError("length and legal-cell arrays must have equal length")
    if pair_budget <= 0 or cell_budget <= 0 or position_cap <= 0:
        raise ValueError("inference budgets and position cap must be positive")
    order = sorted(range(len(lengths)), key=lambda i: -int(lengths[i]))
    chunks: list[list[int]] = []
    chunk: list[int] = []
    max_t = 0
    n_cells = 0
    for index in order:
        t_pad = int(lengths[index])
        width = int(cells[index])
        candidate_t = max(max_t, t_pad)
        if chunk and (
            len(chunk) == position_cap
            or (len(chunk) + 1) * candidate_t * candidate_t > pair_budget
            or n_cells + width > cell_budget
        ):
            chunks.append(chunk)
            chunk = []
            max_t = 0
            n_cells = 0
            candidate_t = t_pad
        chunk.append(index)
        max_t = candidate_t
        n_cells += width
    if chunk:
        chunks.append(chunk)
    return chunks


def collate_samples(
    corpus: FrozenCorpus,
    samples: SampleSplit,
    indices: Sequence[int],
    collate,
) -> tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    target = torch.zeros(batch.n_cells, dtype=torch.float32)
    target[batch.legal_offsets[:-1] + ranks] = 1.0
    return batch, target, ranks, z


def _supervised_heads(model, batch: Batch):
    _stones, windows, token, cells = model.trunk(batch)
    policy, critic = model.cell_head_logits(windows, token, cells, batch)
    value, _value_dist, value_logits = model.value_head(windows, token, batch)
    return policy, critic, value, value_logits


_supervised_heads_compiled = None
_compile_lock = threading.Lock()


def _supervised_fn(compile_model: bool):
    global _supervised_heads_compiled
    if not compile_model:
        return _supervised_heads
    with _compile_lock:
        if _supervised_heads_compiled is None:
            # Packed chunks span tail-to-full sizes and the AOT cache guards
            # them by size range, so the graph set splits past the stock limit
            # of 8 â€” after which every new range runs eager for the rest of
            # the session. The set converges; give it room.
            torch._dynamo.config.recompile_limit = 64
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
    steady: tuple[int, int] | None = None,
) -> dict[str, float | int]:
    """Fit one corpus epoch without creating artifacts.

    Bench mode uses this entry point so its throughput measures exactly the
    same collation, forward, loss, accumulation, and optimizer engine as a lab
    cell.
    """

    frozen = _as_corpus(corpus)
    samples = _fit_samples(frozen, split, cfg)
    lengths, cells = sizes if sizes is not None else sample_sizes(frozen, samples)
    lengths = scoped_attention_lengths(lengths)
    if len(lengths) != len(samples):
        raise ValueError(
            f"precomputed sizes cover {len(lengths)} samples but the fitted "
            f"split has {len(samples)} after the recipe's train_subset cap"
        )
    collate = scoped_collate(variant)
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
        state_value_ce = value_loss(value_logits, z)
        loss = policy_ce + critic_ce + state_value_ce
        return loss, {
            "policy_loss": policy_ce.detach(),
            "critic_ce": critic_ce.detach(),
            "value_loss": state_value_ce.detach(),
        }

    model.train()
    return fit_epoch(
        model,
        optimizer,
        rng,
        lengths=lengths,
        cells=cells,
        budgets=FitBudgets(
            batch_size=cfg.batch_size,
            pair_budget=cfg.pair_budget,
            cell_budget=cfg.cell_budget,
        ),
        prepare=prepare,
        step=step,
        lock=_gpu_lock,
        progress=progress,
        steady=steady,
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
    lengths, cells = sizes if sizes is not None else sample_sizes(frozen, samples)
    lengths = scoped_attention_lengths(lengths)
    chunks = pack_inference_indices(
        lengths,
        cells,
        pair_budget=cfg.collect_pair_budget,
        cell_budget=cfg.collect_cell_budget,
    )
    collate = scoped_collate(variant)
    forward = _supervised_fn(cfg.compile)
    device_type = torch.device(cfg.device).type
    correct = 0
    sign_correct = 0
    absolute_error = 0.0
    # Sums of the three fit losses over the split, per sample, so the
    # per-epoch row carries the same quantities the fit minimizes.
    policy_nll_sum = 0.0
    critic_ce_sum = 0.0
    state_value_ce_sum = 0.0
    model.eval()
    for indices in chunks:
        batch, target, ranks, z = collate_samples(frozen, samples, indices, collate)
        batch = batch.to(cfg.device)
        target = target.to(cfg.device)
        ranks = ranks.to(cfg.device)
        z = z.to(cfg.device)
        with _gpu_lock, torch.autocast(
            device_type, dtype=torch.bfloat16, enabled=cfg.autocast
        ):
            policy, critic, value, value_logits = forward(model, batch)
        rows = len(indices)
        policy_nll_sum += float(policy_loss(policy.float(), batch.legal_offsets, target)) * rows
        taken = critic.index_select(0, batch.legal_offsets[:-1] + ranks).float()
        critic_target = torch.stack(
            (z.clamp(min=0.0), (-z).clamp(min=0.0), 1.0 - z.abs()), dim=-1
        )
        critic_ce_sum += float(
            -(critic_target * F.log_softmax(taken, dim=-1)).sum(dim=-1).sum()
        )
        state_value_ce_sum += float(value_loss(value_logits, z)) * rows
        # One transfer per tensor keeps validation packed on CUDA instead of
        # synchronizing separately for every position's scalar conversions.
        offsets = batch.legal_offsets.cpu().tolist()
        ranks_cpu = ranks.cpu().tolist()
        policy_cpu = policy.float().cpu()
        value_cpu = value.float().cpu()
        z_cpu = z.cpu()
        for row in range(rows):
            lo, hi = offsets[row], offsets[row + 1]
            correct += int(int(policy_cpu[lo:hi].argmax()) == ranks_cpu[row])
        sign_correct += int((torch.sign(value_cpu) == z_cpu).sum())
        absolute_error += float((value_cpu - z_cpu).abs().sum())
    n = len(samples)
    return {
        "samples": n,
        "imitation_top1": correct / n,
        "value_sign_accuracy": sign_correct / n,
        "value_mae": absolute_error / n,
        "policy_nll": policy_nll_sum / n,
        "critic_ce": critic_ce_sum / n,
        "state_value_ce": state_value_ce_sum / n,
    }


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
    cell_budget: int | None = None,
    train_subset: int | None = None,
    train_subset_seed: int | None = None,
    param_budget: int | None = None,
    param_tol: float = 0.02,
    progress=None,
):
    """Train one fresh variant/seed cell and write its complete artifacts.

    ``cell_budget`` caps the accumulation micro-chunks, not the optimizer
    batch: gradients are summed over the same samples either way, so it is
    an execution/memory knob, not a recipe change. Window-heavy arms use it
    to keep their edge tables resident instead of paging.
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
    if cell_budget is not None:
        updates["cell_budget"] = cell_budget
    if train_subset is not None:
        updates["train_subset"] = train_subset
    if train_subset_seed is not None:
        updates["train_subset_seed"] = train_subset_seed
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
    if cfg.freeze:
        frozen_names = [
            name for name, parameter in model.named_parameters()
            if name.endswith(tuple(cfg.freeze))
        ]
        if not frozen_names:
            raise ValueError(f"freeze suffixes {cfg.freeze} match no parameter")
        for name, parameter in model.named_parameters():
            if name in frozen_names:
                parameter.requires_grad_(False)
    optimizer, adam_resolved = make_adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
            "loss_weights": {"policy": 1.0, "critic": 1.0, "state_value": 1.0},
        },
        "seed": seed,
        "versions": versions,
        "param_count": parameter_count,
    }
    _write_json(destination / "config.json", cell_config)

    train_samples = _fit_samples(frozen, "train", cfg)
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
