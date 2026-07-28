"""The KLENT iteration: collect an on-policy buffer, fit once, discard it.

Faithful to the paper's outer loop (design doc §1): a self-play phase fills
the buffer, one fitting epoch consumes it (O2's assumption), and nothing
survives to the next iteration. The loss is eq. 4 — cross-entropy of π_θ
against π′ plus squared error of the taken action's Q against the λ-return —
under plain Adam at the paper's learning rate. The value head appears
nowhere: KLENT has no state-value head, and v̂ = E_{π′}[Q] does its job.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from ..builder import collate_prefixes
from ..losses import policy_loss
from .seeds import line_evaluate, seed_prefixes
from .selfplay import Sample, collection_stats, play_episodes


@dataclass
class KlentConfig:
    """Design doc §2's starting values, expected to move — not defaults to trust."""

    # Verified against the paper's eq. 2 (2026-07-27): reverse KL is the
    # heavier regulariser. Prior exponent tau/(tau+lam) = 0.77.
    tau: float = 0.1  # reverse-KL weight (the paper's beta)
    lam: float = 0.03  # entropy weight (the paper's alpha)
    # e^{-1/16}: the paper's 8-turn horizon at Hexo's two placements per turn
    # (KLENT_PROPOSALS A1). The paper's literal e^{-1/8} would halve it.
    lam_ret: float = 0.939
    ply_cap: int = 512  # §5: capped episodes are dropped whole
    games_per_iteration: int = 128
    seed_fraction: float = 1.0  # §5.2: anneal toward zero as f allows
    # Opponent grounding: this fraction of each iteration's games puts an
    # external whole-turn opponent in one (alternating) seat, unseeded. Only
    # the model's plies are recorded, with Monte-Carlo outcomes; capped
    # grounded games are draws. 0 = pure self-play.
    ground_fraction: float = 0.0
    seed_cut: tuple = (1, 8)  # plies cut from a won seed game's end
    seed_noise: float = 0.1
    batch_size: int = 4096  # paper's *effective* batch: chunks accumulate to it
    lr: float = 1e-3  # paper's Adam rate
    device: str = "cpu"
    autocast: bool = False  # bf16 autocast for the network passes
    compile: bool = False  # torch.compile the policy/Q pass (one-time cost)
    # VRAM is budgeted, not hoped for: every network batch — fit chunk or
    # collection cohort — is packed under both measured memory axes, so the
    # peak is set here rather than by whatever the corpus happens to contain.
    # Attention memory is quadratic in the batch's longest position (padding),
    # decoder memory linear in its total legal cells.
    pair_budget: int = 8_000_000  # padded (stones + token)^2 pairs per batch
    # 800k cells ≈ 5.8 GiB worst-case fit peak on the iteration-0 corpus —
    # still comfortable on 12 GiB, and half the chunk count (so half the
    # per-chunk Python and launch overhead) of the original 400k.
    cell_budget: int = 800_000  # legal cells (decoder rows) per batch


def _policy_q(model, batch):
    """The KLENT pass: trunk + the two heads it trains, never the value head."""
    _s, w, g = model.trunk(batch)
    return model.policy_head(w, g, batch), model.q_head(w, g, batch)


# One symbolic-shape graph serves every batch; compiled lazily, shared by
# collection and fitting (the same graph gets the compiled backward).
_policy_q_compiled = None

# Every compiled-callable invocation holds this lock. The GPU work was
# serialized on one stream anyway; what the lock buys is that when dynamo
# (re)compiles on one thread — sporadic new shape buckets keep this possible
# at any iteration — the other thread sleeps at the lock instead of
# thrashing the GIL against the trace, which was measured to stretch a
# seconds-long compile into a minutes-long crawl.
_gpu_lock = threading.Lock()


def _policy_q_fn(cfg: KlentConfig):
    global _policy_q_compiled
    if not cfg.compile:
        return _policy_q
    if _policy_q_compiled is None:
        _policy_q_compiled = torch.compile(_policy_q, dynamic=True)
    return _policy_q_compiled


def network_evaluate(model, cfg: KlentConfig):
    """The self-play evaluator, returning flat CPU tensors so the collection
    loop stays device-ignorant."""
    policy_q = _policy_q_fn(cfg)

    def evaluate(batch):
        with _gpu_lock:
            b = batch.to(cfg.device)
            with torch.no_grad(), torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
                policy, q = policy_q(model, b)
            return policy.float().cpu(), q.float().cpu()

    return evaluate


def _rebuild(samples: list[Sample]):
    """Buffer states back into one batch by parallel replay (design doc §12:
    a position is a move prefix). Refuses a sample whose stored π′ no longer
    matches its position's legal count — that misalignment trains against
    scrambled targets and has no downstream symptom."""
    batch = collate_prefixes([s.moves for s in samples], [s.t for s in samples])
    counts = (batch.legal_offsets[1:] - batch.legal_offsets[:-1]).tolist()
    for s, count in zip(samples, counts):
        if count != len(s.improved):
            raise ValueError(
                f"sample at ply {s.t}: stored pi' has {len(s.improved)} entries, "
                f"position has {count} legal moves"
            )
    return batch


def _pack(samples: list[Sample], order, cfg: KlentConfig) -> list[list[int]]:
    """Pack sample indices into fit chunks under ``batch_size`` and both
    memory budgets. Sorted by position size (descending, ties in ``order``'s
    random order), so a chunk's padded attention cost is exact — everyone
    pads to its first element — and one long sample can no longer pad a
    whole mixed chunk up to its own square. A sample too big for the budgets
    alone still gets its own chunk: the buffer is never silently dropped."""
    idx = sorted(order, key=lambda i: samples[i].t, reverse=True)
    chunks: list[list[int]] = []
    chunk: list[int] = []
    chunk_t, cells = 0, 0
    for i in idx:
        t_pad = samples[i].t + 1  # + the global token row
        c = len(samples[i].improved)
        if chunk and (
            len(chunk) == cfg.batch_size
            or (len(chunk) + 1) * chunk_t * chunk_t > cfg.pair_budget
            or cells + c > cfg.cell_budget
        ):
            chunks.append(chunk)
            chunk, cells = [], 0
        if not chunk:
            chunk_t = t_pad
        chunk.append(int(i))
        cells += c
    if chunk:
        chunks.append(chunk)
    return chunks


def fit(model, samples: list[Sample], optimizer, cfg: KlentConfig, rng: np.random.Generator):
    """One epoch over the buffer at the paper's *effective* batch.

    The memory budgets cap what one forward may hold, so chunks accumulate
    sample-weighted gradients until ~``batch_size`` samples have contributed
    and the optimizer steps once — the paper's batch statistics under the
    packing. A step's gradient equals the mean loss over its whole
    accumulated batch, so the chunking is an implementation detail of memory
    and not of optimization. Returns the sample-weighted mean losses.

    Two throughput facts. Chunk *preparation* (the Rust replay rebuild and
    the target tensors) runs one chunk ahead on a worker thread, so the GPU
    never waits on it. And the loss scalars accumulate on-device — the only
    host sync is the single read at the end; per-chunk ``float()`` reads
    were measured to stall the stream once per chunk."""
    model.train()
    policy_q = _policy_q_fn(cfg)
    chunks = _pack(samples, rng.permutation(len(samples)), cfg)

    groups: list[tuple[list[int], int]] = []
    group: list[int] = []
    count = 0
    for k in rng.permutation(len(chunks)):
        group.append(int(k))
        count += len(chunks[k])
        if count >= cfg.batch_size:
            groups.append((group, count))
            group, count = [], 0
    if group:
        groups.append((group, count))

    def prep(k: int):
        chunk = [samples[i] for i in chunks[k]]
        batch = _rebuild(chunk)
        target = torch.from_numpy(np.concatenate([s.improved for s in chunk]))
        ranks = torch.tensor([s.rank for s in chunk])
        returns = torch.tensor([s.g for s in chunk], dtype=torch.float32)
        return chunk, batch, target, ranks, returns

    order = [k for group, _ in groups for k in group]
    policy_sum = torch.zeros((), device=cfg.device)
    q_sum = torch.zeros((), device=cfg.device)
    total = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        prepped = {k: pool.submit(prep, k) for k in order[:1]}
        consumed = 0
        for group, group_n in groups:
            optimizer.zero_grad(set_to_none=True)
            for k in group:
                if consumed + 1 < len(order):
                    nxt = order[consumed + 1]
                    prepped[nxt] = pool.submit(prep, nxt)
                chunk, batch, target, ranks, returns = prepped.pop(k).result()
                consumed += 1
                with _gpu_lock:
                    batch = batch.to(cfg.device)
                    target = target.to(cfg.device)
                    ranks = ranks.to(cfg.device)
                    returns = returns.to(cfg.device)

                    with torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
                        policy_logits, q_values = policy_q(model, batch)
                    ce = policy_loss(policy_logits.float(), batch.legal_offsets, target)
                    taken = q_values.float().index_select(
                        0, batch.legal_offsets[:-1] + ranks
                    )
                    q_mse = (taken - returns).square().mean()

                    ((ce + q_mse) * (len(chunk) / group_n)).backward()
                    policy_sum += ce.detach() * len(chunk)
                    q_sum += q_mse.detach() * len(chunk)
            optimizer.step()
            total += group_n
    return {
        "policy_loss": float(policy_sum) / total,
        "q_loss": float(q_sum) / total,
        "fit_steps": len(groups),
    }


def generate_prefixes(seed: int, n: int, seed_fraction: float, seed_cut, seed_noise):
    """One iteration's seed prefixes, from a self-contained RNG. The seed
    games play in lockstep (`seed_prefixes`) — generating them one at a
    time was the loop's single largest measured cost."""
    prng = np.random.default_rng(seed)
    seeded = prng.random(n) < seed_fraction
    drawn = iter(seed_prefixes(prng, int(seeded.sum()), seed_cut, seed_noise))
    return [next(drawn) if s else [] for s in seeded]


def ground_count(cfg: KlentConfig, warm: bool) -> int:
    """How many of an iteration's games are grounded: none during warm —
    the bootstrap clones the line builder before any opponent can judge it."""
    return 0 if warm else round(cfg.ground_fraction * cfg.games_per_iteration)


def collect_episodes(
    model,
    cfg: KlentConfig,
    rng: np.random.Generator,
    warm: bool = False,
    prefixes: list | None = None,
    opponent=None,
) -> tuple[list, dict]:
    """One iteration's corpus: episodes plus the §13 collection metrics.

    Worker-safe by construction — it draws only from the ``rng`` it is
    given and mutates only the episodes it creates, which is what lets the
    run driver collect iteration ``i+1`` on a weight snapshot while
    iteration ``i`` fits on the live model.

    ``warm`` is the bootstrap phase: collection acts through the line
    builder's scores instead of the network, because an honestly-initialized
    π′ is near-uniform and finishes almost no seeded games — measured, not
    supposed. (Warm episodes must also fit against pure Monte-Carlo returns
    — the heuristic's v̂ lives on an arbitrary scale and must not bootstrap;
    the driver owns that choice of ``lam_ret``.)

    ``prefixes`` covers the *self-play* games only — ``ground_count`` games
    are grounded against ``opponent`` (unseeded, alternating seats) and take
    no prefix. Left ``None``, they are drawn here from ``rng``."""
    n_ground = ground_count(cfg, warm)
    if n_ground and opponent is None:
        raise ValueError("ground_fraction > 0 needs an opponent")
    n_self = cfg.games_per_iteration - n_ground
    if prefixes is None:
        prefixes = generate_prefixes(
            int(rng.integers(2**63)),
            n_self,
            cfg.seed_fraction,
            cfg.seed_cut,
            cfg.seed_noise,
        )
    if len(prefixes) != n_self:
        raise ValueError(
            f"{len(prefixes)} prefixes for {n_self} self-play games "
            f"({n_ground} of {cfg.games_per_iteration} are grounded)"
        )
    model.eval()
    episodes, metrics = play_episodes(
        line_evaluate if warm else network_evaluate(model, cfg),
        [[]] * n_ground + list(prefixes),
        cfg.ply_cap,
        cfg.tau,
        cfg.lam,
        rng,
        pair_budget=cfg.pair_budget,
        cell_budget=cfg.cell_budget,
        opponent=opponent,
        opponent_seats=[g % 2 for g in range(n_ground)] + [None] * n_self,
    )
    metrics.update(collection_stats(episodes))
    metrics["warm"] = int(warm)
    return episodes, metrics
