"""The policy debugger: one checkpoint's whole opinion of one position.

The telemetry database stores four scalars per ply and not π′ itself,
because π′ is exactly reproducible from the checkpoint and the move prefix
that reached the position. This is the reproduction — the seam a viewer
calls when it wants the full legal set rather than a summary of it, and the
same seam a branch-and-play frontend calls once per move it explores.

Everything here is the training path's own code: the checkpoint loader that
refuses a version mismatch, the Rust prefix builder, and the closed-form
improvement of `improve.py`. The numbers it returns are therefore the
numbers collection acted on, not a second opinion about them.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..builder import collate_prefixes
from ..segments import segment_log_softmax
from .improve import improved_policy
from .run import load_model


def inspect_position(
    checkpoint: Path | str | torch.nn.Module,
    moves,
    t: int,
    tau: float,
    lam: float,
    device: str = "cpu",
) -> dict:
    """A checkpoint's policy, Q, π′, and v̂ over the legal set at ``moves[:t]``.

    ``checkpoint`` is a path or an already-loaded model — a caller walking a
    line one move at a time loads once and passes the model, which is the
    difference between a responsive branch-and-play view and one that reads
    a checkpoint per node.

    ``tau`` and ``lam`` have no defaults on purpose: π′ is a function of
    them, and a run that used other values would be silently misreported by
    a debugger that assumed the current ones. Read them from the run's
    `config.json`.

    Returns the position's own scalars plus one entry per legal move, in the
    engine's order — which is the order every rank in the database indexes
    into. A model passed in is left in eval mode, since a debugger reporting
    numbers a dropout layer had touched would be reporting nothing.
    """
    import hexo_py

    moves = [tuple(m) for m in moves]
    if not 0 <= t <= len(moves):
        raise ValueError(f"ply {t} is outside the {len(moves)}-move line")
    model = (
        checkpoint
        if isinstance(checkpoint, torch.nn.Module)
        else load_model(Path(checkpoint), device)
    )
    model.eval()

    position = hexo_py.Position.replay(moves[:t])
    if position.is_terminal:
        raise ValueError(f"the position after {t} plies is terminal: nothing to choose")

    batch = collate_prefixes([moves], [t]).to(device)
    with torch.no_grad():
        _s, w, g = model.trunk(batch)
        logits = model.policy_head(w, g, batch).float().cpu()
        q_values = model.q_head(w, g, batch).float().cpu()
    offsets = batch.legal_offsets.cpu()
    log_pi = segment_log_softmax(logits, offsets)
    imp = improved_policy(logits, q_values, offsets, tau, lam)

    legal = position.legal_moves()
    if len(legal) != logits.shape[0]:
        raise RuntimeError(
            f"builder produced {logits.shape[0]} cells for a position with "
            f"{len(legal)} legal moves"
        )
    return {
        "moves": moves[:t],
        "t": t,
        "mover": position.current_player,
        "moves_remaining": position.moves_remaining,
        "stone_count": position.stone_count,
        "legal_count": len(legal),
        "tau": tau,
        "lam": lam,
        "v_hat": float(imp.v_hat[0]),
        "kl": float(imp.kl[0]),
        "norm_entropy": float(imp.norm_entropy[0]),
        # The move actually played from here, when the line continues — the
        # anchor a viewer highlights against the alternatives.
        "played": moves[t] if t < len(moves) else None,
        "legal": [
            {
                "move": move,
                "rank": rank,
                "logit": float(logits[rank]),
                "policy": float(log_pi[rank].exp()),
                "q": float(q_values[rank]),
                "improved": float(imp.probs[rank]),
            }
            for rank, move in enumerate(legal)
        ],
    }
