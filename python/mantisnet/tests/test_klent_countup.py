"""The two-placements-per-turn Count Up Game
(``KLENT_FOR_HEXO.md`` §1.4).

A synthetic with Hexo's mover pattern — one opening placement, then two per
turn — small enough to solve exactly. The KLENT iteration, run through the
*real* `improved_policy`, must converge to the quantal-response fixed point
`π* ∝ exp(Q*/λ)` computed by independent backward induction; and episodes
scored by the *real* sign/λ-return machinery must average back to `Q*`. A
parity-derived sign (K1), a bootstrap from a terminal (K3), or a `v̂` taken
under `π_θ` instead of `π′` (K5) each move the fixed point and fail here.

The game: a counter starts at 0; each placement adds 1 or 2; the mover who
reaches `TARGET` wins. States are `(count, moves_remaining)` — mr = 2 is a
first stone (sign +1, same mover next), mr = 1 an opening or second stone
(sign −1). Wins occur on both halves of a turn, so K2 is exercised too.
"""

from __future__ import annotations

import numpy as np
import torch

from mantisnet.klent import improved_policy, lambda_returns, signs_from_moves_remaining

TARGET = 6
ACTIONS = (1, 2)
TAU, LAM = 0.03, 0.1


def states():
    return [(c, mr) for c in range(TARGET) for mr in (1, 2)]


def sign(state):
    return 1.0 if state[1] == 2 else -1.0


def step(state, action):
    """(next_state, won). The mover keeps the turn after a first stone."""
    c, mr = state
    c2 = c + action
    if c2 >= TARGET:
        return None, True
    return (c2, 2 if mr == 1 else 1), False


def solved_qre():
    """Backward induction of Q* and π* ∝ exp(Q*/λ), counts descending —
    every transition strictly increases the count, so this is exact."""
    q = {}
    v = {}
    for c in range(TARGET - 1, -1, -1):
        for mr in (1, 2):
            s = (c, mr)
            row = []
            for a in ACTIONS:
                nxt, won = step(s, a)
                row.append(1.0 if won else sign(s) * v[nxt])
            q[s] = np.array(row)
            pi = np.exp(q[s] / LAM)
            pi /= pi.sum()
            v[s] = float(pi @ q[s])
    return q, v


def _improve(pi, q):
    """One π′/v̂ pass over every state, through the real operator."""
    order = states()
    logits = torch.tensor(np.log(np.concatenate([pi[s] for s in order])), dtype=torch.float32)
    qs = torch.tensor(np.concatenate([q[s] for s in order]), dtype=torch.float32)
    offsets = torch.arange(0, 2 * len(order) + 1, 2)
    imp = improved_policy(logits, qs, offsets, TAU, LAM)
    probs = imp.probs.numpy()
    v_hat = imp.v_hat.numpy()
    new_pi = {s: probs[2 * i : 2 * i + 2].astype(np.float64) for i, s in enumerate(order)}
    new_v = {s: float(v_hat[i]) for i, s in enumerate(order)}
    return new_pi, new_v


def test_expected_iteration_converges_to_the_qre():
    q_star, v_star = solved_qre()
    pi = {s: np.array([0.5, 0.5]) for s in states()}
    q = {s: np.zeros(2) for s in states()}
    for _ in range(300):
        pi, v_hat = _improve(pi, q)
        # The one-step target: r at a win, else the signed bootstrap on v̂ of
        # the *next* state — the recursion never touches a terminal (K3).
        q = {
            s: np.array(
                [
                    1.0 if step(s, a)[1] else sign(s) * v_hat[step(s, a)[0]]
                    for a in ACTIONS
                ]
            )
            for s in states()
        }
    for s in states():
        assert np.allclose(q[s], q_star[s], atol=1e-4), f"Q mismatch at {s}"
        pi_star = np.exp(q_star[s] / LAM)
        pi_star /= pi_star.sum()
        assert np.allclose(pi[s], pi_star, atol=1e-4), f"policy mismatch at {s}"


def test_sampled_returns_average_to_q_star():
    # Play the solved policy; score episodes with the real sign and λ-return
    # machinery at λ_ret = 1 (the Monte Carlo endpoint). E[G | s, a] is Q*.
    q_star, _ = solved_qre()
    pi_star = {}
    for s in states():
        p = np.exp(q_star[s] / LAM)
        pi_star[s] = p / p.sum()

    rng = np.random.default_rng(11)
    totals = {(s, a): [0.0, 0] for s in states() for a in ACTIONS}
    for _ in range(6000):
        s = (0, 1)  # the opening placement
        visited, mrs = [], []
        while True:
            a = int(rng.choice(2, p=pi_star[s]))
            visited.append((s, a))
            mrs.append(s[1])
            nxt, won = step(s, ACTIONS[a])
            if won:
                break
            s = nxt
        g = lambda_returns(
            signs_from_moves_remaining(mrs), np.zeros(len(mrs)), 1.0, 1.0
        )
        for (state, action), value in zip(visited, g):
            cell = totals[(state, ACTIONS[action])]
            cell[0] += value
            cell[1] += 1
    for (s, a), (total, count) in totals.items():
        if count >= 200:
            idx = ACTIONS.index(a)
            assert abs(total / count - q_star[s][idx]) < 0.06, f"{s} action {a}"
