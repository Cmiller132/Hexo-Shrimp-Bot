"""MantisNet: the trunk and both heads of ``docs/MODEL_SPEC.md`` §5–§7.

The forward consumes a :class:`~mantisnet.builder.Batch` and performs no
data-dependent index discovery: every gather, scatter, and pad slot was
precomputed by the builder. Weights are fp32; the forward is written to run
under bf16 autocast without assuming it (buffers inherit dtype from inputs,
and the scalar value decode is done in fp32).

Linear maps written as bare matrices in the spec (``U``, ``V``, ``P``) are
bias-free — the per-class embedding added alongside them is the additive term.
Attention, FFN, and MLP linears keep the framework-default bias (§10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import fused_attention
from .builder import NEAREST_BUCKETS, NUM_PATTERNS, Batch


@dataclass(frozen=True)
class MantisConfig:
    """The named parameters of MODEL_SPEC §2, at their suggested defaults."""

    h: int = 128  # H: embedding width, everywhere
    blocks: int = 4  # B
    heads: int = 4  # A
    ffn_factor: int = 2  # F
    d_max: int = 12  # D_MAX: hex-distance clamp
    value_queries: int = 4  # Q
    value_bins: int = 65  # K
    policy_hidden: int = 128  # P_H
    value_hidden: int = 128  # V_H
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.h % self.heads != 0:
            raise ValueError(f"H={self.h} must divide into A={self.heads} heads")
        if self.value_bins % 2 == 0:
            raise ValueError(f"K={self.value_bins} must be odd so an exact-zero bin exists")

    # Bucket indices of the attention bias table (§4.1): distances 1..D_MAX
    # occupy 0..D_MAX-1, then SELF, then TOKEN. TOKEN wins on the token-token
    # pair. PAD is not a parameter row: attention appends its finite sentinel
    # after casting the learned table to the compute dtype.
    @property
    def self_bucket(self) -> int:
        return self.d_max

    @property
    def token_bucket(self) -> int:
        return self.d_max + 1

    @property
    def pad_bucket(self) -> int:
        return self.d_max + 2


@dataclass
class ModelOutput:
    """What one full forward answers (§11 plus the appendix-B Q head)."""

    policy_logits: Tensor  # (N_cells,) raw, engine legal order per position
    q_values: Tensor  # (N_cells,) action values, same layout — the KLENT head
    value: Tensor  # (P,) scalar decode, in [-1, 1]
    value_dist: Tensor  # (P, K) softmax over bins, fp32
    value_logits: Tensor  # (P, K) the bins' raw logits — what value_loss trains


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


class _PairMlp(nn.Module):
    """``MLP([a; b])`` with the concatenation folded away.

    A linear over a concatenation is the sum of two linears, so the 2H-wide
    input tensor of the spec's ``MLP_W``/``MLP_S``/``MLP_P`` never has to be
    materialised — on a batch of a few hundred thousand entity rows that cat
    is pure memory traffic. Parameters are identical (one bias on the first
    layer), and so is the arithmetic.
    """

    def __init__(self, h: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.lin_a = nn.Linear(h, d_hidden)
        self.lin_b = nn.Linear(h, d_hidden, bias=False)
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.out(F.relu(self.lin_a(a) + self.lin_b(b)))


class _Block(nn.Module):
    """One trunk block (§5): window <- stones, stone <- windows, attention."""

    def __init__(self, cfg: MantisConfig) -> None:
        super().__init__()
        h = cfg.h
        self.cfg = cfg
        # §5.1 window <- stones
        self.ln_ws_s = nn.LayerNorm(h)
        self.ln_ws_w = nn.LayerNorm(h)
        self.u = nn.Linear(h, h, bias=False)
        self.e_ws = nn.Embedding(3, h)
        self.mlp_w = _PairMlp(h, h, h)
        # §5.2 stone <- windows
        self.ln_sw_w = nn.LayerNorm(h)
        self.ln_sw_s = nn.LayerNorm(h)
        self.v = nn.Linear(h, h, bias=False)
        self.e_sw = nn.Embedding(3, h)
        self.mlp_s = _PairMlp(h, h, h)
        # §5.3 stone self-attention + token
        self.ln_attn = nn.LayerNorm(h)
        self.wq = nn.Linear(h, h)
        self.wk = nn.Linear(h, h)
        self.wv = nn.Linear(h, h)
        self.wo = nn.Linear(h, h)
        self.dist_bias = nn.Parameter(torch.zeros(cfg.heads, cfg.d_max + 2))
        self.ln_ffn = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, cfg.ffn_factor * h), nn.ReLU(), nn.Linear(cfg.ffn_factor * h, h)
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self, s: Tensor, w: Tensor, g: Tensor, batch: Batch, seq_lens: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg
        # Sizes come from tensor shapes, not the Batch's ints: under
        # torch.compile they become symbolic, so one graph serves every shape.
        p, max_t = g.shape[0], batch.attn_valid.shape[1]

        # §5.1: windows aggregate their stones. Sum, not mean — the count is
        # signal.
        x = self.u(self.ln_ws_s(s))
        msg = x.index_select(0, batch.inc_stone) + self.e_ws(batch.inc_class)
        agg = msg.new_zeros(w.shape[0], cfg.h).index_add_(0, batch.inc_window, msg)
        w = w + self.drop(self.mlp_w(self.ln_ws_w(w), agg))

        # §5.2: stones aggregate their windows.
        y = self.v(self.ln_sw_w(w))
        msg = y.index_select(0, batch.inc_window) + self.e_sw(batch.inc_class)
        agg = msg.new_zeros(s.shape[0], cfg.h).index_add_(0, batch.inc_stone, msg)
        s = s + self.drop(self.mlp_s(self.ln_sw_s(s), agg))

        # §5.3: attention over [token; stones], block-diagonal per position.
        rows = s.new_zeros(p * max_t, cfg.h)
        token_slot = torch.arange(p, device=s.device) * max_t
        rows.index_copy_(0, token_slot, g)
        rows.index_copy_(0, batch.stone_slot, s)
        z = self.ln_attn(rows.view(p, max_t, cfg.h))

        hd = cfg.h // cfg.heads
        q = self.wq(z).view(p, max_t, cfg.heads, hd).transpose(1, 2)
        k = self.wk(z).view(p, max_t, cfg.heads, hd).transpose(1, 2)
        v = self.wv(z).view(p, max_t, cfg.heads, hd).transpose(1, 2)

        # Coordinates become distance buckets inside the attention kernel.
        # Each position's key loop stops at its live prefix instead of doing
        # quadratic work over padding.
        out = fused_attention(q, k, v, batch.coords, seq_lens, self.dist_bias)
        out = self.wo(out.transpose(1, 2).reshape(p, max_t, cfg.h)).view(p * max_t, cfg.h)
        s = s + self.drop(out.index_select(0, batch.stone_slot))
        g = g + self.drop(out.index_select(0, token_slot))

        # FFN over the same rows — row-independent, so no padding needed.
        rows = torch.cat([s, g], dim=0)
        rows = self.drop(self.ffn(self.ln_ffn(rows)))
        s = s + rows[: s.shape[0]]
        g = g + rows[s.shape[0] :]
        return s, w, g


class MantisNet(nn.Module):
    """The whole model: embeddings, B trunk blocks, policy and value heads."""

    def __init__(self, cfg: MantisConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or MantisConfig()
        self.cfg = cfg
        h = cfg.h

        # §3: input embeddings.
        self.stone_table = nn.Embedding(2, h)  # own / opp
        self.window_table = nn.Embedding(2 * NUM_PATTERNS, h)  # colour x pattern
        self.token_base = nn.Parameter(torch.empty(h))
        self.token_moves = nn.Embedding(2, h)  # moves_remaining in {1, 2}

        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.blocks))
        self.ln_out = nn.LayerNorm(h)  # shared final LN over S, W, g (§5)

        # §6 policy decoder. MLP_P([h_a; g]) as a _PairMlp, so the g half of
        # its first layer runs per position, not per legal cell.
        self.p = nn.Linear(h, h, bias=False)
        self.e_pw = nn.Embedding(3, h)
        self.e_bg = nn.Embedding(NEAREST_BUCKETS, h)
        self.mlp_p = _PairMlp(h, cfg.policy_hidden, 1)

        # Appendix B action-value decoder: the same shape as §6 with its own
        # parameters everywhere, one raw Q per legal cell. KLENT's head.
        self.q = nn.Linear(h, h, bias=False)
        self.e_qw = nn.Embedding(3, h)
        self.e_qbg = nn.Embedding(NEAREST_BUCKETS, h)
        self.mlp_q = _PairMlp(h, cfg.policy_hidden, 1)

        # §7 value head.
        self.value_queries = nn.Parameter(torch.empty(cfg.value_queries, h))
        self.ln_value = nn.LayerNorm(h)
        self.mlp_v = _mlp(cfg.value_queries * h, cfg.value_hidden, cfg.value_bins)
        self.register_buffer(
            "bin_centers", torch.linspace(-1.0, 1.0, cfg.value_bins), persistent=False
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # §10: embeddings, the token base, and the value queries N(0, 0.02);
        # bias tables zero (already); linears framework-default.
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.token_base, std=0.02)
        nn.init.normal_(self.value_queries, std=0.02)
        # Appendix B: both decoder output layers start at zero — the KLENT
        # reference net's `zero_init` — so π_θ opens uniform and Q ≡ 0 until
        # trained. π′ ∝ π_θ^{τ/(τ+λ)}·exp(Q/(τ+λ)) sharpens *initialization
        # noise* otherwise, measured to train the first fit toward arbitrary
        # sharp targets.
        for head in (self.mlp_p, self.mlp_q):
            nn.init.zeros_(head.out.weight)
            nn.init.zeros_(head.out.bias)

    def trunk(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        """Embeddings through the B blocks and the shared final LN (§5)."""
        s = self.stone_table(batch.stone_own)
        w = self.window_table(batch.window_feat)
        g = self.token_base + self.token_moves(batch.moves_idx)

        seq_lens = batch.attn_valid.sum(dim=1, dtype=torch.int32)
        for block in self.blocks:
            s, w, g = block(s, w, g, batch, seq_lens)
        return self.ln_out(s), self.ln_out(w), self.ln_out(g)

    def _decode_cells(
        self,
        w: Tensor,
        g: Tensor,
        batch: Batch,
        lin: nn.Linear,
        e_w: nn.Embedding,
        e_bg: nn.Embedding,
        mlp: _PairMlp,
    ) -> Tensor:
        """§6's decoder: one scalar per legal cell. Window path is a
        gather-sum over the decoder table; background cells overwrite from
        their bucket table; the token half of the MLP runs per position."""
        msg = lin(w).index_select(0, batch.dec_window) + e_w(batch.dec_class)
        h = g.new_zeros(batch.cell_pos.shape[0], self.cfg.h)
        h.index_add_(0, batch.dec_cell, msg.to(h.dtype))
        if batch.bg_cell.numel():
            h.index_copy_(0, batch.bg_cell, e_bg(batch.bg_bucket).to(h.dtype))
        g_half = mlp.lin_b(g)
        out = mlp.out(F.relu(mlp.lin_a(h) + g_half.index_select(0, batch.cell_pos)))
        return out.squeeze(-1)

    def policy_head(self, w: Tensor, g: Tensor, batch: Batch) -> Tensor:
        """§6: one raw policy logit per legal cell, engine legal order."""
        return self._decode_cells(w, g, batch, self.p, self.e_pw, self.e_bg, self.mlp_p)

    def q_head(self, w: Tensor, g: Tensor, batch: Batch) -> Tensor:
        """Appendix B: one action value per legal cell, same layout, bounded
        to (−1, 1) by tanh as in the KLENT reference net — π′ exponentiates
        Q/(τ+λ), so an unbounded Q could sharpen without limit."""
        return torch.tanh(
            self._decode_cells(w, g, batch, self.q, self.e_qw, self.e_qbg, self.mlp_q)
        )

    def value_head(self, w: Tensor, g: Tensor, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        """§7: (value, value_dist, value_logits). Multi-query attention
        readout over [token; windows]. The LN runs on the concatenated rows
        before padding — row-wise, so identical, and the padded copy is
        written once."""
        cfg = self.cfg
        p, max_w = g.shape[0], batch.value_valid.shape[1]
        rows = g.new_zeros(p * max_w, cfg.h)
        token_slot = torch.arange(p, device=g.device) * max_w
        rows.index_copy_(0, token_slot, self.ln_value(g))
        if batch.window_slot.numel():
            rows.index_copy_(0, batch.window_slot, self.ln_value(w).to(rows.dtype))
        kv = rows.view(p, max_w, cfg.h)
        scores = torch.einsum("qh,pth->pqt", self.value_queries, kv) / math.sqrt(cfg.h)
        scores = scores.masked_fill(
            ~batch.value_valid[:, None, :], torch.finfo(scores.dtype).min
        )
        r = torch.einsum("pqt,pth->pqh", scores.softmax(dim=-1), kv)
        v_logits = self.mlp_v(r.reshape(p, cfg.value_queries * cfg.h))

        # Scalar decode in-forward, fp32, so every consumer sees the same value.
        value_dist = v_logits.float().softmax(dim=-1)
        value = value_dist @ self.bin_centers
        return value, value_dist, v_logits

    def forward(self, batch: Batch) -> ModelOutput:
        """Every head. KLENT's loop composes `trunk` with the two heads it
        trains instead, skipping the value readout it never reads."""
        s_, w, g = self.trunk(batch)
        value, value_dist, value_logits = self.value_head(w, g, batch)
        return ModelOutput(
            policy_logits=self.policy_head(w, g, batch),
            q_values=self.q_head(w, g, batch),
            value=value,
            value_dist=value_dist,
            value_logits=value_logits,
        )
