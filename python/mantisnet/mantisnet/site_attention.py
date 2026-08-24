"""Joint site attention: one softmax over every entity of a position.

The site rows are ``[4 state latents; stones; legal cells]`` per position,
packed with no padding and attended block-diagonally — every pair direction
(stone-stone, stone-cell, cell-cell, and both against the latents) lives in
one softmax. Scores are pure content ``q.k``: the eval-time knock-out showed
the learned geometric logit biases of the §5.3 path are decorative in the
trained function, so this path carries none, and geometry reaches the rows
the way it demonstrably works — through window-structure content.

The packed layout is derived on device from views the batch already carries
(``seq_lens``, ``legal_offsets``, ``cell_pos``, ``stone_slot``), for the same
reason the §5.1c pair tables are: the int64 index views cost several times
more to ship over PCIe than to derive beside the model.

Two backends, the in-tree pattern of the §5.3 kernel: FlexAttention with a
document block-mask on CUDA, and a packed fp32 segment softmax everywhere
else and for equivalence tests. The document mask captures no learned
tensors, so the CUDA backward takes no atomic-add path at all — unlike the
§5.3 bias-table gradient, this attention is deterministic end to end.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

GLOBAL_ROWS = 4


class SiteLayout(NamedTuple):
    """Packed row indices for one batch, all on the batch's device.

    ``total`` is a host int derived from tensor shapes, so compiled graphs
    treat it symbolically and no device sync is ever taken.
    """

    global_rows: Tensor  # (P, 4) long: packed row of each latent
    stone_rows: Tensor  # (N_s,) long
    cell_rows: Tensor  # (N_c,) long
    doc: Tensor  # (total,) int32: position of every packed row
    total: int


def site_layout(
    seq_lens: Tensor,
    legal_offsets: Tensor,
    cell_pos: Tensor,
    stone_slot: Tensor,
    max_t: int,
    n_stones: int,
) -> SiteLayout:
    """Derive the packed ``[latents; stones; cells]`` row layout.

    ``seq_lens`` counts the live §5.3 rows per position (the four latents
    plus that position's stones); ``stone_slot`` already encodes each
    stone's position and within-position rank against the ``max_t`` stride.
    """
    p = seq_lens.shape[0]
    device = seq_lens.device
    live = seq_lens.long()
    cells_per = legal_offsets[1:] - legal_offsets[:-1]
    rows_per = live + cells_per
    site_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), rows_per.cumsum(0)]
    )
    total = GLOBAL_ROWS * p + n_stones + cell_pos.shape[0]

    stone_pos = stone_slot // max_t
    stone_rank = stone_slot - stone_pos * max_t  # includes the 4 lead rows
    stone_rows = site_offsets.index_select(0, stone_pos) + stone_rank

    cell_rank = (
        torch.arange(cell_pos.shape[0], dtype=torch.long, device=device)
        - legal_offsets.index_select(0, cell_pos)
    )
    cell_rows = (
        site_offsets.index_select(0, cell_pos)
        + live.index_select(0, cell_pos)
        + cell_rank
    )

    global_rows = site_offsets[:-1, None] + torch.arange(
        GLOBAL_ROWS, dtype=torch.long, device=device
    )
    doc = torch.repeat_interleave(
        torch.arange(p, dtype=torch.int32, device=device),
        rows_per,
        output_size=total,
    )
    return SiteLayout(global_rows, stone_rows, cell_rows, doc, total)


def pack_rows(g: Tensor, s: Tensor, c: Tensor, layout: SiteLayout) -> Tensor:
    """Scatter latents, stones, and cells into one ``(total, H)`` buffer."""
    h = s.shape[-1]
    rows = s.new_empty(layout.total, h)
    rows.index_copy_(0, layout.global_rows.reshape(-1), g.reshape(-1, h))
    rows.index_copy_(0, layout.stone_rows, s)
    rows.index_copy_(0, layout.cell_rows, c.to(s.dtype))
    return rows


def site_attention_reference(q: Tensor, k: Tensor, v: Tensor, doc: Tensor) -> Tensor:
    """Packed block-diagonal attention, fp32 segment softmax.

    ``q``, ``k``, ``v`` are ``(total, heads, hd)``; rows attend exactly to
    rows of the same document. O(pairs) memory — the CPU/test backend, and
    the numeric ground truth the CUDA backend is checked against.
    """
    total, heads, hd = q.shape
    scores = torch.einsum("qad,kad->qka", q.float(), k.float()) / math.sqrt(hd)
    mask = doc[:, None] == doc[None, :]
    scores = scores.masked_fill(~mask[:, :, None], torch.finfo(torch.float32).min)
    weights = scores.softmax(dim=1)
    out = torch.einsum("qka,kad->qad", weights, v.float())
    return out.to(v.dtype)


@torch.compiler.disable
def document_mask(layout: SiteLayout):
    """The FlexAttention block mask for one batch's document layout.

    Built once per forward in the trunk — the layout is identical for every
    block — and only on CUDA; the reference backend masks from ``doc``
    directly. The mask captures no learned tensor, so no gradient flows
    through it and the flex backward takes no atomic-add path.

    Documents are contiguous row ranges, so each query block's live kv
    blocks are one contiguous span; the ``BlockMask`` is assembled directly
    from the doc array instead of through ``create_block_mask``, whose
    generic mask evaluation does not survive ``torch.compile``'s symbolic
    shapes (inductor ``CantSplit``) and would materialize the full row-pair
    grid in eager. The builder is compiler-disabled: it runs eagerly at
    every forward and the compiled graph consumes the finished mask.
    """
    if not layout.doc.is_cuda:
        return None
    from torch.nn.attention.flex_attention import BlockMask

    block = 128
    total = layout.total
    device = layout.doc.device
    blocks = (total + block - 1) // block
    # Row spans per document: the four latent rows lead each document, so
    # its first packed row is its first latent row.
    doc_start = layout.global_rows[:, 0]
    doc_end = (
        torch.cat(
            [doc_start[1:], torch.tensor([total], device=device)]
        )
        - 1
    )
    doc_first_block = doc_start // block
    doc_last_block = doc_end // block

    starts = torch.arange(blocks, device=device) * block
    ends = torch.minimum(starts + block - 1, starts.new_tensor(total - 1))
    lo = layout.doc.index_select(0, starts).long()
    hi = layout.doc.index_select(0, ends).long()
    kv_first = doc_first_block.index_select(0, lo)
    kv_last = doc_last_block.index_select(0, hi)
    kv_num = (kv_last - kv_first + 1).to(torch.int32)
    span = int(kv_num.max())
    kv_indices = torch.minimum(
        kv_first[:, None] + torch.arange(span, device=device)[None, :],
        kv_first.new_tensor(blocks - 1),
    ).to(torch.int32)

    # The tail of the last block is dead rows; a sentinel keeps the mask_mod
    # defined there without joining any real document.
    doc = torch.full((blocks * block,), -1, dtype=torch.int32, device=device)
    doc[:total] = layout.doc

    def same_doc(b, h, q_idx, kv_idx):
        return doc[q_idx] == doc[kv_idx]

    return BlockMask.from_kv_blocks(
        kv_num.view(1, 1, blocks),
        kv_indices.view(1, 1, blocks, span),
        BLOCK_SIZE=block,
        mask_mod=same_doc,
        seq_lengths=(total, total),
    )


def site_attention(
    q: Tensor, k: Tensor, v: Tensor, layout: SiteLayout, block_mask
) -> Tensor:
    """Dispatch: FlexAttention under the prebuilt document mask on CUDA, the
    packed reference elsewhere. Inputs and output are ``(total, heads, hd)``."""
    if not q.is_cuda:
        return site_attention_reference(q, k, v, layout.doc)
    from torch.nn.attention.flex_attention import flex_attention

    out = flex_attention(
        q.transpose(0, 1)[None],
        k.transpose(0, 1)[None],
        v.transpose(0, 1)[None],
        block_mask=block_mask,
    )
    return out[0].transpose(0, 1)
