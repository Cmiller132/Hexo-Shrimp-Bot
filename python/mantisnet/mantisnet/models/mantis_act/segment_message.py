"""Section 14's relation-gated message, fused into one segment reduction.

    msg = sigmoid(Wg(rel)) * Wv(LN(src)) + Wb(rel)

summed by destination in-register; the per-edge vector never reaches memory.
The backward re-derives the message from saved inputs.  ``message_plan``
sorts edges into three CSR views (by dest, by src, by relation) reused across
blocks.  Accumulators are fp32 (§27).  CPU: torch parity reference of §36.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without triton
    triton = None
    tl = None


# The axis stream runs three channels per node; a program keeps one accumulator
# per channel and Triton wants a power-of-two leading dimension, so the fourth
# is padded and never stored.
_AXIS_PAD = 4

# A run is a destination's or a source's edges: tens of rows for incidence,
# hundreds for a late-position radius destination. One warp covers a 64-wide
# row at two elements per lane, which is the whole block; more warps would sit
# idle on a block this narrow.
_RUN_WARPS = 1

# The relation view is the hostile layout: hex adjacency puts its entire edge
# set on one relation id, so a class run alone is a whole-family reduction.
# Each run is cut across programs, each summing ``_BLOCK_E``-row tiles into its
# own partial.
_BLOCK_E = 32
_MAX_SPLITS = 64
_REL_WARPS = 4
# Splits multiply the partial buffer and the program count by the relation
# count, and a wide vocabulary (e.g. incidence's 2187 rows) has short runs
# whatever the skew, so slicing them further only buys empty programs and a
# larger second-stage reduction. Capping the product avoids that.
_MAX_PARTIAL_ROWS = 1 << 14

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}


# --------------------------------------------------------------------------
# The three CSR views


@dataclass(frozen=True, eq=False)
class MessagePlan:
    """One typed edge family in the three orderings the kernels reduce over.

    ``dst_*`` is the family sorted by destination, ``src_*`` by source, and
    ``rel_*`` by relation; each ``*_ptr`` is the CSR offset array of that key
    and holds one entry per node or class plus one. Index columns are int32,
    halving what every kernel iteration loads.

    ``channels`` is 1 for the invariant stream and 3 for the axis stream. An
    axis plan is built over the routed rows alone, so ``axis`` is never ``-1``
    and a row's value and output slots are ``node * 3 + axis``.
    """

    n_src: int
    n_dst: int
    n_relations: int
    n_edges: int
    channels: int
    dst_ptr: Tensor
    dst_src: Tensor
    dst_rel: Tensor
    dst_axis: Tensor | None
    src_ptr: Tensor
    src_dst: Tensor
    src_rel: Tensor
    src_axis: Tensor | None
    rel_ptr: Tensor
    rel_src: Tensor
    rel_dst: Tensor
    rel_axis: Tensor | None

    def __post_init__(self) -> None:
        """Validate the transport shape without reading device values.

        Plan contents are checked while they are built on the CPU.  This gate
        deliberately limits itself to metadata, shapes, dtypes, contiguity and
        device agreement so reconstructing a plan in :meth:`to` never stalls a
        CUDA stream to inspect a value.
        """
        for name in ("n_src", "n_dst", "n_relations", "n_edges"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"MessagePlan.{name} must be a nonnegative int, got {value!r}")
        if self.channels not in (1, 3):
            raise ValueError(f"MessagePlan.channels must be 1 or 3, got {self.channels}")

        expected = {
            "dst_ptr": self.n_dst + 1,
            "dst_src": self.n_edges,
            "dst_rel": self.n_edges,
            "src_ptr": self.n_src + 1,
            "src_dst": self.n_edges,
            "src_rel": self.n_edges,
            "rel_ptr": self.n_relations + 1,
            "rel_src": self.n_edges,
            "rel_dst": self.n_edges,
        }
        if self.channels == 3:
            expected.update(
                dst_axis=self.n_edges,
                src_axis=self.n_edges,
                rel_axis=self.n_edges,
            )
        elif any(axis is not None for axis in (self.dst_axis, self.src_axis, self.rel_axis)):
            raise ValueError("a one-channel MessagePlan must not carry axis columns")

        device = None
        for name, rows in expected.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"MessagePlan.{name} must be a tensor, got {type(value).__name__}")
            if value.dtype != torch.int32:
                raise TypeError(f"MessagePlan.{name} must be int32, got {value.dtype}")
            if value.ndim != 1 or int(value.shape[0]) != rows:
                raise ValueError(
                    f"MessagePlan.{name} must be ({rows},), got {tuple(value.shape)}"
                )
            if not value.is_contiguous():
                raise ValueError(f"MessagePlan.{name} must be contiguous")
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError(
                    f"MessagePlan.{name} is on {value.device}, other columns on {device}"
                )

        if device is not None and device.type == "cpu":
            for name in ("dst_ptr", "src_ptr", "rel_ptr"):
                ptr = getattr(self, name)
                if int(ptr[0]) != 0 or int(ptr[-1]) != self.n_edges:
                    raise ValueError(
                        f"MessagePlan.{name} must span 0..{self.n_edges}, got "
                        f"{int(ptr[0])}..{int(ptr[-1])}"
                    )
                if bool((ptr[1:] < ptr[:-1]).any()):
                    raise ValueError(f"MessagePlan.{name} must be monotone")
            for name, bound in (
                ("dst_src", self.n_src),
                ("dst_rel", self.n_relations),
                ("src_dst", self.n_dst),
                ("src_rel", self.n_relations),
                ("rel_src", self.n_src),
                ("rel_dst", self.n_dst),
            ):
                values = getattr(self, name)
                if values.numel() and bool(((values < 0) | (values >= bound)).any()):
                    raise ValueError(
                        f"MessagePlan.{name} contains an index outside 0..{bound - 1}"
                    )
            if self.channels == 3:
                for name in ("dst_axis", "src_axis", "rel_axis"):
                    values = getattr(self, name)
                    if values.numel() and bool(((values < 0) | (values >= 3)).any()):
                        raise ValueError(f"MessagePlan.{name} contains an axis outside 0..2")

    def to(self, device, *, non_blocking: bool = True) -> "MessagePlan":
        """The identical immutable plan with every tensor on ``device``."""
        moved = {
            name: (
                value.to(device, non_blocking=non_blocking)
                if isinstance(value, Tensor)
                else value
            )
            for name, value in vars(self).items()
        }
        return MessagePlan(**moved)

    def pin_memory(self) -> "MessagePlan":
        """The identical plan in page-locked host memory."""
        pinned = {
            name: value.pin_memory() if isinstance(value, Tensor) else value
            for name, value in vars(self).items()
        }
        return MessagePlan(**pinned)

    @property
    def device(self) -> torch.device:
        return self.dst_ptr.device

    def edge_destinations(self) -> Tensor:
        """The destination of each row of the destination-major view."""
        counts = (self.dst_ptr[1:] - self.dst_ptr[:-1]).long()
        return torch.repeat_interleave(
            torch.arange(self.n_dst, device=self.device), counts
        )

    def destination_counts(self) -> Tensor:
        """How many rows reach each output slot, as fp32 — the ``mean`` divisor.

        For the invariant stream that is the destination run lengths straight
        off the CSR. For the axis stream a destination's rows split across its
        three channels, so the count is per ``(node, axis)`` slot.
        """
        if self.channels == 1:
            return (self.dst_ptr[1:] - self.dst_ptr[:-1]).float()
        slots = self.edge_destinations() * self.channels + self.dst_axis.long()
        return torch.zeros(
            self.n_dst * self.channels, dtype=torch.float32, device=self.device
        ).index_add_(0, slots, torch.ones_like(slots, dtype=torch.float32))


def _csr(key: Tensor, n_keys: int) -> Tensor:
    """The CSR offsets of an ascending key column."""
    return torch.searchsorted(
        key, torch.arange(n_keys + 1, device=key.device, dtype=key.dtype)
    ).to(torch.int32)


def message_plan(
    src: Tensor,
    dst: Tensor,
    relation: Tensor,
    axis: Tensor | None,
    n_src: int,
    n_dst: int,
    n_relations: int,
    channels: int,
    *,
    dst_sorted: bool,
) -> MessagePlan:
    """Sort one edge family into the three views the fused message reduces over.

    ``axis`` is required when ``channels`` is 3 and must already be free of the
    ``-1`` of an unrouted edge — an axis plan is built over the routed subset,
    since an unrouted row contributes nothing. `messages.TypedEdges.routed` is
    the only builder of one, and it filters on ``axis >= 0``; where it hands the
    column on unfiltered, the family declared ``fully_routed``.

    ``dst_sorted`` says the rows arrive destination-ascending, in which case the
    destination view is adopted rather than rebuilt. It is a structural
    property the caller states directly: §7 orders ordinary graph edges by
    ``(dst, src, relation)`` and the packed batch concatenates positions in
    order, so the concatenation stays destination-ascending. Only the reverse
    direction of the window-major incidence table is unsorted.
    """
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}")
    if channels == 3:
        if axis is None:
            raise ValueError("an axis plan needs the edges' axis routes")
    else:
        axis = None

    def _view(order: Tensor | None, *columns: Tensor | None) -> tuple[Tensor, ...]:
        return tuple(
            None
            if column is None
            else (column if order is None else column.index_select(0, order)).to(
                torch.int32
            )
            for column in columns
        )

    dst_order = None if dst_sorted else torch.argsort(dst, stable=True)
    dst_key = dst if dst_order is None else dst.index_select(0, dst_order)
    dst_src, dst_rel, dst_axis = _view(dst_order, src, relation, axis)

    src_order = torch.argsort(src, stable=True)
    src_key = src.index_select(0, src_order)
    src_dst, src_rel, src_axis = _view(src_order, dst, relation, axis)

    rel_order = torch.argsort(relation, stable=True)
    rel_key = relation.index_select(0, rel_order)
    rel_src, rel_dst, rel_axis = _view(rel_order, src, dst, axis)

    return MessagePlan(
        n_src=int(n_src),
        n_dst=int(n_dst),
        n_relations=int(n_relations),
        n_edges=int(src.shape[0]),
        channels=int(channels),
        dst_ptr=_csr(dst_key, n_dst),
        dst_src=dst_src,
        dst_rel=dst_rel,
        dst_axis=dst_axis,
        src_ptr=_csr(src_key, n_src),
        src_dst=src_dst,
        src_rel=src_rel,
        src_axis=src_axis,
        rel_ptr=_csr(rel_key, n_relations),
        rel_src=rel_src,
        rel_dst=rel_dst,
        rel_axis=rel_axis,
    )


def _splits(n_relations: int, n_edges: int) -> int:
    """How many programs share a relation run (a power of two).

    Sized against the whole edge set rather than the average run, since the
    skew is the point — hex adjacency puts every one of its edges on a single
    relation id, so an average over the vocabulary would starve that one live
    run. A class run shorter than its share simply leaves its later slices
    empty, which costs a program that exits at once; reading the longest run
    instead would cost a device synchronisation on a path that is already
    host-bound.
    """
    wanted = max(1, n_edges // (_BLOCK_E * 4))
    cap = min(_MAX_SPLITS, max(1, _MAX_PARTIAL_ROWS // max(n_relations, 1)))
    splits = 1
    while splits * 2 <= min(wanted, cap):
        splits *= 2
    return splits


# --------------------------------------------------------------------------
# Kernels


if triton is not None:

    @triton.jit
    def _message_forward_kernel(
        values_ptr,
        gate_ptr,
        bias_ptr,
        dst_ptr,
        edge_src,
        edge_rel,
        edge_axis,
        out_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NCHAN: tl.constexpr,
        PAD: tl.constexpr,
        GATED: tl.constexpr,
    ):
        # One program per destination node. It walks that node's contiguous
        # edge run, forms each edge's message in registers, and adds it to the
        # accumulator of the channel the edge routes through. No (E, D) tensor
        # exists at any point.
        dst = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        live = offs < D
        chan = tl.arange(0, PAD)
        start = tl.load(dst_ptr + dst)
        end = tl.load(dst_ptr + dst + 1)
        acc = tl.zeros([PAD, BLOCK_D], dtype=tl.float32)
        for entry in tl.range(start, end):
            source = tl.load(edge_src + entry)
            relation = tl.load(edge_rel + entry)
            route = tl.load(edge_axis + entry) if NCHAN > 1 else 0
            message = tl.load(
                values_ptr + (source * NCHAN + route) * D + offs, mask=live, other=0.0
            ).to(tl.float32)
            if GATED:
                message *= tl.load(
                    gate_ptr + relation * D + offs, mask=live, other=0.0
                ).to(tl.float32)
            message += tl.load(
                bias_ptr + relation * D + offs, mask=live, other=0.0
            ).to(tl.float32)
            acc += tl.where(chan[:, None] == route, message[None, :], 0.0)
        rows = dst * NCHAN + chan
        tl.store(
            out_ptr + rows[:, None] * D + offs[None, :],
            acc,
            mask=(chan[:, None] < NCHAN) & live[None, :],
        )

    @triton.jit
    def _message_dvalue_kernel(
        gate_ptr,
        grad_ptr,
        src_ptr,
        edge_dst,
        edge_rel,
        edge_axis,
        dvalue_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NCHAN: tl.constexpr,
        PAD: tl.constexpr,
        GATED: tl.constexpr,
    ):
        # The mirror of the forward over the source-major view: a source's
        # gradient is its edges' gates times the destinations' output
        # gradients, which is again a contiguous run and again needs no stored
        # message.
        source = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        live = offs < D
        chan = tl.arange(0, PAD)
        start = tl.load(src_ptr + source)
        end = tl.load(src_ptr + source + 1)
        acc = tl.zeros([PAD, BLOCK_D], dtype=tl.float32)
        for entry in tl.range(start, end):
            dst = tl.load(edge_dst + entry)
            route = tl.load(edge_axis + entry) if NCHAN > 1 else 0
            reaching = tl.load(
                grad_ptr + (dst * NCHAN + route) * D + offs, mask=live, other=0.0
            ).to(tl.float32)
            if GATED:
                relation = tl.load(edge_rel + entry)
                reaching *= tl.load(
                    gate_ptr + relation * D + offs, mask=live, other=0.0
                ).to(tl.float32)
            acc += tl.where(chan[:, None] == route, reaching[None, :], 0.0)
        rows = source * NCHAN + chan
        tl.store(
            dvalue_ptr + rows[:, None] * D + offs[None, :],
            acc,
            mask=(chan[:, None] < NCHAN) & live[None, :],
        )

    @triton.jit
    def _message_drelation_kernel(
        values_ptr,
        grad_ptr,
        rel_ptr,
        edge_src,
        edge_dst,
        edge_axis,
        dgate_ptr,
        dbias_ptr,
        SPLITS: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_E: tl.constexpr,
        NCHAN: tl.constexpr,
        GATED: tl.constexpr,
    ):
        # One slice of one relation run. Both table gradients read the same
        # output-gradient tile, so they share the walk. Slice bounds and the
        # in-slice tile order are functions of the plan alone, which is what
        # makes the two-stage reduction deterministic.
        relation = tl.program_id(0)
        part = tl.program_id(1)
        offs = tl.arange(0, BLOCK_D)
        live = offs < D
        start = tl.load(rel_ptr + relation)
        end = tl.load(rel_ptr + relation + 1)
        per = (end - start + SPLITS - 1) // SPLITS
        lo = start + part * per
        hi = tl.minimum(lo + per, end)
        acc_gate = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc_bias = tl.zeros([BLOCK_D], dtype=tl.float32)
        for base in tl.range(lo, hi, BLOCK_E):
            entries = base + tl.arange(0, BLOCK_E)
            inside = entries < hi
            dst = tl.load(edge_dst + entries, mask=inside, other=0)
            route = (
                tl.load(edge_axis + entries, mask=inside, other=0)
                if NCHAN > 1
                else tl.zeros([BLOCK_E], dtype=tl.int32)
            )
            reaching = tl.load(
                grad_ptr + (dst * NCHAN + route)[:, None] * D + offs[None, :],
                mask=inside[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_bias += tl.sum(reaching, axis=0)
            if GATED:
                source = tl.load(edge_src + entries, mask=inside, other=0)
                value = tl.load(
                    values_ptr + (source * NCHAN + route)[:, None] * D + offs[None, :],
                    mask=inside[:, None] & live[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc_gate += tl.sum(value * reaching, axis=0)
        slot = (relation * SPLITS + part) * D + offs
        tl.store(dbias_ptr + slot, acc_bias, mask=live)
        if GATED:
            tl.store(dgate_ptr + slot, acc_gate, mask=live)


# --------------------------------------------------------------------------
# The torch reference (§36) — CPU, unsupported signatures, and parity


def _reference(
    values: Tensor,
    gate: Tensor | None,
    bias: Tensor,
    dst_ptr: Tensor,
    dst_src: Tensor,
    dst_rel: Tensor,
    dst_axis: Tensor | None,
    channels: int,
) -> Tensor:
    """The ``(E, D)`` formulation of §14, aggregated in fp32.

    Literal: gather the value, multiply by the gathered gate, add the gathered
    bias, scatter into the destinations. This is the tensor shape the fused
    path exists to avoid, and keeping it is what makes a parity test mean
    anything.

    Accumulation is *at least* fp32 rather than exactly fp32: §27 asks for a
    promotion, and an fp64 gradient check would be silently demoted by a
    literal ``.float()``.
    """
    n_dst = int(dst_ptr.numel()) - 1
    width = int(values.shape[1])
    accumulate = torch.promote_types(values.dtype, torch.float32)
    out = torch.zeros(
        n_dst * channels, width, dtype=accumulate, device=values.device
    )
    if dst_src.numel() == 0:
        return out
    counts = (dst_ptr[1:] - dst_ptr[:-1]).long()
    edge_dst = torch.repeat_interleave(
        torch.arange(n_dst, device=values.device), counts
    )
    route = (
        torch.zeros_like(edge_dst)
        if channels == 1
        else dst_axis.long()  # type: ignore[union-attr]
    )
    relation = dst_rel.long()
    message = values.to(accumulate).index_select(0, dst_src.long() * channels + route)
    if gate is not None:
        message = message * gate.to(accumulate).index_select(0, relation)
    message = message + bias.to(accumulate).index_select(0, relation)
    return out.index_add_(0, edge_dst * channels + route, message)


def _reference_backward(
    values: Tensor,
    gate: Tensor | None,
    bias: Tensor,
    dst_ptr: Tensor,
    dst_src: Tensor,
    dst_rel: Tensor,
    dst_axis: Tensor | None,
    channels: int,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    n_dst = int(dst_ptr.numel()) - 1
    accumulate = torch.promote_types(values.dtype, torch.float32)
    d_values = torch.zeros(values.shape, dtype=accumulate, device=values.device)
    d_gate = _gate_gradient(gate, bias)
    d_bias = torch.zeros(bias.shape, dtype=accumulate, device=bias.device)
    if dst_src.numel():
        counts = (dst_ptr[1:] - dst_ptr[:-1]).long()
        edge_dst = torch.repeat_interleave(
            torch.arange(n_dst, device=values.device), counts
        )
        route = (
            torch.zeros_like(edge_dst)
            if channels == 1
            else dst_axis.long()  # type: ignore[union-attr]
        )
        relation = dst_rel.long()
        value_row = dst_src.long() * channels + route
        reaching = grad_out.to(accumulate).index_select(
            0, edge_dst * channels + route
        )
        d_bias = d_bias.index_add_(0, relation, reaching)
        if gate is None:
            d_values = d_values.index_add_(0, value_row, reaching)
        else:
            value = values.to(accumulate).index_select(0, value_row)
            d_gate = d_gate.to(accumulate).index_add_(0, relation, value * reaching)
            d_values = d_values.index_add_(
                0, value_row, gate.to(accumulate).index_select(0, relation) * reaching
            )
    return (
        d_values.to(values.dtype),
        d_gate if gate is None else d_gate.to(gate.dtype),
        d_bias.to(bias.dtype),
    )


def _gate_gradient(gate: Tensor | None, bias: Tensor) -> Tensor:
    """A zero gate gradient, or the width-zero stand-in for the additive control.

    A custom op's return type cannot hold an optional tensor, so §14's ungated
    ``incidence_message="additive"`` path answers with a ``(0, D)`` tensor and
    the autograd dispatcher turns it back into ``None``.
    """
    if gate is None:
        return bias.new_zeros((0, bias.shape[1]))
    return torch.zeros_like(gate)


# --------------------------------------------------------------------------
# Guards, launches, and the custom op


def _validate(
    values: Tensor,
    gate: Tensor | None,
    bias: Tensor,
    dst_ptr: Tensor,
    dst_src: Tensor,
    dst_axis: Tensor | None,
    channels: int,
) -> None:
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}")
    if values.ndim != 2:
        raise ValueError(
            f"values must be (N_src * channels, D), got {tuple(values.shape)}"
        )
    if values.shape[0] % channels:
        raise ValueError(
            f"values has {values.shape[0]} rows, not a multiple of {channels} channels"
        )
    if bias.ndim != 2 or bias.shape[1] != values.shape[1]:
        raise ValueError("bias must be (R, D) beside values' D")
    if gate is not None and gate.shape != bias.shape:
        raise ValueError(
            f"gate is {tuple(gate.shape)} against bias' {tuple(bias.shape)}"
        )
    if channels == 3 and dst_axis is None:
        raise ValueError("an axis message needs the edges' axis routes")
    if dst_ptr.numel() < 1:
        raise ValueError("dst_ptr must hold one entry per destination plus one")
    tensors = [values, bias, dst_ptr, dst_src]
    if gate is not None:
        tensors.append(gate)
    if dst_axis is not None:
        tensors.append(dst_axis)
    if any(tensor.device != values.device for tensor in tensors):
        raise ValueError("every segment-message input must be on one device")


def _shape_key(values: Tensor, channels: int, gated: bool) -> tuple[object, ...]:
    return (
        values.device.type,
        values.device.index,
        values.dtype,
        int(values.shape[1]),
        channels,
        gated,
    )


def _supported(values: Tensor, dst_ptr: Tensor) -> bool:
    return (
        triton is not None
        and values.is_cuda
        and values.dtype == torch.float32
        and values.shape[1] <= 512
        and dst_ptr.numel() > 1
    )


def _launch_forward(
    values: Tensor,
    gate: Tensor | None,
    bias: Tensor,
    dst_ptr: Tensor,
    dst_src: Tensor,
    dst_rel: Tensor,
    dst_axis: Tensor | None,
    channels: int,
) -> Tensor:
    width = int(values.shape[1])
    n_dst = int(dst_ptr.numel()) - 1
    out = torch.zeros(
        n_dst * channels, width, dtype=torch.float32, device=values.device
    )
    if dst_src.numel() == 0:
        return out
    # A pointer argument the kernel never dereferences still has to be a
    # pointer: ``GATED`` and ``NCHAN`` are compile-time, so the gate load and
    # the axis load are absent from the ungated and invariant variants, but
    # Triton specialises a ``None`` argument as a constant and then rejects the
    # arithmetic that computes the address. Passing a live tensor of the right
    # dtype is what keeps the two variants one kernel.
    _message_forward_kernel[(n_dst,)](
        values,
        bias if gate is None else gate,
        bias,
        dst_ptr,
        dst_src,
        dst_rel,
        dst_src if dst_axis is None else dst_axis,
        out,
        D=width,
        BLOCK_D=triton.next_power_of_2(width),
        NCHAN=channels,
        PAD=1 if channels == 1 else _AXIS_PAD,
        GATED=gate is not None,
        num_warps=_RUN_WARPS,
    )
    return out


def _launch_backward(
    values: Tensor,
    gate: Tensor | None,
    bias: Tensor,
    src_ptr: Tensor,
    src_dst: Tensor,
    src_rel: Tensor,
    src_axis: Tensor | None,
    rel_ptr: Tensor,
    rel_src: Tensor,
    rel_dst: Tensor,
    rel_axis: Tensor | None,
    channels: int,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    width = int(values.shape[1])
    block_d = triton.next_power_of_2(width)
    n_src = int(src_ptr.numel()) - 1
    n_relations = int(bias.shape[0])
    grad_out = grad_out.contiguous().float()
    if src_dst.numel() == 0:
        return (
            torch.zeros_like(values),
            _gate_gradient(gate, bias),
            torch.zeros_like(bias),
        )

    # Every source row and every relation slice is stored unconditionally — a
    # node or class with no edge writes its zero — so nothing here needs to be
    # zeroed first.
    d_values = torch.empty_like(values)
    _message_dvalue_kernel[(n_src,)](
        bias if gate is None else gate,
        grad_out,
        src_ptr,
        src_dst,
        src_rel,
        src_dst if src_axis is None else src_axis,
        d_values,
        D=width,
        BLOCK_D=block_d,
        NCHAN=channels,
        PAD=1 if channels == 1 else _AXIS_PAD,
        GATED=gate is not None,
        num_warps=_RUN_WARPS,
    )

    splits = _splits(n_relations, int(src_dst.numel()))
    partial_bias = torch.empty(
        n_relations * splits, width, dtype=torch.float32, device=values.device
    )
    partial_gate = (
        None
        if gate is None
        else torch.empty(
            n_relations * splits, width, dtype=torch.float32, device=values.device
        )
    )
    _message_drelation_kernel[(n_relations, splits)](
        values,
        grad_out,
        rel_ptr,
        rel_src,
        rel_dst,
        rel_src if rel_axis is None else rel_axis,
        partial_bias if partial_gate is None else partial_gate,
        partial_bias,
        SPLITS=splits,
        D=width,
        BLOCK_D=block_d,
        BLOCK_E=_BLOCK_E,
        NCHAN=channels,
        GATED=gate is not None,
        num_warps=_REL_WARPS,
    )
    # Summing the slices along a contiguous dimension, in index order: the
    # second stage of the reduction is as deterministic as the first.
    d_bias = partial_bias.view(n_relations, splits, width).sum(dim=1)
    d_gate = (
        _gate_gradient(gate, bias)
        if partial_gate is None
        else partial_gate.view(n_relations, splits, width).sum(dim=1)
    )
    return d_values, d_gate, d_bias


@torch.library.custom_op("mantisnet::act_segment_message", mutates_args=())
def _segment_message_op(
    values: Tensor,
    gate: Tensor | None,
    bias: Tensor,
    dst_ptr: Tensor,
    dst_src: Tensor,
    dst_rel: Tensor,
    dst_axis: Tensor | None,
    src_ptr: Tensor,
    src_dst: Tensor,
    src_rel: Tensor,
    src_axis: Tensor | None,
    rel_ptr: Tensor,
    rel_src: Tensor,
    rel_dst: Tensor,
    rel_axis: Tensor | None,
    channels: int,
) -> Tensor:
    _validate(values, gate, bias, dst_ptr, dst_src, dst_axis, channels)
    reference = lambda: _reference(  # noqa: E731
        values, gate, bias, dst_ptr, dst_src, dst_rel, dst_axis, channels
    )
    if not _supported(values, dst_ptr):
        return reference()
    key = _shape_key(values, channels, gate is not None)
    if key in _FAILED_SHAPES:
        return reference()
    try:
        return _launch_forward(
            values, gate, bias, dst_ptr, dst_src, dst_rel, dst_axis, channels
        )
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused segment message failed for D={values.shape[1]}, "
            f"channels={channels}; gathering instead for this shape: "
            f"{_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_segment_message_op.register_fake
def _(
    values: Tensor,
    gate: Tensor | None,
    bias: Tensor,
    dst_ptr: Tensor,
    dst_src: Tensor,
    dst_rel: Tensor,
    dst_axis: Tensor | None,
    src_ptr: Tensor,
    src_dst: Tensor,
    src_rel: Tensor,
    src_axis: Tensor | None,
    rel_ptr: Tensor,
    rel_src: Tensor,
    rel_dst: Tensor,
    rel_axis: Tensor | None,
    channels: int,
) -> Tensor:
    return values.new_empty(
        ((dst_ptr.numel() - 1) * channels, values.shape[1]),
        dtype=torch.promote_types(values.dtype, torch.float32),
    )


@torch.library.custom_op("mantisnet::act_segment_message_backward", mutates_args=())
def _segment_message_backward_op(
    values: Tensor,
    gate: Tensor | None,
    bias: Tensor,
    dst_ptr: Tensor,
    dst_src: Tensor,
    dst_rel: Tensor,
    dst_axis: Tensor | None,
    src_ptr: Tensor,
    src_dst: Tensor,
    src_rel: Tensor,
    src_axis: Tensor | None,
    rel_ptr: Tensor,
    rel_src: Tensor,
    rel_dst: Tensor,
    rel_axis: Tensor | None,
    channels: int,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    reference = lambda: _reference_backward(  # noqa: E731
        values, gate, bias, dst_ptr, dst_src, dst_rel, dst_axis, channels, grad_out
    )
    if not _supported(values, dst_ptr):
        return reference()
    key = _shape_key(values, channels, gate is not None) + (grad_out.dtype,)
    if key in _FAILED_BACKWARD_SHAPES:
        return reference()
    try:
        return _launch_backward(
            values,
            gate,
            bias,
            src_ptr,
            src_dst,
            src_rel,
            src_axis,
            rel_ptr,
            rel_src,
            rel_dst,
            rel_axis,
            channels,
            grad_out,
        )
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused segment message backward failed for D={values.shape[1]}, "
            f"channels={channels}; scattering instead for this shape: "
            f"{_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_segment_message_backward_op.register_fake
def _(
    values: Tensor,
    gate: Tensor | None,
    bias: Tensor,
    dst_ptr: Tensor,
    dst_src: Tensor,
    dst_rel: Tensor,
    dst_axis: Tensor | None,
    src_ptr: Tensor,
    src_dst: Tensor,
    src_rel: Tensor,
    src_axis: Tensor | None,
    rel_ptr: Tensor,
    rel_src: Tensor,
    rel_dst: Tensor,
    rel_axis: Tensor | None,
    channels: int,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(values),
        _gate_gradient(gate, bias),
        torch.empty_like(bias),
    )


def _setup_context(ctx, inputs, output) -> None:
    ctx.channels = inputs[-1]
    ctx.gated = inputs[1] is not None
    ctx.save_for_backward(*inputs[:-1])


def _dispatch_backward(ctx, grad_out: Tensor):
    d_values, d_gate, d_bias = _segment_message_backward_op(
        *ctx.saved_tensors, ctx.channels, grad_out
    )
    return (d_values, d_gate if ctx.gated else None, d_bias) + (None,) * 13


_segment_message_op.register_autograd(_dispatch_backward, setup_context=_setup_context)


def relation_gated_message(
    values: Tensor,
    gate: Tensor | None,
    bias: Tensor,
    plan: MessagePlan,
) -> Tensor:
    """§14's message summed by destination, without an ``(E, D)`` intermediate.

    ``values`` is the projected source table — ``(N_src, D)`` for the invariant
    stream, ``(N_src * 3, D)`` for the axis stream — and ``gate`` and ``bias``
    are the relation table's projections, ``(R, D)`` each, with ``gate=None``
    for §14's additive control. The result is ``(N_dst * channels, D)`` in fp32
    (§27).
    """
    if plan.n_relations != bias.shape[0]:
        raise ValueError(
            f"the plan has {plan.n_relations} relation classes against the "
            f"{bias.shape[0]}-row table"
        )
    if values.shape[0] != plan.n_src * plan.channels:
        raise ValueError(
            f"values has {values.shape[0]} rows against the plan's "
            f"{plan.n_src} sources over {plan.channels} channels"
        )
    return _segment_message_op(
        values.contiguous(),
        None if gate is None else gate.contiguous(),
        bias.contiguous(),
        plan.dst_ptr,
        plan.dst_src,
        plan.dst_rel,
        plan.dst_axis,
        plan.src_ptr,
        plan.src_dst,
        plan.src_rel,
        plan.src_axis,
        plan.rel_ptr,
        plan.rel_src,
        plan.rel_dst,
        plan.rel_axis,
        plan.channels,
    )


__all__ = [
    "MessagePlan",
    "message_plan",
    "relation_gated_message",
]
