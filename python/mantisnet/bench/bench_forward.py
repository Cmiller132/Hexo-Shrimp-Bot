"""Throughput of the builder (CPU) and the forward (device) at spec defaults.

Positions are random legal playouts spread over game phases, which is what a
self-play sweep actually feeds the model. Run from python/mantisnet:

    uv run python bench/bench_forward.py [--batch 256] [--iters 50]
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import hexo_py
import torch

from mantisnet import MantisConfig, MantisNet, collate, collate_positions, from_position


def make_positions(count: int, seed: int) -> list[hexo_py.Position]:
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        plies = rng.randint(4, 80)
        pos = hexo_py.Position()
        for _ in range(plies):
            if pos.is_terminal:
                break
            pos.advance(*rng.choice(pos.legal_moves()))
        if not pos.is_terminal:
            out.append(pos)
    return out


def time_forward(net, batch, iters: int, device: str, autocast: bool) -> float:
    def once():
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            net(batch)

    with torch.no_grad():
        for _ in range(5):
            once()
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            once()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--compile", action="store_true", help="torch.compile the forward")
    args = ap.parse_args()

    positions = make_positions(args.batch, args.seed)

    t0 = time.perf_counter()
    graphs = [from_position(p) for p in positions]
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    batch = collate(graphs)
    collate_s = time.perf_counter() - t0
    collate_positions(positions)  # warm the rayon pool
    t0 = time.perf_counter()
    batch = collate_positions(positions)
    rust_s = time.perf_counter() - t0

    stones = sum(g.n_stones for g in graphs)
    windows = sum(g.n_windows for g in graphs)
    cells = sum(g.n_legal for g in graphs)
    print(f"pool: {len(graphs)} positions | {stones} stones | {windows} live windows | {cells} legal cells")
    print(f"build[python]: {len(graphs) / build_s:8.0f} pos/s  ({build_s * 1e3 / len(graphs):.3f} ms/pos, single thread)")
    print(f"collate[python]: {len(graphs) / collate_s:6.0f} pos/s")
    print(f"batch[rust]:   {len(graphs) / rust_s:8.0f} pos/s  ({rust_s * 1e3 / len(graphs):.3f} ms/pos, build+collate, all cores)")

    torch.manual_seed(0)
    net = MantisNet(MantisConfig()).eval()
    params = sum(p.numel() for p in net.parameters())
    print(f"model: {params / 1e6:.2f} M parameters, batch max_t={batch.max_t} max_w={batch.max_w}")
    if args.compile:
        net = torch.compile(net, dynamic=True)

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if args.compile:
        devices = [d for d in devices if d == "cuda"]  # the deploy target
    for device in devices:
        net_d, batch_d = net.to(device), batch.to(device)
        modes = [("fp32", False)] + ([("bf16", True)] if device == "cuda" else [])
        for name, ac in modes:
            t = time_forward(net_d, batch_d, args.iters, device, ac)
            print(f"forward[{device}/{name}]: {t * 1e3:7.2f} ms/batch  {len(graphs) / t:8.0f} pos/s")


if __name__ == "__main__":
    main()
