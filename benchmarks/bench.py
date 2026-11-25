"""Forward+backward time and peak memory of one SplineConv layer vs batch size.

Compared:
  * pyg        : torch_geometric SplineConv on the batch seen as B disjoint copies of the mesh
  * aggregate  : FixedMeshSplineConv, order="aggregate" (SpMM then one GEMM)
  * transform  : FixedMeshSplineConv, order="transform" (one GEMM then SpMM)
  * fused      : FixedMeshSplineConv, order="fused" (CUDA kernel, Z never stored)
  * conv2d     : cuDNN conv2d on the regular grid with the same N, K, C (a "speed of light"
                 reference: what the same layer costs when the mesh is a grid)

Mesh: a jittered 2D grid (continuous pseudo-coordinates, (2r+1)^2 neighbours per node).

    python benchmarks/bench.py --size 128 --channels 32
"""
import argparse
import csv
import os
import warnings

import torch
import torch.nn.functional as F

from batched_splineconv import FixedMeshSplineConv, SplineOperator
from batched_splineconv.meshes import jittered_grid

torch.backends.cudnn.allow_tf32 = False   # compare fp32 with fp32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2], (torch.cuda.max_memory_allocated() - base) / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=128, help="grid side, N = size^2")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--out", default="benchmarks/results")
    a = p.parse_args()

    dev = "cuda"
    H = W = a.size
    N, C, r = H * W, a.channels, a.radius
    k = 2 * r + 1
    pos, ei, pseudo = jittered_grid((H, W), r, device=dev)
    op = SplineOperator(ei, pseudo, k, num_nodes=N).to(dev)
    print(op)

    from torch_geometric.nn import SplineConv
    ref = SplineConv(C, C, 2, k).to(dev)
    convs = {o: FixedMeshSplineConv.from_pyg(ref, order=o) for o in ("aggregate", "transform", "fused")}
    w2d = torch.randn(C, C, k, k, device=dev, requires_grad=True)

    rows = []
    for B in a.batches:
        x = torch.randn(B, N, C, device=dev, requires_grad=True)
        shift = torch.arange(B, device=dev)[:, None] * N
        ei_b = torch.stack([(ei[0][None] + shift).reshape(-1), (ei[1][None] + shift).reshape(-1)])
        ps_b = pseudo.repeat(B, 1)
        img = torch.randn(B, C, H, W, device=dev, requires_grad=True)

        def pyg():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref(x.view(B * N, C), ei_b, ps_b).sum().backward()

        cases = {"pyg": pyg,
                 "aggregate": lambda: convs["aggregate"](x, op).sum().backward(),
                 "transform": lambda: convs["transform"](x, op).sum().backward(),
                 "fused": lambda: convs["fused"](x, op).sum().backward(),
                 "conv2d": lambda: F.conv2d(img, w2d, padding=r).sum().backward()}
        for name, fn in cases.items():
            try:
                t, m = timeit(fn)
            except torch.cuda.OutOfMemoryError:
                t, m = float("nan"), float("nan")
            torch.cuda.empty_cache()
            rows.append(dict(method=name, B=B, N=N, C=C, K=k * k, ms=t, peak_MiB=m))
            print(f"B={B:3d} {name:10s} {t:9.2f} ms  {m:9.1f} MiB")
        del x, img, ei_b, ps_b
        torch.cuda.empty_cache()

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"bench_N{N}_C{C}_K{k * k}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, "|", torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
