"""Time each fused kernel separately and report achieved GFLOP/s.

    python benchmarks/profile_fused.py --size 128 --channels 32 --batch 16
"""
import argparse

import torch

from batched_splineconv import SplineOperator
from batched_splineconv.fused import _load
from batched_splineconv.meshes import jittered_grid


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    ext, dev = _load(), "cuda"
    N, C, B = a.size**2, a.channels, a.batch
    _, ei, pseudo = jittered_grid((a.size, a.size), 2, device=dev)
    op = SplineOperator(ei, pseudo, 5, num_nodes=N).to(dev)
    K = op.K
    x = torch.randn(N, B, C, device=dev)
    w = torch.randn(K, C, C, device=dev)
    g = torch.randn(N, B, C, device=dev)
    wt = w.transpose(1, 2).contiguous()
    gemm_flops = 2.0 * N * B * K * C * C
    nnz = op.f_val.numel()
    gather_flops = 2.0 * nnz * B * C
    cases = {
        "forward  gather_gemm": lambda: ext.gather_gemm(x, w, op.f_ptr, op.f_col, op.f_val, N),
        "dX       gather_gemm(T)": lambda: ext.gather_gemm(g, wt, op.ft_ptr, op.ft_col, op.ft_val, N),
        "dW       wgrad": lambda: ext.wgrad(x, g, op.f_ptr, op.f_col, op.f_val, K),
    }
    print(f"N={N} B={B} C={C} K={K} nnz={nnz}  GEMM {gemm_flops / 1e9:.1f} GFLOP + gather {gather_flops / 1e9:.1f} GFLOP")
    for name, fn in cases.items():
        t = bench(fn)
        print(f"{name:26s} {t:7.2f} ms  {(gemm_flops + gather_flops) / t / 1e6:8.0f} GFLOP/s")
    print(torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
