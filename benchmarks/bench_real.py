"""SplineConv layers on real CFD meshes (run ``download_meshes.py`` first).

Connectivity follows the continuous-kernel criterion |x_j - x_i| <= (sqrt(2)/2) n_k h_i with
the local mesh spacing h_i, so the neighbourhood follows the mesh refinement and the degree
varies across the mesh, unlike on the jittered grid of ``bench.py``.

Two kinds of layers:
  * same-mesh layers on each mesh, with three node orderings (mesher / random / RCM) to
    expose the effect of memory locality on the gathers;
  * mesh-to-mesh (coarse -> fine) layers, as in the decoder of a hierarchical surrogate.

    python benchmarks/bench_real.py --batch 16 --channels 32
"""
import argparse
import csv
import glob
import os
import time
import warnings

import torch

from batched_splineconv import FixedMeshSplineConv, SplineOperator
from batched_splineconv.meshes import adaptive_radius_graph, rcm_permutation, read_cgns_tri, read_su2

torch.backends.cuda.matmul.allow_tf32 = False
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load(name):
    if name.startswith("airfrans"):
        path = glob.glob(os.path.join(DATA, name, "**", "*.cgns"), recursive=True)[0]
        return read_cgns_tri(path)[0]
    return read_su2(os.path.join(DATA, "cfdgcn", f"mesh_{name}.su2"))[0]


SAME_MESH = ["RAE2822_xcoarse", "NACA0012_coarse", "NACA4412_lessfine", "NACA0012_fine",
             "airfrans_remeshed", "RAE2822_fine", "NACA4412_fine", "airfrans_clipped"]
MESH_TO_MESH = [("NACA0012_xcoarse", "NACA0012_coarse"), ("NACA0012_coarse", "NACA0012_fine"),
                ("NACA4412_lessfine", "NACA4412_fine"), ("airfrans_remeshed", "airfrans_clipped"),
                ("NACA0012_fine", "NACA0012_coarse"), ("airfrans_clipped", "airfrans_remeshed")]


def timeit(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3, (torch.cuda.max_memory_allocated() - base) / 2**20


def run_case(label, ordering, pos_src, pos_dst, B, C, ks, methods, iters, dev):
    bip = pos_dst is not None
    t0 = time.perf_counter()
    ei, pseudo = adaptive_radius_graph(pos_src, pos_dst, ks)
    n_src = pos_src.size(0)
    n_dst = pos_dst.size(0) if bip else n_src
    ei, pseudo = ei.to(dev), pseudo.to(dev)
    op = SplineOperator(ei, pseudo, ks, num_nodes=(n_src, n_dst)).to(dev)
    build_s = time.perf_counter() - t0
    deg = torch.bincount(ei[1], minlength=n_dst).float()

    from torch_geometric.nn import SplineConv
    ref = SplineConv((C, C) if bip else C, C, 2, ks, root_weight=not bip).to(dev)
    convs = {o: FixedMeshSplineConv.from_pyg(ref, order=o) for o in ("aggregate", "transform", "fused", "auto")}
    x = torch.randn(B, n_src, C, device=dev, requires_grad=True)
    shift = torch.arange(B, device=dev)[:, None]
    ei_b = torch.stack([(ei[0][None] + n_src * shift).reshape(-1), (ei[1][None] + n_dst * shift).reshape(-1)])
    ps_b = pseudo.repeat(B, 1)

    def pyg():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xs = x.reshape(B * n_src, C)
            inp = (xs, None) if bip else xs
            ref(inp, ei_b, ps_b, size=(B * n_src, B * n_dst)).sum().backward()

    cases = {"pyg": pyg,
             "spmm": lambda: convs[spmm_order](x, op).sum().backward(),
             "auto": lambda: convs["auto"](x, op).sum().backward(),
             "fused": lambda: convs["fused"](x, op).sum().backward()}
    agg, trans = n_dst * C, n_src * C
    spmm_order = "aggregate" if agg <= trans else "transform"
    auto_order = convs["auto"]._resolve_order(op, x.detach().transpose(0, 1))
    rows = []
    for name in methods:
        try:
            t, m = timeit(cases[name], iters)
        except torch.cuda.OutOfMemoryError:
            t, m = float("nan"), float("nan")
        torch.cuda.empty_cache()
        rows.append(dict(layer=label, ordering=ordering, method=name, auto_picks=auto_order, B=B, C=C, n_src=n_src, n_dst=n_dst,
                         edges=ei.size(1), deg_mean=round(deg.mean().item(), 1), deg_max=int(deg.max()),
                         build_s=round(build_s, 2), ms=round(t, 2), peak_MiB=round(m, 1)))
        print(f"{label:38s} {ordering:7s} {name:6s} [auto={auto_order:9s}] E={ei.size(1):8d} deg {deg.mean():5.1f}/{int(deg.max()):4d}"
              f"  {t:9.2f} ms {m:9.1f} MiB", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    dev, B, C, ks = "cuda", a.batch, a.channels, a.kernel_size
    g = torch.Generator().manual_seed(0)
    rows = []
    for name in SAME_MESH:
        pos = load(name)
        e0, _ = adaptive_radius_graph(pos, None, ks)
        orderings = {"mesher": torch.arange(pos.size(0)),
                     "random": torch.randperm(pos.size(0), generator=g),
                     "rcm": rcm_permutation(e0, pos.size(0))}
        for o, perm in orderings.items():
            methods = ["pyg", "spmm", "fused"] if o == "mesher" else ["spmm", "fused"]
            rows += run_case(f"{name} ({pos.size(0)})", o, pos[perm], None, B, C, ks, methods, a.iters, dev)
    for src, dst in MESH_TO_MESH:
        ps, pd = load(src), load(dst)
        rows += run_case(f"{src} -> {dst}", "mesher", ps, pd, B, C, ks, ["pyg", "spmm", "fused"], a.iters, dev)
        torch.cuda.empty_cache()

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"bench_real_B{B}_C{C}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, "|", torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
