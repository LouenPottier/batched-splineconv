"""End-to-end hierarchical SplineConv U-Net on a real CFD mesh (run ``download_meshes.py`` first).

The 169k-node AirfRANS mesh is decimated into 7 levels (x3 fewer nodes per level, refinement
preserved), as in a hierarchical mesh surrogate. The U-Net has, per level, one same-mesh
layer in the encoder and one in the decoder, plus down / up mesh-to-mesh layers between
levels (25 SplineConv layers). The same network is built with PyG (batch = B disjoint
copies) and with this library; outputs are checked to match, then forward + backward time
and peak memory are measured.

    python benchmarks/bench_unet.py --batches 1 4 8 16
"""
import argparse
import glob
import os
import time
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from batched_splineconv import FixedMeshSplineConv, SplineOperator
from batched_splineconv.meshes import adaptive_radius_graph, mesh_hierarchy, read_cgns_tri

torch.backends.cuda.matmul.allow_tf32 = False
HERE = os.path.dirname(os.path.abspath(__file__))
KS = 5


class Geometry:
    """All operators of the hierarchy: same-mesh per level, down l->l+1, up l+1->l."""

    def __init__(self, levels, dev):
        self.levels = levels
        self.n = [p.size(0) for p in levels]
        self.graphs = {}
        for l, p in enumerate(levels):
            self.graphs[("same", l)] = adaptive_radius_graph(p, None, KS) + ((self.n[l], self.n[l]),)
            if l + 1 < len(levels):
                q = levels[l + 1]
                self.graphs[("down", l)] = adaptive_radius_graph(p, q, KS) + ((self.n[l], self.n[l + 1]),)
                self.graphs[("up", l)] = adaptive_radius_graph(q, p, KS) + ((self.n[l + 1], self.n[l]),)
        self.ops = {k: SplineOperator(ei.to(dev), ps.to(dev), KS, num_nodes=sz).to(dev)
                    for k, (ei, ps, sz) in self.graphs.items()}
        self._batched = {}

    def batched(self, key, B):
        """PyG-style batched graph (B disjoint copies), cached."""
        if (key, B) not in self._batched:
            ei, ps, (ns, nd) = self.graphs[key]
            ei, ps = ei.to(self.dev), ps.to(self.dev)
            sh = torch.arange(B, device=ei.device)[:, None]
            eb = torch.stack([(ei[0][None] + ns * sh).reshape(-1), (ei[1][None] + nd * sh).reshape(-1)])
            self._batched[(key, B)] = (eb, ps.repeat(B, 1), (B * ns, B * nd))
        return self._batched[(key, B)]


def channels(L):
    return [min(32 * 2 ** (l // 2), 128) for l in range(L)]


class PygUNet(nn.Module):
    def __init__(self, L, c_in=3, c_out=4):
        super().__init__()
        from torch_geometric.nn import SplineConv
        C = channels(L)
        self.L = L
        self.enc = nn.ModuleList([SplineConv(c_in if l == 0 else C[l], C[l], 2, KS) for l in range(L)])
        self.down = nn.ModuleList([SplineConv((C[l], C[l]), C[l + 1], 2, KS, root_weight=False) for l in range(L - 1)])
        self.up = nn.ModuleList([SplineConv((C[l + 1], C[l + 1]), C[l], 2, KS, root_weight=False) for l in range(L - 1)])
        self.dec = nn.ModuleList([SplineConv(2 * C[l], C[l], 2, KS) for l in range(L - 1)])
        self.head = nn.Linear(C[0], c_out)

    def forward(self, x, g: Geometry, B):  # x: [B*N0, c_in]
        def conv(layer, key, h):
            ei, ps, size = g.batched(key, B)
            if key[0] == "same":
                return layer(h, ei, ps)
            return layer((h, None), ei, ps, size=size)
        skips = []
        for l in range(self.L):
            x = F.elu(conv(self.enc[l], ("same", l), x))
            if l + 1 < self.L:
                skips.append(x)
                x = F.elu(conv(self.down[l], ("down", l), x))
        for l in reversed(range(self.L - 1)):
            x = F.elu(conv(self.up[l], ("up", l), x))
            x = F.elu(conv(self.dec[l], ("same", l), torch.cat([x, skips[l]], -1)))
        return self.head(x)


class FixedMeshUNet(nn.Module):
    """Same network with FixedMeshSplineConv; activations stay in the [N, B, C] layout."""

    def __init__(self, ref: PygUNet, order="auto"):
        super().__init__()
        cp = lambda m: FixedMeshSplineConv.from_pyg(m, order=order)
        self.L = ref.L
        self.enc = nn.ModuleList([cp(m) for m in ref.enc])
        self.down = nn.ModuleList([cp(m) for m in ref.down])
        self.up = nn.ModuleList([cp(m) for m in ref.up])
        self.dec = nn.ModuleList([cp(m) for m in ref.dec])
        self.head = ref.head

    def forward(self, x, g: Geometry):  # x: [N0, B, c_in]
        skips = []
        for l in range(self.L):
            x = F.elu(self.enc[l](x, g.ops[("same", l)], layout="NBC"))
            if l + 1 < self.L:
                skips.append(x)
                x = F.elu(self.down[l](x, g.ops[("down", l)], layout="NBC"))
        for l in reversed(range(self.L - 1)):
            x = F.elu(self.up[l](x, g.ops[("up", l)], layout="NBC"))
            x = F.elu(self.dec[l](torch.cat([x, skips[l]], -1), g.ops[("same", l)], layout="NBC"))
        return self.head(x)

    def set_spmm_only(self, g: Geometry):
        """Disable the fused kernel: every layer uses its best pure-PyTorch order."""
        for layers, kind in [(self.enc, "same"), (self.down, "down"), (self.up, "up"), (self.dec, "same")]:
            for l, m in enumerate(layers):
                op = g.ops[(kind, l)]
                m.order = "aggregate" if op.n_dst * m.in_channels <= op.n_src * m.out_channels else "transform"

    def orders(self, g: Geometry, x):
        out = {}
        for layers, kind, name in [(self.enc, "same", "enc"), (self.down, "down", "down"),
                                   (self.up, "up", "up"), (self.dec, "same", "dec")]:
            for l, m in enumerate(layers):
                out[f"{name}{l}"] = m._resolve_order(g.ops[(kind, l)], x)
        return out


def measure(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    # total peak, including the geometry and the model: what has to fit on the GPU
    return (time.perf_counter() - t0) / iters * 1e3, torch.cuda.max_memory_allocated() / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", type=int, default=7)
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    a = p.parse_args()
    dev = "cuda"
    path = glob.glob(os.path.join(HERE, "data", "airfrans_clipped", "**", "*.cgns"), recursive=True)[0]
    pos = read_cgns_tri(path)[0]
    t0 = time.perf_counter()
    levels = mesh_hierarchy(pos, a.levels, a.factor)
    g = Geometry(levels, dev)
    g.dev = dev
    print(f"hierarchy {[p.size(0) for p in levels]} built in {time.perf_counter() - t0:.1f} s, "
          f"{sum(op.num_edges for op in g.ops.values())} edges in {len(g.ops)} operators")

    torch.manual_seed(0)
    ref = PygUNet(a.levels).to(dev)
    ours = FixedMeshUNet(ref)
    spmm = FixedMeshUNet(ref)
    spmm.set_spmm_only(g)
    n_params = sum(q.numel() for q in ref.parameters())
    print(f"U-Net: 25 SplineConv layers, channels {channels(a.levels)}, {n_params / 1e6:.2f} M parameters")

    # correctness: same output as PyG
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = torch.randn(2, g.n[0], 3, device=dev)
        y_ref = ref(x.reshape(-1, 3), g, 2).view(2, g.n[0], -1)
        y = ours(x.transpose(0, 1).contiguous(), g).transpose(0, 1)
        err = ((y - y_ref).abs().max() / y_ref.abs().max()).item()
    print(f"max |ours - PyG| / max |PyG| = {err:.2e}")
    print("auto picks:", ours.orders(g, x.transpose(0, 1)))

    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"geometry + model on GPU: {torch.cuda.memory_allocated() / 2**20:.0f} MiB of {total:.0f} MiB")
    for B in a.batches:
        x = torch.randn(g.n[0], B, 3, device=dev)
        xb = x.transpose(0, 1).reshape(-1, 3).contiguous()

        def run_pyg():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref(xb, g, B).square().mean().backward()

        cases = {"pyg": run_pyg,
                 "spmm only": lambda: spmm(x, g).square().mean().backward(),
                 "auto (fused + spmm)": lambda: ours(x, g).square().mean().backward()}
        for name, fn in cases.items():
            try:
                t, m = measure(fn)
                spill = "  (> GPU memory: spills to host RAM)" if m > 0.95 * total else ""
                print(f"B={B:3d} {name:22s} {t:9.1f} ms  {m:9.0f} MiB{spill}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"B={B:3d} {name:22s}       OOM", flush=True)
            ref.zero_grad(set_to_none=True)
            ours.zero_grad(set_to_none=True)
            spmm.zero_grad(set_to_none=True)
            g._batched.clear()          # PyG's batched copies of the graphs are not kept around
            torch.cuda.empty_cache()
    print(torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
