"""Regenerate the README figures:  python docs/make_figures.py [benchmark.csv]"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from batched_splineconv import SplineOperator, spline_basis
from batched_splineconv.meshes import cartesian_pseudo, grid_positions, stencil_edges

OUT = os.path.join(os.path.dirname(__file__), "figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})


def fig_kernel():
    """h_theta is a bilinear interpolation of a 5x5 grid of weights; grid offsets hit the knots."""
    torch.manual_seed(3)
    theta = torch.randn(25)
    n = 81
    u = torch.linspace(0, 1, n)
    uu = torch.stack(torch.meshgrid(u, u, indexing="xy"), -1).reshape(-1, 2)
    b, wi = spline_basis(uu, 5, True, 1)
    h = (b * theta[wi]).sum(-1).reshape(n, n).numpy()

    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    U, V = np.meshgrid(u.numpy(), u.numpy())
    ax.plot_surface(U, V, h, cmap="Blues", alpha=0.75, linewidth=0, antialiased=True)
    kx, ky = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    tz = theta.numpy().reshape(5, 5)
    for x_, y_, z_ in zip(kx.ravel(), ky.ravel(), tz.ravel()):
        ax.plot([x_, x_], [y_, y_], [0, z_], color=ORANGE, lw=1.5)
    ax.scatter(kx.ravel(), ky.ravel(), tz.ravel(), color=ORANGE, s=14, depthshade=False)
    ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$"); ax.set_zticks([])
    ax.set_title(r"$h_\theta(u)$: one weight $\theta_k$ per knot", pad=0)
    ax.view_init(32, -60)

    ax = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(wspace=0.35)
    ax.set_aspect("equal")
    for t in np.linspace(0, 1, 5):
        ax.axhline(t, color=GRID, lw=1, zorder=0); ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.scatter(kx, ky, s=70, marker="s", facecolor=SURFACE, edgecolor=INK2, lw=1, zorder=1,
               label="spline knots (one trainable weight each)")
    reg = (np.stack(np.meshgrid(np.arange(-2, 3), np.arange(-2, 3)), -1).reshape(-1, 2) / 4 + 0.5)
    ax.scatter(reg[:, 0], reg[:, 1], s=22, color=BLUE, zorder=2,
               label="regular grid: offsets land on knots\n= discrete 5x5 convolution")
    rng = np.random.default_rng(1)
    jit = np.clip(reg + rng.uniform(-0.07, 0.07, reg.shape), 0, 1)
    ax.scatter(jit[:, 0], jit[:, 1], s=22, color=ORANGE, zorder=3,
               label="unstructured mesh: each offset blends\nthe 4 surrounding weights")
    p = jit[7]
    c0 = np.floor(p * 4) / 4
    for dx in (0, .25):
        for dy in (0, .25):
            ax.plot([p[0], c0[0] + dx], [p[1], c0[1] + dy], color=ORANGE, lw=1, zorder=2)
    ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel(r"$u_1 = (x_j - x_i)_1 / 2r + 1/2$"); ax.set_ylabel("$u_2$")
    ax.set_title("Neighbour offsets of one node")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spline_kernel.png"), dpi=150, bbox_inches="tight")


def fig_batching():
    """B disjoint copies (PyG) vs one shared operator with the batch in the dense columns."""
    shape, r, B, C = (5, 5), 1, 3, 2
    pos = grid_positions(shape)
    ei = stencil_edges(shape, shape, r)
    N = pos.size(0)
    op = SplineOperator(ei, cartesian_pseudo(pos, pos, ei, r + 0.3), 2 * r + 1, num_nodes=N)
    A = op.A.to_dense().numpy() != 0

    adj = np.zeros((N, N), bool)
    adj[ei[1].numpy(), ei[0].numpy()] = True
    big = np.kron(np.eye(B, dtype=bool), adj)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), gridspec_kw=dict(width_ratios=[3, 1.1, 1.3]))
    ax = axes[0]
    ax.spy(big, markersize=1.6, color=BLUE)
    ax.set_title(f"PyG batch: one graph, B={B} components\nB·E edges, basis recomputed B times")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.spy(A, markersize=1.2, color=ORANGE, aspect="auto")
    for k in range(1, op.K):
        ax.axhline(k * N - 0.5, color=GRID, lw=0.8)
    ax.set_title(f"fixed mesh: A\n[K·N × N], built once")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    X = np.zeros((N, B * C))
    for b in range(B):
        X[:, b * C:(b + 1) * C] = b + 1
    ax.imshow(X, cmap=plt.matplotlib.colors.ListedColormap([BLUE, AQUA, YELLOW]), aspect="auto")
    for b in range(1, B):
        ax.axvline(b * C - 0.5, color=SURFACE, lw=2)
    ax.set_title("X: [N × B·C]\nbatch in the dense columns")
    ax.set_xticks([C * b + (C - 1) / 2 for b in range(B)], [f"sample {b + 1}" for b in range(B)],
                  rotation=45, ha="right")
    ax.set_yticks([])
    fig.tight_layout(w_pad=3)
    l, r_ = axes[1].get_position(), axes[2].get_position()
    fig.text((l.x1 + r_.x0) / 2, 0.5, "@", fontsize=18, ha="center", va="center", color=INK2)
    fig.savefig(os.path.join(OUT, "batching.png"), dpi=150, bbox_inches="tight")


def fig_benchmark(path):
    rows = list(csv.DictReader(open(path)))
    series = {
        "PyG SplineConv (B disjoint copies)": ("pyg", BLUE),
        "this repo, SpMM + GEMM (pure PyTorch)": ("best", ORANGE),
        "this repo, fused CUDA kernel": ("fused", AQUA),
        "cuDNN conv2d on the grid (reference)": ("conv2d", YELLOW),
    }
    if not any(r["method"] == "fused" for r in rows):
        series.pop("this repo, fused CUDA kernel")
    def get(method, key):
        if method == "best":
            a = get("aggregate", key); t = get("transform", key)
            return a[0], np.minimum(a[1], t[1])
        sel = [r for r in rows if r["method"] == method]
        return np.array([int(r["B"]) for r in sel]), np.array([float(r[key]) for r in sel])

    N, C, K = rows[0]["N"], rows[0]["C"], rows[0]["K"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in [(axes[0], "ms", "forward + backward time (ms)"),
                           (axes[1], "peak_MiB", "peak memory (MiB)")]:
        for label, (m, col) in series.items():
            x, y = get(m, key)
            ax.plot(x, y, color=col, marker="o", markersize=5, label=label)
            ax.annotate(f"{y[-1]:.0f}" if y[-1] >= 10 else f"{y[-1]:.1f}", (x[-1], y[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("batch size B"); ax.set_title(title, loc="left")
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.suptitle(f"One SplineConv layer, N={N} nodes, ~25 neighbours, C_in=C_out={C}, K={K}, fp32, "
                 f"RTX 4070 Ti SUPER", fontsize=9.5, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "benchmark.png"), dpi=150, bbox_inches="tight")


def fig_real_mesh():
    """AirfRANS mesh around the airfoil, nodes coloured by their number of neighbours."""
    import glob
    from batched_splineconv.meshes import adaptive_radius_graph, read_cgns_tri
    paths = glob.glob(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "data",
                                   "airfrans_remeshed", "**", "*.cgns"), recursive=True)
    if not paths:
        return
    pos, tri = read_cgns_tri(paths[0])
    ei, _ = adaptive_radius_graph(pos, None, 5)
    deg = torch.bincount(ei[1], minlength=pos.size(0)).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (xl, yl), title in [(axes[0], ((-1.5, 2.5), (-1.0, 1.0)), "AirfRANS mesh (8 304 nodes)"),
                                (axes[1], ((-0.05, 0.25), (-0.08, 0.1)), "leading edge")]:
        ax.triplot(pos[:, 0], pos[:, 1], tri.numpy(), color=GRID, lw=0.3)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=np.clip(deg, 0, 80), s=2, cmap="Blues", vmin=0, vmax=80)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("neighbours per node (clipped at 80)")
    fig.savefig(os.path.join(OUT, "real_mesh.png"), dpi=150, bbox_inches="tight")


def fig_real_benchmark(path):
    rows = [r for r in csv.DictReader(open(path)) if r["ordering"] == "mesher" and "->" not in r["layer"]]
    series = {"PyG SplineConv (B disjoint copies)": ("pyg", BLUE),
              "this repo, SpMM + GEMM (pure PyTorch)": ("spmm", ORANGE),
              "this repo, fused CUDA kernel": ("fused", AQUA)}
    B, C = rows[0]["B"], rows[0]["C"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in [(axes[0], "ms", "forward + backward time (ms)"),
                           (axes[1], "peak_MiB", "peak memory (MiB)")]:
        for label, (m, col) in series.items():
            sel = sorted([r for r in rows if r["method"] == m], key=lambda r: int(r["n_dst"]))
            x = np.array([int(r["n_dst"]) for r in sel]); y = np.array([float(r[key]) for r in sel])
            ax.plot(x, y, color=col, marker="o", markersize=6, ls="none", label=label)
            ax.annotate(f"{y[-1]:.0f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)
        ax.axhline(16376, color=INK2, lw=1, ls="--") if key == "peak_MiB" else None
        if key == "peak_MiB":
            ax.text(400, 16376 * 1.15, "GPU memory (16 GB)", fontsize=8.5, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("mesh nodes N"); ax.set_title(title, loc="left")
        ax.grid(True, which="major", color=GRID, lw=0.8)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.suptitle(f"One SplineConv layer on real CFD meshes (CFD-GCN, AirfRANS), adaptive radius "
                 f"n_k=5, B={B}, C={C}, fp32, RTX 4070 Ti SUPER", fontsize=9.5, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "benchmark_real.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_kernel()
    fig_batching()
    fig_real_mesh()
    for path in sys.argv[1:]:
        if "bench_real" in path:
            fig_real_benchmark(path)
        else:
            fig_benchmark(path)
