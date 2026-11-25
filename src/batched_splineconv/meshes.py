"""Synthetic meshes used by the tests and benchmarks."""
from __future__ import annotations

import itertools
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor


def grid_positions(shape: Sequence[int], spacing: float = 1.0, device=None) -> Tensor:
    """Node coordinates of a regular grid, node index = row-major (last axis fastest).

    Coordinate ``d`` of the returned positions is the index along axis ``D-1-d``, i.e.
    ``pos[:, 0]`` is the fastest-varying axis ("x"), matching the spline weight ordering.
    """
    axes = [torch.arange(n, dtype=torch.float32, device=device) * spacing for n in shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in reversed(mesh)], dim=-1)


def stencil_edges(src_shape: Sequence[int], dst_shape: Sequence[int], radius: int,
                  stride: int = 1, device=None) -> Tensor:
    """Edges of a (transposed) convolution stencil between two regular grids.

    Source node at grid index ``p`` sits at fine coordinate ``stride * p``; it is linked
    to every destination node ``q`` with ``|q - stride * p|_inf <= radius``. ``stride=1``
    and equal shapes gives the stencil of an ordinary convolution with kernel ``2r+1``.
    """
    D = len(src_shape)
    src_idx = torch.stack(torch.meshgrid(*[torch.arange(n, device=device) for n in src_shape],
                                         indexing="ij"), -1).reshape(-1, D)
    offsets = torch.tensor(list(itertools.product(range(-radius, radius + 1), repeat=D)),
                           device=device)
    q = stride * src_idx[:, None, :] + offsets[None]               # [N_src, O, D]
    dst_shape_t = torch.tensor(dst_shape, device=device)
    valid = ((q >= 0) & (q < dst_shape_t)).all(-1)
    src = torch.arange(src_idx.size(0), device=device)[:, None].expand_as(valid)[valid]
    q = q[valid]
    strides = torch.cumprod(torch.tensor([1] + list(dst_shape[::-1][:-1]), device=device), 0).flip(0)
    dst = (q * strides).sum(-1)
    return torch.stack([src, dst])


def cartesian_pseudo(pos_src: Tensor, pos_dst: Tensor, edge_index: Tensor, radius: float) -> Tensor:
    """``u = (x_src - x_dst) / (2 r) + 1/2``: offsets in ``[-r, r]^D`` mapped to ``[0, 1]^D``."""
    d = pos_src[edge_index[0]] - pos_dst[edge_index[1]]
    return (d / (2 * radius) + 0.5).clamp(0, 1)


def radius_graph(pos_src: Tensor, pos_dst: Optional[Tensor], radius: float, p: float = float("inf"),
                 chunk: int = 4096) -> Tensor:
    """All (src, dst) pairs with ``|x_src - x_dst|_p <= radius`` (brute force, chunked)."""
    if pos_dst is None:
        pos_dst = pos_src
    edges = []
    for start in range(0, pos_dst.size(0), chunk):
        d = torch.cdist(pos_dst[start:start + chunk], pos_src, p=p)
        dst, src = (d <= radius).nonzero(as_tuple=True)
        edges.append(torch.stack([src, dst + start]))
    return torch.cat(edges, 1)


def jittered_grid(shape: Sequence[int], radius: int, jitter: float = 0.3, seed: int = 0,
                  device=None) -> Tuple[Tensor, Tensor, Tensor]:
    """An unstructured-looking mesh at scale: a regular grid whose nodes are displaced at random.

    Connectivity is the grid stencil of radius ``radius`` (so ~(2r+1)^D neighbours, like a
    convolution), but the pseudo-coordinates are continuous, so every edge activates
    ``(degree+1)^D`` spline weights as on a real unstructured mesh.

    Returns ``(pos, edge_index, pseudo)``.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = grid_positions(shape)
    pos = pos + jitter * (torch.rand(pos.shape, generator=g) - 0.5)
    pos = pos.to(device)
    edge_index = stencil_edges(shape, shape, radius, device=device)
    pseudo = cartesian_pseudo(pos, pos, edge_index, radius + jitter)
    return pos, edge_index, pseudo


# ---------------------------------------------------------------------------------------
# Real meshes
# ---------------------------------------------------------------------------------------

def read_su2(path: str) -> Tuple[Tensor, Tensor]:
    """Nodes and triangles of a 2D SU2 mesh (quads are split in two triangles).

    Returns ``(pos [N, 2] float32, triangles [T, 3] long)``.
    """
    import numpy as np

    with open(path) as f:
        lines = [l.split("%")[0].strip() for l in f]
    it = iter(lines)
    pos, tris = None, []
    for line in it:
        if line.startswith("NELEM="):
            n = int(line.split("=")[1])
            for _ in range(n):
                v = [int(t) for t in next(it).split()]
                if v[0] == 5:                       # triangle
                    tris.append(v[1:4])
                elif v[0] == 9:                     # quadrilateral
                    tris += [[v[1], v[2], v[3]], [v[1], v[3], v[4]]]
        elif line.startswith("NPOIN="):
            n = int(line.split("=")[1].split()[0])
            pos = np.array([[float(t) for t in next(it).split()[:2]] for _ in range(n)])
    return torch.tensor(pos, dtype=torch.float32), torch.tensor(tris, dtype=torch.long)


def read_cgns_tri(path: str) -> Tuple[Tensor, Tensor]:
    """Nodes and triangles of a 2D unstructured CGNS/HDF5 zone (e.g. AirfRANS). Needs h5py."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        zone = f["Base_2_2/Zone"]
        x = zone["GridCoordinates/CoordinateX/ data"][()]
        y = zone["GridCoordinates/CoordinateY/ data"][()]
        conn = zone["Elements_TRI_3/ElementConnectivity/ data"][()]
    pos = torch.tensor(np.stack([x, y], -1), dtype=torch.float32)
    return pos, torch.tensor(conn.reshape(-1, 3) - 1, dtype=torch.long)  # CGNS is 1-based


def local_spacing(pos_src: Tensor, pos_dst: Optional[Tensor] = None, k: int = 2) -> Tensor:
    """Characteristic length ``h_i`` at each destination node: mean distance to its ``k``
    nearest source nodes (excluding itself when both meshes are the same). Needs scipy."""
    from scipy.spatial import cKDTree

    same = pos_dst is None
    pos_dst = pos_src if same else pos_dst
    tree = cKDTree(pos_src.cpu().numpy())
    d, _ = tree.query(pos_dst.cpu().numpy(), k=k + int(same))
    d = d[:, 1:] if same else d
    return torch.tensor(d.mean(1), dtype=pos_src.dtype)


def adaptive_radius_graph(pos_src: Tensor, pos_dst: Optional[Tensor] = None, kernel_size: int = 5,
                          spacing: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Connectivity of a continuous-kernel layer replacing a discrete kernel of size ``n_k``.

    ``j -> i`` iff ``|x_j - x_i|_2 <= R_i``, with ``R_i = (sqrt(2)/2) * n_k * h_i`` and
    ``h_i`` the local mesh spacing (between two meshes: the coarser of the two near ``x_i``).
    The radius follows the mesh refinement, so every node
    sees a comparable number of neighbours, as a discrete kernel would on a grid.
    Pseudo-coordinates are ``(x_j - x_i) / (2 R_i) + 1/2``. Needs scipy.

    Returns ``(edge_index [2, E], pseudo [E, D])``.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    same = pos_dst is None
    pos_dst = pos_src if same else pos_dst
    if spacing is None:
        if same:
            spacing = local_spacing(pos_src)
        else:  # between two meshes the coarser one sets the length scale (up- and down-sampling)
            spacing = torch.maximum(local_spacing(pos_src, pos_dst), local_spacing(pos_dst))
    R =(2 ** 0.5 / 2) * kernel_size * spacing.cpu().numpy()
    tree = cKDTree(pos_src.cpu().numpy())
    nbrs = tree.query_ball_point(pos_dst.cpu().numpy(), R)
    dst = np.repeat(np.arange(len(nbrs)), [len(n) for n in nbrs])
    src = np.concatenate([np.asarray(n, dtype=np.int64) for n in nbrs])
    ei = torch.from_numpy(np.stack([src, dst])).long()
    d = pos_src[ei[0]] - pos_dst[ei[1]]
    r = torch.from_numpy(R).to(pos_src.dtype)[ei[1], None]
    return ei, (d / (2 * r) + 0.5).clamp(0, 1)


def rcm_permutation(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Reverse Cuthill-McKee node ordering (improves memory locality of the gathers)."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    e = edge_index.cpu().numpy()
    a = sp.coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(num_nodes, num_nodes)).tocsr()
    return torch.from_numpy(reverse_cuthill_mckee(a + a.T, symmetric_mode=True).copy()).long()


def poisson_decimate(pos: Tensor, c: float, spacing: Optional[Tensor] = None, seed: int = 0) -> Tensor:
    """Density-preserving decimation: greedy Poisson-disk sampling with a variable radius.

    Nodes are visited in random order; a kept node ``i`` removes every node within
    ``c * h_i`` (``h_i`` = local spacing). The kept set follows the original refinement
    (fine near walls, coarse far away) with a spacing about ``c`` times larger.
    Returns the sorted indices of the kept nodes. Needs scipy.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    h = (local_spacing(pos) if spacing is None else spacing).cpu().numpy()
    p = pos.cpu().numpy()
    nbrs = cKDTree(p).query_ball_point(p, c * h)
    blocked = np.zeros(len(p), dtype=bool)
    keep = []
    for i in np.random.default_rng(seed).permutation(len(p)):
        if not blocked[i]:
            keep.append(i)
            blocked[nbrs[i]] = True
    return torch.from_numpy(np.sort(np.asarray(keep))).long()


def mesh_hierarchy(pos: Tensor, levels: int, factor: float = 3.0, seed: int = 0, tol: float = 0.05):
    """Sequence of ``levels`` node sets, each ~``factor`` times smaller than the previous one.

    Level 0 is ``pos`` itself; each next level is a :func:`poisson_decimate` of the previous
    one, with ``c`` found by bisection to hit the target size. Returns a list of positions.
    """
    out = [pos]
    for _ in range(levels - 1):
        cur = out[-1]
        h = local_spacing(cur)
        target = cur.size(0) / factor
        lo, hi = 0.5, 8.0
        best = None
        for _ in range(20):
            c = 0.5 * (lo + hi)
            idx = poisson_decimate(cur, c, h, seed)
            best = idx
            if abs(idx.numel() - target) <= tol * target:
                break
            if idx.numel() > target:
                lo = c
            else:
                hi = c
        out.append(cur[best])
    return out
