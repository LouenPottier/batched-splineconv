"""The fixed-mesh spline operator.

A SplineConv layer computes, for every destination node i,

    out_i = c_i * sum_{j in N(i)} sum_s B_s(u_ij) x_j W_{k_s(u_ij)}  (+ x_i W_root + b)

with ``c_i = 1 / |N(i)|`` for mean aggregation, 1 for add. When the mesh is fixed,
``B_s``, ``k_s`` and ``c_i`` are constants. Collecting them gives a sparse tensor

    A[(k, i), j] = c_i * sum_{s : k_s(u_ij) = k} B_s(u_ij)       shape [K * N_dst, N_src]

so that ``out = sum_k A_k X W_k``: one sparse operator shared by every sample of the
batch and every channel. This module computes its entries once per mesh, and builds the
storage layouts that the forward and backward passes need on first use.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from .basis import spline_basis


class _SpMM(torch.autograd.Function):
    """``A @ X`` with a precomputed ``A^T`` for a deterministic, atomic-free backward."""

    @staticmethod
    def forward(ctx, A: Tensor, At: Tensor, X: Tensor) -> Tensor:
        ctx.At = At
        return A @ X

    @staticmethod
    def backward(ctx, grad: Tensor):
        return None, None, ctx.At @ grad.contiguous()


def spmm(A: Tensor, At: Tensor, X: Tensor) -> Tensor:
    return _SpMM.apply(A, At, X.contiguous())


def _coalesce(k, i, j, v, n_dst, n_src):
    """Sum duplicate (k, i, j) entries. numpy on purpose: torch 2.4.0's CPU integer sort is
    wrong for >= 32768 elements on some CPUs (fixed in 2.4.1), which would corrupt the operator."""
    key = (k * n_dst + i) * n_src + j
    uk, inv = np.unique(key, return_inverse=True)
    vals = np.bincount(inv.reshape(-1), weights=v, minlength=uk.size)
    j = uk % n_src
    t = uk // n_src
    return (t // n_dst).astype(np.int32), (t % n_dst).astype(np.int32), j.astype(np.int32), vals


def _csr_arrays(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_rows: int, n_cols: int):
    order = np.argsort(rows.astype(np.int64) * n_cols + cols, kind="stable")
    crow = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n_rows), out=crow[1:])
    if crow[-1] >= 2**31 or n_cols >= 2**31:
        raise ValueError("operator too large for int32 indices")
    return crow.astype(np.int32), cols[order].astype(np.int32), vals[order]


class SplineOperator(nn.Module):
    """Geometry of one (possibly bipartite) mesh-to-mesh SplineConv, built once.

    The basis, weight indices and aggregation weights are evaluated in the constructor; the
    sparse layouts (``A``/``At`` for the aggregate-first order, ``Ap``/``Apt`` for the
    transform-first order, ``f_*``/``ft_*`` for the fused kernel) are built on first use,
    on the device of the module, so a layer only pays for the layouts it actually uses.

    Args:
        edge_index: ``[2, E]`` (source, destination) pairs, PyG convention.
        pseudo: ``[E, D]`` pseudo-coordinates in ``[0, 1]^D``.
        kernel_size, is_open_spline, degree: as in ``torch_geometric.nn.SplineConv``.
        num_nodes: ``N`` or ``(N_src, N_dst)``. Inferred from ``edge_index`` if omitted.
        aggr: ``"mean"`` (PyG default) or ``"add"``.
    """

    def __init__(
        self,
        edge_index: Tensor,
        pseudo: Tensor,
        kernel_size: Union[int, Sequence[int]],
        is_open_spline: Union[bool, Sequence[bool]] = True,
        degree: int = 1,
        num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
        aggr: str = "mean",
    ):
        super().__init__()
        if pseudo.dim() == 1:
            pseudo = pseudo.unsqueeze(-1)
        src, dst = edge_index[0].long(), edge_index[1].long()
        if num_nodes is None:
            num_nodes = int(edge_index.max()) + 1
        if isinstance(num_nodes, int):
            num_nodes = (num_nodes, num_nodes)
        n_src, n_dst = num_nodes
        D = pseudo.size(1)
        ks = torch.as_tensor(kernel_size, dtype=torch.long)
        if ks.dim() == 0:
            ks = ks.repeat(D)
        K = int(ks.prod())

        basis, wi = spline_basis(pseudo, ks.to(pseudo.device), is_open_spline, degree)
        if aggr == "mean":
            deg = torch.bincount(dst, minlength=n_dst).clamp(min=1).to(basis.dtype)
            basis = basis / deg[dst, None]
        elif aggr != "add":
            raise ValueError(f"aggr must be 'mean' or 'add', got {aggr!r}")

        S = basis.size(1)
        np_ = lambda t: t.detach().cpu().numpy()
        self._entries = _coalesce(np_(wi.reshape(-1)).astype(np.int64),
                                  np_(dst[:, None].expand(-1, S).reshape(-1)),
                                  np_(src[:, None].expand(-1, S).reshape(-1)),
                                  np_(basis.reshape(-1)).astype(np.float64), n_dst, n_src)
        # follows .to(device / dtype); the layouts are built on its device, in its dtype
        self.register_buffer("_anchor", torch.empty(0, dtype=basis.dtype, device=pseudo.device),
                             persistent=False)
        self._cache = {}

        self.n_src, self.n_dst, self.K, self.dim = n_src, n_dst, K, D
        self.kernel_size = tuple(ks.tolist())
        self.degree, self.aggr = degree, aggr
        self.num_edges = src.numel()
        self.nnz = int(self._entries[3].size)

    # ---------------------------------------------------------------------------------
    def _apply(self, *args, **kwargs):
        self._cache = {}  # device / dtype change: rebuild layouts lazily
        return super()._apply(*args, **kwargs)

    def _layout(self, name: str):
        if name in self._cache:
            return self._cache[name]
        k, i, j, v = self._entries
        K, nd, ns = self.K, self.n_dst, self.n_src
        k64, i64, j64 = k.astype(np.int64), i.astype(np.int64), j.astype(np.int64)
        spec = {
            "A": (k64 * nd + i64, j, K * nd, ns),     # aggregate-first, rows (k, i)
            "At": (j64, k64 * nd + i64, ns, K * nd),
            "Ap": (i64, k64 * ns + j64, nd, K * ns),  # transform-first, cols (k, j)
            "Apt": (k64 * ns + j64, i, K * ns, nd),
            "f": (i64 * K + k64, j, nd * K, ns),      # fused kernel, rows (i, k) node-major
            "ft": (j64 * K + k64, i, ns * K, nd),     # its transpose, rows (j, k)
        }[name]
        crow, col, val = _csr_arrays(*spec[:2], v, *spec[2:])
        dev = self._anchor.device
        crow, col = torch.from_numpy(crow).to(dev), torch.from_numpy(col).to(dev)
        if name in ("f", "ft"):  # raw int32 CSR arrays, float32 values, for the CUDA kernel
            out = (crow, col, torch.from_numpy(val.astype(np.float32)).to(dev))
        else:
            out = torch.sparse_csr_tensor(crow, col, torch.from_numpy(val).to(dev, self._anchor.dtype),
                                          (spec[2], spec[3]))
        self._cache[name] = out
        return out

    A = property(lambda self: self._layout("A"))
    At = property(lambda self: self._layout("At"))
    Ap = property(lambda self: self._layout("Ap"))
    Apt = property(lambda self: self._layout("Apt"))
    f_ptr = property(lambda self: self._layout("f")[0])
    f_col = property(lambda self: self._layout("f")[1])
    f_val = property(lambda self: self._layout("f")[2])
    ft_ptr = property(lambda self: self._layout("ft")[0])
    ft_col = property(lambda self: self._layout("ft")[1])
    ft_val = property(lambda self: self._layout("ft")[2])

    def extra_repr(self) -> str:
        return (f"n_src={self.n_src}, n_dst={self.n_dst}, edges={self.num_edges}, "
                f"kernel_size={self.kernel_size}, degree={self.degree}, aggr={self.aggr!r}, "
                f"nnz={self.nnz}, built={sorted(self._cache)}")
