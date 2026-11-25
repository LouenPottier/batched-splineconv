"""Batched SplineConv layer for inputs that all live on the same mesh."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn

from .operator import SplineOperator, spmm
from .fused import fused_spline_conv
from .fused import is_available as fused_available


class FixedMeshSplineConv(nn.Module):
    """Drop-in equivalent of ``torch_geometric.nn.SplineConv`` for a batch of fields on one mesh.

    Input ``x`` has shape ``[B, N_src, C_in]`` (``layout="BNC"``) or ``[N_src, B, C_in]``
    (``layout="NBC"``, which avoids two transposes when layers are stacked). The mesh
    geometry is a :class:`SplineOperator`, passed at call time so that several layers can
    share it.

    ``order`` selects the contraction order:

    * ``"aggregate"``: ``Z = A @ X`` (one SpMM, batch and channels in the dense columns),
      then a single dense GEMM ``[N*B, K*C_in] @ [K*C_in, C_out]``. Intermediate: ``K*N_dst*B*C_in``.
    * ``"transform"``: ``Y = X @ [W_1 .. W_K]`` (one GEMM), then one SpMM with ``A'``.
      Intermediate: ``K*N_src*B*C_out``.
    * ``"fused"``: CUDA kernel that builds tiles of Z in shared memory and accumulates
      ``Z_k @ W_k`` in registers; no intermediate in global memory (fp32, CUDA only).
    * ``"auto"``: the SpMM order with the smaller intermediate when that intermediate is
      small (resampling between meshes of different sizes), else ``"fused"`` when available
      (CUDA, fp32), else the SpMM order with the smaller intermediate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        kernel_size: Union[int, Sequence[int]],
        root_weight: bool = True,
        bias: bool = True,
        order: str = "auto",
    ):
        super().__init__()
        ks = torch.as_tensor(kernel_size, dtype=torch.long)
        if ks.dim() == 0:
            ks = ks.repeat(dim)
        self.in_channels, self.out_channels, self.dim = in_channels, out_channels, dim
        self.kernel_size = tuple(ks.tolist())
        self.K = int(ks.prod())
        if order not in ("auto", "aggregate", "transform", "fused"):
            raise ValueError(order)
        self.order = order

        self.weight = nn.Parameter(torch.empty(self.K, in_channels, out_channels))
        self.root = nn.Parameter(torch.empty(in_channels, out_channels)) if root_weight else None
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # same initialisation as PyG
        bound = 1.0 / math.sqrt(self.K * self.in_channels)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.root is not None:
            bound = 1.0 / math.sqrt(self.in_channels)
            nn.init.uniform_(self.root, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_pyg(cls, conv, order: str = "auto") -> "FixedMeshSplineConv":
        """Copy the parameters of a ``torch_geometric.nn.SplineConv``."""
        in_c = conv.in_channels if isinstance(conv.in_channels, int) else conv.in_channels[0]
        new = cls(in_c, conv.out_channels, conv.dim, conv.kernel_size.tolist(),
                  root_weight=conv.root_weight, bias=conv.bias is not None, order=order)
        new = new.to(conv.weight.device, conv.weight.dtype)  # before copying: no fp32 round-trip
        with torch.no_grad():
            new.weight.copy_(conv.weight)
            if conv.root_weight:
                new.root.copy_(conv.lin.weight.t())
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        return new

    # SpMM path when its intermediate is at most this many times the layer's input + output.
    # Measured on real meshes: resampling between meshes of different sizes (ratio ~1-5) is
    # faster with the "transform"/"aggregate" orders, same-mesh layers (ratio ~K/2) with "fused".
    AUTO_SPMM_RATIO = 8.0

    def _resolve_order(self, op: SplineOperator, x: Tensor) -> str:
        if self.order != "auto":
            return self.order
        agg = self.K * op.n_dst * self.in_channels      # aggregate-first intermediate (per sample)
        trans = self.K * op.n_src * self.out_channels   # transform-first intermediate (per sample)
        best = "aggregate" if agg <= trans else "transform"
        io = op.n_src * self.in_channels + op.n_dst * self.out_channels
        if min(agg, trans) <= self.AUTO_SPMM_RATIO * io:
            return best
        if x.is_cuda and x.dtype == torch.float32 and fused_available():
            return "fused"
        return best

    def forward(self, x: Tensor, op: SplineOperator, x_dst: Optional[Tensor] = None,
                layout: str = "BNC") -> Tensor:
        if op.K != self.K:
            raise ValueError(f"operator has K={op.K}, layer has K={self.K}")
        if layout == "BNC":
            x = x.transpose(0, 1)
            if x_dst is not None:
                x_dst = x_dst.transpose(0, 1)
        elif layout != "NBC":
            raise ValueError(layout)
        x = x.contiguous()  # [N_src, B, C_in], a node's batch of features is contiguous
        n_src, B, c_in = x.shape
        K, c_out, n_dst = self.K, self.out_channels, op.n_dst

        order = self._resolve_order(op, x)
        if order == "fused":
            out = fused_spline_conv(x, self.weight, op)
        elif order == "aggregate":
            z = spmm(op.A, op.At, x.view(n_src, B * c_in))                      # [K*N_dst, B*C_in]
            z = z.view(K, n_dst, B, c_in).permute(1, 2, 0, 3).reshape(n_dst * B, K * c_in)
            out = (z @ self.weight.reshape(K * c_in, c_out)).view(n_dst, B, c_out)
        else:
            w = self.weight.permute(1, 0, 2).reshape(c_in, K * c_out)
            y = (x.view(n_src * B, c_in) @ w).view(n_src, B, K, c_out)
            y = y.permute(2, 0, 1, 3).reshape(K * n_src, B * c_out)               # [K*N_src, B*C_out]
            out = spmm(op.Ap, op.Apt, y).view(n_dst, B, c_out)

        if self.root is not None:
            if x_dst is None:
                if n_src != n_dst:
                    raise ValueError("bipartite operator: pass x_dst to use the root weight")
                x_dst = x
            out = out + x_dst @ self.root
        if self.bias is not None:
            out = out + self.bias
        return out.transpose(0, 1) if layout == "BNC" else out

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, dim={self.dim}, "
                f"kernel_size={self.kernel_size}, order={self.order!r}")
