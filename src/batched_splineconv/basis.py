"""Pure-PyTorch B-spline basis, numerically identical to ``torch_spline_conv.spline_basis``.

For an edge with pseudo-coordinates ``u in [0, 1]^D`` the continuous kernel is

    h_theta(u) = sum_s  basis[s](u) * W[weight_index[s](u)]

where the sum runs over the ``S = (degree + 1)^D`` tensor-product B-splines whose
support contains ``u``. On a fixed mesh ``u`` never changes, so this function is
called once, when the operator is built, instead of at every forward pass.
"""
from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import Tensor


def _basis_1d(v: Tensor, k: Tensor, degree: int) -> Tensor:
    """Value of the k-th (k in [0, degree]) uniform B-spline piece at local coordinate v in [0, 1)."""
    if degree == 1:
        return torch.where(k == 0, 1 - v, v)
    if degree == 2:
        b0 = 0.5 * v * v - v + 0.5
        b1 = -v * v + v + 0.5
        b2 = 0.5 * v * v
        return torch.where(k == 0, b0, torch.where(k == 1, b1, b2))
    if degree == 3:
        b0 = (1 - v) ** 3 / 6.0
        b1 = (3 * v**3 - 6 * v**2 + 4) / 6.0
        b2 = (-3 * v**3 + 3 * v**2 + 3 * v + 1) / 6.0
        b3 = v**3 / 6.0
        return torch.where(k == 0, b0, torch.where(k == 1, b1, torch.where(k == 2, b2, b3)))
    raise ValueError(f"degree must be 1, 2 or 3, got {degree}")


def _as_tensor(value, dim: int, dtype, device) -> Tensor:
    t = torch.as_tensor(value, dtype=dtype, device=device)
    if t.dim() == 0:
        t = t.repeat(dim)
    return t


def spline_basis(
    pseudo: Tensor,
    kernel_size: Union[int, Sequence[int], Tensor],
    is_open_spline: Union[bool, Sequence[bool], Tensor] = True,
    degree: int = 1,
) -> tuple[Tensor, Tensor]:
    """Return ``(basis [E, S], weight_index [E, S])`` with ``S = (degree + 1) ** D``."""
    if pseudo.dim() == 1:
        pseudo = pseudo.unsqueeze(-1)
    E, D = pseudo.shape
    device = pseudo.device
    ks = _as_tensor(kernel_size, D, torch.long, device)
    is_open = _as_tensor(is_open_spline, D, torch.long, device)

    S = (degree + 1) ** D
    s = torch.arange(S, device=device)
    pow_ = (degree + 1) ** torch.arange(D, device=device)
    k_mod = (s[:, None] // pow_[None, :]) % (degree + 1)  # [S, D]

    v = pseudo * (ks - degree * is_open).to(pseudo.dtype)  # [E, D]
    v_floor = v.floor()
    frac = v - v_floor

    offset = torch.cumprod(torch.cat([ks.new_ones(1), ks[:-1]]), 0)  # [D]
    wi = (v_floor.long()[:, None, :] + k_mod[None]) % ks  # [E, S, D]
    weight_index = (wi * offset).sum(-1)

    basis = _basis_1d(frac[:, None, :], k_mod[None], degree).prod(-1)  # [E, S]
    return basis, weight_index
