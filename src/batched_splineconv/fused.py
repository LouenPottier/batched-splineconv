"""Fused CUDA path: the intermediate Z = A X is never written to global memory.

Build it once ahead of time with ``BSC_BUILD_CUDA=1 pip install -e .`` (or
``python setup.py build_ext --inplace``). Otherwise it is JIT-compiled on first use with
``torch.utils.cpp_extension.load``, which needs ``nvcc`` and a host compiler at runtime
(on Windows: a shell where ``cl.exe`` is on PATH, e.g. after ``vcvars64.bat``).
"""
from __future__ import annotations

import os

import torch
from torch import Tensor

_ext = None


def _load():
    global _ext
    if _ext is None:
        try:  # ahead-of-time build (BSC_BUILD_CUDA=1 pip install -e .)
            from . import _fused_C
            _ext = _fused_C
            return _ext
        except ImportError:
            pass
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "csrc", "fused.cu")
        _ext = load(name="batched_splineconv_fused", sources=[src],
                    extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _ext


_available = None


def is_available() -> bool:
    """True if the extension is importable (prebuilt) or can be JIT-compiled. Cached."""
    global _available
    if _available is None:
        try:
            _available = torch.cuda.is_available() and _load() is not None
        except Exception:
            _available = False
    return _available


class _FusedSplineConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, op) -> Tensor:
        ext = _load()
        x = x.contiguous()
        w = weight.contiguous()
        out = ext.gather_gemm(x, w, op.f_ptr, op.f_col, op.f_val, op.n_dst)
        ctx.save_for_backward(x, w)
        ctx.op = op
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        ext = _load()
        x, w = ctx.saved_tensors
        op = ctx.op
        g = grad_out.contiguous()
        gx = gw = None
        if ctx.needs_input_grad[0]:
            gx = ext.gather_gemm(g, w.transpose(1, 2).contiguous(),
                                 op.ft_ptr, op.ft_col, op.ft_val, op.n_src)
        if ctx.needs_input_grad[1]:
            gw = ext.wgrad(x, g, op.f_ptr, op.f_col, op.f_val, op.K)
        return gx, gw, None


def fused_spline_conv(x: Tensor, weight: Tensor, op) -> Tensor:
    """``x: [N_src, B, C_in]``, ``weight: [K, C_in, C_out]`` -> ``[N_dst, B, C_out]`` (fp32, CUDA)."""
    if x.dtype != torch.float32:
        raise TypeError("the fused kernel supports float32 only")
    return _FusedSplineConv.apply(x, weight, op)
