"""Optional ahead-of-time build of the fused CUDA kernel.

    BSC_BUILD_CUDA=1 pip install -e .        # or: python setup.py build_ext --inplace

Without it, the kernel is JIT-compiled on first use (needs nvcc and a host compiler at runtime).
"""
import os

from setuptools import setup

ext_modules, cmdclass = [], {}
if os.environ.get("BSC_BUILD_CUDA", "0") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules = [CUDAExtension("batched_splineconv._fused_C", ["src/batched_splineconv/csrc/fused.cu"],
                                 extra_compile_args={"cxx": [], "nvcc": ["-O3", "-lineinfo"]})]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
