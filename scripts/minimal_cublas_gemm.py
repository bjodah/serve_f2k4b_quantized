#!/usr/bin/env python3
"""Minimal cuBLAS GEMM diagnostic script.

Verifies that PyTorch can perform a basic matrix multiplication on CUDA using
cuBLAS.  Run this before launching the server to confirm your GPU environment
is configured correctly.

Usage::

    python scripts/minimal_cublas_gemm.py
"""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed.", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available — cuBLAS GEMM test skipped.")
        print("  If you expected CUDA support, check your driver and PyTorch installation.")
        return

    device = torch.device("cuda")
    dtype = torch.float16

    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch version: {torch.__version__}")

    # Warm-up + GEMM
    a = torch.randn(512, 512, device=device, dtype=dtype)
    b = torch.randn(512, 512, device=device, dtype=dtype)

    # Ensure cuBLAS is initialised
    _ = torch.mm(a, b)
    torch.cuda.synchronize()

    c = torch.mm(a, b)
    torch.cuda.synchronize()

    print(f"GEMM result shape: {c.shape}  (dtype={c.dtype})")
    print("cuBLAS GEMM OK ✓")


if __name__ == "__main__":
    main()
