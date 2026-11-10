"""Input checks shared by complex64 CUDA pointwise operators."""

from __future__ import annotations

from numbers import Integral

import torch


def check_complex64_cuda(field: torch.Tensor, label: str) -> None:
    if not isinstance(field, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if field.device.type != "cuda":
        raise ValueError(f"{label} must reside on CUDA")
    if field.dtype != torch.complex64:
        raise TypeError(f"{label} must have dtype torch.complex64")
    if field.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional field")
    if not field.is_contiguous():
        raise ValueError(f"{label} must be contiguous")


def check_num_warps(num_warps: int) -> int:
    """Validate and normalize the Triton launch's warp count."""
    if not isinstance(num_warps, Integral) or isinstance(num_warps, bool) or num_warps < 1:
        raise ValueError("num_warps must be a positive integer")
    return int(num_warps)
