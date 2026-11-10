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
    return _check_power_of_two(num_warps, "num_warps")


def check_block(block: int) -> int:
    """Validate and normalize a Triton block size."""
    return _check_power_of_two(block, "block")


def _check_power_of_two(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive power of two")
    value = int(value)
    if value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return value
