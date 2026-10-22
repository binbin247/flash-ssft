"""Fused temporal Kerr phase rotation."""

from __future__ import annotations


import torch
import triton
import triton.language as tl

from ._validation import check_complex64_cuda


@triton.jit
def _kerr_kernel(angular_ptr, h, N: tl.constexpr, BLOCK: tl.constexpr):
    # angular_ptr: GPU float32 pointer to interleaved complex64 data [re0, im0, re1, im1, ...].
    # h: time step for the Kerr phase rotation.
    # N: total number of complex elements, not the number of float32 components.
    # BLOCK: complex elements handled by each Triton program (currently 256).
    # N and BLOCK are compile-time constants because they use tl.constexpr.

    # Compute this program's complex-element offsets and mask out-of-bounds lanes.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load real and imaginary components; masked lanes receive zero.
    re = tl.load(angular_ptr + 2 * offsets, mask, 0)
    im = tl.load(angular_ptr + 2 * offsets + 1, mask, 0)

    # Compute the intensity-dependent phase: h * |psi|^2.
    phase = h * (re * re + im * im)
    c = tl.cos(phase)
    s = tl.sin(phase)

    # Apply psi *= exp(i * phase) and write both components back in place.
    tl.store(angular_ptr + 2 * offsets, re * c - im * s, mask)
    tl.store(angular_ptr + 2 * offsets + 1, re * s + im * c, mask)


def kerr_(angular: torch.Tensor, h: float) -> torch.Tensor:
    """Fuse intensity, phase rotation and complex multiplication, in place."""
    check_complex64_cuda(angular, "angular")
    N = angular.numel()
    _kerr_kernel[(triton.cdiv(N, 256),)](
        angular_ptr=angular.view(torch.float32),
        h=float(h),
        N=N,
        BLOCK=256,
    )
    return angular
