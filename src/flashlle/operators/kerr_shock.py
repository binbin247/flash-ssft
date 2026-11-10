"""Exact Kerr rotation followed by an Euler self-steepening step (gamma0 = 1)."""

from __future__ import annotations

from math import isfinite

import torch
import triton
import triton.language as tl

from ._validation import check_block, check_complex64_cuda, check_num_warps


@triton.jit
def _kerr_cubic_kernel(
    angular_ptr,          # Input A; overwritten with B, as interleaved float32.
    cubic_ptr,            # Output |B|²B; unused when WRITE_CUBIC is false.
    h,                   # Propagation step for the Kerr phase rotation.
    N: tl.constexpr,      # Total number of complex samples.
    BLOCK: tl.constexpr,  # Complex samples handled by each Triton program.
    WRITE_CUBIC: tl.constexpr,  # Generate the cubic term only for self-steepening.
):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(start=0, end=BLOCK)
    mask = offsets < N
    re = tl.load(pointer=angular_ptr + 2 * offsets, mask=mask, other=0)
    im = tl.load(pointer=angular_ptr + 2 * offsets + 1, mask=mask, other=0)
    phase = h * (re * re + im * im)
    c = tl.cos(phase)
    s = tl.sin(phase)
    b_re = re * c - im * s
    b_im = re * s + im * c
    tl.store(pointer=angular_ptr + 2 * offsets, value=b_re, mask=mask)
    tl.store(pointer=angular_ptr + 2 * offsets + 1, value=b_im, mask=mask)
    if WRITE_CUBIC:
        # Form |B|²B from the rotated field without reading it back from memory.
        intensity_b = b_re * b_re + b_im * b_im
        tl.store(pointer=cubic_ptr + 2 * offsets, value=intensity_b * b_re, mask=mask)
        tl.store(pointer=cubic_ptr + 2 * offsets + 1, value=intensity_b * b_im, mask=mask)


@triton.jit
def _shock_derivative_kernel(
    spectrum_ptr,         # FFT of |B|²B, as interleaved float32; updated in place.
    w_ptr,               # Unshifted angular frequencies, stored as float32/float64.
    omega0: tl.constexpr, # Carrier angular frequency, in the units of w.
    N: tl.constexpr,      # Total number of complex frequency bins.
    BLOCK: tl.constexpr,  # Complex frequency bins handled by each Triton program.
):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(start=0, end=BLOCK)
    mask = offsets < N
    re = tl.load(pointer=spectrum_ptr + 2 * offsets, mask=mask, other=0)
    im = tl.load(pointer=spectrum_ptr + 2 * offsets + 1, mask=mask, other=0)
    # Compute w/omega0 in float32, matching the complex64 spectrum.
    ratio = tl.load(pointer=w_ptr + offsets, mask=mask, other=0).to(dtype=tl.float32) / omega0
    # Multiply by i*w/omega0, retaining the complex FFT's Nyquist frequency.
    tl.store(pointer=spectrum_ptr + 2 * offsets, value=-ratio * im, mask=mask)
    tl.store(pointer=spectrum_ptr + 2 * offsets + 1, value=ratio * re, mask=mask)


@triton.jit
def _shock_update_kernel(
    angular_ptr,          # Field B, as interleaved float32; updated in place.
    derivative_ptr,       # (1/omega0)*d(|B|²B)/dT, in the same interleaved layout.
    h,                    # Propagation step along z.
    N: tl.constexpr,      # Total number of complex samples.
    BLOCK: tl.constexpr,  # Complex samples handled by each Triton program.
):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(start=0, end=BLOCK)
    mask = offsets < N
    re = tl.load(pointer=angular_ptr + 2 * offsets, mask=mask, other=0)
    im = tl.load(pointer=angular_ptr + 2 * offsets + 1, mask=mask, other=0)
    dre = tl.load(pointer=derivative_ptr + 2 * offsets, mask=mask, other=0)
    dim = tl.load(pointer=derivative_ptr + 2 * offsets + 1, mask=mask, other=0)
    tl.store(pointer=angular_ptr + 2 * offsets, value=re - h * dre, mask=mask)
    tl.store(pointer=angular_ptr + 2 * offsets + 1, value=im - h * dim, mask=mask)


def kerr_shock_(
    angular: torch.Tensor,  # Time-domain field; updated in place.
    w: torch.Tensor,        # Unshifted angular frequencies: 2*pi*fftfreq(N, dT).
    h: float,               # Propagation step along z.
    omega0: float,          # Positive carrier angular frequency, in the units of w.
    *,
    block: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:          # Returns the original angular tensor.
    """Apply B = A*exp(i*h*|A|²), then A = B - h/omega0 * dT(|B|²B).

    Uses a periodic spectral derivative and gamma0 = 1. This first-order split
    step retains Euler error and stability limits; check step-size convergence.
    angular is contiguous 1D complex64 CUDA; w is matching float32/float64.
    Kerr rotation and |B|²B generation share one kernel. Positive infinite omega0
    runs only the rotation. Warm up before CUDA Graph capture; omega0 specializes
    the derivative kernel.
    """
    check_complex64_cuda(field=angular, label="angular")
    N = angular.numel()
    if N == 0:
        raise ValueError("angular must not be empty")
    if not isinstance(w, torch.Tensor) or w.dtype not in (torch.float32, torch.float64):
        raise TypeError("w must be a float32 or float64 tensor")
    if w.shape != angular.shape or w.device != angular.device:
        raise ValueError("angular and w must have matching shape and device")
    if not w.is_contiguous():
        raise ValueError("w must be contiguous")
    h, omega0 = float(h), float(omega0)
    if not isfinite(h):
        raise ValueError("h must be finite")
    if not omega0 > 0:
        raise ValueError("omega0 must be positive")
    block = check_block(block)
    num_warps = check_num_warps(num_warps)
    if h == 0:
        return angular
    grid = (triton.cdiv(N, block),)
    shock_enabled = omega0 != float("inf")
    cubic = torch.empty_like(input=angular) if shock_enabled else None
    _kerr_cubic_kernel[grid](
        angular_ptr=angular.view(dtype=torch.float32),
        cubic_ptr=cubic.view(dtype=torch.float32) if shock_enabled else None,
        h=h,
        N=N,
        BLOCK=block,
        WRITE_CUBIC=shock_enabled,
        num_warps=num_warps,
    )
    if not shock_enabled:
        return angular

    spectrum = torch.fft.fft(cubic)
    _shock_derivative_kernel[grid](
        spectrum_ptr=spectrum.view(dtype=torch.float32),
        w_ptr=w,
        omega0=omega0,
        N=N,
        BLOCK=block,
        num_warps=num_warps,
    )
    derivative = torch.fft.ifft(spectrum)
    _shock_update_kernel[grid](
        angular_ptr=angular.view(dtype=torch.float32),
        derivative_ptr=derivative.view(dtype=torch.float32),
        h=h,
        N=N,
        BLOCK=block,
        num_warps=num_warps,
    )
    return angular
