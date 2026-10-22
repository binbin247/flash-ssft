"""Kerr–Raman phase rotation with a periodic spectral intensity derivative."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._validation import check_complex64_cuda
from .kerr import kerr_


@triton.jit
def _intensity_kernel(angular_ptr, intensity_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    re = tl.load(angular_ptr + 2 * offsets, mask, 0)
    im = tl.load(angular_ptr + 2 * offsets + 1, mask, 0)
    tl.store(intensity_ptr + offsets, re * re + im * im, mask)


@triton.jit
def _spectral_derivative_kernel(spectrum_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N // 2 + 1
    re = tl.load(spectrum_ptr + 2 * offsets, mask, 0)
    im = tl.load(spectrum_ptr + 2 * offsets + 1, mask, 0)
    # Integer wavenumbers on a 2π-periodic grid; even-N Nyquist has zero derivative.
    active = (offsets != 0) & ((N % 2 != 0) | (offsets != N // 2))
    tl.store(spectrum_ptr + 2 * offsets, tl.where(active, -offsets * im, 0.0), mask)
    tl.store(spectrum_ptr + 2 * offsets + 1, tl.where(active, offsets * re, 0.0), mask)


@triton.jit
def _kerr_raman_kernel(
    angular_ptr, derivative_ptr, h, raman, N: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    re = tl.load(angular_ptr + 2 * offsets, mask, 0)
    im = tl.load(angular_ptr + 2 * offsets + 1, mask, 0)
    derivative = tl.load(derivative_ptr + offsets, mask, 0)
    phase = h * (re * re + im * im - raman * derivative)
    c = tl.cos(phase)
    s = tl.sin(phase)
    tl.store(angular_ptr + 2 * offsets, re * c - im * s, mask)
    tl.store(angular_ptr + 2 * offsets + 1, re * s + im * c, mask)


def kerr_raman_(angular: torch.Tensor, h: float, raman: float) -> torch.Tensor:
    """Apply exp(i*h*(|ψ|² - raman*∂θ|ψ|²)) in place on a 2π-periodic grid."""
    check_complex64_cuda(angular, "angular")
    N = angular.numel()
    if N == 0:
        raise ValueError("angular must not be empty")
    h, raman = float(h), float(raman)
    if raman == 0:
        return kerr_(angular, h)

    intensity = torch.empty(N, dtype=torch.float32, device=angular.device)
    _intensity_kernel[(triton.cdiv(N, 256),)](
        angular_ptr=angular.view(torch.float32), intensity_ptr=intensity, N=N, BLOCK=256
    )
    spectrum = torch.fft.rfft(intensity)
    _spectral_derivative_kernel[(triton.cdiv(N // 2 + 1, 256),)](
        spectrum_ptr=spectrum.view(torch.float32), N=N, BLOCK=256
    )
    derivative = torch.fft.irfft(spectrum, n=N)
    _kerr_raman_kernel[(triton.cdiv(N, 256),)](
        angular_ptr=angular.view(torch.float32), derivative_ptr=derivative,
        h=h, raman=raman, N=N, BLOCK=256,
    )
    return angular
