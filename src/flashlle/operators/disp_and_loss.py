"""Fused spectral dispersion and loss for the nonlinear Schrodinger equation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ._validation import check_block, check_complex64_cuda, check_num_warps


@dataclass(frozen=True)
class DispParams:
    """Dispersion coefficients through fourth order, in consistent T/z units."""

    beta2: float = 0.0
    beta3: float = 0.0
    beta4: float = 0.0


def make_linear(
    w: torch.Tensor,          # Angular frequency grid: 2*pi*fftfreq(N, dT), unshifted.
    dz: float,                # Propagation step along z.
    alpha: float,             # Power loss coefficient per unit length (not dB).
    disp_params: DispParams,  # Dispersion coefficients beta2, beta3, beta4.
) -> torch.Tensor:  # Complex64 linear propagation factors, matching w's shape and device.
    """Precompute dispersion and loss factors for a propagation step dz.

    Use the unshifted angular frequency grid w = 2*pi*fftfreq(N, dT).
    Alpha is the power attenuation coefficient; field amplitude decays by
    exp(-alpha*dz/2). Reuse the factors while the grid, parameters, and dz are fixed.
    """
    if w.ndim != 1 or not w.is_floating_point():
        raise ValueError("w must be a one-dimensional real floating-point tensor")
    # Optical angular frequencies raised to the fourth power can overflow float32.
    w = w.to(torch.float64)
    D = disp_params.beta2*w**2/2 - disp_params.beta3*w**3/6 + disp_params.beta4*w**4/24
    return torch.exp(dz * (-alpha/2 + 1j*D)).to(torch.complex64).contiguous()


@triton.jit
def _disp_and_loss_kernel(
    spectrum_ptr, linear_ptr, N: tl.constexpr, BLOCK: tl.constexpr
):
    # Interleaved complex64 data: [re0, im0, re1, im1, ...].
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    re = tl.load(spectrum_ptr + 2 * offsets, mask, 0)
    im = tl.load(spectrum_ptr + 2 * offsets + 1, mask, 0)
    linear_re = tl.load(linear_ptr + 2 * offsets, mask, 0)
    linear_im = tl.load(linear_ptr + 2 * offsets + 1, mask, 0)

    tl.store(spectrum_ptr + 2 * offsets, re * linear_re - im * linear_im, mask)
    tl.store(spectrum_ptr + 2 * offsets + 1, re * linear_im + im * linear_re, mask)


def disp_and_loss_(
    spectrum: torch.Tensor,  # FFT of the field A; updated in place.
    linear: torch.Tensor,    # Precomputed dispersion/loss factors from make_linear.
    *,
    block: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """Multiply the FFT spectrum by precomputed dispersion/loss factors in place.

    Build linear with make_linear. Inputs are contiguous 1D complex64 CUDA
    tensors. Compose as fft(A) -> disp_and_loss_(spectrum, linear) -> ifft.
    """
    check_complex64_cuda(spectrum, "spectrum")
    check_complex64_cuda(linear, "linear")
    if spectrum.shape != linear.shape or spectrum.device != linear.device:
        raise ValueError("spectrum and linear must have matching shape and device")
    block = check_block(block)
    num_warps = check_num_warps(num_warps)

    N = spectrum.numel()
    _disp_and_loss_kernel[(triton.cdiv(N, block),)](
        spectrum_ptr=spectrum.view(torch.float32),
        linear_ptr=linear.view(torch.float32),
        N=N,
        BLOCK=block,
        num_warps=num_warps,
    )
    return spectrum
