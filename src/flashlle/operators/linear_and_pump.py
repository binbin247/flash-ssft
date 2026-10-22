"""Fused spectral linear evolution and zero-mode pump."""

from __future__ import annotations


import torch
import triton
import triton.language as tl

from ._validation import check_complex64_cuda


@triton.jit
def _linear_and_pump_kernel(
    spectrum_ptr, linear_ptr, pump_re, pump_im, N: tl.constexpr, BLOCK: tl.constexpr
):
    # spectrum_ptr, linear_ptr: GPU float32 pointers to interleaved complex64 values.
    # linear and pump already include the full time step.
    # N: total number of complex elements, not the number of float32 components.
    # BLOCK: complex elements handled by each Triton program (currently 256).
    # N and BLOCK are compile-time constants because they use tl.constexpr.

    # Compute this program's complex-element offsets and mask out-of-bounds lanes.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load real and imaginary components; masked lanes receive zero.
    re = tl.load(spectrum_ptr + 2 * offsets, mask, 0)
    im = tl.load(spectrum_ptr + 2 * offsets + 1, mask, 0)
    linear_re = tl.load(linear_ptr + 2 * offsets, mask, 0)
    linear_im = tl.load(linear_ptr + 2 * offsets + 1, mask, 0)

    # Apply the precomputed linear factor and add the zero-mode pump.
    out_re = re * linear_re - im * linear_im + tl.where(offsets == 0, pump_re, 0.0)
    out_im = re * linear_im + im * linear_re + tl.where(offsets == 0, pump_im, 0.0)

    # Write both components back in place.
    tl.store(spectrum_ptr + 2 * offsets, out_re, mask)
    tl.store(spectrum_ptr + 2 * offsets + 1, out_im, mask)


def linear_and_pump_(
    spectrum: torch.Tensor, linear: torch.Tensor, pump: complex
) -> torch.Tensor:
    """Apply precomputed linear factors and the zero-mode pump increment in place."""
    check_complex64_cuda(spectrum, "spectrum")
    check_complex64_cuda(linear, "linear")
    if spectrum.shape != linear.shape or spectrum.device != linear.device:
        raise ValueError("spectrum and linear must have matching shape and device")
    pump = complex(pump)
    N = spectrum.numel()
    _linear_and_pump_kernel[(triton.cdiv(N, 256),)](
        spectrum_ptr=spectrum.view(torch.float32),
        linear_ptr=linear.view(torch.float32),
        pump_re=float(pump.real),
        pump_im=float(pump.imag),
        N=N,
        BLOCK=256,
    )
    return spectrum
