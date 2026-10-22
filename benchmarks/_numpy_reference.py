"""Independent complex128 reference for the periodic Kerr–Raman substep."""

import numpy as np


def numpy_kerr_raman(angular, h, raman, calls=1):
    """Evaluate repeated phase rotations using their invariant intensity."""
    field = np.asarray(angular, dtype=np.complex128)
    intensity = field.real**2 + field.imag**2
    n = field.size
    wavenumbers = np.fft.fftfreq(n, d=1.0 / n)
    if n % 2 == 0:
        wavenumbers[n // 2] = 0
    derivative = np.fft.ifft(1j * wavenumbers * np.fft.fft(intensity)).real
    return field * np.exp(1j * h * calls * (intensity - raman * derivative))
