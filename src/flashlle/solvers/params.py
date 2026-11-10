"""NLSE grid, physical parameters and spectral factors, without GPU dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, isfinite
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class NLSEParams:
    """Constant coefficients on a periodic retarded-time grid of length N*dt.

    Use consistent T/z/power units: alpha is power loss per z (not dB), beta_m
    has units T**m/z, gamma0 has units 1/(power*z), and omega0 has units 1/T.
    beta_higher contains beta5, beta6, ...; omitted orders are zero. Positive
    infinite omega0 explicitly disables self-steepening. Defaults are an
    illustrative normalized problem, not measured material parameters.
    """

    n_samples: int = 1024
    dt: float = 0.02
    dz: float = 0.001
    alpha: float = 0.0
    beta2: float = -1.0
    beta3: float = 0.0
    beta4: float = 0.0
    gamma0: float = 1.0
    omega0: float = 20.0
    beta_higher: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.n_samples, Integral)
                or isinstance(self.n_samples, bool) or self.n_samples < 1):
            raise ValueError("n_samples must be a positive integer")
        object.__setattr__(self, "n_samples", int(self.n_samples))
        for name in ("dt", "dz", "alpha", "beta2", "beta3", "beta4", "gamma0", "omega0"):
            value = getattr(self, name)
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(f"{name} must be real")
            value = float(value)
            if not isfinite(value) and not (name == "omega0" and value == float("inf")):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.dt <= 0 or self.omega0 <= 0:
            raise ValueError("dt and omega0 must be positive")
        if self.dz < 0 or self.alpha < 0:
            raise ValueError("dz and alpha must be nonnegative")
        higher = tuple(self.beta_higher)
        if any(not isinstance(b, Real) or isinstance(b, bool) or not isfinite(b) for b in higher):
            raise ValueError("beta_higher must contain finite real coefficients")
        object.__setattr__(self, "beta_higher", tuple(float(b) for b in higher))
        if not isfinite(self.gamma0 * self.dz):
            raise ValueError("gamma0 * dz must be finite")


def time_grid(params: NLSEParams) -> np.ndarray:
    """Centered retarded-time samples, separated by dt; no duplicate endpoint."""
    return (np.arange(params.n_samples) - params.n_samples // 2) * params.dt


def angular_frequencies(params: NLSEParams) -> np.ndarray:
    """Unshifted FFT angular frequencies; d/dT corresponds to multiplication by i*w."""
    return 2 * np.pi * np.fft.fftfreq(params.n_samples, d=params.dt)


def linear_coefficients(params: NLSEParams) -> np.ndarray:
    """Return exp(dz*(-alpha/2 + i*sum((-1)**m*beta_m*w**m/m!))).

    This sign follows directly from the stated NLSE and NumPy/PyTorch's FFT
    convention. Evaluate the polynomial with Horner's rule in float64, avoiding
    unnecessarily large intermediate powers of optical angular frequencies.
    """
    if params.dz == 0:
        return np.ones(params.n_samples, dtype=np.complex128)
    beta = (params.beta2, params.beta3, params.beta4, *params.beta_higher)
    coefficients = [0.0, 0.0] + [(-1)**m * b / factorial(m) for m, b in enumerate(beta, 2)]
    w = angular_frequencies(params)
    dispersion = np.polynomial.polynomial.polyval(w, coefficients)
    return np.exp(params.dz * (-params.alpha / 2 + 1j * dispersion))
