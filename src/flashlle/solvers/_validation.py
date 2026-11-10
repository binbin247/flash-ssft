"""Input checks shared by the NumPy and GPU NLSE solvers."""

from numbers import Integral

import numpy as np


def check_steps(steps: int) -> None:
    if not isinstance(steps, Integral) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")


def check_graph_chunk(chunk: int) -> None:
    if not isinstance(chunk, Integral) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError("graph_chunk must be a positive integer")


def check_input(a0: np.ndarray, n_samples: int, steps: int) -> np.ndarray:
    """Validate a finite complex time-domain field and nonnegative step count."""
    field = np.asarray(a0)
    if field.shape != (n_samples,):
        raise ValueError(f"a0 must have shape ({n_samples},), got {field.shape}")
    if not np.issubdtype(field.dtype, np.complexfloating):
        raise TypeError("a0 must contain complex values")
    if not np.isfinite(field).all():
        raise ValueError("a0 must contain finite values")
    check_steps(steps)
    return field
