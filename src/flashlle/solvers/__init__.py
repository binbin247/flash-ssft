"""NLSE solvers composed from independent operators.

Parameters and the NumPy reference require only NumPy. Import GPU execution
explicitly from ``flashlle.solvers.nlse`` when CUDA and Triton are available.
"""

from .params import NLSEParams, angular_frequencies, linear_coefficients, time_grid
from .reference import simulate_numpy

__all__ = ["NLSEParams", "time_grid", "angular_frequencies", "linear_coefficients", "simulate_numpy"]
