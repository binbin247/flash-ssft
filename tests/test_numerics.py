"""NLSE formula, convergence, input validation and GPU execution checks."""

from dataclasses import replace
from math import factorial

import numpy as np
import pytest

from flashlle.solvers import (
    NLSEParams, angular_frequencies, linear_coefficients, simulate_numpy, time_grid,
)


def smooth_field(n):
    theta = 2 * np.pi * np.arange(n) / n
    return ((0.6 + 0.1 * np.cos(theta)) * np.exp(0.2j * np.sin(2 * theta))).astype(np.complex128)


def relative_l2(actual, expected):
    return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)


@pytest.fixture(scope="module")
def cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the NLSE GPU checks")
    return torch


@pytest.fixture(scope="module")
def simulate_gpu(cuda):
    from flashlle.solvers.nlse import simulate_gpu

    return simulate_gpu


@pytest.mark.parametrize("n", [31, 32])
def test_time_and_frequency_grid_units(n):
    params = NLSEParams(n_samples=n, dt=0.17)
    t = time_grid(params)
    np.testing.assert_allclose(np.diff(t), params.dt, rtol=1e-14)
    assert t[n // 2] == 0
    # A physical harmonic is differentiated with i*w, including negative modes.
    w = angular_frequencies(params)
    frequency = -3 * 2 * np.pi / (n * params.dt)
    field = np.exp(1j * frequency * t)
    derivative = np.fft.ifft(1j * w * np.fft.fft(field))
    np.testing.assert_allclose(derivative, 1j * frequency * field, atol=5e-14)


@pytest.mark.parametrize("n", [31, 32])
@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_linear_dispersion_matches_equation_fourier_modes(n, order):
    coefficients = {m: (0.012 if m == order else 0.0) for m in range(2, 7)}
    params = NLSEParams(n_samples=n, dt=0.3, dz=0.003, alpha=0.2, gamma0=0,
                        beta2=coefficients[2], beta3=coefficients[3], beta4=coefficients[4],
                        beta_higher=(coefficients[5], coefficients[6]))
    t = time_grid(params)
    seed = np.zeros(n, dtype=np.complex128)
    expected = seed.copy()
    for mode, amplitude in [(0, 0.3), (3, 0.2 + 0.1j), (-3, -0.4j), (-(n // 2), 0.1j)]:
        frequency = 2 * np.pi * mode / (n * params.dt)
        wave = amplitude * np.exp(1j * frequency * t)
        # Substitute d_T**m -> (i*w)**m directly into the user's PDE.
        generator = -params.alpha / 2 - sum(
            1j**(m - 1) * beta / factorial(m) * (1j * frequency)**m
            for m, beta in coefficients.items()
        )
        seed += wave
        expected += wave * np.exp(7 * params.dz * generator)
    np.testing.assert_allclose(simulate_numpy(seed, params, 7), expected, rtol=2e-13, atol=3e-14)


def test_linear_factors_with_optical_frequencies_and_higher_dispersion():
    params = NLSEParams(n_samples=32, dt=1e-14, dz=0.01, beta2=-2e-26,
                        beta3=1e-40, beta4=1e-54, beta_higher=(1e-68, 1e-82), gamma0=0)
    w = angular_frequencies(params)
    generator = -params.alpha / 2 - sum(
        1j**(m - 1) * b / factorial(m) * (1j * w)**m
        for m, b in enumerate((params.beta2, params.beta3, params.beta4, *params.beta_higher), 2)
    )
    np.testing.assert_allclose(linear_coefficients(params), np.exp(params.dz * generator), atol=1e-14)


def test_loss_is_power_attenuation():
    params = NLSEParams(n_samples=64, dt=0.1, dz=0.02, alpha=0.7, beta2=0, gamma0=0)
    seed = smooth_field(params.n_samples)
    actual = simulate_numpy(seed, params, 11)
    np.testing.assert_allclose(actual, seed * np.exp(-params.alpha * 11 * params.dz / 2), atol=2e-15)
    assert np.sum(np.abs(actual)**2) / np.sum(np.abs(seed)**2) == pytest.approx(
        np.exp(-params.alpha * 11 * params.dz), rel=1e-13)


@pytest.mark.parametrize("gamma0", [0.0, 0.7, 2.3, -1.2])
def test_pure_kerr_has_exact_phase_and_preserves_intensity(gamma0):
    params = NLSEParams(n_samples=65, dt=0.1, dz=0.03, beta2=0, gamma0=gamma0, omega0=float("inf"))
    seed = smooth_field(params.n_samples)
    actual = simulate_numpy(seed, params, 9)
    expected = seed * np.exp(1j * gamma0 * params.dz * 9 * np.abs(seed)**2)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)
    np.testing.assert_allclose(np.abs(actual)**2, np.abs(seed)**2, rtol=3e-14)


def test_zero_field_has_no_external_source():
    params = NLSEParams(n_samples=32, alpha=0.2, beta3=0.02, gamma0=2.0, omega0=3.0)
    np.testing.assert_array_equal(simulate_numpy(np.zeros(32, dtype=complex), params, 10), 0)


@pytest.mark.parametrize("mode", [3, -3, -16])
def test_shock_sign_and_gamma_scaling_for_single_mode(mode):
    params = NLSEParams(n_samples=32, dt=0.2, dz=0.004, beta2=0, gamma0=2.7, omega0=7.0)
    frequency = 2 * np.pi * mode / (params.n_samples * params.dt)
    seed = 0.6 * np.exp(1j * frequency * time_grid(params))
    h = params.gamma0 * params.dz
    rotated = seed * np.exp(1j * h * 0.6**2)
    expected = rotated * (1 - 1j * h * frequency * 0.6**2 / params.omega0)
    np.testing.assert_allclose(simulate_numpy(seed, params, 1), expected, atol=1e-14)


def test_uniform_field_full_step_orders_loss_before_nonlinearity():
    params = NLSEParams(n_samples=32, dz=0.05, alpha=0.4, gamma0=1.8, omega0=5.0)
    seed = np.full(32, 0.6 + 0.2j)
    after_loss = seed * np.exp(-params.alpha * params.dz / 2)
    expected = after_loss * np.exp(1j * params.gamma0 * params.dz * np.abs(after_loss)**2)
    np.testing.assert_allclose(simulate_numpy(seed, params, 1), expected, atol=2e-15)


def test_full_nlse_converges_to_independent_rk4_equation():
    """Halving dz should halve global error for the existing Euler shock split."""
    n, distance = 32, 0.2
    params = NLSEParams(n_samples=n, dt=2 * np.pi / n, alpha=0.1,
                        beta2=-0.2, beta3=0.03, beta4=0.002, gamma0=1.7, omega0=4.0)
    seed = smooth_field(n)
    w = angular_frequencies(params)
    # Direct PDE right-hand side, independent of split-step coefficients/updates.
    def rhs(field):
        spectrum = np.fft.fft(field)
        linear = -params.alpha * field / 2
        for m, beta in enumerate((params.beta2, params.beta3, params.beta4), 2):
            linear -= 1j**(m - 1) * beta / factorial(m) * np.fft.ifft((1j * w)**m * spectrum)
        cubic = np.abs(field)**2 * field
        return (linear + 1j * params.gamma0 * cubic
                - params.gamma0 / params.omega0 * np.fft.ifft(1j * w * np.fft.fft(cubic)))

    reference = seed.copy()
    dt = distance / 2048
    for _ in range(2048):
        k1 = rhs(reference)
        k2 = rhs(reference + dt * k1 / 2)
        k3 = rhs(reference + dt * k2 / 2)
        k4 = rhs(reference + dt * k3)
        reference += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    errors = [relative_l2(simulate_numpy(seed, replace(params, dz=distance / s), s), reference)
              for s in (8, 16, 32)]
    ratios = np.array(errors[:-1]) / errors[1:]
    assert np.all((ratios > 1.8) & (ratios < 2.2)), (errors, ratios)
    assert errors[-1] < 1e-3


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("steps,dz", [(0, 0.01), (4, 0.0)])
def test_zero_propagation_returns_independent_copy(dtype, steps, dz):
    params = NLSEParams(n_samples=32, dz=dz)
    source = smooth_field(64).astype(dtype)
    seed = source[::2]
    original = source.copy()
    actual = simulate_numpy(seed, params, steps)
    np.testing.assert_array_equal(actual, seed)
    assert actual.dtype == np.complex128
    assert not np.shares_memory(actual, source)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("kwargs", [
    {"n_samples": 0}, {"n_samples": 2.5}, {"n_samples": True}, {"dt": 0}, {"dt": float("nan")},
    {"dz": -1}, {"dz": float("inf")}, {"alpha": -1}, {"gamma0": float("nan")},
    {"omega0": 0}, {"omega0": -1}, {"omega0": float("nan")}, {"omega0": -float("inf")},
    {"beta3": 1j}, {"beta_higher": (float("inf"),)},
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        NLSEParams(**kwargs)


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_step_count(steps):
    params = NLSEParams(n_samples=32)
    with pytest.raises(ValueError, match="steps"):
        simulate_numpy(smooth_field(32), params, steps)


@pytest.mark.parametrize("seed,error", [
    (np.ones(32), TypeError), (np.ones((2, 16), dtype=complex), ValueError),
    (np.full(32, complex(float("nan"), 0)), ValueError),
])
def test_invalid_input_field(seed, error):
    with pytest.raises(error):
        simulate_numpy(seed, NLSEParams(n_samples=32), 1)


@pytest.mark.parametrize("gamma0,omega0", [(0.0, 5.0), (2.3, 5.0), (2.3, float("inf"))])
def test_torch_nlse_benchmark_baseline_on_cpu(gamma0, omega0):
    torch = pytest.importorskip("torch")
    from benchmarks._torch_reference import torch_nlse_step

    params = NLSEParams(n_samples=65, dt=0.2, dz=0.003, alpha=0.1, beta3=0.02,
                        gamma0=gamma0, omega0=omega0)
    seed = smooth_field(65)
    field = torch.tensor(seed, dtype=torch.complex64)
    original = field.clone()
    linear = torch.tensor(linear_coefficients(params), dtype=torch.complex64)
    w = torch.tensor(angular_frequencies(params), dtype=torch.float64)
    actual = torch_nlse_step(field, linear, w, params.gamma0 * params.dz, params.omega0)
    torch.testing.assert_close(field, original, rtol=0, atol=0)
    assert relative_l2(actual.numpy(), simulate_numpy(seed, params, 1)) < 2e-6


@pytest.mark.parametrize("n", [31, 32, 257])
@pytest.mark.parametrize("steps", [0, 1, 4, 5])
@pytest.mark.parametrize("omega0", [5.0, float("inf")])
def test_gpu_matches_numpy_at_graph_boundaries(n, steps, omega0, simulate_gpu):
    params = NLSEParams(n_samples=n, dt=2*np.pi/n, dz=0.005, alpha=0.1,
                        beta2=-0.2, beta3=0.01, beta4=0.001, gamma0=1.7, omega0=omega0,
                        beta_higher=(0.0001,))
    seed = smooth_field(n)
    expected = simulate_numpy(seed, params, steps)
    actual = simulate_gpu(seed, params, steps, graph_chunk=4)
    assert actual.dtype == np.complex64
    assert relative_l2(actual, expected) < 2e-5


def test_gpu_single_step_uses_graph_and_preserves_seed(cuda, simulate_gpu):
    from unittest.mock import patch

    params = NLSEParams(n_samples=32, dt=0.2, dz=0.01, gamma0=2.0, omega0=5.0)
    seed = smooth_field(32)
    original = seed.copy()
    with patch.object(cuda.cuda.CUDAGraph, "replay", autospec=True,
                      side_effect=cuda.cuda.CUDAGraph.replay) as replay:
        actual = simulate_gpu(seed, params, 1, graph_chunk=16)
    replay.assert_called_once()
    np.testing.assert_array_equal(seed, original)
    assert relative_l2(actual, simulate_numpy(seed, params, 1)) < 2e-6


def test_nlse_step_and_torch_baseline_match_reference(cuda):
    from benchmarks._torch_reference import torch_nlse_step
    from flashlle.solvers.nlse import nlse_step

    params = NLSEParams(n_samples=65, dt=0.2, gamma0=2.3, beta3=0.02, omega0=8.0)
    seed = smooth_field(65)
    field = cuda.tensor(seed, dtype=cuda.complex64, device="cuda")
    original = field.clone()
    linear = cuda.tensor(linear_coefficients(params), dtype=cuda.complex64, device="cuda")
    w = cuda.tensor(angular_frequencies(params), dtype=cuda.float64, device="cuda")
    for step in (nlse_step, torch_nlse_step):
        actual = step(field, linear, w, params.gamma0 * params.dz, params.omega0)
        cuda.testing.assert_close(field, original, rtol=0, atol=0)
        assert relative_l2(actual.cpu().numpy(), simulate_numpy(seed, params, 1)) < 2e-6


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_gpu_accepts_sliced_inputs(dtype, simulate_gpu):
    source = smooth_field(128).astype(dtype)
    seed = source[::2]
    original = source.copy()
    params = NLSEParams(n_samples=64, dt=0.2)
    actual = simulate_gpu(seed, params, 5, graph_chunk=4)
    assert relative_l2(actual, simulate_numpy(seed, params, 5)) < 2e-5
    np.testing.assert_array_equal(source, original)


def test_graph_cache_separates_parameters_and_restores_each_seed(simulate_gpu):
    from flashlle.solvers.nlse import clear_graph_cache

    base = NLSEParams(n_samples=64, dt=0.2, dz=0.01)
    seed = smooth_field(64)
    clear_graph_cache()
    reference = simulate_gpu(seed, base, 5, graph_chunk=4)
    for params in (replace(base, gamma0=0), replace(base, omega0=2), replace(base, dt=0.3),
                   replace(base, beta3=0.04), replace(base, alpha=0.2),
                   replace(base, beta_higher=(0.003,))):
        changed_seed = seed * 0.8
        actual = simulate_gpu(changed_seed, params, 5, graph_chunk=4)
        assert relative_l2(actual, simulate_numpy(changed_seed, params, 5)) < 2e-5
    again = simulate_gpu(seed, base, 5, graph_chunk=4)
    np.testing.assert_array_equal(again, reference)
    clear_graph_cache()


@pytest.mark.parametrize("chunk", [0, -1, 1.5, True])
def test_invalid_graph_chunks(chunk, simulate_gpu):
    with pytest.raises(ValueError, match="graph_chunk"):
        simulate_gpu(smooth_field(32), NLSEParams(n_samples=32), 0, graph_chunk=chunk)


@pytest.mark.parametrize("in_place", [False, True], ids=["out-of-place", "in-place"])
def test_graph_stepper_preserves_inputs_and_previous_outputs(in_place, cuda):
    from functools import partial
    from operator import methodcaller
    from flashlle.solvers.nlse import CudaGraphStepper

    increment = 1 + 2j
    step = methodcaller("add_", increment) if in_place else partial(cuda.add, other=increment)
    seed_a = cuda.arange(17, dtype=cuda.float32, device="cuda").to(cuda.complex64) + 2j
    seed_b = seed_a * (-1 + 1j)
    original_a, original_b = seed_a.clone(), seed_b.clone()
    runner = CudaGraphStepper(seed_a, step, chunk=4)
    cases = [(seed_a, original_a, 0), (seed_b, original_b, 1),
             (seed_a, original_a, 4), (seed_b, original_b, 5)]
    results = [runner.run(seed, steps) for seed, _, steps in cases]
    for actual, (_, original, steps) in zip(results, cases):
        cuda.testing.assert_close(actual, original + steps * increment, rtol=0, atol=0)
    cuda.testing.assert_close(seed_a, original_a, rtol=0, atol=0)
    cuda.testing.assert_close(seed_b, original_b, rtol=0, atol=0)
