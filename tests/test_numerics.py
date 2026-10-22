"""Numerical checks for the public, composable LLE solver.

Run from the project root with ``python -m pytest tests/test_numerics.py``.
GPU checks are skipped on machines without CUDA; the CPU reference still runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from flashlle import LLEParams, initial_field, simulate_gpu, simulate_numpy


def _relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-30))


@pytest.fixture(scope="module")
def cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton operator checks")
    return torch


def test_numpy_zero_steps_returns_initial_field() -> None:
    params = LLEParams(n_modes=1024)
    seed = initial_field(params)
    final = simulate_numpy(seed, params, 0)
    assert final.dtype == np.complex128
    assert _relative_l2(final, seed) < 1e-13


def test_linear_coefficients_scale_with_step_size() -> None:
    from dataclasses import replace

    from flashlle.gpu import linear_coefficients

    params = LLEParams(n_modes=257, pump_f=1.8, h=0.005, d3=1e-4)
    linear, pump = linear_coefficients(params)
    double_linear, double_pump = linear_coefficients(replace(params, h=2 * params.h))
    identity, zero_pump = linear_coefficients(replace(params, h=0))

    np.testing.assert_allclose(double_linear, linear * linear, rtol=1e-14, atol=0)
    np.testing.assert_allclose(double_pump, 2 * pump, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(identity, np.ones(params.n_modes, dtype=np.complex128))
    assert zero_pump == 0


@pytest.fixture
def uniform_step_case() -> tuple[LLEParams, np.ndarray, np.ndarray]:
    """Uniform-field result for linear evolution, additive pumping, then Kerr."""
    params = LLEParams(n_modes=128, pump_f=1.8, detuning_alpha=3.2, h=0.15)
    initial = 0.8 + 0.6j
    lam = -1.0 - 1j * params.detuning_alpha
    after_linear = np.exp(lam * params.h) * initial + params.pump_f * params.h
    expected = after_linear * np.exp(1j * params.h * abs(after_linear) ** 2)
    return (
        params,
        np.full(params.n_modes, initial, dtype=np.complex128),
        np.full(params.n_modes, expected, dtype=np.complex128),
    )


def test_numpy_step_is_full_linear_then_kerr(uniform_step_case) -> None:
    params, seed, expected = uniform_step_case

    actual = simulate_numpy(seed, params, 1)

    assert _relative_l2(actual, expected) < 1e-13


def test_gpu_single_step_replays_graph_and_matches_full_step(uniform_step_case, cuda) -> None:
    """A call shorter than graph_chunk must still replay a real CUDA Graph."""
    from unittest.mock import patch

    params, seed, expected = uniform_step_case

    with patch.object(
        cuda.cuda.CUDAGraph,
        "replay",
        autospec=True,
        side_effect=cuda.cuda.CUDAGraph.replay,
    ) as replay:
        actual = simulate_gpu(seed, params, 1, graph_chunk=16)

    replay.assert_called()

    assert _relative_l2(actual, expected) < 1e-6


@pytest.mark.parametrize("steps", [1, 3])
def test_lle_step_accepts_and_returns_angular_field(steps: int, cuda) -> None:
    """Direct steps compose in the time domain and preserve the caller's input."""
    from flashlle.gpu import linear_coefficients, lle_step

    params = LLEParams(
        n_modes=257, pump_f=1.8, detuning_alpha=3.2, d2=0.01, d3=1e-4, h=0.025
    )
    theta = np.linspace(-np.pi, np.pi, params.n_modes, endpoint=False)
    seed = np.asarray(
        0.6 + 0.2j + 0.3 * np.cos(3 * theta) + 0.15j * np.sin(5 * theta),
        dtype=np.complex64,
    )
    initial_gpu = cuda.as_tensor(seed, device="cuda")
    linear, pump = linear_coefficients(params)
    linear_gpu = cuda.as_tensor(linear, dtype=cuda.complex64, device="cuda")

    actual = initial_gpu
    for _ in range(steps):
        actual = lle_step(actual, linear_gpu, pump, params.h)

    assert actual.shape == initial_gpu.shape
    assert actual.dtype == cuda.complex64
    assert actual.device == initial_gpu.device
    assert _relative_l2(actual.cpu().numpy(), simulate_numpy(seed, params, steps)) < 1e-5
    np.testing.assert_array_equal(initial_gpu.cpu().numpy(), seed)


@pytest.mark.parametrize("n_modes", [1024, 2048, 4096])
@pytest.mark.parametrize("d3", [0.0, 1e-4])
@pytest.mark.parametrize("steps", [1, 16, 17])
def test_gpu_matches_numpy(n_modes: int, d3: float, steps: int, cuda) -> None:
    """Check one step, a full graph chunk, and its remainder at every target size."""
    params = LLEParams(n_modes=n_modes, d3=d3)
    seed = initial_field(params)
    reference = simulate_numpy(seed, params, steps)
    actual = simulate_gpu(seed, params, steps, graph_chunk=16)

    assert reference.shape == actual.shape == (n_modes,)
    assert reference.dtype == np.complex128
    assert actual.dtype == np.complex64
    assert np.isfinite(actual).all()
    assert _relative_l2(actual, reference) < 1e-4

    # Power and the entire modal power spectrum catch normalization mistakes.
    reference_power = np.mean(np.abs(reference) ** 2)
    actual_power = np.mean(np.abs(actual) ** 2)
    assert abs(actual_power - reference_power) / reference_power < 1e-4
    reference_spectrum = np.abs(np.fft.fft(reference)) ** 2
    actual_spectrum = np.abs(np.fft.fft(actual)) ** 2
    assert _relative_l2(actual_spectrum, reference_spectrum) < 1e-4


@pytest.mark.parametrize(
    ("steps", "graph_chunk"),
    [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (7, 4), (8, 4), (9, 4), (2, 1), (3, 1)],
)
def test_full_steps_match_numpy_at_graph_boundaries(
    steps: int, graph_chunk: int, cuda
) -> None:
    """Check zero/one step and remainders before, at, and after graph chunk edges."""
    params = LLEParams(
        n_modes=128, pump_f=1.8, detuning_alpha=3.2, d2=0.01, d3=1e-4, h=0.025
    )
    theta = np.linspace(-np.pi, np.pi, params.n_modes, endpoint=False)
    seed = 0.6 + 0.2j + 0.3 * np.cos(3 * theta) + 0.15j * np.sin(5 * theta)
    reference = simulate_numpy(seed, params, steps)

    actual = simulate_gpu(seed, params, steps, graph_chunk=graph_chunk)

    assert actual.shape == seed.shape
    assert actual.dtype == np.complex64
    assert np.isfinite(actual).all()
    assert _relative_l2(actual, reference) < 1e-5


@pytest.mark.parametrize(
    "input_dtype",
    [np.complex64, np.complex128],
    ids=["complex64", "complex128"],
)
def test_gpu_accepts_sliced_initial_fields(input_dtype, cuda) -> None:
    """Preserve positive-stride complex NumPy inputs when converting to CUDA."""
    params = LLEParams(n_modes=128, h=0.025)
    theta = np.linspace(-np.pi, np.pi, 2 * params.n_modes, endpoint=False)
    source = (0.6 + 0.2j + 0.3 * np.cos(3 * theta) + 0.15j * np.sin(5 * theta)).astype(input_dtype)
    original = source.copy()
    seed = source[::2]

    actual = simulate_gpu(seed, params, 5, graph_chunk=4)

    assert actual.dtype == np.complex64
    assert _relative_l2(actual, simulate_numpy(seed, params, 5)) < 1e-5
    np.testing.assert_array_equal(source, original)


def test_graph_cache_restores_each_new_seed(cuda) -> None:
    """A cached graph must not resume the preceding call's final field."""
    params = LLEParams(n_modes=1024, d3=1e-4)
    seed_a = initial_field(params)
    theta = np.linspace(-np.pi, np.pi, params.n_modes, endpoint=False)
    seed_b = seed_a * np.exp(0.07j * np.cos(3 * theta))

    result_a = simulate_gpu(seed_a, params, 32, graph_chunk=16)
    result_b = simulate_gpu(seed_b, params, 32, graph_chunk=16)
    result_a_again = simulate_gpu(seed_a, params, 32, graph_chunk=16)

    assert _relative_l2(result_b, result_a) > 1e-4
    assert _relative_l2(result_b, simulate_numpy(seed_b, params, 32)) < 1e-4
    assert _relative_l2(result_a_again, result_a) < 1e-7


@pytest.mark.parametrize("in_place", [False, True], ids=["out-of-place", "in-place"])
def test_graph_stepper_preserves_inputs_and_previous_outputs(in_place: bool, cuda) -> None:
    """Repeated runs keep GPU results independent across graph and remainder paths."""
    from functools import partial
    from operator import methodcaller

    from flashlle.gpu import CudaGraphStepper

    increment = 1 + 2j
    step = methodcaller("add_", increment) if in_place else partial(cuda.add, other=increment)
    seed_a = cuda.arange(17, dtype=cuda.float32, device="cuda").to(cuda.complex64) + 2j
    seed_b = seed_a * (-1 + 1j)
    original_a, original_b = seed_a.clone(), seed_b.clone()

    runner = CudaGraphStepper(seed_a, step, chunk=4)
    cuda.testing.assert_close(seed_a, original_a, rtol=0, atol=0)

    cases = [
        (seed_a, original_a, 0),
        (seed_b, original_b, 1),
        (seed_a, original_a, 4),
        (seed_b, original_b, 5),
    ]
    # Keep the original GPU tensors until later replays could have overwritten them.
    results = [runner.run(seed, steps) for seed, _, steps in cases]
    for actual, (_, original, steps) in zip(results, cases):
        cuda.testing.assert_close(actual, original + steps * increment, rtol=0, atol=0)
    cuda.testing.assert_close(seed_a, original_a, rtol=0, atol=0)
    cuda.testing.assert_close(seed_b, original_b, rtol=0, atol=0)


def test_fused_pointwise_operators_in_place(cuda) -> None:
    """Verify the independent building blocks against explicit NumPy formulas."""
    from flashlle.operators import kerr_, linear_and_pump_

    rng = np.random.default_rng(20261020)
    n = 1024
    spectrum = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    linear = np.exp((-0.002 - 0.01j) * np.arange(n) / n).astype(np.complex64)
    pump = 0.125 - 0.375j

    spectrum_gpu = cuda.as_tensor(spectrum.copy(), device="cuda")
    linear_gpu = cuda.as_tensor(linear, device="cuda")
    assert linear_and_pump_(spectrum_gpu, linear_gpu, pump) is spectrum_gpu
    expected = spectrum.astype(np.complex128) * linear.astype(np.complex128)
    expected[0] += pump
    assert _relative_l2(spectrum_gpu.cpu().numpy(), expected) < 2e-6

    angular = (0.5 * rng.normal(size=n) + 0.5j * rng.normal(size=n)).astype(np.complex64)
    angular_gpu = cuda.as_tensor(angular.copy(), device="cuda")
    h = 0.005
    assert kerr_(angular_gpu, h) is angular_gpu
    expected_kerr = angular * np.exp(1j * h * np.abs(angular) ** 2)
    assert _relative_l2(angular_gpu.cpu().numpy(), expected_kerr) < 2e-6


@pytest.mark.parametrize(
    ("h", "pump_source"),
    [(0.0, 1.8), (1e-9, 0.6 - 0.8j), (0.15, 0.6 - 0.8j)],
    ids=["zero-step", "tiny-step", "complex-pump"],
)
def test_linear_and_pump_accepts_precomputed_coefficients(
    h: float, pump_source: complex, cuda
) -> None:
    """Apply precomputed zero/tiny steps and complex pumping across a kernel tail."""
    from flashlle.operators import linear_and_pump_

    rng = np.random.default_rng(20261021)
    n = 257
    spectrum = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    # Isolate the pump so a lost tiny increment cannot hide beneath the initial field.
    spectrum[0] = 0
    generator = np.asarray((-0.2 - 0.7j) * (1 + np.arange(n) / n), dtype=np.complex128)
    linear = np.exp(generator * h).astype(np.complex64)
    pump = complex(pump_source * h)
    linear_gpu = cuda.as_tensor(linear, device="cuda")

    spectrum_gpu = cuda.as_tensor(spectrum.copy(), device="cuda")
    assert linear_and_pump_(spectrum_gpu, linear_gpu, pump) is spectrum_gpu
    actual = spectrum_gpu.cpu().numpy()

    expected = spectrum.astype(np.complex128) * np.exp(generator * h)
    expected[0] += pump
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-6, atol=0)
    if h == 0:
        np.testing.assert_array_equal(actual, spectrum)
    np.testing.assert_array_equal(linear_gpu.cpu().numpy(), linear)
