"""Independent spectral, analytic and graph checks for the Kerr–Raman operator."""

import numpy as np
import pytest

from benchmarks._numpy_reference import numpy_kerr_raman
from benchmarks._torch_reference import torch_kerr_raman_
from flashlle.operators import kerr_, kerr_raman_


@pytest.fixture(scope="module")
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Kerr–Raman GPU checks")
    return torch


def relative_l2(actual, expected):
    return np.linalg.norm(np.asarray(actual, dtype=np.complex128) - expected) / np.linalg.norm(expected)


def smooth_field(n):
    theta = 2 * np.pi * np.arange(n) / n
    return (np.sqrt(1 + 0.2 * np.cos(3 * theta)) * np.exp(0.1j * np.sin(theta))).astype(np.complex64)


@pytest.mark.parametrize("n,frequency", [(257, 3), (255, 127), (256, 128)])
def test_numpy_reference_matches_analytic_derivative(n, frequency):
    theta = 2 * np.pi * np.arange(n) / n
    intensity = 1 + 0.2 * np.cos(frequency * theta)
    derivative = -0.2 * frequency * np.sin(frequency * theta)
    field = np.sqrt(intensity).astype(np.complex128)
    expected = field * np.exp(0.1j * (intensity - 0.02 * derivative))
    assert relative_l2(numpy_kerr_raman(field, 0.1, 0.02), expected) < 1e-13


@pytest.mark.parametrize("n", [1, 2, 255, 256, 257, 1024])
@pytest.mark.parametrize("raman", [-0.03, 0.03])
def test_kerr_raman_matches_complex128_reference_in_place(cuda, n, raman):
    rng = np.random.default_rng(21)
    field = (0.5 * (rng.normal(size=n) + 1j * rng.normal(size=n))).astype(np.complex64)
    angular = cuda.as_tensor(field.copy(), device="cuda")
    pointer = angular.data_ptr()
    assert kerr_raman_(angular, 0.005, raman) is angular
    assert angular.data_ptr() == pointer
    actual = angular.cpu().numpy()
    assert relative_l2(actual, numpy_kerr_raman(field, 0.005, raman)) < 2e-6
    np.testing.assert_allclose(np.abs(actual)**2, np.abs(field)**2, rtol=5e-7, atol=2e-7)


@pytest.mark.parametrize("n,frequency", [(257, 3), (255, 127), (256, 128)])
def test_kerr_raman_analytic_phase_and_highest_frequency(cuda, n, frequency):
    theta = 2 * np.pi * np.arange(n) / n
    intensity = 1 + 0.2 * np.cos(frequency * theta)
    derivative = -0.2 * frequency * np.sin(frequency * theta)
    field = np.sqrt(intensity).astype(np.complex64)
    expected = field * np.exp(0.1j * (intensity - 0.02 * derivative))
    angular = cuda.as_tensor(field.copy(), device="cuda")
    kerr_raman_(angular, 0.1, 0.02)
    assert relative_l2(angular.cpu().numpy(), expected) < 2e-6


@pytest.mark.parametrize("n", [256, 257])
def test_spectral_derivative_endpoint_convention(cuda, n):
    from flashlle.operators.kerr_raman import _spectral_derivative_kernel

    spectrum = cuda.full((n // 2 + 1,), 1 + 2j, dtype=cuda.complex64, device="cuda")
    _spectral_derivative_kernel[((n // 2 + 256) // 256,)](
        spectrum_ptr=spectrum.view(cuda.float32), N=n, BLOCK=256
    )
    actual = spectrum.cpu().numpy()
    assert actual[0] == 0
    expected_last = 0 if n % 2 == 0 else (-2 + 1j) * (n // 2)
    assert actual[-1] == expected_last


def test_kerr_raman_zero_coefficient_and_zero_step(cuda):
    field = cuda.as_tensor(smooth_field(257), device="cuda")
    expected = kerr_(field.clone(), 0.1)
    actual = kerr_raman_(field.clone(), 0.1, 0)
    cuda.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual = kerr_raman_(field.clone(), 0, 0.01)
    cuda.testing.assert_close(actual, field, rtol=0, atol=0)


def test_kerr_raman_uniform_intensity(cuda):
    n = 257
    phase = 0.1 * np.sin(2 * np.pi * np.arange(n) / n)
    field = (0.75 * np.exp(1j * phase)).astype(np.complex64)
    angular = cuda.as_tensor(field.copy(), device="cuda")
    kerr_raman_(angular, 0.1, 0.2)
    assert relative_l2(angular.cpu().numpy(), field * np.exp(0.1j * 0.75**2)) < 2e-6


@pytest.mark.parametrize("n", [257, 1024])
@pytest.mark.parametrize("operation", [kerr_raman_, torch_kerr_raman_], ids=["triton", "torch"])
def test_kerr_raman_repeated_graph_replay_and_new_input(cuda, n, operation):
    seed = cuda.as_tensor(smooth_field(n), device="cuda")
    state = seed.clone()
    stream = cuda.cuda.Stream()
    stream.wait_stream(cuda.cuda.current_stream())
    with cuda.cuda.stream(stream):
        for _ in range(2):
            operation(state, 0.005, 0.01)
    cuda.cuda.current_stream().wait_stream(stream)
    graph = cuda.cuda.CUDAGraph()
    with cuda.cuda.graph(graph):
        for _ in range(4):
            operation(state, 0.005, 0.01)

    for scale in (1.0, 0.7):
        initial = seed * scale
        state.copy_(initial)
        eager = initial.clone()
        for _ in range(8):
            graph.replay()
        for _ in range(32):
            operation(eager, 0.005, 0.01)
        cuda.testing.assert_close(state, eager, rtol=1e-6, atol=1e-7)
        field = initial.cpu().numpy()
        actual = state.cpu().numpy()
        assert relative_l2(actual, numpy_kerr_raman(field, 0.005, 0.01, calls=32)) < 1e-3
        np.testing.assert_allclose(np.abs(actual)**2, np.abs(field)**2, rtol=2e-5, atol=2e-6)


def test_kerr_raman_rejects_empty_and_noncontiguous_input(cuda):
    with pytest.raises(ValueError, match="empty"):
        kerr_raman_(cuda.empty(0, dtype=cuda.complex64, device="cuda"), 0.005, 0.01)
    field = cuda.ones(10, dtype=cuda.complex64, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        kerr_raman_(field[::2], 0.005, 0.01)
