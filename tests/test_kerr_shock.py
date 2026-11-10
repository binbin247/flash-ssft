"""Numerical, validation, and CUDA Graph checks for Kerr/self-steepening."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from flashlle.operators import kerr_, kerr_shock_


@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Kerr/self-steepening checks")
    return torch.device("cuda")


def frequencies(n, device, dtype=torch.float64):
    return 2 * torch.pi * torch.fft.fftfreq(n, d=2 * np.pi / n, device=device, dtype=dtype)


def smooth_field(n):
    t = 2 * np.pi * np.arange(n) / n
    return ((0.8 + 0.15 * np.cos(t)) * np.exp(0.2j * np.sin(2*t))).astype(np.complex64)


def numpy_step(field, w, h, omega0):
    field = np.asarray(field, dtype=np.complex128)
    b = field * np.exp(1j * h * np.abs(field)**2)
    derivative = np.fft.ifft(1j * w * np.fft.fft(np.abs(b)**2 * b))
    return b - h / omega0 * derivative


@pytest.mark.parametrize("n", [1, 2, 255, 256, 257, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kerr_shock_matches_numpy_in_place(cuda, n, dtype):
    rng = np.random.default_rng(42)
    seed = (0.3 * (rng.normal(size=n) + 1j * rng.normal(size=n))).astype(np.complex64)
    angular = torch.tensor(seed, device=cuda)
    w = frequencies(n, cuda, dtype)
    original_w = w.clone()
    pointer = angular.data_ptr()
    expected = numpy_step(seed, w.cpu().numpy(), 0.01, 20.0)
    assert kerr_shock_(angular, w, 0.01, 20.0) is angular
    assert angular.data_ptr() == pointer
    np.testing.assert_allclose(angular.cpu().numpy(), expected, rtol=4e-6, atol=3e-7)
    torch.testing.assert_close(w, original_w, rtol=0, atol=0)


@pytest.mark.parametrize("n,mode", [(257, 3), (257, -3), (255, 127), (256, -128)])
def test_kerr_shock_single_mode_sign_and_nyquist(cuda, n, mode):
    t = 2 * np.pi * np.arange(n) / n
    intensity, h, omega0 = 0.49, 0.03, 100.0
    seed = np.sqrt(intensity) * np.exp(1j * mode * t)
    expected = seed * np.exp(1j * h * intensity) * (1 - 1j * h * mode * intensity / omega0)
    angular = torch.tensor(seed, dtype=torch.complex64, device=cuda)
    kerr_shock_(angular, frequencies(n, cuda), h, omega0)
    np.testing.assert_allclose(angular.cpu().numpy(), expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("value", [0j, 0.6 + 0.8j])
def test_kerr_shock_constant_field(cuda, value):
    angular = torch.full((257,), value, dtype=torch.complex64, device=cuda)
    kerr_shock_(angular, frequencies(257, cuda), 0.1, 10.0)
    expected = value * np.exp(0.1j * abs(value)**2)
    np.testing.assert_allclose(angular.cpu().numpy(), expected, rtol=2e-6, atol=1e-7)


def test_kerr_shock_zero_step_and_disabled_shock(cuda):
    seed = torch.tensor(smooth_field(257), device=cuda)
    w = frequencies(257, cuda)
    actual = seed.clone()
    assert kerr_shock_(actual, w, 0.0, 10.0) is actual
    torch.testing.assert_close(actual, seed, rtol=0, atol=0)
    actual = kerr_shock_(seed.clone(), w, 0.1, float("inf"))
    torch.testing.assert_close(actual, kerr_(seed.clone(), 0.1), rtol=0, atol=0)


def test_kerr_shock_first_order_convergence(cuda):
    n, distance, omega0 = 64, 0.2, 5.0
    seed = smooth_field(n)
    w = frequencies(n, cuda)
    wn = w.cpu().numpy()

    # Independent high-accuracy solution of the combined semidiscrete equation.
    def rhs(a):
        q = np.abs(a)**2 * a
        return 1j*q - np.fft.ifft(1j*wn*np.fft.fft(q)) / omega0

    reference = seed.astype(np.complex128)
    dt = distance / 1024
    for _ in range(1024):
        k1 = rhs(reference)
        k2 = rhs(reference + dt*k1/2)
        k3 = rhs(reference + dt*k2/2)
        k4 = rhs(reference + dt*k3)
        reference += dt*(k1 + 2*k2 + 2*k3 + k4)/6

    errors = []
    for steps in (8, 16, 32):
        actual = torch.tensor(seed, device=cuda)
        for _ in range(steps):
            kerr_shock_(actual, w, distance / steps, omega0)
        errors.append(np.linalg.norm(actual.cpu().numpy() - reference))
    ratios = np.array(errors[:-1]) / errors[1:]
    assert np.all((ratios > 1.8) & (ratios < 2.2)), (errors, ratios)
    assert np.max(np.abs(np.abs(reference) - np.abs(seed))) > 1e-3


@pytest.mark.parametrize("n", [256, 257])
@pytest.mark.parametrize("omega0", [20.0, float("inf")], ids=["shock", "kerr-only"])
def test_kerr_shock_graph_replay_and_new_input(cuda, n, omega0):
    seed = torch.tensor(smooth_field(n), device=cuda)
    state = seed.clone()
    w = frequencies(n, cuda)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            kerr_shock_(state, w, 0.005, omega0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            kerr_shock_(state, w, 0.005, omega0)
    for scale in (1.0, 0.7):
        initial = seed * scale
        state.copy_(initial)
        eager = initial.clone()
        for _ in range(4):
            graph.replay()
        for _ in range(12):
            kerr_shock_(eager, w, 0.005, omega0)
        torch.testing.assert_close(state, eager, rtol=1e-6, atol=1e-7)
        reference = initial.cpu().numpy()
        for _ in range(12):
            reference = numpy_step(reference, w.cpu().numpy(), 0.005, omega0)
        np.testing.assert_allclose(state.cpu().numpy(), reference, rtol=8e-6, atol=1e-6)


@pytest.mark.parametrize("case", [
    "empty", "angular_cpu", "angular_dtype", "angular_ndim", "angular_stride",
    "w_type", "w_dtype", "w_shape", "w_ndim", "w_cpu", "w_stride",
    "h_nan", "h_inf", "omega0_zero", "omega0_negative", "omega0_nan", "omega0_negative_inf",
])
def test_kerr_shock_rejects_invalid_input_before_mutation(cuda, case):
    angular = torch.ones(16, dtype=torch.complex64, device=cuda)
    w = frequencies(16, cuda)
    h, omega0 = 0.1, 10.0
    if case == "empty":
        angular, w = angular[:0], w[:0]
    elif case == "angular_cpu":
        angular = angular.cpu()
    elif case == "angular_dtype":
        angular = angular.to(torch.complex128)
    elif case == "angular_ndim":
        angular = angular.reshape(4, 4)
    elif case == "angular_stride":
        angular, w = angular[::2], w[:8]
    elif case == "w_type":
        w = [0.0] * 16
    elif case == "w_dtype":
        w = w.to(torch.int64)
    elif case == "w_shape":
        w = w[:-1]
    elif case == "w_ndim":
        w = w.reshape(4, 4)
    elif case == "w_cpu":
        w = w.cpu()
    elif case == "w_stride":
        w = frequencies(32, cuda)[::2]
    elif case == "h_nan":
        h = float("nan")
    elif case == "h_inf":
        h = float("inf")
    elif case == "omega0_zero":
        omega0 = 0.0
    elif case == "omega0_negative":
        omega0 = -1.0
    elif case == "omega0_nan":
        omega0 = float("nan")
    elif case == "omega0_negative_inf":
        omega0 = -float("inf")
    original = angular.clone()
    with pytest.raises((TypeError, ValueError)):
        kerr_shock_(angular, w, h, omega0)
    torch.testing.assert_close(angular, original, rtol=0, atol=0)


def test_kerr_shock_optical_frequency_scaling(cuda):
    seed = torch.tensor(smooth_field(257), device=cuda)
    w = frequencies(257, cuda)
    reference = kerr_shock_(seed.clone(), w, 0.01, 10.0)
    actual = kerr_shock_(seed.clone(), w * 1e14, 0.01, 1e15)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
