"""Independent regression checks for the existing Kerr and linear/pump operators."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton operator checks")
    return torch


def _relative_l2(actual, reference):
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-30))


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
