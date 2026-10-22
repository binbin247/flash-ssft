"""cuFFT plus fused Triton pointwise LLE steps and CUDA Graph execution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import Callable

import numpy as np
import torch

from ._validation import check_input
from .operators import kerr_, linear_and_pump_


@dataclass(frozen=True)
class LLEParams:
    n_modes: int = 1024
    pump_f: float = 2.5
    detuning_alpha: float = 5.0
    d2: float = 0.02
    h: float = 0.005
    d3: float = 0.0

def linear_coefficients(params: LLEParams) -> tuple[np.ndarray, complex]:
    """Precompute linear factors and the first-order zero-mode pump increment N*F*h."""
    n = params.n_modes
    mu = np.fft.fftfreq(n, d=1.0 / n)
    lam = -1.0 - 1j * (
        params.detuning_alpha
        + params.d2 * mu**2 / 2.0
        + params.d3 * mu**3 / 6.0
    )
    h = params.h
    linear = np.exp(lam * h)
    pump = n * params.pump_f * h
    return linear, complex(pump)

def lle_step(
    angular: torch.Tensor,
    linear: torch.Tensor,
    pump: complex,
    h: float,
) -> torch.Tensor:
    """Advance a time-domain field via FFT -> linear/pump -> IFFT -> Kerr."""
    spectrum = torch.fft.fft(angular)
    linear_and_pump_(spectrum, linear, pump)
    angular = torch.fft.ifft(spectrum)
    return kerr_(angular, h)


class CudaGraphStepper:
    """Replay GPU step chunks; use each instance sequentially on one CUDA stream."""

    def __init__(
            self,
            seed: torch.Tensor,
            step: Callable[[torch.Tensor],torch.Tensor],
            chunk: int
            ):
        self.step = step
        self.chunk = chunk
        self.state = seed.clone()
        self.graph = torch.cuda.CUDAGraph()

        # Warm up on a temporary stream before automatic graph capture.
        torch.cuda.synchronize(self.state.device)
        with torch.cuda.stream(torch.cuda.Stream(device=self.state.device)):
            scratch = self.state
            for _ in range(2):
                scratch = step(scratch)
        torch.cuda.synchronize(self.state.device)

        with torch.cuda.graph(self.graph):
            next_state = self.state
            for _ in range(chunk):
                next_state = step(next_state)
            self.state.copy_(next_state)

    def run(self, initial_state: torch.Tensor, steps: int) -> torch.Tensor:
        if initial_state.shape != self.state.shape or initial_state.device != self.state.device:
            raise ValueError("initial_state shape and device must match the captured graph")

        self.state.copy_(initial_state)
        for _ in range(steps // self.chunk):
            self.graph.replay()
        state = self.state
        for _ in range(steps % self.chunk):
            state = self.step(state)
        return state.clone()


_GRAPH_CACHE: dict[tuple[LLEParams, int, int], CudaGraphStepper] = {}
_GRAPH_CACHE_LOCK = RLock()


def clear_graph_cache() -> None:
    """Release this process's cached CUDA Graph runners."""
    with _GRAPH_CACHE_LOCK:
        _GRAPH_CACHE.clear()


def simulate_gpu(
    psi0: np.ndarray,
    params: LLEParams,
    steps: int,
    *,
    graph_chunk: int = 16,
) -> np.ndarray:
    """Evolve a time-domain field on the GPU using cached CUDA Graphs."""
    field = check_input(psi0, params.n_modes, steps)

    device = torch.device("cuda", torch.cuda.current_device())
    field_gpu = torch.as_tensor(field, dtype=torch.complex64, device=device).contiguous()
    if steps == 0:
        return field_gpu.cpu().numpy().copy()

    graph_chunk = min(graph_chunk, steps)
    key = (params, graph_chunk, device.index)
    with _GRAPH_CACHE_LOCK:
        runner = _GRAPH_CACHE.get(key)
        if runner is None:
            linear, pump = linear_coefficients(params)
            linear_gpu = torch.as_tensor(linear, dtype=torch.complex64, device=device)
            step = partial(lle_step, linear=linear_gpu, pump=pump, h=params.h)
            runner = CudaGraphStepper(field_gpu, step, graph_chunk)
            _GRAPH_CACHE[key] = runner
        angular = runner.run(field_gpu, steps)
        return angular.cpu().numpy().copy()
