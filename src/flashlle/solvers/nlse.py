"""NLSE propagation using existing dispersion/loss and Kerr/shock operators."""

from __future__ import annotations

from functools import partial
from threading import RLock
from typing import Callable

import numpy as np
import torch

from ._validation import check_graph_chunk, check_input, check_steps
from .params import NLSEParams, angular_frequencies, linear_coefficients
from ..operators import disp_and_loss_, kerr_shock_


def nlse_step(
    field: torch.Tensor,
    linear: torch.Tensor,
    w: torch.Tensor,
    h: float,
    omega0: float,
) -> torch.Tensor:
    """One Lie step: linear propagation, exact Kerr, then Euler self-steepening.

    linear contains the full dz propagation factors, w=2*pi*fftfreq(N, dt),
    and h=gamma0*dz scales both nonlinear terms. Returns a new time-domain
    field and preserves the caller's input. omega0=+inf disables shock.
    """
    spectrum = torch.fft.fft(field)
    disp_and_loss_(spectrum, linear)
    field = torch.fft.ifft(spectrum)
    return kerr_shock_(field, w, h, omega0)


class CudaGraphStepper:
    """Replay GPU step chunks; use each instance sequentially on one CUDA stream."""

    def __init__(
        self,
        seed: torch.Tensor,
        step: Callable[[torch.Tensor], torch.Tensor],
        chunk: int,
    ):
        check_graph_chunk(chunk)
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
        check_steps(steps)
        if (initial_state.shape != self.state.shape or initial_state.device != self.state.device
                or initial_state.dtype != self.state.dtype):
            raise ValueError("initial_state shape, device and dtype must match the captured graph")

        self.state.copy_(initial_state)
        for _ in range(steps // self.chunk):
            self.graph.replay()
        state = self.state
        for _ in range(steps % self.chunk):
            state = self.step(state)
        return state.clone()


_GRAPH_CACHE: dict[tuple[NLSEParams, int, int], CudaGraphStepper] = {}
_GRAPH_CACHE_LOCK = RLock()


def clear_graph_cache() -> None:
    """Release this process's cached CUDA Graph runners."""
    with _GRAPH_CACHE_LOCK:
        _GRAPH_CACHE.clear()


def simulate_gpu(
    a0: np.ndarray,
    params: NLSEParams,
    steps: int,
    *,
    graph_chunk: int = 16,
) -> np.ndarray:
    """Propagate to z=steps*dz; return a new NumPy complex64 time-domain field."""
    field = check_input(a0, params.n_samples, steps)
    check_graph_chunk(graph_chunk)
    if steps == 0 or params.dz == 0:
        return np.array(field, dtype=np.complex64, copy=True)

    device = torch.device("cuda", torch.cuda.current_device())
    field_gpu = torch.as_tensor(field, dtype=torch.complex64, device=device).contiguous()
    graph_chunk = min(graph_chunk, steps)
    key = (params, graph_chunk, device.index)
    with _GRAPH_CACHE_LOCK:
        runner = _GRAPH_CACHE.get(key)
        if runner is None:
            linear = linear_coefficients(params)
            linear_gpu = torch.as_tensor(linear, dtype=torch.complex64, device=device)
            w = torch.as_tensor(angular_frequencies(params), dtype=torch.float64, device=device)
            step = partial(nlse_step, linear=linear_gpu, w=w,
                           h=params.gamma0 * params.dz, omega0=params.omega0)
            runner = CudaGraphStepper(field_gpu, step, graph_chunk)
            _GRAPH_CACHE[key] = runner
        angular = runner.run(field_gpu, steps)
        return angular.cpu().numpy().copy()
