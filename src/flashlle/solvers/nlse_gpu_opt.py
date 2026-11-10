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
from ..operators._validation import check_block, check_num_warps


def nlse_step(
    field: torch.Tensor,
    linear: torch.Tensor,
    w: torch.Tensor,
    h: float,
    omega0: float,
    *,
    block: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """One Lie step: linear propagation, exact Kerr, then Euler self-steepening.

    linear contains the full dz propagation factors, w=2*pi*fftfreq(N, dt),
    and h=gamma0*dz scales both nonlinear terms. Returns a new time-domain
    field and preserves the caller's input. omega0=+inf disables shock.
    """
    # Keep the linear operator in the frequency domain, where dispersion and
    # loss are represented by one elementwise propagation factor.
    block = check_block(block)
    num_warps = check_num_warps(num_warps)
    spectrum = torch.fft.fft(field)
    disp_and_loss_(spectrum, linear, block=block, num_warps=num_warps)
    # The nonlinear operator works in the time domain and returns a new field.
    return kerr_shock_(
        torch.fft.ifft(spectrum), w, h, omega0,
        block=block, num_warps=num_warps,
    )


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

        # Warm up on a temporary stream before automatic graph capture. This
        # lets CUDA initialize the kernels and memory allocations first.
        torch.cuda.synchronize(self.state.device)
        with torch.cuda.stream(torch.cuda.Stream(device=self.state.device)):
            scratch = self.state
            for _ in range(2):
                scratch = step(scratch)
        torch.cuda.synchronize(self.state.device)

        # Capture one fixed-size batch. The graph reuses self.state as its
        # input and output buffer on every replay.
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

        # Restore the caller's input before replaying the captured graph.
        self.state.copy_(initial_state)
        # A graph always advances exactly self.chunk steps, so handle the
        # complete chunks with replay and execute the remaining steps normally.
        full_chunks, remainder = divmod(steps, self.chunk)
        for _ in range(full_chunks):
            self.graph.replay()
        state = self.state
        for _ in range(remainder):
            state = self.step(state)
        return state.clone()


# A runner owns mutable CUDA buffers, so cached runners must be used serially.
_GRAPH_CACHE: dict[tuple[NLSEParams, int, int, int, int], CudaGraphStepper] = {}
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
    block: int = 256,
    num_warps: int = 4,
) -> np.ndarray:
    """Propagate to z=steps*dz; return a new NumPy complex64 time-domain field."""
    field = check_input(a0, params.n_samples, steps)
    check_graph_chunk(graph_chunk)
    block = check_block(block)
    num_warps = check_num_warps(num_warps)
    if steps == 0 or params.dz == 0:
        return np.array(field, dtype=np.complex64, copy=True)

    # Keep the propagation state on one CUDA device for the whole run.
    device = torch.device("cuda", torch.cuda.current_device())
    field_gpu = torch.as_tensor(field, dtype=torch.complex64, device=device).contiguous()
    # The captured graph has a fixed length; a shorter graph avoids doing
    # unnecessary work when the requested propagation is shorter than chunk.
    graph_chunk = min(graph_chunk, steps)
    key = (params, graph_chunk, block, num_warps, device.index)
    with _GRAPH_CACHE_LOCK:
        runner = _GRAPH_CACHE.get(key)
        if runner is None:
            # These tensors and the bound step function become part of the
            # cached graph and are reused for later calls with the same key.
            linear = linear_coefficients(params)
            linear_gpu = torch.as_tensor(linear, dtype=torch.complex64, device=device)
            w = torch.as_tensor(angular_frequencies(params), dtype=torch.float64, device=device)
            step = partial(nlse_step, linear=linear_gpu, w=w,
                           h=params.gamma0 * params.dz, omega0=params.omega0,
                           block=block, num_warps=num_warps)
            runner = CudaGraphStepper(field_gpu, step, graph_chunk)
            _GRAPH_CACHE[key] = runner
        final_field = runner.run(field_gpu, steps)
        return final_field.cpu().numpy().copy()
