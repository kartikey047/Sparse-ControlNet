"""Toy DDIM sampler with a gated control branch and honest cost accounting.

Skipped steps really skip the branch: it is not called at all, and the cost report counts the calls
that were actually made.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .timesteps import StepInfo


@dataclass
class CostReport:
    n_steps: int
    control_evals: int  # actual ControlNet forward calls
    backbone_evals: int
    reused_steps: int = 0
    control_time_s: float = 0.0
    total_time_s: float = 0.0
    peak_vram_bytes: int = None
    ran_mask: list = field(default_factory=list)
    relative_control_compute: float = 0.0  # control_evals / n_steps


class CountingModule:
    """Wraps a module and counts how often it is called."""

    def __init__(self, module):
        self.module = module
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.module(*args, **kwargs)


def ddim_alpha_bar(infos):
    """Cumulative alpha per step plus a final 1.0, from sigma^2 = (1 - ab) / ab."""
    return np.array([1.0 / (1.0 + info.sigma**2) for info in infos] + [1.0])


@torch.no_grad()
def sample(
    backbone,
    control_branch,
    control_image,
    mask,
    infos,
    shape,
    seed=0,
    reuse="none",
    device="cpu",
):
    """Deterministic DDIM sampling (eta = 0). Returns (x0, CostReport).

    mask[i] True runs the control branch at step i, False skips it. What a skipped step gets depends on
    `reuse`:
      none    no residual at all
      hold    the last computed residual again (approximate)
      linear  a blend of the last residual and one computed for the next enabled step on the current
              latents (approximate). That needs an extra branch call, which is counted.
    """
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(infos):
        raise ValueError(f"mask length {len(mask)} != steps {len(infos)}")
    if reuse not in ("none", "hold", "linear"):
        raise ValueError(reuse)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=generator).to(device)
    alpha_bar = ddim_alpha_bar(infos)
    branch = CountingModule(control_branch) if control_branch is not None else None
    use_cuda = device != "cpu" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    enabled_steps = np.flatnonzero(mask)
    last_residual = None
    reused = 0
    control_time = 0.0
    start = time.perf_counter()

    def timed_branch_call(timestep):
        nonlocal control_time
        t_branch = time.perf_counter()
        residual = branch(
            x, torch.full((shape[0],), timestep, device=device), control_image
        )
        control_time += time.perf_counter() - t_branch
        return residual

    for i, info in enumerate(infos):
        t = torch.full((shape[0],), info.timestep, device=device)
        residual = None

        if mask[i] and branch is not None:
            residual = timed_branch_call(info.timestep)
            last_residual = residual
        elif reuse == "hold" and last_residual is not None and branch is not None:
            residual = last_residual
            reused += 1
        elif reuse == "linear" and last_residual is not None and branch is not None:
            later = enabled_steps[enabled_steps > i]
            if len(later) == 0:
                residual = last_residual
            else:
                previous = enabled_steps[enabled_steps < i][-1]
                weight = (i - previous) / (later[0] - previous)
                next_residual = timed_branch_call(infos[later[0]].timestep)
                if next_residual.shape != last_residual.shape:
                    raise ValueError("residual shape mismatch in interpolation")
                residual = (1 - weight) * last_residual + weight * next_residual
            reused += 1

        eps = backbone(x, t, residual)
        a_t, a_next = alpha_bar[i], alpha_bar[i + 1]
        x0_estimate = (x - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t)
        x = np.sqrt(a_next) * x0_estimate + np.sqrt(1 - a_next) * eps

    n_control_calls = branch.calls if branch else 0
    report = CostReport(
        n_steps=len(infos),
        control_evals=n_control_calls,
        backbone_evals=len(infos),
        reused_steps=reused,
        control_time_s=control_time,
        total_time_s=time.perf_counter() - start,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated()) if use_cuda else None,
        ran_mask=mask.tolist(),
        relative_control_compute=n_control_calls / len(infos),
    )
    return x, report
