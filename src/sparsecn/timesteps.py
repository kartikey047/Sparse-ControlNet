"""Sampling-step bookkeeping shared by the toy sampler and the Diffusers adapter.

Terminology used everywhere in this package:

* A step *index* i = 0..N-1 is the position in the sampling loop; i = 0 is the first step.
* In normal reverse diffusion the first steps have high noise (large sigma) and the last steps have
  low noise. So "early" and "high noise" describe the same steps for a monotone schedule, but they are
  different selection rules: `early` picks by index, `high_noise` picks by sigma. They only diverge for
  non-monotone or custom sigma sequences, which is why both exist and are tested separately.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepInfo:
    index: int  # position in the sampling loop (0 = first step)
    timestep: float  # timestep value passed to the network
    sigma: float  # noise level (noise std relative to signal)
    sigma_rank: int  # 0 = highest sigma among the N steps
    n_steps: int


def linear_ddpm_sigmas(num_train=1000, beta_start=1e-4, beta_end=2e-2):
    """Sigma for every training timestep of the linear-beta DDPM schedule."""
    betas = np.linspace(beta_start, beta_end, num_train)
    alphas_cumprod = np.cumprod(1 - betas)
    return np.sqrt((1 - alphas_cumprod) / alphas_cumprod)


def build_step_infos(timesteps, sigmas):
    """StepInfo list from per-step timesteps and sigmas (ranks are by descending sigma)."""
    order = np.argsort(-sigmas, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(sigmas))
    return [
        StepInfo(i, float(timesteps[i]), float(sigmas[i]), int(rank[i]), len(timesteps))
        for i in range(len(timesteps))
    ]


def make_step_infos(n_steps, num_train=1000, sigmas_full=None, timesteps=None):
    """Step list for a DDIM-style sampler.

    By default the timesteps are evenly spaced and descending, with sigmas from the linear-beta DDPM
    table. Pass `timesteps` / `sigmas_full` to use another scheduler's values (for a Diffusers
    scheduler see `diffusers_adapter.step_infos_from_scheduler`).
    """
    sigma_table = (
        linear_ddpm_sigmas(num_train)
        if sigmas_full is None
        else np.asarray(sigmas_full, float)
    )
    if timesteps is None:
        timesteps = np.linspace(len(sigma_table) - 1, 0, n_steps).round()
    timesteps = np.asarray(timesteps, float)
    sigmas = np.interp(timesteps, np.arange(len(sigma_table)), sigma_table)
    return build_step_infos(timesteps, sigmas)


def sigma_order(infos):
    """'high_to_low' (the usual case), 'low_to_high' or 'non_monotone'."""
    steps = np.diff([info.sigma for info in infos])
    if np.all(steps <= 0):
        return "high_to_low"
    if np.all(steps >= 0):
        return "low_to_high"
    return "non_monotone"
