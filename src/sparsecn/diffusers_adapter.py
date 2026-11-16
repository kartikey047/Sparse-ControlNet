"""Gated denoising on real Diffusers components (UNet2DConditionModel + ControlNetModel + a scheduler).

`gated_denoise` works directly on the model and scheduler objects instead of a pipeline, so the tests
can run it on tiny randomly initialised models with no weights or network. On a skipped step the
ControlNet forward is simply not called (a forward hook counts the real calls) and the UNet runs
without extra residuals. `gated_pipeline_call` is a thin wrapper for StableDiffusionControlNetPipeline
objects; the full-ControlNet path was checked against the stock pipeline with real SD1.5 weights.
"""

import time

import numpy as np
import torch

from .sampler import CostReport
from .timesteps import build_step_infos


def step_infos_from_scheduler(scheduler, n_steps):
    """Set the scheduler's timesteps and describe each step as a StepInfo.

    Sigmas come from `scheduler.sigmas` when it has them (Euler, Karras, ...). Otherwise they are
    derived from alphas_cumprod as sqrt((1 - abar) / abar) at the scheduler's timesteps.
    """
    scheduler.set_timesteps(n_steps)
    timesteps = scheduler.timesteps.detach().cpu().numpy().astype(float)

    scheduler_sigmas = getattr(scheduler, "sigmas", None)
    if scheduler_sigmas is not None and len(scheduler_sigmas) >= len(timesteps):
        sigmas = scheduler_sigmas.detach().cpu().numpy().astype(float)[: len(timesteps)]
    else:
        alphas_cumprod = scheduler.alphas_cumprod.detach().cpu().numpy()
        abar = alphas_cumprod[
            np.clip(timesteps.astype(int), 0, len(alphas_cumprod) - 1)
        ]
        sigmas = np.sqrt((1 - abar) / abar)
    return build_step_infos(timesteps, sigmas)


class _CallCounter:
    """Forward hooks that count real calls of a module and add up the time spent in them."""

    def __init__(self, module):
        self.calls = 0
        self.time_s = 0.0
        self._started = 0.0
        self._on_cuda = torch.cuda.is_available() and next(module.parameters()).is_cuda
        self._pre_handle = module.register_forward_pre_hook(self._before)
        self._post_handle = module.register_forward_hook(self._after)

    def _before(self, *_):
        self.calls += 1
        self._started = time.perf_counter()

    def _after(self, *_):
        if self._on_cuda:
            torch.cuda.synchronize()
        self.time_s += time.perf_counter() - self._started

    def close(self):
        self._pre_handle.remove()
        self._post_handle.remove()


def gated_denoise(
    unet,
    controlnet,
    scheduler,
    latents,
    encoder_hidden_states,
    control_image,
    mask,
    *,
    uncond_embeds=None,
    guidance_scale=1.0,
    conditioning_scale=1.0,
    reuse="none",
    generator=None,
    scheduler_kwargs=None,
):
    """Denoise `latents`, running ControlNet only on the steps where `mask` is True.

    - A skipped step does not call ControlNet and passes no residuals to the UNet.
    - reuse="hold" (an approximation, kept as a separate method) feeds the last computed residuals
      to the skipped steps. ControlNet is still not called on them.
    - With classifier-free guidance (guidance_scale > 1, needs `uncond_embeds`) ControlNet and the UNet
      see the doubled batch, so an executed step costs one ControlNet call, as in the stock pipeline.

    Call `step_infos_from_scheduler(scheduler, n)` first; it sets the timesteps. Returns
    (latents, CostReport).
    """
    if reuse not in ("none", "hold"):
        raise ValueError(f"reuse must be 'none' or 'hold', got {reuse!r}")
    mask = np.asarray(mask, dtype=bool)
    use_cfg = guidance_scale > 1.0
    if use_cfg and uncond_embeds is None:
        raise ValueError("guidance_scale>1 requires uncond_embeds")
    n_steps = len(scheduler.timesteps)
    if len(mask) != n_steps:
        raise ValueError(
            f"mask length {len(mask)} != scheduler steps {n_steps}; "
            "call step_infos_from_scheduler first"
        )

    embeds = (
        torch.cat([uncond_embeds, encoder_hidden_states])
        if use_cfg
        else encoder_hidden_states
    )
    control = torch.cat([control_image, control_image]) if use_cfg else control_image
    step_kwargs = dict(scheduler_kwargs or {})
    if generator is not None:
        step_kwargs["generator"] = generator

    on_cuda = latents.device.type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    counter = _CallCounter(controlnet)
    last_residuals = None
    reused = 0
    start = time.perf_counter()

    try:
        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps):
                model_input = torch.cat([latents, latents]) if use_cfg else latents
                model_input = scheduler.scale_model_input(model_input, t)

                down_residuals = mid_residual = None
                if mask[i]:
                    down_residuals, mid_residual = controlnet(
                        model_input,
                        t,
                        encoder_hidden_states=embeds,
                        controlnet_cond=control,
                        conditioning_scale=conditioning_scale,
                        return_dict=False,
                    )
                    if reuse == "hold":
                        last_residuals = (down_residuals, mid_residual)
                elif reuse == "hold" and last_residuals is not None:
                    if last_residuals[0][0].shape[0] != model_input.shape[0]:
                        raise ValueError("batch size changed; cannot reuse residuals")
                    down_residuals, mid_residual = last_residuals
                    reused += 1

                noise = unet(
                    model_input,
                    t,
                    encoder_hidden_states=embeds,
                    down_block_additional_residuals=down_residuals,
                    mid_block_additional_residual=mid_residual,
                ).sample
                if use_cfg:
                    noise_uncond, noise_cond = noise.chunk(2)
                    noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                latents = scheduler.step(noise, t, latents, **step_kwargs).prev_sample
    finally:
        counter.close()

    report = CostReport(
        n_steps=n_steps,
        control_evals=counter.calls,
        backbone_evals=n_steps,
        reused_steps=reused,
        control_time_s=counter.time_s,
        total_time_s=time.perf_counter() - start,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated()) if on_cuda else None,
        ran_mask=mask.tolist(),
        relative_control_compute=counter.calls / n_steps,
    )
    return latents, report


def gated_pipeline_call(
    pipe,
    prompt,
    control_image,
    mask,
    n_steps=20,
    seed=0,
    guidance_scale=7.5,
    conditioning_scale=1.0,
    reuse="none",
):
    """`gated_denoise` for a StableDiffusionControlNetPipeline-like object.

    Uses pipe.unet / controlnet / scheduler, `encode_prompt` and, if present, `prepare_image`. The
    experiment code in `runner.py` does the same job with more control over seeds and caching.
    """
    device = pipe.unet.device
    pipe_infos = step_infos_from_scheduler(
        pipe.scheduler, n_steps
    )  # also sets the timesteps
    if len(mask) != len(pipe_infos):
        raise ValueError("mask length != n_steps")

    use_cfg = guidance_scale > 1.0
    cond, uncond = pipe.encode_prompt(prompt, device, 1, use_cfg)[:2]

    config = pipe.unet.config
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (1, config.in_channels, config.sample_size, config.sample_size),
        generator=generator,
    ).to(device)
    latents = latents * getattr(pipe.scheduler, "init_noise_sigma", 1.0)

    side = config.sample_size * getattr(pipe, "vae_scale_factor", 8)
    if hasattr(pipe, "prepare_image"):
        control = pipe.prepare_image(
            control_image, side, side, 1, 1, device, pipe.controlnet.dtype
        )
    else:
        control = control_image

    return gated_denoise(
        pipe.unet,
        pipe.controlnet,
        pipe.scheduler,
        latents,
        cond,
        control,
        mask,
        uncond_embeds=uncond if use_cfg else None,
        guidance_scale=guidance_scale,
        conditioning_scale=conditioning_scale,
        reuse=reuse,
    )
