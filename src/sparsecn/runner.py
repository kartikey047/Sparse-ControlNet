"""GPU side of the experiment: Stable Diffusion 1.5 + Canny ControlNet, and the DINO structure metric.

Needs diffusers, transformers and a CUDA device.
"""

import time

import numpy as np
import torch

from .diffusers_adapter import gated_denoise, step_infos_from_scheduler
from .settings import setting_mask

SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"
NEGATIVE_PROMPT = "low quality, blurry, distorted"


class SDRunner:
    """SD1.5 + ControlNet in fp16 with a gated denoising loop.

    `generate(prompt, edges, condition, seed)` runs one condition. Initial latents depend only on the
    seed, so every condition (and the stock diffusers pipeline) can start from identical noise.
    """

    def __init__(
        self,
        sd_id=SD_ID,
        controlnet_id=CONTROLNET_ID,
        steps=20,
        guidance=7.5,
        size=512,
        negative_prompt=NEGATIVE_PROMPT,
        device="cuda",
    ):
        from diffusers import (
            ControlNetModel,
            DDIMScheduler,
            StableDiffusionControlNetPipeline,
        )

        self.device = device
        self.steps = steps
        self.guidance = guidance
        self.size = size
        self.negative_prompt = negative_prompt

        controlnet = ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=torch.float16
        )
        load_args = dict(
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
        )
        try:
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                sd_id, variant="fp16", **load_args
            )
        except Exception as err:  # some mirrors have no fp16 variant files
            print(
                "fp16 variant not available, loading default weights:", repr(err)[:200]
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                sd_id, **load_args
            )

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.step_infos = step_infos_from_scheduler(self.pipe.scheduler, steps)
        self._embed_cache = {}

    def _prompt_embeds(self, prompt):
        if prompt not in self._embed_cache:
            if len(self._embed_cache) > 64:
                self._embed_cache.clear()
            with torch.no_grad():
                cond, uncond, *_ = self.pipe.encode_prompt(
                    prompt, self.device, 1, True, self.negative_prompt
                )
            self._embed_cache[prompt] = (cond, uncond)
        return self._embed_cache[prompt]

    def initial_latents(self, seed):
        """Standard normal latents from a CPU generator, so they are the same on every machine."""
        generator = torch.Generator("cpu").manual_seed(int(seed))
        shape = (1, 4, self.size // 8, self.size // 8)
        return torch.randn(shape, generator=generator).to(self.device, torch.float16)

    def _control_tensor(self, edges):
        if (
            edges is None
        ):  # the "none" condition never reads it, but the loop still wants a tensor
            return torch.zeros(
                1, 3, self.size, self.size, device=self.device, dtype=torch.float16
            )
        three_channel = np.repeat(edges[None].astype(np.float32), 3, axis=0)
        return torch.from_numpy(three_channel)[None].to(self.device, torch.float16)

    def generate(self, prompt, edges, condition, seed, reuse="none"):
        """Generate one image.

        Returns the image as float32 (H, W, 3) in [0, 1] and a dict with timing and ControlNet cost.
        `condition` is any name accepted by `settings.setting_mask`.
        """
        cond, uncond = self._prompt_embeds(prompt)
        mask = setting_mask(condition, self.step_infos)
        step_infos_from_scheduler(
            self.pipe.scheduler, self.steps
        )  # resets the scheduler timesteps
        sigma0 = getattr(self.pipe.scheduler, "init_noise_sigma", 1.0)
        latents = (self.initial_latents(seed) * sigma0).to(torch.float16)

        torch.cuda.synchronize()
        start = time.perf_counter()
        latents, report = gated_denoise(
            self.pipe.unet,
            self.pipe.controlnet,
            self.pipe.scheduler,
            latents,
            cond,
            self._control_tensor(edges),
            mask,
            uncond_embeds=uncond,
            guidance_scale=self.guidance,
            reuse=reuse,
        )
        torch.cuda.synchronize()
        denoise_seconds = time.perf_counter() - start

        with torch.no_grad():
            decoded = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor
            ).sample
        image = (
            (decoded / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        )

        cost = {
            "gen_s": denoise_seconds,
            "control_s": report.control_time_s,
            "control_evals": report.control_evals,
            "n_steps": report.n_steps,
            "peak_vram_mb": (report.peak_vram_bytes or 0) / 2**20,
        }
        return image, cost

    def reference_pipeline_image(self, prompt, edges, seed):
        """Same image from the stock diffusers pipeline, used to check that our loop matches it."""
        from PIL import Image

        control_image = Image.fromarray(
            np.repeat((edges * 255).astype(np.uint8)[..., None], 3, axis=2)
        )
        result = self.pipe(
            prompt,
            image=control_image,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            negative_prompt=self.negative_prompt,
            latents=self.initial_latents(seed),
            output_type="np",
            height=self.size,
            width=self.size,
        )
        return result.images[0].astype(np.float32)


class DinoStructure:
    """Layout distance between two images, in the spirit of Splice (Tumanyan et al.).

    Compares the self-similarity matrices of DINO ViT patch features, which ignore colour and texture
    but react to layout. Lower means more similar structure. Uses the last-layer patch tokens of
    facebook/dino-vits8 at 224x224 (a 28x28 patch grid).
    """

    def __init__(self, model_id="facebook/dino-vits8", device="cuda"):
        from transformers import ViTModel

        self.device = device
        self.model = (
            ViTModel.from_pretrained(model_id, add_pooling_layer=False)
            .to(device)
            .half()
            .eval()
        )
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def self_similarity(self, img01):
        """Patch-by-patch cosine similarity matrix for an (H, W, 3) image in [0, 1]."""
        x = (
            torch.from_numpy(img01.astype(np.float32))
            .permute(2, 0, 1)[None]
            .to(self.device)
        )
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False
        )
        with torch.no_grad():
            tokens = self.model(
                pixel_values=((x - self.mean) / self.std).half()
            ).last_hidden_state
        patches = torch.nn.functional.normalize(
            tokens[0, 1:].float(), dim=-1
        )  # drop the CLS token
        return patches @ patches.T

    @staticmethod
    def distance(sim_a, sim_b):
        return float((sim_a - sim_b).abs().mean())
