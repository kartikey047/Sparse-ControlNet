"""Gated ControlNet on REAL Diffusers modules (tiny, randomly initialised; no weights, no network)."""

import numpy as np, pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
from diffusers import (
    UNet2DConditionModel,
    ControlNetModel,
    DDIMScheduler,
    EulerDiscreteScheduler,
)
from sparsecn.diffusers_adapter import step_infos_from_scheduler, gated_denoise
from sparsecn.schedules import make_mask
from sparsecn.timesteps import sigma_order


def _unet():
    return UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=8,
        attention_head_dim=2,
        norm_num_groups=4,
    ).eval()


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    unet = _unet()
    cn = ControlNetModel.from_unet(
        unet, conditioning_embedding_out_channels=(8, 16)
    ).eval()
    # ControlNet zero-inits its output convs => residual == 0 and control would be a no-op. Randomise them so
    # control has a real effect and the tests can tell executed from skipped steps.
    with torch.no_grad():
        for m in list(cn.controlnet_down_blocks) + [cn.controlnet_mid_block]:
            m.weight.normal_(0, 0.5)
            m.bias.normal_(0, 0.1)
    return unet, cn


def _inputs(seed=0, batch=1):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, 4, 16, 16, generator=g),
        torch.randn(batch, 5, 8, generator=g),
        torch.rand(batch, 3, 32, 32, generator=g),
    )


def _run(models, mask_name, n=8, sched=DDIMScheduler, reuse="none", budget=None, **kw):
    unet, cn = models
    s = sched()
    infos = step_infos_from_scheduler(s, n)
    mask = make_mask(mask_name, infos, frac=0.5, budget=budget)
    lat, emb, img = _inputs()
    out, rep = gated_denoise(
        unet, cn, s, lat.clone(), emb, img, mask, reuse=reuse, **kw
    )
    return out, rep, mask, infos


def test_full_calls_N_and_budget_calls_exactly_K(models):
    _, rep, _, _ = _run(models, "full")
    assert rep.control_evals == 8 and rep.relative_control_compute == 1.0
    for K in (0, 1, 3, 8):
        _, rep, mask, _ = _run(models, "budgeted", budget=K)
        assert rep.control_evals == K == mask.sum() and rep.backbone_evals == 8


def test_skipped_steps_never_call_controlnet_and_unet_gets_no_residual(models):
    unet, cn = models
    s = DDIMScheduler()
    step_infos_from_scheduler(s, 6)
    mask = np.array([1, 0, 0, 1, 0, 0], bool)
    lat, emb, img = _inputs()
    seen = []
    h1 = cn.register_forward_pre_hook(lambda m, a: seen.append("cn"))

    def unet_hook(m, args, kwargs):
        seen.append(kwargs.get("down_block_additional_residuals") is not None)

    h2 = unet.register_forward_pre_hook(unet_hook, with_kwargs=True)
    try:
        gated_denoise(unet, cn, s, lat, emb, img, mask)
    finally:
        h1.remove()
        h2.remove()
    assert seen == ["cn", True, False, False, "cn", True, False, False]


def test_none_schedule_equals_plain_unet_and_control_has_effect(models):
    unet, cn = models
    s = DDIMScheduler()
    step_infos_from_scheduler(s, 6)
    lat, emb, img = _inputs()
    out0, rep0 = gated_denoise(unet, cn, s, lat.clone(), emb, img, np.zeros(6, bool))
    assert rep0.control_evals == 0
    ref = lat.clone()
    with torch.no_grad():
        for t in s.timesteps:
            ref = s.step(
                unet(s.scale_model_input(ref, t), t, encoder_hidden_states=emb).sample,
                t,
                ref,
            ).prev_sample
    assert torch.equal(out0, ref)  # bit-identical to a plain no-control UNet loop
    outf, _ = gated_denoise(unet, cn, s, lat.clone(), emb, img, np.ones(6, bool))
    assert (outf - out0).abs().max() > 1e-4  # control really changes the result


def test_zero_initialised_controlnet_equals_no_control():
    torch.manual_seed(1)
    unet = _unet()
    cn = ControlNetModel.from_unet(
        unet, conditioning_embedding_out_channels=(8, 16)
    ).eval()
    s = DDIMScheduler()
    step_infos_from_scheduler(s, 4)
    lat, emb, img = _inputs()
    a, _ = gated_denoise(unet, cn, s, lat.clone(), emb, img, np.ones(4, bool))
    b, _ = gated_denoise(unet, cn, s, lat.clone(), emb, img, np.zeros(4, bool))
    assert torch.allclose(a, b, atol=1e-6)


def test_cfg_one_controlnet_call_per_executed_step(models):
    unet, cn = models
    s = DDIMScheduler()
    infos = step_infos_from_scheduler(s, 6)
    lat, emb, img = _inputs()
    out, rep = gated_denoise(
        unet,
        cn,
        s,
        lat,
        emb,
        img,
        make_mask("budgeted", infos, budget=2),
        uncond_embeds=torch.zeros_like(emb),
        guidance_scale=5.0,
    )
    assert rep.control_evals == 2 and torch.isfinite(out).all()
    with pytest.raises(ValueError):
        gated_denoise(unet, cn, s, lat, emb, img, np.ones(6, bool), guidance_scale=5.0)


def test_hold_reuse_adds_no_control_calls_and_differs_from_none(models):
    a, ra, _, _ = _run(models, "early", reuse="none")
    b, rb, _, _ = _run(models, "early", reuse="hold")
    assert (
        ra.control_evals == rb.control_evals == 4
        and rb.reused_steps == 4
        and ra.reused_steps == 0
    )
    assert (a - b).abs().max() > 1e-6


def test_deterministic_and_bad_inputs(models):
    a, _, _, _ = _run(models, "budgeted", budget=3)
    b, _, _, _ = _run(models, "budgeted", budget=3)
    assert torch.equal(a, b)
    unet, cn = models
    s = DDIMScheduler()
    step_infos_from_scheduler(s, 5)
    lat, emb, img = _inputs()
    with pytest.raises(ValueError):
        gated_denoise(unet, cn, s, lat, emb, img, np.ones(4, bool))
    with pytest.raises(ValueError):
        gated_denoise(unet, cn, s, lat, emb, img, np.ones(5, bool), reuse="linear")


def test_scheduler_normalisation_ddim_and_euler_sigma_order():
    d = step_infos_from_scheduler(DDIMScheduler(), 10)
    e = step_infos_from_scheduler(EulerDiscreteScheduler(), 10)
    assert sigma_order(d) == "high_to_low" and sigma_order(e) == "high_to_low"
    assert (
        d[0].timestep > d[-1].timestep
        and d[0].sigma > d[-1].sigma
        and e[0].sigma > e[-1].sigma
    )
    assert [i.sigma_rank for i in d] == list(range(10))
