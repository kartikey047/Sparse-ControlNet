"""CPU tests for the real-SD experiment helpers (no GPU, no downloads)."""

import numpy as np, pytest
from sparsecn.settings import SETTINGS, setting_mask
from sparsecn.prompts import make_prompts
from sparsecn.edge_metrics import (
    edge_f1,
    dilate,
    effect_recovery_img,
    soft_recall,
    grad_corr,
)
from sparsecn.stats import bootstrap_ci, paired_diff_ci
from sparsecn.timesteps import make_step_infos


def test_settings_structure():
    infos = make_step_infos(20)
    early, low, comb = (setting_mask(s, infos) for s in SETTINGS)
    assert early.sum() == 6 and low.sum() == comb.sum() == 10
    assert early[:6].all() and not early[6:].any()  # first 10 steps (high-noise end)
    assert low[10:].all() and not low[:10].any()  # last 10 steps (lowest sigma)
    assert (
        comb[:5].all() and comb[-5:].all() and not comb[5:15].any()
    )  # outer quarters, middle skipped
    assert setting_mask("full", infos).all() and not setting_mask("none", infos).any()
    sig = np.array([i.sigma for i in infos])
    assert sig[early].min() > sig[low].max()  # early really is the higher-noise half
    with pytest.raises(ValueError):
        setting_mask("bogus", infos)


def test_prompts_deterministic_unique_and_bounded():
    a, b = make_prompts(50, 1), make_prompts(50, 1)
    assert a == b and len(set(a)) == 50 and make_prompts(50, 2) != a
    with pytest.raises(ValueError):
        make_prompts(10**6)


def test_edge_f1_behaviour():
    m = np.zeros((40, 40), bool)
    m[20, 5:35] = True
    assert edge_f1(m, m)["f1"] == pytest.approx(1.0) and edge_f1(m, m)[
        "iou"
    ] == pytest.approx(1.0)
    shifted = np.roll(m, 2, 0)
    r = edge_f1(shifted, m, tol=2)
    assert r["f1"] == pytest.approx(1.0) and r["iou"] == 0.0  # within tolerance
    assert edge_f1(np.roll(m, 8, 0), m, tol=2)["f1"] == 0.0
    assert edge_f1(np.zeros_like(m), m)["f1"] == 0.0
    half = m.copy()
    half[:, 20:] = False
    r = edge_f1(half, m)
    assert r["precision"] == pytest.approx(1.0) and r["recall"] < 0.7
    assert dilate(m, 1).sum() > m.sum()


def test_effect_recovery_img_endpoints():
    a, b = np.random.default_rng(0).random((8, 8, 3)), np.random.default_rng(1).random(
        (8, 8, 3)
    )
    assert (
        effect_recovery_img(a, a, b) == pytest.approx(1.0)
        and effect_recovery_img(b, a, b) == pytest.approx(0.0)
        and np.isnan(effect_recovery_img(a, a, a))
    )


def test_bootstrap_ci_contains_mean_and_handles_edge_cases():
    x = np.random.default_rng(0).normal(1.0, 0.1, 200)
    c = bootstrap_ci(x)
    assert c["lo"] < c["mean"] < c["hi"] and c["n"] == 200
    assert bootstrap_ci([])["n"] == 0 and np.isnan(bootstrap_ci([np.nan])["mean"])
    d = paired_diff_ci(x + 0.5, x)
    assert (
        d["lo"] <= 0.5 + 1e-9 and d["hi"] >= 0.5 - 1e-9 and abs(d["mean"] - 0.5) < 1e-9
    )  # constant shift -> degenerate CI
    y = x + np.random.default_rng(1).normal(0, 0.05, 200)
    d2 = paired_diff_ci(y, x)
    assert d2["lo"] < d2["hi"]


def test_soft_recall_is_tolerant_ignores_extra_edges_and_degrades_with_distance():
    ctrl = np.zeros((60, 60), bool)
    ctrl[30, 10:50] = True
    assert (
        soft_recall(ctrl, ctrl)["soft_recall"] == pytest.approx(1.0)
        and soft_recall(ctrl, ctrl)["chamfer_px"] == 0.0
    )
    noisy = ctrl.copy()
    noisy[5, :] = True
    noisy[:, 55] = True
    noisy[50, 20:40] = True  # lots of extra texture edges
    assert soft_recall(noisy, ctrl)["soft_recall"] == pytest.approx(
        1.0
    )  # extra edges do not hurt recall
    near, far = np.roll(ctrl, 2, 0), np.roll(ctrl, 8, 0)
    a, b, c = (soft_recall(g, ctrl) for g in (ctrl, near, far))
    assert (
        a["soft_recall"] > b["soft_recall"] > c["soft_recall"]
        and a["chamfer_px"] < b["chamfer_px"] < c["chamfer_px"]
    )
    assert soft_recall(np.zeros_like(ctrl), ctrl)["soft_recall"] == 0.0 and np.isnan(
        soft_recall(ctrl, np.zeros_like(ctrl))["soft_recall"]
    )


def test_grad_corr_threshold_free_structure_agreement():
    rng = np.random.default_rng(0)
    img = np.zeros((64, 64, 3))
    img[16:48, 16:48] = 1.0
    img += rng.normal(0, 0.02, img.shape)
    other = np.zeros((64, 64, 3))
    other[5:20, 40:60] = 1.0
    other[40:60, 5:25] = 0.8
    assert (
        grad_corr(img, img) == pytest.approx(1.0)
        and grad_corr(img, 0.5 * img + 0.2) > 0.95
    )  # invariant to contrast/brightness
    assert (
        grad_corr(img, np.roll(img, 3, 0))
        > grad_corr(img, np.roll(img, 12, 0))
        > grad_corr(img, other)
    )
    assert np.isnan(grad_corr(img, np.full_like(img, 0.5)))
