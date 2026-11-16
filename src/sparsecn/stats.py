"""Small statistics helpers: bootstrap confidence intervals and the "share of gain kept" ratio."""

import numpy as np


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Mean with a percentile bootstrap interval over the samples. NaNs are dropped."""
    values = _finite(values)
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    rng = np.random.default_rng(seed)
    boot_means = rng.choice(values, (n_boot, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(boot_means, alpha / 2)),
        "hi": float(np.quantile(boot_means, 1 - alpha / 2)),
        "n": int(len(values)),
    }


def paired_diff_ci(a, b, **kwargs):
    """Bootstrap interval of the per-sample difference a - b (pairs with a NaN are dropped)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both_ok = np.isfinite(a) & np.isfinite(b)
    return bootstrap_ci((a - b)[both_ok], **kwargs)


def retention(setting, full, none, n_boot=2000, seed=0):
    """Share of full ControlNet's improvement over no control that a setting keeps.

    (mean(setting) - mean(none)) / (mean(full) - mean(none)), so 1 is as good as full and 0 as bad as
    no control. It works for lower-is-better metrics too because the direction cancels out. The
    interval is a bootstrap over images, resampling the three conditions together.
    """
    setting, full, none = (np.asarray(x, dtype=float) for x in (setting, full, none))
    ok = np.isfinite(setting) & np.isfinite(full) & np.isfinite(none)
    setting, full, none = setting[ok], full[ok], none[ok]
    n = len(setting)

    def ratio(idx):
        gain_full = full[idx].mean() - none[idx].mean()
        if abs(gain_full) < 1e-12:
            return float("nan")
        return (setting[idx].mean() - none[idx].mean()) / gain_full

    rng = np.random.default_rng(seed)
    boots = np.array([ratio(rng.integers(0, n, n)) for _ in range(n_boot)])
    return {
        "mean": float(ratio(np.arange(n))),
        "lo": float(np.nanquantile(boots, 0.025)),
        "hi": float(np.nanquantile(boots, 0.975)),
    }
