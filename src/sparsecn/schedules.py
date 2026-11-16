"""Gating policies: a list of StepInfo becomes a boolean mask where True means "run ControlNet here".

The policies encode hypotheses about where in the trajectory structure gets decided, not facts:

    full, none     run on every step / on no step
    early, late    first / last fraction of the steps by sampling index
    high_noise     the highest-sigma fraction of the steps
    low_noise      the lowest-sigma fraction of the steps
    middle         the band of steps around the median sigma
    window         one contiguous index window (start, end)
    budgeted       exactly K steps, chosen by a strategy
"""

import numpy as np

SCHEDULE_NAMES = [
    "full",
    "none",
    "early",
    "late",
    "high_noise",
    "low_noise",
    "middle",
    "window",
    "budgeted",
]


def _by_index(n, frac, from_start):
    k = int(round(frac * n))
    mask = np.zeros(n, dtype=bool)
    if k > 0:
        if from_start:
            mask[:k] = True
        else:
            mask[n - k :] = True
    return mask


def _by_sigma_rank(infos, frac, where):
    n = len(infos)
    k = int(round(frac * n))
    ranks = np.array([info.sigma_rank for info in infos])
    if where == "high":
        return ranks < k
    if where == "low":
        return ranks >= n - k
    first = (n - k) // 2  # "mid": k ranks centred on the median
    return (ranks >= first) & (ranks < first + k)


def budgeted_mask(infos, budget, strategy="uniform"):
    """Mask with exactly `budget` steps enabled."""
    n = len(infos)
    if not 0 <= budget <= n:
        raise ValueError(f"budget K={budget} outside [0,{n}]")
    mask = np.zeros(n, dtype=bool)
    if budget == 0:
        return mask

    if strategy == "uniform":
        chosen = np.unique(np.round(np.linspace(0, n - 1, budget)).astype(int))
    elif strategy == "first":
        chosen = np.arange(budget)
    elif strategy == "last":
        chosen = np.arange(n - budget, n)
    elif strategy == "high_sigma":
        chosen = np.argsort([-info.sigma for info in infos], kind="stable")[:budget]
    elif strategy == "low_sigma":
        chosen = np.argsort([info.sigma for info in infos], kind="stable")[:budget]
    else:
        raise ValueError(f"unknown budget strategy {strategy}")
    mask[chosen] = True

    # Rounding in the uniform strategy can pick the same step twice; fill up from the front.
    for i in range(n):
        if mask.sum() == budget:
            break
        mask[i] = True
    return mask


def make_mask(
    name, infos, frac=0.5, budget=None, window=None, budget_strategy="uniform"
):
    """Mask for the policy `name` (see the module docstring)."""
    n = len(infos)
    if name == "full":
        return np.ones(n, dtype=bool)
    if name == "none":
        return np.zeros(n, dtype=bool)
    if name == "early":
        return _by_index(n, frac, from_start=True)
    if name == "late":
        return _by_index(n, frac, from_start=False)
    if name == "high_noise":
        return _by_sigma_rank(infos, frac, "high")
    if name == "low_noise":
        return _by_sigma_rank(infos, frac, "low")
    if name == "middle":
        return _by_sigma_rank(infos, frac, "mid")
    if name == "window":
        if window is None:
            raise ValueError("window schedule needs window=(start, end)")
        start, end = window
        if not 0 <= start <= end <= n:
            raise ValueError(f"window {window} outside [0,{n}]")
        mask = np.zeros(n, dtype=bool)
        mask[start:end] = True
        return mask
    if name == "budgeted":
        if budget is None:
            raise ValueError("budgeted schedule needs a budget K")
        return budgeted_mask(infos, budget, budget_strategy)
    raise ValueError(f"unknown schedule {name!r}")
