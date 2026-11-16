"""The experiment conditions and which denoising steps each one runs ControlNet on.

References:   none       ControlNet never runs
              full       ControlNet runs on every step
Gated:        early      first EARLY_FRACTION of the steps (the high-noise end)
              low_noise  the lowest-sigma half of the steps
              combined   first quarter + lowest-sigma quarter, skipping the middle

With 20 DDIM steps, early runs 6 ControlNet calls and the other two run 10 each, so early is not an
equal-compute comparison. DDIM sigmas fall monotonically, so "early" and "low_noise" never overlap.
"""

import numpy as np

from .schedules import make_mask

EARLY_FRACTION = 0.3  # 6 of 20 steps

REFERENCES = ("none", "full")
SETTINGS = ("early", "low_noise", "combined")
ALL_CONDITIONS = REFERENCES + SETTINGS

# Extra conditions that can be generated on top of an existing run (see `experiment rerun`):
#   early_10    first 10 steps (what "early" meant before it was cut to 6)
#   low_noise_6 last 6 steps by sigma, the equal-compute counterpart of the 6-step early
EXTRA_CONDITIONS = ("early_10", "low_noise_6")


def setting_mask(name, infos):
    """Boolean mask over steps, True where ControlNet runs."""
    n = len(infos)
    if name == "full":
        return np.ones(n, dtype=bool)
    if name == "none":
        return np.zeros(n, dtype=bool)
    if name == "early":
        return make_mask("early", infos, frac=EARLY_FRACTION)
    if name == "early_10":
        return make_mask("early", infos, frac=0.5)
    if name == "low_noise_6":
        return make_mask("low_noise", infos, frac=EARLY_FRACTION)
    if name == "low_noise":
        return make_mask("low_noise", infos, frac=0.5)
    if name == "combined":
        return make_mask("early", infos, frac=0.25) | make_mask(
            "low_noise", infos, frac=0.25
        )
    raise ValueError(f"unknown setting {name!r}")
