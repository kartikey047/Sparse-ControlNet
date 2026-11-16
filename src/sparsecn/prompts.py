"""Prompt pool for the real-model experiment.

Prompts are built from three small lists (subject x scene x style), so every prompt is unique and the
pool is the same on every machine. They are deliberately simple: the experiment needs images with clear
structure for Canny edges, not a realistic prompt distribution.
"""

import itertools

import numpy as np

SUBJECTS = [
    "a red vintage car",
    "a wooden house",
    "a lighthouse",
    "a bowl of fruit",
    "a bicycle",
    "a castle",
    "a sailing boat",
    "a mountain cabin",
    "a street lamp",
    "a stone bridge",
    "a windmill",
    "a teapot",
    "a city skyline",
    "a barn",
    "a train station",
    "a market stall",
    "a clock tower",
    "a garden gate",
    "a rocking chair",
    "a camper van",
]
SCENES = [
    "in a misty forest",
    "on a rainy street",
    "at sunrise",
    "in a snowy field",
    "by a calm lake",
    "in a desert",
    "at night",
    "in an autumn park",
    "next to the sea",
    "in a quiet village",
]
STYLES = [
    "photograph, natural light",
    "oil painting",
    "watercolor illustration",
    "cinematic photo",
    "detailed pencil drawing colored",
    "3d render",
]


def make_prompts(n, seed=0):
    """Return `n` distinct prompts in a seeded random order."""
    pool = [
        f"{subject} {scene}, {style}"
        for subject, scene, style in itertools.product(SUBJECTS, SCENES, STYLES)
    ]
    if n > len(pool):
        raise ValueError(f"only {len(pool)} distinct prompts available, requested {n}")
    order = np.random.default_rng(seed).permutation(len(pool))
    return [pool[i] for i in order[:n]]
