"""Deterministic seeding utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

SeededRng = np.random.Generator


@dataclass(frozen=True)
class SeedBundle:
    """Deterministic seed values for Python and NumPy consumers."""

    seed: int


def seeded_rng(seed: int | None) -> SeededRng:
    """Create a NumPy generator and seed Python's random module consistently."""
    normalized_seed = 0 if seed is None else int(seed)
    random.seed(normalized_seed)
    return np.random.default_rng(normalized_seed)
