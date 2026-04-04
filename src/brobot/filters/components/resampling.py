"""Systematic resampling for particle filters."""

import numpy as np


def systematic_resample(
    weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Systematic resampling.

    Parameters
    ----------
    weights : ndarray (N,)
        Normalized particle weights (sum to 1).
    rng : Generator

    Returns
    -------
    indices : ndarray (N,) int
        Resampled particle indices.
    """
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    # Clip to valid range in case of floating-point issues
    return np.clip(indices, 0, n - 1)
