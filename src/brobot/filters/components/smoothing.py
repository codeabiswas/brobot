"""Kernel smoothing for regularized particle filter."""

import numpy as np


def kernel_smooth(
    particles: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply kernel smoothing (RA-PF style).

    Each particle receives: x_tilde = x + h * epsilon
    where epsilon ~ N(0, Sigma_hat) and h = N^{-1/7} (Silverman bandwidth for d=3).

    Parameters
    ----------
    particles : ndarray (N, 3)
        Particle poses [px, py, theta].
    rng : Generator

    Returns
    -------
    smoothed : ndarray (N, 3)
    """
    n = particles.shape[0]
    h = n ** (-1.0 / 7.0)  # Silverman bandwidth for d=3

    # Empirical covariance
    cov = np.cov(particles.T)  # (3, 3)

    # Sample perturbations
    perturbations = rng.multivariate_normal(np.zeros(3), cov, size=n)

    smoothed = particles + h * perturbations

    # Wrap theta
    smoothed[:, 2] = (smoothed[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return smoothed
