"""Epanechnikov kernel regularization for the RAPF (Liu et al. 2011)."""

import warnings

import numpy as np


def epanechnikov_sample(
    n_dim: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from the n-dimensional Epanechnikov kernel.

    K_opt(x) = (n+2)/(2*c_n) * (1 - ||x||^2)  for ||x|| <= 1

    Uses the Silverman (1986) trick: project uniform samples from
    the (n+2)-dimensional unit sphere surface onto the first n coordinates.
    This produces exact Epanechnikov samples with no rejection loop.

    Parameters
    ----------
    n_dim : int
        Dimension of the state space.
    n_samples : int
        Number of samples to draw.
    rng : Generator

    Returns
    -------
    samples : ndarray (n_samples, n_dim)
        Each sample has ||sample|| <= 1.
    """
    z = rng.standard_normal((n_samples, n_dim + 2))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z[:, :n_dim]


def regularize(
    particles: np.ndarray,
    weights: np.ndarray,
    original_particles: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Regularize resampled particles via Epanechnikov kernel smoothing.

    Implements Stage 2 of the RAPF (Liu et al. 2011, Table 1):
    1. Compute weighted empirical covariance S from pre-resampling particles.
    2. Cholesky decompose: A @ A.T = S.
    3. Compute optimal bandwidth h_opt (Eq. 15).
    4. Perturb each particle: x* = x + h_opt * A @ epsilon,
       where epsilon ~ Epanechnikov kernel.

    Parameters
    ----------
    particles : ndarray (N, 3)
        Post-resampling particles (equal weights).
    weights : ndarray (N,)
        Pre-resampling normalized auxiliary weights (for covariance).
    original_particles : ndarray (N, 3)
        Pre-resampling particles.
    rng : Generator

    Returns
    -------
    regularized : ndarray (N, 3)
    """
    n = particles.shape[1]  # state dimension (3)
    M = particles.shape[0]  # number of particles

    # Weighted empirical covariance from pre-resampling distribution.
    # np.cov can warn when effective DOF <= 0 (e.g., all weight on one
    # particle). We catch that and fall back to unweighted covariance.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = np.cov(original_particles.T, aweights=weights)
    if np.any(np.isnan(S)) or np.any(np.isinf(S)):
        S = np.cov(original_particles.T)
        if np.any(np.isnan(S)) or np.any(np.isinf(S)):
            S = 1e-10 * np.eye(n)

    # Ensure positive-definiteness
    S += 1e-10 * np.eye(n)

    # Cholesky decomposition: A @ A.T = S
    try:
        A = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Fallback: identity perturbation only
        A = 1e-5 * np.eye(n)

    # Optimal bandwidth (Eq. 15)
    # h_opt = [8 * c_n^{-1} * (n+4) * (2*pi)^n]^{1/(n+4)} * M^{-1/(n+4)}
    # For n=3: c_3 = 4*pi/3
    c_n = 4.0 * np.pi / 3.0
    h_opt = (8.0 / c_n * (n + 4) * (2 * np.pi) ** n) ** (1.0 / (n + 4)) * M ** (
        -1.0 / (n + 4)
    )

    # Sample Epanechnikov perturbations
    eps = epanechnikov_sample(n, M, rng)  # (M, 3)

    # Apply perturbation: x* = x + h_opt * A @ eps
    regularized = particles + h_opt * (eps @ A.T)

    # Wrap theta to [-pi, pi]
    regularized[:, 2] = (regularized[:, 2] + np.pi) % (2 * np.pi) - np.pi

    return regularized
