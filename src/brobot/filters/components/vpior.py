"""Rényi divergence-based outlier detection and VPIOR beam removal."""

import numpy as np


def renyi_divergence_gaussian(
    mu_pred: np.ndarray,
    var_pred: np.ndarray,
    observed: np.ndarray,
    sigma: float,
) -> float:
    """KL divergence between predictive N(mu_k, sigma_k^2) and observed N(d_k, sigma^2).

    D1 = sum_k [log(sigma/sigma_k) + (sigma_k^2 + (mu_k - d_k)^2) / (2*sigma^2) - 0.5]

    Parameters
    ----------
    mu_pred : ndarray (K,)
        Predictive mean per beam.
    var_pred : ndarray (K,)
        Predictive variance per beam (sigma_k^2).
    observed : ndarray (K,)
        Observed beam ranges.
    sigma : float
        Sensor noise std.

    Returns
    -------
    d1 : float
        Rényi divergence of order 1 (= KL divergence).
    """
    sigma_pred = np.sqrt(var_pred)
    sigma2 = sigma ** 2

    d1 = np.sum(
        np.log(sigma / (sigma_pred + 1e-10))
        + (var_pred + (mu_pred - observed) ** 2) / (2 * sigma2)
        - 0.5
    )
    return d1


def detect_outlier_scan(
    mu_pred: np.ndarray,
    var_pred: np.ndarray,
    observed: np.ndarray,
    sigma: float,
    threshold: float = 3.81,
) -> bool:
    """Detect if scan contains outliers using Rényi divergence.

    When outliers corrupt a scan, the KL divergence between the predictive
    distribution and the observed scan increases. We flag the scan as
    outlier-corrupted when D1 >= threshold (chi-squared critical value
    at alpha=0.05, 1 DOF = 3.81).
    """
    d1 = renyi_divergence_gaussian(mu_pred, var_pred, observed, sigma)
    return d1 >= threshold


def compute_predictive_stats(
    weights: np.ndarray,
    expected_ranges: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute predictive scan distribution from particle set.

    mu_k = sum_i w_i * d_k*(x_i)
    sigma_k^2 = sum_i w_i * (d_k*(x_i) - mu_k)^2 + sigma^2

    Parameters
    ----------
    weights : ndarray (N,)
        Normalized particle weights.
    expected_ranges : ndarray (N, K)
        Expected ranges per particle per beam.
    sigma : float
        Sensor noise std.

    Returns
    -------
    mu_pred : ndarray (K,)
    var_pred : ndarray (K,)
    """
    mu_pred = np.average(expected_ranges, weights=weights, axis=0)
    diff = expected_ranges - mu_pred[np.newaxis, :]
    var_pred = np.average(diff ** 2, weights=weights, axis=0) + sigma ** 2
    return mu_pred, var_pred


def vpior_remove_particles(
    particles: np.ndarray,
    rng: np.random.Generator,
    threshold_factor: float = 6.66,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis VP pruning of the particle set (paper Algorithm 1 outlier step).

    For each pose dimension, flag particles whose value lies more than
    ``threshold_factor * sigma_axis`` from the axis mean. This is the
    Vysochanskij-Petunin bound at alpha=0.01 (factor 6.66) generalized to
    each pose axis independently — the closest 3-D analogue to the paper's
    1-D `[TH_L, TH_U]` rejection region.

    Theta uses circular statistics: the mean is `atan2(<sin>, <cos>)` and
    the per-particle deviation is wrapped into `[-pi, pi]` before being
    compared with `threshold_factor * sigma_theta`, so the bound straddles
    the wrap-around correctly.

    Removed particles are replaced by uniform-with-replacement draws from
    the surviving subset (the cloud is equal-weight by the time this runs,
    so no weighted resample is needed).

    Parameters
    ----------
    particles : ndarray (N, 3)
        Pose particles ``[x, y, theta]``.
    rng : np.random.Generator
        RNG for the survivor draw.
    threshold_factor : float
        VP inequality factor (6.66 at alpha=0.01).

    Returns
    -------
    cleaned : ndarray (N, 3)
        Particle cloud with outliers replaced by survivor draws.
    removed_mask : ndarray (N,) bool
        True at indices that were pruned.
    """
    n = particles.shape[0]
    if n <= 1:
        return particles, np.zeros(n, dtype=bool)

    x = particles[:, 0]
    y = particles[:, 1]
    theta = particles[:, 2]

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.std()
    sigma_y = y.std()

    mu_theta = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())
    dtheta = (theta - mu_theta + np.pi) % (2 * np.pi) - np.pi
    sigma_theta = np.sqrt(np.mean(dtheta ** 2))

    eps = 1e-12
    removed = (
        (np.abs(x - mu_x) > threshold_factor * (sigma_x + eps))
        | (np.abs(y - mu_y) > threshold_factor * (sigma_y + eps))
        | (np.abs(dtheta) > threshold_factor * (sigma_theta + eps))
    )

    n_removed = int(removed.sum())
    if n_removed == 0:
        return particles, removed

    survivors = np.where(~removed)[0]
    if survivors.size == 0:
        # Degenerate: every particle is its own outlier. Leave the cloud
        # untouched rather than collapse it.
        return particles, np.zeros(n, dtype=bool)

    cleaned = particles.copy()
    replacements = rng.choice(survivors, size=n_removed, replace=True)
    cleaned[removed] = particles[replacements]
    return cleaned, removed
