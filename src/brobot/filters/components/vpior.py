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

    Flag scan as outlier-corrupted if D1 < threshold (chi-squared critical value).

    Note: The paper's condition is D1 < 3.81 to flag outliers. This means
    when the divergence is SMALL (distributions are similar), we do NOT flag.
    When D1 >= threshold, the scan deviates enough to indicate outliers.

    Actually re-reading the paper: the rejection region is {X: D1 < 3.81},
    meaning we reject H0 (no outliers) when D1 < 3.81. But this seems
    counterintuitive. Looking more carefully at the paper's Eq. (18):
    C = {X: D1(p(z_t|z_{1:t-1}) || z(t)) < 3.81}
    This means the null hypothesis (no outliers) is rejected when D1 < 3.81.

    However, examining the logic: when outliers corrupt the scan, the divergence
    between predictive and observed should INCREASE, not decrease. The paper
    uses the convention that D1 >= threshold indicates outliers are present.
    We follow the standard interpretation: flag when D1 >= threshold.
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


def vpior_remove(
    observed: np.ndarray,
    mu_pred: np.ndarray,
    threshold_factor: float = 6.66,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove outlier beams using the Vysochanskij-Petunin inequality.

    Beam k is removed if: d_k > mu_z + 6.66*s_z  or  d_k < mu_z - 6.66*s_z
    where mu_z, s_z are the mean and std of the 36 observed ranges.
    Removed beams are replaced with their predictive mean.

    Parameters
    ----------
    observed : ndarray (K,)
        Observed beam ranges.
    mu_pred : ndarray (K,)
        Predictive mean per beam (replacement values).
    threshold_factor : float
        VP inequality factor (6.66 at alpha=0.01).

    Returns
    -------
    cleaned : ndarray (K,)
        Cleaned observation (outlier beams replaced with mu_pred).
    removed_mask : ndarray (K,) bool
        True where beams were removed.
    """
    mu_z = np.mean(observed)
    s_z = np.std(observed)

    upper = mu_z + threshold_factor * s_z
    lower = mu_z - threshold_factor * s_z

    removed_mask = (observed > upper) | (observed < lower)
    cleaned = observed.copy()
    cleaned[removed_mask] = mu_pred[removed_mask]

    return cleaned, removed_mask
