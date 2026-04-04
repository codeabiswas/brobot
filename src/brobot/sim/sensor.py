"""LiDAR sensor model: simulation, outlier injection, and likelihood."""

import numpy as np

from brobot.sim.raycast import batch_raycast

# LiDAR configuration
N_BEAMS = 36
FOV_DEG = 240.0
D_MAX = 5.6  # meters


def beam_angles(n_beams: int = N_BEAMS, fov_deg: float = FOV_DEG) -> np.ndarray:
    """Compute beam angle offsets relative to robot heading.

    Beams are evenly spaced over the FOV, centered at 0 (forward).
    """
    fov_rad = np.deg2rad(fov_deg)
    return np.linspace(-fov_rad / 2, fov_rad / 2, n_beams)


def simulate_lidar(
    pose: np.ndarray,
    occ_map: np.ndarray,
    resolution: float,
    sigma: float,
    rng: np.random.Generator,
    d_max: float = D_MAX,
    n_beams: int = N_BEAMS,
    fov_deg: float = FOV_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a noisy LiDAR scan from a single pose.

    Returns
    -------
    noisy_ranges : ndarray (n_beams,)
    expected_ranges : ndarray (n_beams,)
    """
    angles = beam_angles(n_beams, fov_deg)
    pose_2d = pose.reshape(1, 3)
    expected = batch_raycast(pose_2d, angles, occ_map, resolution, d_max)[0]

    # Add Gaussian noise
    noisy = expected + rng.normal(0, sigma, n_beams)
    noisy = np.clip(noisy, 0.0, d_max)

    return noisy, expected


def inject_outliers(
    ranges: np.ndarray,
    r: float,
    rng: np.random.Generator,
    d_max: float = D_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace beams with uniform random values at rate r.

    Returns
    -------
    corrupted : ndarray (n_beams,)
    outlier_mask : ndarray (n_beams,) bool
        True where beam was replaced with an outlier.
    """
    n_beams = ranges.shape[0]
    outlier_mask = rng.random(n_beams) < r
    corrupted = ranges.copy()
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        corrupted[outlier_mask] = rng.uniform(0, d_max, n_outliers)
    return corrupted, outlier_mask


def log_likelihood(
    observed: np.ndarray,
    expected: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Compute per-particle log-likelihood of observations.

    Parameters
    ----------
    observed : ndarray (K,)
        Observed beam ranges.
    expected : ndarray (N, K)
        Expected ranges for N particles, K beams.
    sigma : float
        Sensor noise std.

    Returns
    -------
    log_weights : ndarray (N,)
    """
    diff = observed[np.newaxis, :] - expected  # (N, K)
    return -np.sum(diff**2, axis=1) / (2 * sigma**2)
