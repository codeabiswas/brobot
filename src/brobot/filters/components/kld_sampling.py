"""KLD-sampling for adaptive particle count."""

import numpy as np
from scipy.stats import norm


def kld_particle_count(
    particles: np.ndarray,
    bin_sizes: tuple[float, float, float] = (0.5, 0.5, np.deg2rad(10)),
    eps: float = 0.05,
    delta: float = 0.01,
    n_min: int = 50,
    n_max: int = 500,
) -> int:
    """Compute required number of particles via KLD-sampling.

    Parameters
    ----------
    particles : ndarray (N, 3)
        Current particle poses [px, py, theta].
    bin_sizes : tuple
        Bin sizes for x, y, theta.
    eps : float
        Maximum KL distance.
    delta : float
        Probability bound.
    n_min, n_max : int
        Bounds on particle count.

    Returns
    -------
    n_required : int
    """
    # Count occupied bins
    bins_x = np.floor(particles[:, 0] / bin_sizes[0]).astype(int)
    bins_y = np.floor(particles[:, 1] / bin_sizes[1]).astype(int)
    # Wrap theta to [0, 2pi] before binning
    theta_wrapped = particles[:, 2] % (2 * np.pi)
    bins_t = np.floor(theta_wrapped / bin_sizes[2]).astype(int)

    # Unique bins
    bin_ids = np.column_stack([bins_x, bins_y, bins_t])
    k = len(np.unique(bin_ids, axis=0))

    if k <= 1:
        return n_min

    # Wilson-Hilferty approximation
    z = norm.ppf(1 - delta)  # z_{1-delta}
    n_kld = (k - 1) / (2 * eps) * (
        1 - 2 / (9 * (k - 1)) + np.sqrt(2 / (9 * (k - 1))) * z
    ) ** 3

    return int(np.clip(n_kld, n_min, n_max))
