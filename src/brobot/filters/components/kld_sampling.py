"""KLD-sampling for adaptive particle count.

Implements the canonical incremental KLD_Sampling_MCL of Fox (2003) / Thrun
et al. Probabilistic Robotics (2005), Table 8.4. Particles are generated one
at a time; after each draw the bin histogram is updated and the Wilson-Hilferty
target M_chi is recomputed. Sampling stops as soon as M >= M_chi and M >= n_min.

The helper ``kld_adaptive_sample`` implements a semantically-equivalent chunked
version of the loop that reuses the project's vectorized motion, raycast, and
log-likelihood kernels. Overshoot is bounded by ``chunk_size - 1`` particles.
"""

import numpy as np
from scipy.stats import norm

from brobot.sim.motion import sample_motion_batch
from brobot.sim.raycast import batch_raycast
from brobot.sim.sensor import log_likelihood


def compute_kld_threshold(
    k: int,
    eps: float = 0.05,
    delta: float = 0.01,
) -> float:
    """Wilson-Hilferty KLD sample-size bound given k occupied bins.

    M_chi = (k-1)/(2*eps) * [1 - 2/(9(k-1)) + sqrt(2/(9(k-1))) * z_{1-delta}]^3

    Returns 0.0 when k <= 1 (bound undefined; the caller's n_min floor applies).
    """
    if k <= 1:
        return 0.0
    z = norm.ppf(1 - delta)
    return (
        (k - 1)
        / (2 * eps)
        * (1 - 2 / (9 * (k - 1)) + np.sqrt(2 / (9 * (k - 1))) * z) ** 3
    )


def kld_adaptive_sample(
    prev_particles: np.ndarray,
    prev_weights: np.ndarray,
    v: float,
    omega: float,
    observation: np.ndarray,
    occ_map: np.ndarray,
    resolution: float,
    sigma: float,
    d_max: float,
    beam_angles: np.ndarray,
    rng: np.random.Generator,
    eps: float = 0.05,
    delta: float = 0.01,
    n_min: int = 50,
    n_max: int = 500,
    bin_sizes: tuple[float, float, float] = (0.5, 0.5, np.deg2rad(10)),
    chunk_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Incremental KLD-sampling MCL step (chunked for vectorization).

    Generates new particles one at a time conceptually: draw an ancestor from
    Categorical(prev_weights), propagate through the motion model, score with
    the measurement model, update the bin histogram, recompute M_chi, and stop
    when both ``M >= M_chi`` and ``M >= n_min`` are satisfied (or ``n_max`` is
    hit). The implementation generates particles in chunks of ``chunk_size`` to
    amortize Python overhead on motion/raycast; termination is checked
    per-particle inside the chunk so the only inaccuracy versus a strict
    per-particle loop is at most ``chunk_size - 1`` extra particles drawn in
    the final chunk (always bounded above by ``n_max``).

    Parameters
    ----------
    prev_particles : ndarray (N_prev, 3)
        Particle set from the previous step.
    prev_weights : ndarray (N_prev,)
        Previous-step importance weights. Must be normalized (sum = 1). The
        caller is responsible for fallback-to-uniform on degeneracy.
    v, omega : float
        Control input for the motion model.
    observation : ndarray (K,)
        Sensor scan used in the measurement model. Pass VPIOR-cleaned scan
        if upstream outlier removal was applied.
    occ_map, resolution, sigma, d_max, beam_angles :
        Map + sensor parameters forwarded to ``batch_raycast`` and
        ``log_likelihood``.
    rng : Generator
    eps, delta : float
        KLD bound parameters (max KL distance, confidence).
    n_min, n_max : int
        Hard bounds on the returned particle count.
    bin_sizes : tuple
        (dx, dy, dtheta) bin sizes for the histogram.
    chunk_size : int
        Vectorization batch size.

    Returns
    -------
    particles_out : ndarray (M, 3)
    log_w_out : ndarray (M,)
        Unnormalized log-likelihoods of the returned particles. Caller is
        responsible for normalization.
    """
    occupied_bins: set[tuple[int, int, int]] = set()
    k = 0
    m_chi = 0.0
    M = 0
    particles_out: list[np.ndarray] = []
    log_w_out: list[float] = []

    # Safety: normalize / fall back to uniform if prev_weights is degenerate.
    w = np.asarray(prev_weights, dtype=np.float64)
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        w = np.ones(len(prev_particles)) / len(prev_particles)
    else:
        w = w / w.sum()

    inv_bx = 1.0 / bin_sizes[0]
    inv_by = 1.0 / bin_sizes[1]
    inv_bt = 1.0 / bin_sizes[2]
    two_pi = 2 * np.pi

    while M < max(n_min, m_chi) and M < n_max:
        chunk = int(min(chunk_size, n_max - M))
        if chunk <= 0:
            break

        # Draw ancestors ~ Categorical(prev_weights)
        ancestor_idx = rng.choice(len(prev_particles), size=chunk, p=w)
        ancestors = prev_particles[ancestor_idx]

        # Batch motion + raycast + log-likelihood
        new_chunk = sample_motion_batch(ancestors, v, omega, rng)
        expected = batch_raycast(
            new_chunk,
            beam_angles,
            occ_map,
            resolution,
            d_max,
        )
        log_w_chunk = log_likelihood(observation, expected, sigma)

        # Walk through chunk sequentially for bin counting + termination.
        stop = False
        for i in range(chunk):
            p = new_chunk[i]
            particles_out.append(p)
            log_w_out.append(float(log_w_chunk[i]))
            M += 1

            bx = int(np.floor(p[0] * inv_bx))
            by = int(np.floor(p[1] * inv_by))
            bt = int(np.floor((p[2] % two_pi) * inv_bt))
            key = (bx, by, bt)
            if key not in occupied_bins:
                occupied_bins.add(key)
                k += 1
                if k > 1:
                    m_chi = compute_kld_threshold(k, eps, delta)

            # Termination: met the KLD bound AND the minimum floor.
            if M >= n_min and M >= m_chi and k > 1:
                stop = True
                break
        if stop:
            break

    return np.asarray(particles_out), np.asarray(log_w_out)
