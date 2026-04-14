"""KLD-AMCL: Adaptive Monte Carlo Localization via KLD-sampling.

Implements the canonical incremental KLD_Sampling_MCL of Fox (2003) / Thrun
et al. (2005), Table 8.4. All the heavy lifting lives in
``kld_adaptive_sample``; this class is a thin wrapper that forwards state and
normalizes the returned importance weights.
"""

import numpy as np

from brobot.filters.base import BaseFilter
from brobot.filters.components.kld_sampling import kld_adaptive_sample


class KLDAMCL(BaseFilter):
    """Adaptive particle count via KLD-sampling. No mutation, no VPIOR."""

    def step(
        self,
        control: np.ndarray,
        observation: np.ndarray,
        timestep: int,
        rng: np.random.Generator,
    ) -> dict:
        v, omega = control

        new_particles, new_log_w = kld_adaptive_sample(
            self.particles,
            self.weights,
            v,
            omega,
            observation,
            self.occ_map,
            self.resolution,
            self.sigma,
            self.d_max,
            self.beam_angles,
            rng,
        )

        # Normalize the returned log-likelihoods (log-sum-exp for stability).
        log_w = new_log_w - np.max(new_log_w)
        w = np.exp(log_w)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones_like(w) / len(w)

        self.particles = new_particles
        self.weights = w
        # Kept for interface consistency with MPF's mutation path.
        self.weights_prenorm = np.exp(log_w)
        self.N = len(new_particles)

        return {}
