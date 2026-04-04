"""KLD-AMCL: Adaptive Monte Carlo Localization via KLD-sampling."""

import numpy as np

from brobot.filters.base import BaseFilter
from brobot.filters.components.resampling import systematic_resample
from brobot.filters.components.kld_sampling import kld_particle_count
from brobot.sim.motion import sample_motion_batch


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

        # 1. Motion update
        self.particles = sample_motion_batch(self.particles, v, omega, rng)

        # 2. Weight update
        expected = self.compute_expected_ranges()
        log_w = self.compute_log_weights(observation, expected)
        log_w -= np.max(log_w)
        self.weights_prenorm = np.exp(log_w)
        w_sum = self.weights_prenorm.sum()
        if w_sum > 0:
            self.weights = self.weights_prenorm / w_sum
        else:
            self.weights[:] = 1.0 / self.N

        # 3. KLD-adaptive resampling
        n_new = kld_particle_count(self.particles)
        indices = systematic_resample(self.weights, rng)

        # Resize particle set
        if n_new != self.N:
            # Sample n_new particles from resampled set
            selected = rng.choice(indices, size=n_new, replace=True)
            self.particles = self.particles[selected]
            self.N = n_new
            self.weights = np.ones(n_new) / n_new
            self.weights_prenorm = np.zeros(n_new)
        else:
            self.particles = self.particles[indices]
            self.weights[:] = 1.0 / self.N

        return {}
