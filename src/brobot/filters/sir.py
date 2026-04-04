"""SIR-PF: Standard Sequential Importance Resampling Particle Filter."""

import numpy as np

from brobot.filters.base import BaseFilter
from brobot.filters.components.resampling import systematic_resample
from brobot.sim.motion import sample_motion_batch


class SIRPF(BaseFilter):
    """Standard SIR particle filter.

    Motion update → weight update → systematic resampling → reset weights.
    """

    def step(
        self,
        control: np.ndarray,
        observation: np.ndarray,
        timestep: int,
        rng: np.random.Generator,
    ) -> dict:
        v, omega = control

        # 1. Motion update (predict)
        self.particles = sample_motion_batch(
            self.particles, v, omega, rng,
        )

        # 2. Weight update (correct)
        expected = self.compute_expected_ranges()
        log_w = self.compute_log_weights(observation, expected)

        # Numerical stability: subtract max before exp
        log_w -= np.max(log_w)
        self.weights_prenorm = np.exp(log_w)
        w_sum = self.weights_prenorm.sum()
        if w_sum > 0:
            self.weights = self.weights_prenorm / w_sum
        else:
            self.weights[:] = 1.0 / self.N

        # 3. Systematic resampling
        indices = systematic_resample(self.weights, rng)
        self.particles = self.particles[indices]

        # 4. Reset weights
        self.weights[:] = 1.0 / self.N

        return {}
