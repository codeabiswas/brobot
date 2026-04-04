"""RA-PF: Regularized Auxiliary Particle Filter (SIR + kernel smoothing)."""

import numpy as np

from brobot.filters.base import BaseFilter
from brobot.filters.components.resampling import systematic_resample
from brobot.filters.components.smoothing import kernel_smooth
from brobot.sim.motion import sample_motion_batch


class RAPF(BaseFilter):
    """SIR-PF with kernel smoothing at resampling."""

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

        # 3. Resampling
        indices = systematic_resample(self.weights, rng)
        self.particles = self.particles[indices]

        # 4. Kernel smoothing
        self.particles = kernel_smooth(self.particles, rng)

        # 5. Reset weights
        self.weights[:] = 1.0 / self.N

        return {}
