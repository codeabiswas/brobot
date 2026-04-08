"""MPF and ablation variants: MPF, KLD+Mut, KLD+VPIOR, MPF-E."""

import numpy as np

from brobot.filters.base import BaseFilter
from brobot.filters.components.resampling import systematic_resample
from brobot.filters.components.kld_sampling import kld_particle_count
from brobot.filters.components.vpior import (
    compute_predictive_stats,
    detect_outlier_scan,
    vpior_remove,
)
from brobot.filters.components.mutation import (
    tst_mutation,
    tst_mutation_entropy,
    shannon_entropy,
)
from brobot.sim.motion import sample_motion_batch


class MPF(BaseFilter):
    """Modified Particle Filter with configurable components.

    Components (all on by default for full MPF):
    - VPIOR: outlier detection via Rényi divergence + VP inequality beam removal
    - Mutation: TST mutation operator
    - Entropy mutation: use Shannon entropy instead of timestep (MPF-E)
    - KLD: adaptive particle count via KLD-sampling
    """

    def __init__(
        self,
        *args,
        use_vpior: bool = True,
        use_mutation: bool = True,
        entropy_mutation: bool = False,
        use_kld: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_vpior = use_vpior
        self.use_mutation = use_mutation
        self.entropy_mutation = entropy_mutation
        self.use_kld = use_kld

    def step(
        self,
        control: np.ndarray,
        observation: np.ndarray,
        timestep: int,
        rng: np.random.Generator,
    ) -> dict:
        v, omega = control
        info = {}

        # 1. Motion update
        self.particles = sample_motion_batch(self.particles, v, omega, rng)

        # 2. Raycast once for both VPIOR and weight update
        expected = self.compute_expected_ranges()

        # 3. VPIOR: outlier detection and beam removal
        obs_for_weights = observation
        if self.use_vpior:
            mu_pred, var_pred = compute_predictive_stats(
                self.weights, expected, self.sigma
            )

            if detect_outlier_scan(mu_pred, var_pred, observation, self.sigma):
                cleaned, removed_mask = vpior_remove(observation, mu_pred, var_pred)
                obs_for_weights = cleaned
                info["vpior_removed_mask"] = removed_mask
            else:
                info["vpior_removed_mask"] = np.zeros(self.n_beams, dtype=bool)

        # 4. Weight update
        log_w = self.compute_log_weights(obs_for_weights, expected)
        log_w -= np.max(log_w)
        self.weights_prenorm = np.exp(log_w)
        w_sum = self.weights_prenorm.sum()
        if w_sum > 0:
            self.weights = self.weights_prenorm / w_sum
        else:
            self.weights[:] = 1.0 / self.N

        # Save entropy before resampling (weights are still non-uniform)
        if self.entropy_mutation:
            self._entropy = shannon_entropy(self.weights)

        # 5. Resampling (with optional KLD adaptation)
        if self.use_kld:
            n_new = kld_particle_count(self.particles)
            indices = systematic_resample(self.weights, rng)
            if n_new != self.N:
                selected = rng.choice(indices, size=n_new, replace=True)
                self.particles = self.particles[selected]
                self.weights_prenorm = self.weights_prenorm[selected]
                self.N = n_new
                self.weights = np.ones(n_new) / n_new
            else:
                self.particles = self.particles[indices]
                self.weights_prenorm = self.weights_prenorm[indices]
                self.weights[:] = 1.0 / self.N
        else:
            indices = systematic_resample(self.weights, rng)
            self.weights_prenorm = self.weights_prenorm[indices]
            self.particles = self.particles[indices]
            self.weights[:] = 1.0 / self.N

        # 5. Mutation (post-resampling)
        if self.use_mutation:
            best_idx = np.argmax(self.weights_prenorm)
            best_particle = self.particles[best_idx]

            if self.entropy_mutation:
                H = self._entropy
                self.particles = tst_mutation_entropy(
                    self.particles, self.weights_prenorm, best_particle, H,
                )
            else:
                self.particles = tst_mutation(
                    self.particles, self.weights_prenorm, best_particle, timestep,
                )

        return info


class KLDMut(MPF):
    """KLD-AMCL + TST mutation only (no VPIOR)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_vpior", False)
        kwargs.setdefault("use_mutation", True)
        kwargs.setdefault("entropy_mutation", False)
        kwargs.setdefault("use_kld", True)
        super().__init__(*args, **kwargs)


class KLDVpior(MPF):
    """KLD-AMCL + VPIOR only (no mutation)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_vpior", True)
        kwargs.setdefault("use_mutation", False)
        kwargs.setdefault("entropy_mutation", False)
        kwargs.setdefault("use_kld", True)
        super().__init__(*args, **kwargs)


class MPFE(MPF):
    """MPF-E: MPF with entropy-based mutation."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_vpior", True)
        kwargs.setdefault("use_mutation", True)
        kwargs.setdefault("entropy_mutation", True)
        kwargs.setdefault("use_kld", False)
        super().__init__(*args, **kwargs)
