"""MPF and ablation variants: MPF, KLD+Mut, KLD+VPIOR, MPF-E."""

import numpy as np

from brobot.filters.base import BaseFilter
from brobot.filters.components.resampling import systematic_resample
from brobot.filters.components.kld_sampling import kld_adaptive_sample
from brobot.filters.components.vpior import (
    compute_predictive_stats,
    detect_outlier_scan,
    vpior_remove_particles,
)
from brobot.filters.components.mutation import (
    tst_mutation,
    tst_mutation_entropy,
    shannon_entropy,
)
from brobot.sim.motion import sample_motion_batch
from brobot.sim.raycast import batch_raycast


class MPF(BaseFilter):
    """Modified Particle Filter with configurable components.

    Components (all on by default for full MPF):
    - VPIOR: outlier detection via Rényi divergence + VP-inequality particle pruning
    - Mutation: TST mutation operator
    - Entropy mutation: use Shannon entropy instead of timestep (MPF-E)
    - KLD: adaptive particle count via incremental KLD-sampling (Thrun et al. 8.4)
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
        self._entropy = 0.0

    def step(
        self,
        control: np.ndarray,
        observation: np.ndarray,
        timestep: int,
        rng: np.random.Generator,
    ) -> dict:
        if self.use_kld:
            return self._step_kld(control, observation, timestep, rng)
        return self._step_fixed(control, observation, timestep, rng)

    # ------------------------------------------------------------------
    # Path A: fixed-size MPF / MPF-E (unchanged from original MPF flow)
    # ------------------------------------------------------------------
    def _step_fixed(
        self,
        control: np.ndarray,
        observation: np.ndarray,
        timestep: int,
        rng: np.random.Generator,
    ) -> dict:
        v, omega = control
        info: dict = {}

        # 1. Motion update
        self.particles = sample_motion_batch(self.particles, v, omega, rng)

        # 2. Raycast once (used for both the D1 trigger stats and the weight update)
        expected = self.compute_expected_ranges()

        # 3. VPIOR D1 trigger + diagnostic per-beam mask. The observation itself
        #    is NOT modified — paper Algorithm 1 prunes particles after
        #    resampling, not beams before weighting.
        vpior_trigger = False
        if self.use_vpior:
            mu_pred, var_pred = compute_predictive_stats(
                self.weights, expected, self.sigma
            )
            if detect_outlier_scan(mu_pred, var_pred, observation, self.sigma):
                vpior_trigger = True
                sigma_pred = np.sqrt(var_pred)
                beam_mask = np.abs(observation - mu_pred) > 6.66 * sigma_pred
            else:
                beam_mask = np.zeros(self.n_beams, dtype=bool)
            info["vpior_beam_mask"] = beam_mask
        else:
            info["vpior_beam_mask"] = np.zeros(self.n_beams, dtype=bool)

        # 4. Weight update on the RAW observation
        log_w = self.compute_log_weights(observation, expected)
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

        # 5. Systematic resampling
        indices = systematic_resample(self.weights, rng)
        self.weights_prenorm = self.weights_prenorm[indices]
        self.particles = self.particles[indices]
        self.weights[:] = 1.0 / self.N

        # 6. Mutation (post-resampling)
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

        # 7. VPIOR particle pruning, gated by the D1 trigger (paper Algorithm 1).
        if self.use_vpior and vpior_trigger:
            self.particles, particle_mask = vpior_remove_particles(
                self.particles, rng,
            )
            info["vpior_particle_mask"] = particle_mask
        else:
            info["vpior_particle_mask"] = np.zeros(self.N, dtype=bool)

        return info

    # ------------------------------------------------------------------
    # Path B: adaptive KLD-AMCL (KLDMut, KLDVpior)
    # ------------------------------------------------------------------
    def _step_kld(
        self,
        control: np.ndarray,
        observation: np.ndarray,
        timestep: int,
        rng: np.random.Generator,
    ) -> dict:
        v, omega = control
        info: dict = {}

        # --- VPIOR pre-pass: D1 trigger only ------------------------------
        # The D1 trigger needs the predictive scan distribution across the
        # current particle set, which must be computed BEFORE we start
        # drawing new particles. We do one extra batch motion+raycast on the
        # previous particles; those motion-updated particles are DISCARDED
        # (kld_adaptive_sample makes its own independent draws). The
        # observation itself is NOT modified — pruning happens on the
        # particles after sampling+mutation.
        vpior_trigger = False
        if self.use_vpior:
            prev_motion = sample_motion_batch(self.particles, v, omega, rng)
            expected_prev = batch_raycast(
                prev_motion,
                self.beam_angles,
                self.occ_map,
                self.resolution,
                self.d_max,
            )
            mu_pred, var_pred = compute_predictive_stats(
                self.weights, expected_prev, self.sigma,
            )
            if detect_outlier_scan(mu_pred, var_pred, observation, self.sigma):
                vpior_trigger = True
                sigma_pred = np.sqrt(var_pred)
                beam_mask = np.abs(observation - mu_pred) > 6.66 * sigma_pred
            else:
                beam_mask = np.zeros(self.n_beams, dtype=bool)
            info["vpior_beam_mask"] = beam_mask
        else:
            info["vpior_beam_mask"] = np.zeros(self.n_beams, dtype=bool)

        # --- Incremental KLD-AMCL sampling on the RAW observation ---------
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

        # Normalize the returned log-likelihoods (log-sum-exp).
        log_w = new_log_w - np.max(new_log_w)
        w = np.exp(log_w)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones_like(w) / len(w)

        self.particles = new_particles
        self.weights = w
        self.weights_prenorm = np.exp(log_w)
        self.N = len(new_particles)

        # --- Mutation (post-sampling, if enabled) -------------------------
        if self.use_mutation:
            # Mirror the fixed-size path: entropy is computed on the
            # non-uniform importance weights before mutation.
            if self.entropy_mutation:
                self._entropy = shannon_entropy(self.weights)

            best_idx = int(np.argmax(self.weights_prenorm))
            best_particle = self.particles[best_idx]

            if self.entropy_mutation:
                self.particles = tst_mutation_entropy(
                    self.particles,
                    self.weights_prenorm,
                    best_particle,
                    self._entropy,
                )
            else:
                self.particles = tst_mutation(
                    self.particles,
                    self.weights_prenorm,
                    best_particle,
                    timestep,
                )

        # --- VPIOR particle pruning, gated by the D1 trigger --------------
        if self.use_vpior and vpior_trigger:
            self.particles, particle_mask = vpior_remove_particles(
                self.particles, rng,
            )
            info["vpior_particle_mask"] = particle_mask
        else:
            info["vpior_particle_mask"] = np.zeros(self.N, dtype=bool)

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
