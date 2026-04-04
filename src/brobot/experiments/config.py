"""Experiment configuration and sweep grid."""

from dataclasses import dataclass
from itertools import product

from brobot.filters.sir import SIRPF
from brobot.filters.ra import RAPF
from brobot.filters.kld import KLDAMCL
from brobot.filters.mpf import MPF, KLDMut, KLDVpior, MPFE


@dataclass
class SimConfig:
    map_name: str
    method: str
    sigma: float
    r: float
    N: int
    T: int
    seed: int


METHOD_REGISTRY = {
    "SIR": SIRPF,
    "RA": RAPF,
    "KLD": KLDAMCL,
    "MPF": MPF,
    "KLD_mut": KLDMut,
    "KLD_vpior": KLDVpior,
    "MPFE": MPFE,
}

SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.5]
OUTLIER_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
MAPS = ["open", "corridor"]
METHODS = list(METHOD_REGISTRY.keys())
N_PARTICLES = 500
T_STEPS = 200
N_TRIALS = 30


def generate_sweep_configs(
    sigmas=SIGMAS,
    rates=OUTLIER_RATES,
    maps=MAPS,
    methods=METHODS,
    n_particles=N_PARTICLES,
    t_steps=T_STEPS,
    n_trials=N_TRIALS,
    base_seed=0,
) -> list[SimConfig]:
    """Generate all configurations for the parameter sweep."""
    configs = []
    for sigma, r, map_name, method in product(sigmas, rates, maps, methods):
        for trial in range(n_trials):
            seed = hash((base_seed, map_name, trial)) % (2**31)
            configs.append(SimConfig(
                map_name=map_name,
                method=method,
                sigma=sigma,
                r=r,
                N=n_particles,
                T=t_steps,
                seed=seed,
            ))
    return configs
