"""Reproduction check tables: compare results against paper Tables 3-4."""

import numpy as np
import pandas as pd

from brobot.sim.world import World
from brobot.experiments.config import METHOD_REGISTRY


def reproduction_check(
    n_trials: int = 30,
    sigma: float = 0.05,
    r: float = 0.0,
) -> pd.DataFrame:
    """Run SIR-PF, RA-PF, and MPF on clean data and report mean RMSE.

    Compares against the paper's Tables 3-4 values.
    """
    methods = ["SIR", "RA", "MPF"]
    results = {m: [] for m in methods}

    for trial in range(n_trials):
        seed = hash(("open", trial)) % (2**31)
        world = World(map_name="open", sigma=sigma, r=r, T=200, seed=seed)

        for method in methods:
            filter_cls = METHOD_REGISTRY[method]
            filt = filter_cls(
                N=500,
                occ_map_grid=world.occ_map.grid,
                resolution=world.occ_map.resolution,
                sigma=sigma,
                beam_angles_arr=world.beam_angles_arr,
            )
            filter_seed = hash((seed, method)) % (2**31)
            result = filt.run(world, seed=filter_seed)
            results[method].append(result.mean_rmse)

    rows = []
    for method in methods:
        rmses = results[method]
        rows.append({
            "Method": method,
            "Mean RMSE": np.mean(rmses),
            "Std RMSE": np.std(rmses),
            "N trials": n_trials,
        })

    df = pd.DataFrame(rows)
    print("\nReproduction Check (σ=0.05, r=0, open map, N=500)")
    print("=" * 50)
    print(df.to_string(index=False))
    return df
