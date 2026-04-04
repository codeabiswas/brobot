"""Experiment runner: single run and parallel sweep."""

import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from brobot.experiments.config import SimConfig, METHOD_REGISTRY
from brobot.filters.base import FilterResult
from brobot.sim.world import World


def run_single(config: SimConfig) -> dict:
    """Run a single experiment configuration.

    Returns a flat dict suitable for DataFrame construction.
    """
    world = World(
        map_name=config.map_name,
        sigma=config.sigma,
        r=config.r,
        T=config.T,
        seed=config.seed,
    )

    filter_cls = METHOD_REGISTRY[config.method]
    filt = filter_cls(
        N=config.N,
        occ_map_grid=world.occ_map.grid,
        resolution=world.occ_map.resolution,
        sigma=config.sigma,
        beam_angles_arr=world.beam_angles_arr,
    )

    # Use a different seed for the filter's RNG (independent of world generation)
    filter_seed = hash((config.seed, config.method)) % (2**31)
    result = filt.run(world, seed=filter_seed)

    return {
        "map_name": config.map_name,
        "method": config.method,
        "sigma": config.sigma,
        "r": config.r,
        "N": config.N,
        "T": config.T,
        "seed": config.seed,
        "mean_rmse": result.mean_rmse,
        "std_rmse": result.std_rmse,
        "diverged": result.diverged,
        "vpior_recall": result.vpior_recall,
        "runtime_sec": result.runtime_sec,
    }


def _run_single_wrapper(config_dict: dict) -> dict:
    """Wrapper for multiprocessing (pickling support)."""
    config = SimConfig(**config_dict)
    return run_single(config)


def run_sweep(
    configs: list[SimConfig],
    n_workers: int | None = None,
    output_path: str = "results/sweep.csv",
) -> pd.DataFrame:
    """Run the full parameter sweep with multiprocessing.

    Parameters
    ----------
    configs : list[SimConfig]
    n_workers : int or None
        Number of worker processes. None = cpu_count.
    output_path : str
        Path to save results CSV.

    Returns
    -------
    df : DataFrame
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    # Convert to dicts for pickling
    config_dicts = [
        {
            "map_name": c.map_name,
            "method": c.method,
            "sigma": c.sigma,
            "r": c.r,
            "N": c.N,
            "T": c.T,
            "seed": c.seed,
        }
        for c in configs
    ]

    results = []
    with mp.Pool(n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_run_single_wrapper, config_dicts),
            total=len(config_dicts),
            desc="Running sweep",
        ):
            results.append(result)

    df = pd.DataFrame(results)

    # Save results
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    return df
