"""Generic filter conformance tests."""

import numpy as np
import pytest

from brobot.sim.world import World
from brobot.filters.sir import SIRPF
from brobot.filters.ra import RAPF
from brobot.filters.kld import KLDAMCL
from brobot.filters.mpf import MPF, KLDMut, KLDVpior, MPFE


def _make_world():
    return World("open", sigma=0.05, r=0.0, seed=42)


def _make_filter(cls, world):
    return cls(
        N=100,
        occ_map_grid=world.occ_map.grid,
        resolution=world.occ_map.resolution,
        sigma=world.sigma,
        beam_angles_arr=world.beam_angles_arr,
    )


FILTER_CLASSES = [SIRPF, RAPF, KLDAMCL, MPF, KLDMut, KLDVpior, MPFE]


@pytest.mark.parametrize("cls", FILTER_CLASSES)
def test_filter_runs(cls):
    world = _make_world()
    filt = _make_filter(cls, world)
    result = filt.run(world, seed=42)
    assert result.rmse_per_step.shape == (200,)
    assert result.mean_rmse >= 0
    assert result.runtime_sec > 0


@pytest.mark.parametrize("cls", FILTER_CLASSES)
def test_filter_low_noise_localizes(cls):
    world = World("open", sigma=0.01, r=0.0, seed=42)
    filt = _make_filter(cls, world)
    result = filt.run(world, seed=42)
    # At very low noise, all filters should localize reasonably
    assert result.mean_rmse < 0.5
    assert not result.diverged


def test_sir_integration_checkpoint():
    """SIR-PF at low noise must achieve < 0.1m mean RMSE."""
    world = World("open", sigma=0.01, r=0.0, seed=42)
    filt = SIRPF(
        N=500,
        occ_map_grid=world.occ_map.grid,
        resolution=world.occ_map.resolution,
        sigma=world.sigma,
        beam_angles_arr=world.beam_angles_arr,
    )
    result = filt.run(world, seed=42)
    assert result.mean_rmse < 0.1
