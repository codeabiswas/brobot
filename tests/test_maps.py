"""Tests for occupancy grid maps."""

import numpy as np
from brobot.sim.maps import open_map, corridor_map, four_rooms_map, OccupancyGrid


def test_open_map_shape():
    m = open_map()
    assert m.grid.shape == (200, 200)
    assert m.resolution == 0.05


def test_open_map_walls():
    m = open_map()
    # Perimeter is walls
    assert m.grid[0, :].sum() > 0
    assert m.grid[-1, :].sum() > 0
    assert m.grid[:, 0].sum() > 0
    assert m.grid[:, -1].sum() > 0


def test_open_map_center_free():
    m = open_map()
    assert m.is_free(5.0, 5.0)
    assert not m.is_occupied(5.0, 5.0)


def test_open_map_wall_occupied():
    m = open_map()
    assert m.is_occupied(0.0, 0.0)


def test_corridor_map_has_internal_walls():
    m = corridor_map()
    # Should have more walls than open map
    open_walls = open_map().grid.sum()
    assert m.grid.sum() > open_walls


def test_corridor_map_has_gaps():
    m = corridor_map()
    # Wall at row 65, gap at cols 30-44
    assert m.grid[65, 35] == 0  # gap
    assert m.grid[65, 10] == 1  # wall


def test_out_of_bounds_is_occupied():
    m = open_map()
    assert m.is_occupied(-1.0, -1.0)
    assert m.is_occupied(100.0, 100.0)


def test_four_rooms_doorways_open():
    m = four_rooms_map()
    # All four doorways must be free (traversable)
    assert m.is_free(4.95, 2.45)   # vertical wall, bottom doorway
    assert m.is_free(4.95, 7.45)   # vertical wall, top doorway
    assert m.is_free(2.45, 4.95)   # horizontal wall, left doorway
    assert m.is_free(7.45, 4.95)   # horizontal wall, right doorway


def test_four_rooms_walls_exist():
    m = four_rooms_map()
    # Center walls must be solid between doorways
    assert m.is_occupied(4.95, 5.5)   # vertical wall center
    assert m.is_occupied(2.0, 4.95)   # horizontal wall center


def test_four_rooms_interiors_free():
    m = four_rooms_map()
    # Each room interior must be navigable
    assert m.is_free(2.0, 2.0)    # room 1 (bottom-left)
    assert m.is_free(7.5, 2.0)    # room 2 (bottom-right)
    assert m.is_free(2.0, 7.5)    # room 3 (top-left)
    assert m.is_free(7.5, 7.5)    # room 4 (top-right)
