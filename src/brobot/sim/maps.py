"""Occupancy grid maps for 2D localization simulation."""

import numpy as np


class OccupancyGrid:
    """Binary occupancy grid. 1 = occupied (wall), 0 = free."""

    def __init__(self, grid: np.ndarray, resolution: float = 0.05):
        self.grid = grid.astype(np.int8)
        self.resolution = resolution
        self.height, self.width = grid.shape
        self.world_height = self.height * resolution
        self.world_width = self.width * resolution

    def is_occupied(self, px: float, py: float) -> bool:
        col = int(px / self.resolution)
        row = int(py / self.resolution)
        if 0 <= row < self.height and 0 <= col < self.width:
            return bool(self.grid[row, col])
        return True  # out of bounds = occupied

    def is_free(self, px: float, py: float) -> bool:
        return not self.is_occupied(px, py)


def open_map(size: int = 200, resolution: float = 0.05) -> OccupancyGrid:
    """Single rectangular room with perimeter walls.

    200x200 at 0.05 m/cell = 10m x 10m world.
    """
    grid = np.zeros((size, size), dtype=np.int8)
    # Perimeter walls (2 cells thick for robustness)
    grid[0:2, :] = 1
    grid[-2:, :] = 1
    grid[:, 0:2] = 1
    grid[:, -2:] = 1
    return OccupancyGrid(grid, resolution)


def corridor_map(size: int = 200, resolution: float = 0.05) -> OccupancyGrid:
    """Parallel hallways connected by perpendicular passages.

    Layout (200x200 grid):
    - Perimeter walls
    - Two horizontal walls creating 3 corridors
    - Gaps in the horizontal walls for passage connections
    """
    grid = np.zeros((size, size), dtype=np.int8)

    # Perimeter walls (2 cells thick)
    grid[0:2, :] = 1
    grid[-2:, :] = 1
    grid[:, 0:2] = 1
    grid[:, -2:] = 1

    wall_thickness = 3

    # Horizontal wall 1 at ~1/3 height (row 65)
    wall1_row = 65
    grid[wall1_row : wall1_row + wall_thickness, :] = 1
    # Gaps for passages
    grid[wall1_row : wall1_row + wall_thickness, 30:45] = 0
    grid[wall1_row : wall1_row + wall_thickness, 100:115] = 0
    grid[wall1_row : wall1_row + wall_thickness, 160:175] = 0

    # Horizontal wall 2 at ~2/3 height (row 130)
    wall2_row = 130
    grid[wall2_row : wall2_row + wall_thickness, :] = 1
    # Gaps offset from wall 1 for interesting navigation
    grid[wall2_row : wall2_row + wall_thickness, 50:65] = 0
    grid[wall2_row : wall2_row + wall_thickness, 130:145] = 0

    return OccupancyGrid(grid, resolution)


MAP_REGISTRY = {
    "open": open_map,
    "corridor": corridor_map,
}


def get_map(name: str) -> OccupancyGrid:
    """Get a map by name."""
    return MAP_REGISTRY[name]()
