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


def four_rooms_map(size: int = 200, resolution: float = 0.05) -> OccupancyGrid:
    """Four identical rooms connected by narrow doorways.

    Layout (200x200 grid = 10m x 10m world):
    - Perimeter walls (2 cells thick)
    - Vertical center wall at cols 98–101 (x ≈ 4.9–5.05m)
    - Horizontal center wall at rows 98–101 (y ≈ 4.9–5.05m)
    - Four doorways (15-cell = 0.75m gaps), one per wall segment:
        - Vertical wall, bottom half: rows 42–56  (y ≈ 2.1–2.8m)
        - Vertical wall, top half:    rows 143–157 (y ≈ 7.15–7.85m)
        - Horizontal wall, left half: cols 42–56   (x ≈ 2.1–2.8m)
        - Horizontal wall, right half: cols 143–157 (x ≈ 7.15–7.85m)

    All four rooms are geometrically identical, creating global localization
    ambiguity — the sensor returns look nearly the same in each room.
    """
    grid = np.zeros((size, size), dtype=np.int8)

    # Perimeter walls (2 cells thick)
    grid[0:2, :] = 1
    grid[-2:, :] = 1
    grid[:, 0:2] = 1
    grid[:, -2:] = 1

    wall_thickness = 4  # cols/rows 98-101

    # Vertical center wall
    grid[:, 98 : 98 + wall_thickness] = 1
    # Doorway: bottom half — rows 42–56
    grid[42:57, 98 : 98 + wall_thickness] = 0
    # Doorway: top half — rows 143–157
    grid[143:158, 98 : 98 + wall_thickness] = 0

    # Horizontal center wall
    grid[98 : 98 + wall_thickness, :] = 1
    # Doorway: left half — cols 42–56
    grid[98 : 98 + wall_thickness, 42:57] = 0
    # Doorway: right half — cols 143–157
    grid[98 : 98 + wall_thickness, 143:158] = 0

    return OccupancyGrid(grid, resolution)


def snake_map(size: int = 200, resolution: float = 0.05) -> OccupancyGrid:
    """Snake-like corridor winding from bottom-left to top-right.

    Layout (200x200 grid = 10m x 10m world):
    - Perimeter walls (2 cells thick)
    - Five horizontal corridors (~36 cells / 1.8m wide each)
    - Four horizontal dividing walls (3 cells thick) at rows 41, 80, 119, 158
    - 25-cell (1.25m) gaps alternate sides:
        - Walls 1 & 3 (rows 41, 119): gap on left  (cols 5–29)
        - Walls 2 & 4 (rows 80, 158): gap on right (cols 170–194)
    - Path snakes: bottom-left → right → up-right → left → up-left → ... → top-right
    """
    grid = np.zeros((size, size), dtype=np.int8)

    # Perimeter walls (2 cells thick)
    grid[0:2, :] = 1
    grid[-2:, :] = 1
    grid[:, 0:2] = 1
    grid[:, -2:] = 1

    wall_thickness = 3
    gap_width = 25  # cells

    wall_rows = [41, 80, 119, 158]
    for i, wr in enumerate(wall_rows):
        grid[wr : wr + wall_thickness, :] = 1
        if i % 2 == 0:
            # Gap on left side
            grid[wr : wr + wall_thickness, 5 : 5 + gap_width] = 0
        else:
            # Gap on right side
            grid[wr : wr + wall_thickness, 170 : 170 + gap_width] = 0

    return OccupancyGrid(grid, resolution)


MAP_REGISTRY = {
    "open": open_map,
    "corridor": corridor_map,
    "four_rooms": four_rooms_map,
    "snake": snake_map,
}


def get_map(name: str) -> OccupancyGrid:
    """Get a map by name."""
    return MAP_REGISTRY[name]()
