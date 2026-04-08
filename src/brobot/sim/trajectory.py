"""Pre-defined 200-step trajectories for each map."""

import numpy as np

from brobot.sim.maps import OccupancyGrid
from brobot.sim.motion import DT


def _waypoints_open() -> np.ndarray:
    """Waypoints for the open room map (10m x 10m world).

    A rectangular patrol path with some diagonal cuts.
    """
    return np.array([
        [1.0, 1.0],
        [8.0, 1.0],
        [8.0, 5.0],
        [5.0, 5.0],
        [5.0, 8.0],
        [2.0, 8.0],
        [2.0, 3.0],
        [6.0, 3.0],
        [6.0, 7.0],
        [1.0, 7.0],
        [1.0, 1.0],
    ])


def _waypoints_corridor() -> np.ndarray:
    """Waypoints for the corridor map.

    Navigates through hallways and passage gaps.
    Grid: walls at rows 65-67 (y=3.25-3.35m) and 130-132 (y=6.50-6.60m).
    Gaps in wall1 at x: [1.5-2.25], [5.0-5.75], [8.0-8.75]
    Gaps in wall2 at x: [2.5-3.25], [6.5-7.25]
    """
    return np.array([
        [1.5, 1.5],    # Start in bottom corridor
        [5.0, 1.5],    # Move right
        [5.5, 1.5],    # Near gap in wall1
        [5.5, 5.0],    # Up through gap into middle corridor
        [3.0, 5.0],    # Left in middle corridor
        [3.0, 5.0],    # Near gap in wall2
        [3.0, 8.0],    # Up through gap into top corridor
        [7.0, 8.0],    # Right in top corridor
        [7.0, 8.0],    # Near gap in wall2
        [7.0, 5.0],    # Down through gap into middle corridor
        [2.0, 5.0],    # Left in middle
        [2.0, 1.5],    # Down through gap in wall1
        [8.0, 1.5],    # Right in bottom
    ])


def _waypoints_four_rooms() -> np.ndarray:
    """Waypoints for the four-rooms map.

    Navigates through all 4 rooms via the narrow doorways, making a full loop.
    Doorway centers:
      - Vertical wall, bottom: x≈4.95m, y≈2.45m  (connects room1 ↔ room2)
      - Horizontal wall, right: x≈7.45m, y≈4.95m (connects room2 ↔ room4)
      - Vertical wall, top:    x≈4.95m, y≈7.45m  (connects room4 ↔ room3)
      - Horizontal wall, left: x≈2.45m, y≈4.95m  (connects room3 ↔ room1)
    """
    return np.array([
        [1.5, 1.5],    # Room 1 (bottom-left)
        [3.5, 1.5],    # Room 1, moving toward doorway
        [4.7, 2.45],   # Approach bottom vertical doorway
        [5.3, 2.45],   # Through into Room 2 (bottom-right)
        [7.5, 1.5],    # Room 2 interior
        [8.5, 3.5],    # Room 2 upper area
        [7.45, 4.7],   # Approach right horizontal doorway
        [7.45, 5.3],   # Through into Room 4 (top-right)
        [8.5, 7.5],    # Room 4 interior
        [5.3, 7.45],   # Approach top vertical doorway
        [4.7, 7.45],   # Through into Room 3 (top-left)
        [1.5, 8.5],    # Room 3 interior
        [2.45, 5.3],   # Approach left horizontal doorway
        [2.45, 4.7],   # Through back into Room 1
        [1.5, 1.5],    # Back to start
    ])


def _waypoints_snake() -> np.ndarray:
    """Waypoints for the snake map.

    Navigates the zigzag corridors from bottom-left to top-right.
    Grid: 4 horizontal walls at rows 41, 80, 119, 158 (3 cells thick).
    Gaps alternate sides:
      - Walls at rows 41, 119: gap on left  (cols 5–29,  x ≈ 0.25–1.45m)
      - Walls at rows 80, 158: gap on right (cols 170–194, x ≈ 8.50–9.70m)

    Five corridors (bottom to top):
      Corridor 1: y ≈ 0.10–2.05m  (rows 2–40)
      Corridor 2: y ≈ 2.20–4.00m  (rows 44–79)
      Corridor 3: y ≈ 4.15–5.95m  (rows 83–118)
      Corridor 4: y ≈ 6.10–7.90m  (rows 122–157)
      Corridor 5: y ≈ 8.05–9.90m  (rows 161–197)
    """
    return np.array([
        [9.0, 1.0],    # Start bottom-right, corridor 1
        [0.85, 1.0],   # Back left toward gap (wall 41 gap on left)
        [0.85, 3.0],   # Through left gap into corridor 2
        [9.0, 3.0],    # Right across corridor 2
        [9.0, 5.0],    # Through right gap (wall 80) into corridor 3
        [0.85, 5.0],   # Left across corridor 3
        [0.85, 7.0],   # Through left gap (wall 119) into corridor 4
        [9.0, 7.0],    # Right across corridor 4
        [9.0, 9.0],    # Through right gap (wall 158) into corridor 5
        [9.0, 9.0],    # Finish top-right
    ])


def generate_trajectory(
    occ_map: OccupancyGrid,
    map_name: str,
    T: int = 200,
    dt: float = DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a T-step trajectory through waypoints.

    Uses simple proportional control to follow waypoints.

    Returns
    -------
    gt_poses : ndarray (T+1, 3)
        Ground truth poses [px, py, theta] at each timestep.
    controls : ndarray (T, 2)
        Control inputs [v, omega] for each timestep.
    """
    if map_name == "open":
        waypoints = _waypoints_open()
    elif map_name == "corridor":
        waypoints = _waypoints_corridor()
    elif map_name == "four_rooms":
        waypoints = _waypoints_four_rooms()
    elif map_name == "snake":
        waypoints = _waypoints_snake()
    else:
        raise ValueError(f"Unknown map: {map_name}")

    # Speed parameters
    max_v = 0.15       # m/s
    max_omega = 0.3    # rad/s
    wp_threshold = 0.3  # m, distance to switch waypoint

    gt_poses = np.zeros((T + 1, 3))
    controls = np.zeros((T, 2))

    # Start at first waypoint, facing toward second
    gt_poses[0, 0] = waypoints[0, 0]
    gt_poses[0, 1] = waypoints[0, 1]
    dx = waypoints[1, 0] - waypoints[0, 0]
    dy = waypoints[1, 1] - waypoints[0, 1]
    gt_poses[0, 2] = np.arctan2(dy, dx)

    wp_idx = 1

    for t in range(T):
        px, py, theta = gt_poses[t]

        # Current target waypoint
        if wp_idx < len(waypoints):
            target = waypoints[wp_idx]
        else:
            # Cycle back
            wp_idx = 0
            target = waypoints[wp_idx]

        dx = target[0] - px
        dy = target[1] - py
        dist = np.sqrt(dx**2 + dy**2)

        # Switch waypoint if close enough
        if dist < wp_threshold and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            target = waypoints[wp_idx]
            dx = target[0] - px
            dy = target[1] - py
            dist = np.sqrt(dx**2 + dy**2)

        # Desired heading
        desired_theta = np.arctan2(dy, dx)
        angle_error = desired_theta - theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # Proportional control
        omega = np.clip(2.0 * angle_error, -max_omega, max_omega)
        # Slow down when turning sharply
        v = max_v * max(0.2, 1.0 - abs(angle_error) / np.pi)

        controls[t] = [v, omega]

        # Apply motion (ground truth, no noise)
        ds = v * dt
        dtheta = omega * dt
        gt_poses[t + 1, 0] = px + ds * np.cos(theta + dtheta / 2)
        gt_poses[t + 1, 1] = py + ds * np.sin(theta + dtheta / 2)
        gt_poses[t + 1, 2] = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi

    return gt_poses, controls
