from collections import deque
import random

import numpy as np


def generate_shooter_map(size=25, spawns=None, seed=None):
    if spawns is None:
        spawns = []

    random.seed(seed)
    size = max(5, min(25, size))

    grid = [[0]*size for _ in range(size)]

    # ---- Borders ----
    for i in range(size):
        grid[0][i] = grid[size-1][i] = 1
        grid[i][0] = grid[i][size-1] = 1

    # ---- Density scales nicely with size ----
    density = 0.15 + (size - 5) * 0.004

    # ---- Interior random walls ----
    for x in range(1, size-1):
        for y in range(1, size-1):
            if (x, y) in spawns:
                continue
            if random.random() < density:
                grid[x][y] = 1

    # ---- Guaranteed open corridors (prevents isolation) ----
    for i in range(2, size-2, 4):
        grid[i][random.randint(1, size-2)] = 0
        grid[random.randint(1, size-2)][i] = 0

    # ---- Short wall segments (LOS blockers) ----
    for _ in range(size // 5):
        x = random.randint(2, size-3)
        y = random.randint(2, size-3)

        if random.random() < 0.5:
            # horizontal
            for dx in range(2):
                if (x+dx, y) not in spawns:
                    grid[x+dx][y] = 1
        else:
            # vertical
            for dy in range(2):
                if (x, y+dy) not in spawns:
                    grid[x][y+dy] = 1

    return grid


def get_surrounding(grid, x, y):
    surrounding = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):  
            nx, ny = x + dx, y + dy
            surrounding.append(grid[ny][nx])
    
    return surrounding

# ── BFS pathfinder ────────────────────────────────────────────────────────
def bfs_path(grid_map: np.ndarray, start: tuple[int, int],
                goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Return a list of (x, y) grid cells from start (exclusive) to goal
    (inclusive).  grid_map is indexed [y, x] where 1 = wall.
    Returns an empty list if no path exists."""
    if start == goal:
        return []

    h, w = grid_map.shape
    visited = {start}
    # queue holds (current_pos, path_so_far)
    queue = deque([(start, [])])

    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):  # N S W E
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid_map[ny, nx] == 1:   # wall
                continue
            npos = (nx, ny)
            if npos in visited:
                continue
            visited.add(npos)
            new_path = path + [npos]
            if npos == goal:
                return new_path
            queue.append((npos, new_path))

    return []   # unreachable