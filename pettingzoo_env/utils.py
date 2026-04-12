import random

def generate_shooter_map(size=25, spawns=[], seed=None):
    random.seed(seed)

    # Initialize empty map
    grid = [[0 for _ in range(size)] for _ in range(size)]

    # Add borders
    for i in range(size):
        grid[0][i] = 1
        grid[size-1][i] = 1
        grid[i][0] = 1
        grid[i][size-1] = 1

    # Parameters controlling structure
    num_blocks = size // 3          # number of wall clusters
    block_size = 2                 # size of clusters

    for _ in range(num_blocks):
        # Work only in one quadrant for symmetry
        x = random.randint(2, size//2 - 2)
        y = random.randint(2, size//2 - 2)

        for dx in range(block_size):
            for dy in range(block_size):
                coords = [
                    (x+dx, y+dy),
                    (size-1-(x+dx), y+dy),
                    (x+dx, size-1-(y+dy)),
                    (size-1-(x+dx), size-1-(y+dy))
                ]

                for cx, cy in coords:
                    # Check if this cell is a spawn point; if so, skip it
                    if (cx, cy) in spawns:
                        continue
                    if 0 < cx < size-1 and 0 < cy < size-1:
                        grid[cx][cy] = 1

    # Optional: add some vertical/horizontal bars
    for _ in range(size // 6):
        row = random.randint(2, size-3)
        col = random.randint(2, size-3)

        for i in range(2, size-2):
            if random.random() < 0.15:
                grid[row][i] = 1
                grid[size-1-row][i] = 1
            if random.random() < 0.15:
                grid[i][col] = 1
                grid[i][size-1-col] = 1

    return grid