import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.distributions.categorical import Categorical
from pettingzoo_env.shooter_env import GRID
from pettingzoo_env.utils import bfs_path, time_average

DEG_EPS = 2

class ScriptedShooterAgent(nn.Module):
    def __init__(self, num_agents):
        super().__init__()
        
        self._num_agents = num_agents
        self._grid_size = GRID
        self.path = None
        self.goal = None
        self._not_moved_counter = 0

        self._MOVE_MAP = {
            ( 0, -1): 1,   # N
            ( 0,  1): 2,   # S
            (-1,  0): 3,   # W
            ( 1,  0): 4,   # E
        }
        
    
    def same_pos(self, a, b, eps):
        return abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps

    # ── helpers ───────────────────────────────────────────────────────────────────
    def angle_to_target(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.degrees(math.atan2(dy, dx)) % 360

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        return layer
    
    def _denorm(self, norm_pos) -> int:
        # Convert a normalised position (0-1) back to an integer grid cell
        return round(norm_pos * (self._grid_size - 1))

    # ── Core Module ────────────────────────────────────────────────────────────────
    def get_value(self, x):
        return self.critic(self.network(x))
    
    def get_action_and_value(self, x, action=None):
        grid_cells    = self._grid_size * self._grid_size
        obs           = x[0]
        agent_half    = self._num_agents >> 1   # // 2 via bit shift
        pos_x_start   = grid_cells
        pos_y_start   = grid_cells + self._num_agents
        hp_start      = grid_cells + (self._num_agents << 1)  # * 2 via bit shift

        # --- Decode self position once ---
        self_x = self._denorm(obs[pos_x_start].item())
        self_y = self._denorm(obs[pos_y_start].item())

        # --- Find first alive enemy (vectorized, no Python loop) ---
        hp_enemy_slice = obs[hp_start + agent_half : hp_start + self._num_agents]
        alive          = (hp_enemy_slice > 0).nonzero(as_tuple=False)
        assert len(alive) > 0, "No living enemy found — episode should have ended."
        e_idx          = alive[0].item() + agent_half          # offset back to full index
        target_x       = self._denorm(obs[pos_x_start + e_idx].item())
        target_y       = self._denorm(obs[pos_y_start + e_idx].item())
        target         = (target_x, target_y)

        # --- Recompute path only when target moved ---
        if self.path is None:
            recompute = True
        else:
            eps        = 1e-3 if len(self.path) < 4 else (5.0 / GRID)
            recompute  = not self.same_pos(self.goal, target, eps=eps)

        if recompute:
            self.goal  = target
            grid_obs   = obs[:grid_cells].reshape(self._grid_size, self._grid_size)
            self.path  = bfs_path(grid_obs, (self_x, self_y), target)

        # --- Close to target: turn to face it ---
        if len(self.path) < 3:
            if self._not_moved_counter > 5:
                return np.random.randint(1, 5)   # random move (1–4), avoids Python list alloc

            # Random heading turn every N steps to skip costly atan2 sometimes
            if self._not_moved_counter % 3 != 0:     # <-- tune the modulo to taste
                return np.random.choice([5, 6])

            heading_idx    = hp_start + self._num_agents + agent_half
            cos_h, sin_h   = obs[heading_idx].item(), obs[heading_idx + 1].item()
            # Fast integer atan2 approximation avoids math.degrees + two conversions
            current_heading = math.degrees(math.atan2(sin_h, cos_h)) % 360
            target_heading  = self.angle_to_target(self_x, self_y, target_x, target_y)
            heading_diff    = (target_heading - current_heading + 180.0) % 360.0 - 180.0

            if heading_diff > DEG_EPS:
                self._not_moved_counter = 0
                return 6   # turn right
            if heading_diff < -DEG_EPS:
                self._not_moved_counter = 0
                return 5   # turn left
            self._not_moved_counter += 1
            return 0

        # --- Follow path ---
        next_x, next_y = self.path.pop(0)
        dx = next_x - self_x
        dy = next_y - self_y

        action = self._MOVE_MAP.get((dx, dy), 0)
        self._not_moved_counter = 0 if action else self._not_moved_counter + 1
        return action
    
    def update(self, rb_obs, rb_actions, rb_logprobs, rb_rewards, rb_terms, rb_values, end_step):
        return