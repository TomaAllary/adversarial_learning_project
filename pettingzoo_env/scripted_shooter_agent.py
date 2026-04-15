import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.distributions.categorical import Categorical
from pettingzoo_env.shooter_env import GRID
from pettingzoo_env.utils import bfs_path

DEG_EPS = 2

class ScriptedShooterAgent(nn.Module):
    def __init__(self, num_agents):
        super().__init__()
        
        self._num_agents = num_agents
        self._grid_size = GRID
        self.path = None
        self.goal = None
        self._not_moved_counter = 0
        
    
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
        grid_cells = self._grid_size * self._grid_size
        obs        = x[0]
 
        pos_x_start = grid_cells
        pos_y_start = grid_cells + self._num_agents
 
        self_x = self._denorm(obs[pos_x_start].item())
        self_y = self._denorm(obs[pos_y_start].item())
 
        agent_per_team = self._num_agents // 2
        hp_start = grid_cells + (2*self._num_agents)

        # hp indices for enemies start at hp_start + agent_per_team
        # Find the first alive enemy (hp > 0) and grab their position
        target_x, target_y = None, None
        for e_idx in range(agent_per_team, self._num_agents):
            hp_idx = hp_start + e_idx
            if obs[hp_idx].item() > 0:
                target_x = self._denorm(obs[pos_x_start + e_idx].item())
                target_y = self._denorm(obs[pos_y_start + e_idx].item())
                break

        if target_x is None:
            assert target_x is not None, "No living enemy found in observation! This should not happen since episode should end when all enemies are dead."
 
        target  = (target_x, target_y)
        start = (self_x, self_y)
 
        # Pathfind
        compute_new_path = self.path is None
        if not compute_new_path:
            not_moved = self.same_pos(self.goal, target, eps=1e-3) if len(self.path) < 4 else self.same_pos(self.goal, target, eps=(5/GRID))
            compute_new_path = not not_moved
        if compute_new_path:
            self.goal = target
            observed_grid = obs[:grid_cells].reshape(self._grid_size, self._grid_size)
            self.path = bfs_path(observed_grid, start, self.goal)

        # If close, turn to face target
        if len(self.path) < 3:
            if(self._not_moved_counter > 5):
                # Random movement if stuck for too long
                return np.random.choice([1, 2, 3, 4])
            else:
                cos_h, sin_h = obs[-2].item(), obs[-1].item()
                current_heading = math.degrees(math.atan2(sin_h, cos_h)) % 360
                target_heading = self.angle_to_target(self_x, self_y, target_x, target_y)

                # Shortest angular difference in (-180, 180)
                heading_diff = (target_heading - current_heading + 180) % 360 - 180
                if   heading_diff >  DEG_EPS: 
                    self._not_moved_counter = 0
                    return 6   # turn right  (counter-clockwise)
                elif heading_diff < -DEG_EPS: 
                    self._not_moved_counter = 0
                    return 5   # turn left (clockwise)
                else:        
                    self._not_moved_counter += 1           
                    return 0 

        # Otherwise, follow path and consume waypoint
        next_x, next_y = self.path.pop(0)
        dx = next_x - self_x
        dy = next_y - self_y
 
        action = 0
        if   dx == 0 and dy == -1: action = 1   # N
        elif dx == 0 and dy ==  1: action = 2   # S
        elif dx == -1 and dy == 0: action = 3   # W
        elif dx == 1  and dy == 0: action = 4   # E
        else:
            action = 0 # stay
        
        self._not_moved_counter = 0 if action != 0 else self._not_moved_counter + 1
        return action

    def update(self, rb_obs, rb_actions, rb_logprobs, rb_rewards, rb_terms, rb_values, end_step):
        return