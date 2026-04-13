import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.distributions.categorical import Categorical


class ScriptedShooterAgent(nn.Module):
    def __init__(self, num_agents):
        super().__init__()
        
        self._num_agents = num_agents
        
    
    # ── helpers ───────────────────────────────────────────────────────────────────
    def angle_to_target(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.degrees(math.atan2(dy, dx)) % 360

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        return layer

    # ── Core Module ────────────────────────────────────────────────────────────────
    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        
        # Enemies hp observation start-end
        start_idx = 9 + self._num_agents * 3
        end_idx = int(start_idx + (self._num_agents / 2))
        
        obs = x[0]
        
        # Find first alive enemy, use their position
        agent_idx = int(self._num_agents / 2)
        for i in range(start_idx, end_idx):
            if(obs[i] > 0):
                break
            agent_idx += 1
            
        # Get enemy position
        target_pos_x = obs[agent_idx].item()
        target_pos_y = obs[agent_idx + self._num_agents].item()
        
        # Get self position and heading
        self_x = obs[9].item()
        self_y = obs[self._num_agents].item()
        
        heading = (obs[-2].item(), obs[-1].item())
        
        # Go toward target, if far. Turn toward if close
        action = 0
        x_dist = self_x - target_pos_x
        y_dist = self_y - target_pos_y
        if abs(x_dist) < 4 and abs(y_dist) < 4:
            # Heading
                heading_diff = self.angle_to_target(self_x, self_y, target_pos_x, target_pos_y)
            
                if heading_diff > 0:
                    action = 5 # Turn left
                elif heading_diff < 0:
                    action = 6 # Turn right
                else:
                    action = 0 # On target
        else:
            # Go near
            if abs(x_dist) > 3:
                if x_dist < 0:  # Target is toward east
                    action = 4
                else:           # Target is toward west
                    action = 3
            else:
                if y_dist < 0:  # Target is toward south
                    action = 2
                else:           # Target is toward north
                    action = 1
                    
        return action

    def update(self, rb_obs, rb_actions, rb_logprobs, rb_rewards, rb_terms, rb_values, end_step):
        return