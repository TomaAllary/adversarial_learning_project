"""
3 v 3 tactical shooter environment (PettingZoo ParallelEnv)

Teams
-----
  Red:  red_0, red_1, red_2   (start top-left cluster)
  Blue: blue_0, blue_1, blue_2 (start bottom-right cluster)

Observation per agent  (flat float32 vector, length = OBS_DIM)
--------------------------------------------------------------
  For each of the 6 agents (self first, then teammates, then enemies):
    - norm_x, norm_y          (0-1 each)
    - hp_ratio                (0-1)
    - alive                   (0 or 1)
    - in_my_cone              (0 or 1, 1 if that agent is inside MY vision cone)
  = 5 features x 6 agents = 30
  Plus self heading (sin, cos) = 2
  Total = 32  ->  OBS_DIM = 32

Actions (Discrete 8)
--------------------
  0 = stay / shoot
  1 = move N       5 = rotate left  (-45°)
  2 = move S       6 = rotate right (+45°)
  3 = move W       
  4 = move E

Vision cone
-----------
  Half-angle  = 45°  (90° total)
  Max range   = 4 cells?
  An enemy inside the cone + no wall blocking -> shoot probability applied each step

Map
---
  grid, 0 = floor, 1 = wall
  Agents cannot enter wall cells.
"""

import functools
import math
import random
from copy import deepcopy
from typing import Optional

import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from pettingzoo_env.utils import generate_shooter_map

# ── constants ─────────────────────────────────────────────────────────────────
GRID       = 17
CELL       = 48          # pixels per cell
MAX_STEPS  = 200
HP_MAX     = 5
SHOOT_PROB = 0.4         # hit probability per step if enemy is in cone
CONE_HALF  = math.radians(45)
CONE_RANGE = 4           # cells
OBS_DIM    = 32

HEADINGS = [0, 45, 90, 135, 180, 225, 270, 315]   # degrees, 0 = East

# Static map:  1 = wall, 0 = open
MAP = np.array(generate_shooter_map(GRID), dtype=np.int8)

RED_SPAWNS  = [(1,1),(2,1),(1,2)]
BLUE_SPAWNS = [(GRID-2,1),(GRID-2,2),(GRID-2,3)]

# ── colour palette ────────────────────────────────────────────────────────────
C_BG        = ( 20,  22,  28)
C_WALL      = ( 55,  60,  80)
C_FLOOR     = ( 32,  36,  48)
C_GRID      = ( 40,  45,  60)
C_RED       = (230,  70,  70)
C_RED_HIT   = (150,  70,  70)
C_BLUE      = ( 70, 140, 230)
C_BLUE_HIT  = ( 70, 210, 230)
C_DEAD      = ( 80,  80,  90)
C_CONE_R    = (230,  70,  70,  35)
C_CONE_B    = ( 70, 140, 230,  35)
C_HP_BG     = ( 20,  20,  25)
C_HP_RED    = (220,  50,  50)
C_HP_BLUE   = ( 50, 130, 220)
C_HP_OK     = ( 50, 220,  90)


# ── helpers ───────────────────────────────────────────────────────────────────

def _deg_to_vec(deg: float):
    rad = math.radians(deg)
    return math.cos(rad), math.sin(rad)   # (dx, dy) in grid space  (y-down)


def _has_los(grid, x0, y0, x1, y1) -> bool:
    """Bresenham line-of-sight: returns False if any wall blocks the path."""
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    cx, cy = x0, y0
    while (cx, cy) != (x1, y1):
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; cx += sx
        if e2 < dx:
            err += dx; cy += sy
        if grid[cy, cx] == 1:
            return False
    return True


def _in_cone(ax, ay, a_deg, bx, by) -> bool:
    """Is (bx,by) inside the vision cone of an agent at (ax,ay) facing a_deg?"""
    dist = math.hypot(bx-ax, by-ay)
    if dist < 1e-6 or dist > CONE_RANGE:
        return False
    vx, vy = _deg_to_vec(a_deg)
    tx, ty = (bx-ax)/dist, (by-ay)/dist
    dot = vx*tx + vy*ty
    return dot >= math.cos(CONE_HALF)


# ── environment ───────────────────────────────────────────────────────────────

class ShooterEnvironment_3v3(ParallelEnv):
    metadata = {"name": "shooter_3v3_v0", "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None,
                 cell_size: int = CELL, fps: int = 10):
        self.render_mode = render_mode
        self.cell_size   = cell_size
        self.fps         = fps

        self.possible_agents = [
            "red_0","red_1","red_2",
            "blue_0","blue_1","blue_2",
        ]
        self._agent_team = {a: a.split("_")[0] for a in self.possible_agents}
        self._agent_idx  = {a: i for i, a in enumerate(self.possible_agents)}

        # pygame state
        self._screen = None
        self._clock  = None
        self._hit_flash: dict = {}   # agent → flash_ttl

        # game state (filled in reset)
        self._x    = {}
        self._y    = {}
        self._hp   = {}
        self._deg  = {}
        self._alive= {}
        self.timestep = 0

    # ── spaces ────────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(7)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        self.agents   = list(self.possible_agents)
        self.timestep = 0
        self._hit_flash = {}

        spawns = RED_SPAWNS + BLUE_SPAWNS
        for i, a in enumerate(self.possible_agents):
            self._x[a]     = spawns[i][0]
            self._y[a]     = spawns[i][1]
            self._hp[a]    = HP_MAX
            self._deg[a]   = 0.0 if "red" in a else 180.0
            self._alive[a] = True

        obs   = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, actions):
        rewards     = {a: 0.0 for a in self.agents}
        terminations= {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        # — move / rotate —
        for a in self.agents:
            if not self._alive[a]:
                continue
            act = actions[a]
            nx, ny, nd = self._x[a], self._y[a], self._deg[a]

            if   act == 1: ny -= 1                          # N
            elif act == 2: ny += 1                          # S
            elif act == 3: nx -= 1                          # W
            elif act == 4: nx += 1                          # E
            elif act == 5: nd = (nd - 45) % 360            # rotate L
            elif act == 6: nd = (nd + 45) % 360            # rotate R

            # clamp + wall check
            nx = max(0, min(GRID-1, nx))
            ny = max(0, min(GRID-1, ny))
            if MAP[ny, nx] == 0:
                self._x[a], self._y[a] = nx, ny
            self._deg[a] = nd

        # — shooting —
        for a in self.agents:
            if not self._alive[a]:
                continue
            my_team = self._agent_team[a]
            for e in self.agents:
                if e == a or self._agent_team[e] == my_team or not self._alive[e]:
                    continue
                if _in_cone(self._x[a], self._y[a], self._deg[a],
                            self._x[e], self._y[e]):
                    if _has_los(MAP, self._x[a], self._y[a],
                                self._x[e], self._y[e]):
                        if random.random() < SHOOT_PROB:
                            self._hp[e] -= 1
                            rewards[a]  += 0.1
                            self._hit_flash[e] = 4   # flash frames
                            if self._hp[e] <= 0:
                                self._hp[e]    = 0
                                self._alive[e] = False
                                rewards[a]    += 1.0
                                rewards[e]    -= 1.0

        # — check win condition —
        red_alive  = any(self._alive[a] for a in self.possible_agents if "red"  in a)
        blue_alive = any(self._alive[a] for a in self.possible_agents if "blue" in a)
        game_over  = not red_alive or not blue_alive

        if game_over:
            for a in self.agents:
                terminations[a] = True
                if "red"  in a and not red_alive:  rewards[a] -= 2.0
                if "blue" in a and not blue_alive: rewards[a] -= 2.0
                if "red"  in a and not blue_alive: rewards[a] += 2.0
                if "blue" in a and not red_alive:  rewards[a] += 2.0

        self.timestep += 1
        if self.timestep >= MAX_STEPS:
            for a in self.agents:
                truncations[a] = True

        obs   = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        if game_over or self.timestep >= MAX_STEPS:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # ── observation builder ───────────────────────────────────────────────────

    def _observe(self, agent) -> np.ndarray:
        team  = self._agent_team[agent]
        order = [agent]
        # teammates first, then enemies
        for a in self.possible_agents:
            if a != agent and self._agent_team[a] == team:
                order.append(a)
        for a in self.possible_agents:
            if self._agent_team[a] != team:
                order.append(a)

        feats = []
        for a in order:
            feats += [
                self._x[a] / (GRID-1),
                self._y[a] / (GRID-1),
                self._hp[a] / HP_MAX,
                float(self._alive[a]),
                float(_in_cone(self._x[agent], self._y[agent], self._deg[agent],
                               self._x[a], self._y[a])) if a != agent else 0.0,
            ]
        vx, vy = _deg_to_vec(self._deg[agent])
        feats += [vx, vy]
        return np.array(feats, dtype=np.float32)

    # ── pygame rendering ──────────────────────────────────────────────────────

    def _init_pygame(self):
        if self._screen is not None:
            return
        pygame.init()
        W = GRID * self.cell_size
        self._screen = pygame.display.set_mode((W, W))
        pygame.display.set_caption("3v3 Tactical Shooter")
        self._clock = pygame.time.Clock()
        self._font  = pygame.font.SysFont("monospace", 11, bold=True)
        self._font_big = pygame.font.SysFont("monospace", 16, bold=True)

    def render(self):
        if self.render_mode != "human":
            return
        self._init_pygame()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close(); raise SystemExit

        surf = self._screen
        W    = GRID * self.cell_size
        C    = self.cell_size

        # — background & grid —
        surf.fill(C_BG)
        for gy in range(GRID):
            for gx in range(GRID):
                r = pygame.Rect(gx*C, gy*C, C, C)
                color = C_WALL if MAP[gy, gx] == 1 else C_FLOOR
                pygame.draw.rect(surf, color, r)
                pygame.draw.rect(surf, C_GRID, r, 1)

        # — vision cones (semi-transparent layer) —
        cone_surf = pygame.Surface((W, W), pygame.SRCALPHA)
        for a in self.possible_agents:
            if not self._alive[a]: continue
            col = (*C_RED[:3], 35) if "red" in a else (*C_BLUE[:3], 35)
            self._draw_cone(cone_surf, a, col)
        surf.blit(cone_surf, (0, 0))

        # — agents —
        for a in self.possible_agents:
            ax = self._x[a] * C + C//2
            ay = self._y[a] * C + C//2
            radius = C//2 - 6

            if not self._alive[a]:
                pygame.draw.circle(surf, C_DEAD, (ax, ay), radius//2)
                continue

            col  = C_RED if "red" in a else C_BLUE
            # flash on hit
            if self._hit_flash.get(a, 0) > 0:
                col = C_RED_HIT if "red" in a else C_BLUE_HIT
                self._hit_flash[a] -= 1
                
            pygame.draw.circle(surf, col, (ax, ay), radius)
            pygame.draw.circle(surf, (255,255,255), (ax, ay), radius, 2)

            # agent label
            label = self._font.render(a[-1], True, (255,255,255))
            surf.blit(label, (ax - label.get_width()//2, ay - label.get_height()//2))

            # HP bar
            bar_w = C - 12
            bar_h = 5
            bx = self._x[a] * C + 6
            by = self._y[a] * C + 2
            pygame.draw.rect(surf, C_HP_BG,  (bx, by, bar_w, bar_h))
            hp_frac = self._hp[a] / HP_MAX
            hp_col  = C_HP_OK if hp_frac > 0.5 else C_HP_RED
            pygame.draw.rect(surf, hp_col, (bx, by, int(bar_w * hp_frac), bar_h))

        # — HUD: score / step —
        red_hp  = sum(self._hp[a] for a in self.possible_agents if "red"  in a)
        blue_hp = sum(self._hp[a] for a in self.possible_agents if "blue" in a)
        hud = self._font_big.render(
            f"RED HP:{red_hp:2d}   STEP:{self.timestep:3d}   BLUE HP:{blue_hp:2d}",
            True, (200, 200, 210))
        surf.blit(hud, (W//2 - hud.get_width()//2, W - 20))

        pygame.display.flip()
        self._clock.tick(self.fps)

    def _draw_cone(self, surf, agent, color):
        C = self.cell_size
        ax = self._x[agent] * C + C//2
        ay = self._y[agent] * C + C//2
        deg = self._deg[agent]
        pts = [(ax, ay)]
        steps = 20
        for i in range(steps + 1):
            a = math.radians(deg - 45 + 90 * i / steps)
            px = ax + math.cos(a) * CONE_RANGE * C
            py = ay + math.sin(a) * CONE_RANGE * C
            pts.append((px, py))
        if len(pts) >= 3:
            pygame.draw.polygon(surf, color, pts)

    def close(self):
        if self._screen:
            pygame.quit()
            self._screen = None
            self._clock  = None


# ── quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test
    env = ShooterEnvironment_3v3()
    parallel_api_test(env, num_cycles=10_000)
    print("API test passed.")
