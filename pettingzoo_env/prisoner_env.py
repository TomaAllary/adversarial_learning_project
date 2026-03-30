import functools
import random
from copy import copy
import pygame

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test, api_test

class PrisonerEnvironment(ParallelEnv):
    """
    The metadata holds environment constants.
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {
        "name": "pettingzoo_env_prisoner_v0",
    }

    def __init__(self, render_mode=None, grid_size=7, cell_size=80, fps=10):
        """
        - escape x and y coordinates
        - guard x and y coordinates
        - prisoner x and y coordinates
        - timestamp
        - possible_agents

        These attributes should not be changed after initialization.
        """
        self.escape_y = None
        self.escape_x = None
        self.guard_y = None
        self.guard_x = None
        self.prisoner_y = None
        self.prisoner_x = None
        self.timestep = None
        self.possible_agents = ["prisoner", "guard"]


        #### PYGAME RENDERING SETUP ####
        self.render_mode = render_mode   # None | "human" | "rgb_array" | "ansi"
        self.grid_size = grid_size       # e.g., 7x7
        self.cell_size = cell_size       # pixel size of each grid cell
        self.fps = fps                   # cap frame rate for "human"

        # Lazy Pygame handles
        self._screen = None
        self._clock = None
        self._surface_size = (self.grid_size * self.cell_size,
                              self.grid_size * self.cell_size)
        self._closed = False





    def reset(self, seed=None, options=None):
        """
        Reset set the environment to a starting point.
        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.prisoner_x = 0
        self.prisoner_y = 0

        self.guard_x = 6
        self.guard_y = 6

        self.escape_x = random.randint(2, 5)
        self.escape_y = random.randint(2, 5)

        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guard_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """
        Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        # Execute actions
        prisoner_action = actions["prisoner"]
        guard_action = actions["guard"]

        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < 6:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < 6:
            self.prisoner_y += 1

        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < 6:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < 6:
            self.guard_y += 1

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}

        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
        self.timestep += 1

        # Get observations
        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guard_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos


    def _init_pygame(self):
        if self._screen is not None:
            return

        pygame.init()
        pygame.display.set_caption("Prisoner vs Guard")
        self._screen = pygame.display.set_mode(self._surface_size)
        self._clock = pygame.time.Clock()

        # Images
        self.prisoner_img = pygame.image.load("assets/prisoner.png").convert_alpha()
        self.guard_img = pygame.image.load("assets/guard.png").convert_alpha()
        self.escape_img = pygame.image.load("assets/prison_exit.png").convert_alpha()
        # Scale to fit inside a cell
        img_size = self.cell_size - 12
        self.prisoner_img = pygame.transform.smoothscale(self.prisoner_img, (img_size, img_size))
        self.guard_img = pygame.transform.smoothscale(self.guard_img, (img_size, img_size))
        self.escape_img = pygame.transform.smoothscale(self.escape_img, (img_size, img_size))

    def render(self):
        if self.render_mode != "human":
            grid = np.full((self.grid_size, self.grid_size), " ", dtype="<U1")
            grid[self.prisoner_y, self.prisoner_x] = "P"
            grid[self.guard_y, self.guard_x] = "G"
            grid[self.escape_y, self.escape_x] = "E"
            print(grid, "\n")
            return
        else:
            if self._screen is None:
                self._init_pygame()

            # keep the OS window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit
                
            self._draw_scene(self._screen)
            pygame.display.flip()
            self._clock.tick(60)


    def _draw_scene(self, surface):
        # colors (define once in __init__ if you prefer)
        COLOR_BG = (30, 30, 35)
        COLOR_GRID = (60, 60, 70)
        COLOR_PRISONER = (80, 200, 120)
        COLOR_GUARD = (220, 80, 80)
        COLOR_ESCAPE = (80, 150, 240)

        surface.fill(COLOR_BG)

        # grid lines
        for i in range(self.grid_size + 1):
            x = i * self.cell_size
            y = i * self.cell_size
            pygame.draw.line(surface, COLOR_GRID, (x, 0), (x, self._surface_size[1]), 1)
            pygame.draw.line(surface, COLOR_GRID, (0, y), (self._surface_size[0], y), 1)

        # helper: grid cell rectangle
        def cell_rect(x, y, pad=6):
            return pygame.Rect(
                x * self.cell_size + pad,
                y * self.cell_size + pad,
                self.cell_size - 2 * pad,
                self.cell_size - 2 * pad,
            )
        def draw_image(img, x, y):
            rect = img.get_rect(center=cell_rect(x, y).center)
            surface.blit(img, rect)


        # draw escape, guard, prisoner (order decides which is “on top”)
        draw_image(self.escape_img,   self.escape_x,   self.escape_y)
        draw_image(self.guard_img,    self.guard_x,    self.guard_y)
        draw_image(self.prisoner_img, self.prisoner_x, self.prisoner_y)


    
    def close(self):
        try:
            if pygame.display.get_init():
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
        self._screen = None
        self._clock = None
        self._closed = True



    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([7 * 7] * 3)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)
    

if __name__ == "__main__":
    env = PrisonerEnvironment()
    #api_test(env, num_cycles=1_000_000, verbose_progress=True)
    parallel_api_test(env, num_cycles=1_000_000)
    print("hello")