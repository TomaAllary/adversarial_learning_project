"""
Gymnasium wrapper for ShooterEnvironment.

Two operating modes
-------------------
self_play=True  (default, for R-NaD)
    Converts the simultaneous PettingZoo game into a sequential two-player
    interface.  Turns alternate: player 0 (red) → player 1 (blue) → …

    At player 0's turn:
        • Red's action is buffered; the underlying env does NOT step yet.
        • Returns blue's current observation so the algorithm has the right
          context for the next call.
        • Reward is [0.0, 0.0] (no env step has occurred).

    At player 1's turn:
        • Both buffered actions are applied; the env steps once.
        • Returns red's new observation.
        • Reward is [r_red, r_blue] — a length-2 float32 array.

    V-trace in R-NaD naturally handles the one-turn delay: it accumulates
    discounted returns between a player's own turns, so the zero-reward
    half-step does not break the credit-assignment signal.

    Extra methods for R-NaD compatibility:
        current_player()      → int   (0 or 1)
        legal_actions_mask()  → np.ndarray shape (7,), all ones

self_play=False  (for PPO / SB3)
    The agent always controls red; blue is driven by an opponent policy.
    step() returns a scalar float reward for the red agent.

    opponent kwarg:
        "random"   — blue samples uniformly from Discrete(7)
        "scripted" — blue uses the BFS ScriptedShooterAgent
        callable   — fn(blue_obs: np.ndarray) -> int

Usage examples
--------------
# R-NaD self-play
from pettingzoo_env.shooter_gym_env import ShooterGymEnv
from new_rnad import RNaD, RNaDConfig

env_fn = lambda: ShooterGymEnv(self_play=True)
model  = RNaD(env_fn=env_fn, config=RNaDConfig(num_players=2))
model.learn(total_timesteps=500_000)

# PPO single-agent (stable-baselines3)
from stable_baselines3 import PPO
env = ShooterGymEnv(self_play=False, opponent="random")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium.spaces import Box, Discrete
from typing import Callable, Optional, Union

from pettingzoo_env.shooter_env import ShooterEnvironment, OBS_DIM

# Number of discrete actions (stay, N, S, W, E, rot-L, rot-R)
_N_ACTIONS = 7


class ShooterGymEnv(gymnasium.Env):
    """Gymnasium wrapper around ShooterEnvironment.

    See module docstring for full description of the two modes.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        self_play: bool = True,
        opponent: Union[str, Callable] = "random",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.self_play   = self_play
        self.render_mode = render_mode

        self._env = ShooterEnvironment(render_mode=render_mode)

        self.observation_space = Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = Discrete(_N_ACTIONS)

        # ── self-play state ───────────────────────────────────────────────
        self._current_player:       int        = 0
        self._buffered_red_action:  int        = 0
        self._last_red_obs:  Optional[np.ndarray] = None
        self._last_blue_obs: Optional[np.ndarray] = None
        self._done:          bool               = False
        self._should_render: bool               = False  # set True after real env steps

        # ── single-agent opponent ─────────────────────────────────────────
        if not self_play:
            self._opponent_fn = self._build_opponent(opponent)

    # ── R-NaD interface ───────────────────────────────────────────────────────

    def current_player(self) -> int:
        """Which player acts next: 0 = red, 1 = blue."""
        return self._current_player

    def legal_actions_mask(self) -> np.ndarray:
        """All 7 actions are always legal (walls silently block movement)."""
        return np.ones(_N_ACTIONS, dtype=np.float32)

    # ── gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        obs_dict, _ = self._env.reset(seed=seed)

        self._current_player      = 0
        self._done                = False
        self._buffered_red_action = 0
        self._last_red_obs  = obs_dict["red_0"].copy()
        self._last_blue_obs = obs_dict["blue_0"].copy()

        if self.render_mode == "human":
            self._env.render()

        # Player 0 (red) always acts first
        return self._last_red_obs.copy(), {}

    def step(self, action: int):
        if self._done:
            raise RuntimeError(
                "step() called on a terminated environment; call reset() first."
            )
        if self.self_play:
            result = self._step_self_play(int(action))
        else:
            result = self._step_single_agent(int(action))

        # Auto-render after each full env step (only on blue's half-step for
        # self_play, or every step for single-agent mode).
        if self.render_mode == "human" and self._should_render:
            self._env.render()

        return result

    def render(self):
        """Explicit render call — also works when render_mode is None."""
        return self._env.render()

    def close(self):
        self._env.close()

    # ── self-play step ────────────────────────────────────────────────────────

    def _step_self_play(self, action: int):
        if self._current_player == 0:
            # ── Red's half-step: buffer action, no env step ────────────────
            self._buffered_red_action = action
            self._current_player = 1
            self._should_render = False  # nothing to render yet

            # Return blue's current obs so the algorithm has the right
            # observation context on the next call.
            blue_obs = self._last_blue_obs.copy()
            return (
                blue_obs,
                np.array([0.0, 0.0], dtype=np.float32),
                False,
                False,
                {},
            )

        # ── Blue's half-step: apply both actions, step the env ─────────────
        obs_dict, rew_dict, terms, truncs, _ = self._env.step(
            {"red_0": self._buffered_red_action, "blue_0": action}
        )

        r_red  = float(rew_dict.get("red_0",  0.0))
        r_blue = float(rew_dict.get("blue_0", 0.0))

        terminated = bool(any(terms.values())) if terms else False
        truncated  = bool(any(truncs.values())) if truncs else False

        red_obs  = obs_dict["red_0"].copy()
        blue_obs = obs_dict["blue_0"].copy()

        self._last_red_obs  = red_obs
        self._last_blue_obs = blue_obs
        self._current_player = 0
        self._should_render  = True  # env stepped — render this frame

        if terminated or truncated:
            self._done = True

        # Return red's new obs — player 0 acts next
        return (
            red_obs,
            np.array([r_red, r_blue], dtype=np.float32),
            terminated,
            truncated,
            {"r_red": r_red, "r_blue": r_blue},
        )

    # ── single-agent step ─────────────────────────────────────────────────────

    def _step_single_agent(self, action: int):
        blue_obs    = self._last_blue_obs if self._last_blue_obs is not None \
                      else np.zeros(OBS_DIM, dtype=np.float32)
        blue_action = int(self._opponent_fn(blue_obs))

        obs_dict, rew_dict, terms, truncs, _ = self._env.step(
            {"red_0": action, "blue_0": blue_action}
        )

        r_red = float(rew_dict.get("red_0", 0.0))

        terminated = bool(any(terms.values())) if terms else False
        truncated  = bool(any(truncs.values())) if truncs else False

        self._last_blue_obs = obs_dict["blue_0"].copy()
        self._should_render = True  # env always steps in single-agent mode

        if terminated or truncated:
            self._done = True

        return obs_dict["red_0"].copy(), r_red, terminated, truncated, {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_opponent(self, opponent: Union[str, Callable]) -> Callable:
        if callable(opponent):
            return opponent
        if opponent == "random":
            rng = np.random.default_rng()
            return lambda _obs: int(rng.integers(_N_ACTIONS))
        if opponent == "scripted":
            from pettingzoo_env.scripted_shooter_agent import ScriptedShooterAgent
            agent = ScriptedShooterAgent()
            return lambda obs: agent.act(obs, self._env)
        raise ValueError(
            f"Unknown opponent '{opponent}'. "
            "Use 'random', 'scripted', or a callable fn(obs) -> int."
        )


# ── sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("=== self_play=True (R-NaD mode) ===")
    env = ShooterGymEnv(self_play=True)
    check_env(env, warn=True)
    obs, info = env.reset()
    print(f"  obs shape: {obs.shape}, player: {env.current_player()}")
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"  player={env.current_player()}  reward={r}  done={term or trunc}")
        if term or trunc:
            obs, info = env.reset()
    env.close()

    print("\n=== self_play=False (PPO / single-agent mode) ===")
    env = ShooterGymEnv(self_play=False, opponent="random")
    check_env(env, warn=True)
    obs, info = env.reset()
    print(f"  obs shape: {obs.shape}")
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"  reward={r:.3f}  done={term or trunc}")
        if term or trunc:
            obs, info = env.reset()
    env.close()

    print("\nAll checks passed.")
