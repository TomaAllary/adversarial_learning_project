#!/usr/bin/env python3
"""
Visualise a trained R-NaD model playing the Shooter environment.

Loads a saved model, opens a Pygame window, and loops through games
until you close the window or press Ctrl+C.

Usage
-----
  # Use the best model from a run directory
  python animate_rnad.py --run runs/rnad_20240101_120000

  # Point directly to any .pt file
  python animate_rnad.py --model runs/rnad_20240101_120000/final_model.pt

  # Both sides use the trained policy (Nash self-play)
  python animate_rnad.py --run runs/... --mode self_play

  # Red=trained, Blue=random (shows learned advantage)
  python animate_rnad.py --run runs/... --mode vs_random

  # Slow down for analysis
  python animate_rnad.py --run runs/... --fps 3

  # Run only N episodes then exit
  python animate_rnad.py --run runs/... --episodes 5
"""

import argparse
import time
from pathlib import Path

import numpy as np

from pettingzoo_env.shooter_gym_env import ShooterGymEnv
from new_rnad import RNaD


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Animate a trained R-NaD model in the Shooter environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--run", type=str,
        help="Path to a run directory; loads best_model.pt (falls back to final_model.pt)",
    )
    src.add_argument(
        "--model", type=str,
        help="Direct path to a .pt model file",
    )

    p.add_argument(
        "--mode", choices=["self_play", "vs_random"], default="self_play",
        help=(
            "self_play: both players use the trained policy (Nash equilibrium demo). "
            "vs_random: red=trained, blue=random (shows learned advantage)."
        ),
    )
    p.add_argument("--fps",      type=int,  default=5,
                   help="Frames per second for the pygame window")
    p.add_argument("--episodes", type=int,  default=0,
                   help="Episodes to play; 0 = loop forever")
    p.add_argument("--deterministic", action="store_true",
                   help="Use argmax policy instead of sampling")
    p.add_argument("--device", type=str, default="cpu")

    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(args) -> RNaD:
    if args.run:
        run_dir    = Path(args.run)
        candidates = [run_dir / "best_model.pt", run_dir / "final_model.pt"]
        model_path = next((c for c in candidates if c.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                f"No best_model.pt or final_model.pt found in {run_dir}"
            )
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model: {model_path}")
    env_fn = lambda: ShooterGymEnv(self_play=True)
    model  = RNaD.load(str(model_path), env_fn=env_fn, device=args.device)
    print(f"  Learner steps: {model.learner_steps:,}  |  Actor steps: {model.actor_steps:,}")
    return model


# ── episode runner ────────────────────────────────────────────────────────────

def run_episode(env: ShooterGymEnv, model: RNaD, deterministic: bool) -> dict:
    """Play one episode and return summary stats."""
    obs, _ = env.reset()
    ep_rewards  = [0.0, 0.0]   # [red_total, blue_total]
    game_steps  = 0             # counts actual env ticks (not sequential half-steps)
    done        = False

    while not done:
        legal  = env.legal_actions_mask()
        action, _ = model.predict(obs, legal_actions=legal, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(int(action))
        done = term or trunc

        if isinstance(r, np.ndarray):
            ep_rewards[0] += float(r[0])
            ep_rewards[1] += float(r[1])
            # Only count game ticks (blue's half-step), not buffer half-steps
            if r[0] != 0.0 or r[1] != 0.0 or (term or trunc):
                game_steps += 1
        else:
            ep_rewards[0] += float(r)
            game_steps += 1

    return {
        "red_reward":  ep_rewards[0],
        "blue_reward": ep_rewards[1],
        "game_steps":  game_steps,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    model = load_model(args)

    # Build the rendering environment
    if args.mode == "self_play":
        env = ShooterGymEnv(
            self_play=True,
            render_mode="human",
            fps=args.fps,
        )
        mode_desc = "self-play (both players: trained model)"
    else:
        env = ShooterGymEnv(
            self_play=False,
            opponent="random",
            render_mode="human",
            fps=args.fps,
        )
        mode_desc = "vs random (red=trained, blue=random)"

    policy_desc = "deterministic" if args.deterministic else "stochastic"
    max_ep      = args.episodes if args.episodes > 0 else "inf"

    print(f"\nMode:    {mode_desc}")
    print(f"Policy:  {policy_desc}")
    print(f"FPS:     {args.fps}")
    print(f"Episodes: {max_ep}")
    print("\nClose the Pygame window or press Ctrl+C to stop.\n")

    ep          = 0
    all_rewards = []

    try:
        while args.episodes == 0 or ep < args.episodes:
            stats = run_episode(env, model, args.deterministic)
            ep += 1
            all_rewards.append(stats["red_reward"])

            outcome = (
                "WIN " if stats["red_reward"] > 15.0
                else "LOSS" if stats["red_reward"] < -15.0
                else "DRAW"
            )
            print(
                f"  ep {ep:4d}  [{outcome}]"
                f"  steps={stats['game_steps']:3d}"
                f"  red={stats['red_reward']:+7.2f}"
                f"  blue={stats['blue_reward']:+7.2f}"
            )

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        env.close()

    if all_rewards:
        import numpy as np
        arr = np.array(all_rewards)
        wins = int((arr > 15.0).sum())
        print(f"\n--- Summary ({'deterministic' if args.deterministic else 'stochastic'}) ---")
        print(f"  Episodes:    {ep}")
        print(f"  Mean reward: {arr.mean():+.3f}  +/-{arr.std():.3f}")
        print(f"  Win rate:    {wins/ep:.0%}  ({wins}/{ep})")


if __name__ == "__main__":
    main()
