#!/usr/bin/env python3
"""
Visualise a trained model playing the Shooter environment.

Loads a saved model, opens a Pygame window, and loops through games
until you close the window or press Ctrl+C.

Usage
-----
  # R-NaD: best model from a run directory
  python animate.py rnad --run runs/rnad_20240101_120000

  # R-NaD: point directly to any .pt file
  python animate.py rnad --model runs/rnad_20240101_120000/final_model.pt

  # PPO: best model from a run directory
  python animate.py ppo --run runs/ppo_20240101_120000

  # PPO: point directly to any .zip file
  python animate.py ppo --model runs/ppo_20240101_120000/best_model.zip

  # Modes
  python animate.py rnad --run runs/... --mode self_play    # both players: trained model (R-NaD only)
  python animate.py rnad --run runs/... --mode vs_random    # red=trained, blue=random
  python animate.py rnad --run runs/... --mode vs_scripted  # red=trained, blue=scripted

  # Adversary model (overrides --mode)
  python animate.py rnad --run runs/... --adversary runs/ppo_run/best_model.zip
  python animate.py ppo  --run runs/... --adversary runs/rnad_run/best_model.pt

  # Slow down for analysis / run only N episodes
  python animate.py ppo --run runs/... --fps 3 --episodes 5
"""

import argparse
import time
from pathlib import Path

import numpy as np

from environments.shooter_gym_env import ShooterGymEnv


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Animate a trained model in the Shooter environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="algo", required=True)

    def _add_common(sp):
        src = sp.add_mutually_exclusive_group(required=True)
        src.add_argument(
            "--run", type=str,
            help="Path to a run directory; loads best_model (falls back to final_model)",
        )
        src.add_argument(
            "--model", type=str,
            help="Direct path to a model file (.pt for rnad, .zip for ppo)",
        )
        sp.add_argument("--fps",      type=int,  default=5,
                        help="Frames per second for the pygame window")
        sp.add_argument("--episodes", type=int,  default=0,
                        help="Episodes to play; 0 = loop forever")
        sp.add_argument("--deterministic", action="store_true",
                        help="Use argmax policy instead of sampling")
        sp.add_argument("--device", type=str, default="cpu")
        sp.add_argument(
            "--adversary", type=str, default=None, metavar="PATH",
            help="Path to an adversary model (.zip for PPO, .pt for R-NaD). "
                 "Overrides --mode: red=main model, blue=adversary.",
        )

    rnad = sub.add_parser("rnad", help="Animate an R-NaD model",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common(rnad)
    rnad.add_argument(
        "--mode", choices=["self_play", "vs_random", "vs_scripted"], default="self_play",
        help=(
            "self_play: both players use the trained policy. "
            "vs_random: red=trained, blue=random. "
            "vs_scripted: red=trained, blue=scripted."
        ),
    )

    ppo = sub.add_parser("ppo", help="Animate a PPO model",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common(ppo)
    ppo.add_argument(
        "--mode", choices=["vs_random", "vs_scripted"], default="vs_scripted",
        help=(
            "vs_random: red=trained, blue=random. "
            "vs_scripted: red=trained, blue=scripted."
        ),
    )

    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def load_rnad(args):
    from rnad import RNaD

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

    print(f"Loading R-NaD model: {model_path}")
    model = RNaD.load(str(model_path),
                      env_fn=lambda: ShooterGymEnv(self_play=True),
                      device=args.device)
    print(f"  Learner steps: {model.learner_steps:,}  |  Actor steps: {model.actor_steps:,}")
    return model


def load_ppo(args):
    from stable_baselines3 import PPO

    if args.run:
        run_dir    = Path(args.run)
        candidates = [run_dir / "best_model.zip", run_dir / "final_model.zip"]
        model_path = next((c for c in candidates if c.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                f"No best_model.zip or final_model.zip found in {run_dir}"
            )
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading PPO model: {model_path}")
    model = PPO.load(str(model_path), device=args.device)
    return model


# ── adversary loader ─────────────────────────────────────────────────────────

def load_adversary_fn(path: str, device: str):
    """Return an opponent callable (obs -> action) from a .zip or .pt checkpoint."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Adversary model not found: {p}")
    if p.suffix == ".zip":
        from stable_baselines3 import PPO
        print(f"Loading PPO adversary: {p}")
        adv = PPO.load(str(p), device=device)
        return lambda obs: int(adv.predict(obs, deterministic=False)[0])
    elif p.suffix == ".pt":
        from rnad import RNaD
        print(f"Loading R-NaD adversary: {p}")
        adv = RNaD.load(str(p), env_fn=lambda: ShooterGymEnv(self_play=True), device=device)
        _mask = np.ones(7, dtype=np.float32)
        return lambda obs: int(adv.predict(obs, legal_actions=_mask, deterministic=False)[0])
    else:
        raise ValueError(f"Unsupported adversary file type '{p.suffix}' — expected .zip or .pt")


# ── episode runners ───────────────────────────────────────────────────────────

def run_episode_rnad(env: ShooterGymEnv, model, deterministic: bool) -> dict:
    """Play one episode with R-NaD (supports self-play and legal action mask)."""
    obs, _ = env.reset()
    ep_rewards = [0.0, 0.0]
    game_steps = 0
    done       = False

    while not done:
        legal     = env.legal_actions_mask() if hasattr(env, "legal_actions_mask") else np.ones(7, dtype=np.float32)
        action, _ = model.predict(obs, legal_actions=legal, deterministic=deterministic)
        obs, r, term, trunc, _ = env.step(int(action))
        done = term or trunc

        if isinstance(r, np.ndarray):
            ep_rewards[0] += float(r[0])
            ep_rewards[1] += float(r[1])
            if r[0] != 0.0 or r[1] != 0.0 or done:
                game_steps += 1
        else:
            ep_rewards[0] += float(r)
            game_steps += 1

    return {
        "red_reward":  ep_rewards[0],
        "blue_reward": ep_rewards[1],
        "game_steps":  game_steps,
    }


def run_episode_ppo(env: ShooterGymEnv, model, deterministic: bool) -> dict:
    """Play one episode with a PPO model (single-agent, no legal mask)."""
    obs, _ = env.reset()
    ep_reward  = 0.0
    game_steps = 0
    done       = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, _ = env.step(int(action))
        ep_reward  += float(r)
        game_steps += 1
        done = term or trunc

    return {
        "red_reward":  ep_reward,
        "blue_reward": 0.0,
        "game_steps":  game_steps,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    _OPPONENT = {
        "self_play":   None,        # handled separately for rnad
        "vs_random":   "random",
        "vs_scripted": "scripted",
    }

    if args.adversary:
        adversary_fn = load_adversary_fn(args.adversary, args.device)
        adv_name     = Path(args.adversary).name
        if args.algo == "rnad":
            model       = load_rnad(args)
            run_episode = lambda env: run_episode_rnad(env, model, args.deterministic)
        else:
            model       = load_ppo(args)
            run_episode = lambda env: run_episode_ppo(env, model, args.deterministic)
        env       = ShooterGymEnv(self_play=False, opponent=adversary_fn,
                                  render_mode="human", fps=args.fps)
        mode_desc = f"vs model (red=trained {args.algo.upper()}, blue={adv_name})"
    elif args.algo == "rnad":
        model        = load_rnad(args)
        run_episode  = lambda env: run_episode_rnad(env, model, args.deterministic)
        if args.mode == "self_play":
            env       = ShooterGymEnv(self_play=True, render_mode="human", fps=args.fps)
            mode_desc = "self-play (both players: trained R-NaD model)"
        else:
            opponent  = _OPPONENT[args.mode]
            env       = ShooterGymEnv(self_play=False, opponent=opponent,
                                      render_mode="human", fps=args.fps)
            mode_desc = f"vs {opponent} (red=trained R-NaD, blue={opponent})"
    else:
        model        = load_ppo(args)
        opponent     = _OPPONENT[args.mode]
        run_episode  = lambda env: run_episode_ppo(env, model, args.deterministic)
        env          = ShooterGymEnv(self_play=False, opponent=opponent,
                                     render_mode="human", fps=args.fps)
        mode_desc    = f"vs {opponent} (red=trained PPO, blue={opponent})"

    policy_desc = "deterministic" if args.deterministic else "stochastic"
    max_ep      = args.episodes if args.episodes > 0 else "inf"

    print(f"\nMode:     {mode_desc}")
    print(f"Policy:   {policy_desc}")
    print(f"FPS:      {args.fps}")
    print(f"Episodes: {max_ep}")
    print("\nClose the Pygame window or press Ctrl+C to stop.\n")

    ep         = 0
    red_wins   = 0
    blue_wins  = 0
    all_rewards = []

    try:
        while args.episodes == 0 or ep < args.episodes:
            stats = run_episode(env)
            ep   += 1
            all_rewards.append(stats["red_reward"])

            if stats["red_reward"] > 15.0:
                red_wins  += 1
                outcome = "WIN "
            elif stats["red_reward"] < -15.0:
                blue_wins += 1
                outcome = "LOSS"
            else:
                outcome = "DRAW"

            if hasattr(env, "set_render_stats"):
                env.set_render_stats(red_wins, blue_wins, ep)

            print(
                f"  ep {ep:4d}  [{outcome}]"
                f"  steps={stats['game_steps']:3d}"
                f"  red={stats['red_reward']:+7.2f}"
                f"  blue={stats['blue_reward']:+7.2f}"
                f"  WR={red_wins/ep:.0%}"
            )

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        env.close()

    if all_rewards:
        arr  = np.array(all_rewards)
        wins = int((arr > 15.0).sum())
        print(f"\n--- Summary ({'deterministic' if args.deterministic else 'stochastic'}) ---")
        print(f"  Episodes:    {ep}")
        print(f"  Mean reward: {arr.mean():+.3f}  +/-{arr.std():.3f}")
        print(f"  Win rate:    {wins/ep:.0%}  ({wins}/{ep})")


if __name__ == "__main__":
    main()
