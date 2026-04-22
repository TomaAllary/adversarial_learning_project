"""
Train a PPO agent (Stable Baselines 3) to play Red in the Shooter environment.

Red controls a single agent; Blue is driven by the scripted BFS opponent by
default. Uses ShooterGymEnv with self_play=False so the env is a standard
single-agent gymnasium.Env that SB3 can consume directly.

Usage
-----
    python -m pettingzoo_env.train_ppo
    python -m pettingzoo_env.train_ppo --total-timesteps 2_000_000 --n-envs 8
    python -m pettingzoo_env.train_ppo --opponent random --device cuda
    python -m pettingzoo_env.train_ppo --load runs/ppo_20260101_120000/best_model.zip
"""

import argparse
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from pettingzoo_env.shooter_gym_env import ShooterGymEnv


def _env_factory(opponent: str):
    return lambda: ShooterGymEnv(self_play=False, opponent=opponent)


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on the Shooter environment (SB3).")
    p.add_argument("--total-timesteps", type=int,   default=1_000_000,  help="Training budget in env steps")
    p.add_argument("--n-envs",          type=int,   default=8,           help="Parallel training environments")
    p.add_argument("--opponent",        type=str,   default="scripted",  choices=["random", "scripted"],
                   help="Blue opponent policy")
    p.add_argument("--run-name",        type=str,   default="ppo",       help="Base name; timestamp is appended")
    p.add_argument("--runs-dir",        type=str,   default="runs",      help="Root output directory")
    p.add_argument("--load",            type=str,   default=None,        help="Path to a .zip checkpoint to resume from")
    p.add_argument("--eval-freq",       type=int,   default=10_000,      help="Eval every N total env steps")
    p.add_argument("--eval-episodes",   type=int,   default=20,          help="Episodes per evaluation round")
    p.add_argument("--checkpoint-freq", type=int,   default=50_000,      help="Save checkpoint every N total env steps")
    p.add_argument("--device",          type=str,   default="auto",      help="Torch device: auto, cpu, cuda")
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(args.runs_dir, f"{args.run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    train_env = make_vec_env(_env_factory(args.opponent), n_envs=args.n_envs, seed=args.seed)
    eval_env  = make_vec_env(_env_factory(args.opponent), n_envs=1,            seed=args.seed + 1)

    if args.load:
        print(f"Resuming from {args.load}")
        model = PPO.load(args.load, env=train_env, device=args.device)
        model.set_logger(model.logger)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=run_dir,
            device=args.device,
            verbose=1,
            seed=args.seed,
        )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=os.path.join(run_dir, "checkpoints"),
            name_prefix="ppo_shooter",
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not bool(args.load),
    )

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Outputs saved to {run_dir}/")
    print(f"  final model : {final_path}.zip")
    print(f"  best model  : {os.path.join(run_dir, 'best_model.zip')}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
