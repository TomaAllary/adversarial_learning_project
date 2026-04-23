#!/usr/bin/env python3
"""
Train R-NaD on the Shooter environment with TensorBoard logging and checkpointing.

Quickstart
----------
  python train_rnad.py
  python train_rnad.py --run-name my_run --total-steps 1_000_000
  python train_rnad.py --batch-size 128 --learning-rate 1e-4 --device cuda

After training
--------------
  tensorboard --logdir runs/
  python animate_rnad.py --run runs/<run_name>

Run directory layout
--------------------
  runs/<run_name>/
    config.json            full reproducible config
    best_model.pt          model with highest eval reward
    final_model.pt         model at end of training
    checkpoints/
      model_step_0001000.pt
      model_step_0002000.pt
      ...
    events.out.tfevents.*  TensorBoard logs
"""

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pettingzoo_env.shooter_gym_env import ShooterGymEnv
from new_rnad import RNaD, RNaDConfig, AdamConfig, NerdConfig, FineTuningConfig


# ── helpers ───────────────────────────────────────────────────────────────────

def _unique_run_name(base: str | None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}" if base else f"rnad_{ts}"


def _build_rnad_config(args) -> RNaDConfig:
    return RNaDConfig(
        policy_network_layers=tuple(args.hidden_layers),
        batch_size=args.batch_size,
        trajectory_max=args.trajectory_max,
        learning_rate=args.learning_rate,
        adam=AdamConfig(b1=args.adam_b1, b2=args.adam_b2, eps=args.adam_eps),
        clip_gradient=args.clip_gradient,
        target_network_avg=args.target_network_avg,
        entropy_schedule_repeats=tuple(args.entropy_schedule_repeats),
        entropy_schedule_size=tuple(args.entropy_schedule_size),
        eta_reward_transform=args.eta_reward_transform,
        nerd=NerdConfig(beta=args.nerd_beta, clip=args.nerd_clip),
        c_vtrace=args.c_vtrace,
        num_players=2,
        seed=args.seed,
    )


def _config_to_dict(run_name: str, args, cfg: RNaDConfig) -> dict:
    return {
        "run_name": run_name,
        "total_steps": args.total_steps,
        "device": args.device,
        "log_interval": args.log_interval,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "checkpoint_interval": args.checkpoint_interval,
        "rnad": {
            "policy_network_layers":    list(cfg.policy_network_layers),
            "batch_size":               cfg.batch_size,
            "trajectory_max":           cfg.trajectory_max,
            "learning_rate":            cfg.learning_rate,
            "clip_gradient":            cfg.clip_gradient,
            "target_network_avg":       cfg.target_network_avg,
            "entropy_schedule_size":    list(cfg.entropy_schedule_size),
            "entropy_schedule_repeats": list(cfg.entropy_schedule_repeats),
            "eta_reward_transform":     cfg.eta_reward_transform,
            "c_vtrace":                 cfg.c_vtrace,
            "num_players":              cfg.num_players,
            "seed":                     cfg.seed,
            "adam": {"b1": cfg.adam.b1, "b2": cfg.adam.b2, "eps": cfg.adam.eps},
            "nerd": {"beta": cfg.nerd.beta, "clip": cfg.nerd.clip},
        },
    }


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: RNaD, num_episodes: int) -> dict:
    """Run `num_episodes` games with red=trained model vs blue=random.

    Returns mean/std episode reward for red and estimated win rate.
    A "win" is when red's episode reward exceeds 15 (needs kill bonus to clear
    the step-penalty baseline of -10 for a full 200-step episode).
    """
    env = ShooterGymEnv(self_play=False, opponent="random")
    rewards = []
    wins = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(
                obs,
                legal_actions=env.legal_actions_mask(),
                deterministic=True,
            )
            obs, r, term, trunc, _ = env.step(int(action))
            ep_reward += float(r)
            done = term or trunc

        rewards.append(ep_reward)
        if ep_reward > 15.0:
            wins += 1

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "win_rate":    wins / num_episodes,
    }


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train R-NaD on the Shooter environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── run metadata ──────────────────────────────────────────────────────────
    p.add_argument("--run-name", type=str, default=None,
                   help="Base run name; timestamp is always appended for uniqueness")
    p.add_argument("--runs-dir", type=str, default="runs",
                   help="Root directory for all runs")

    # ── training schedule ─────────────────────────────────────────────────────
    p.add_argument("--total-steps", type=int, default=500_000,
                   help="Stop after this many actor steps")
    p.add_argument("--log-interval", type=int, default=50,
                   help="Log training scalars every N learner steps")
    p.add_argument("--eval-interval", type=int, default=500,
                   help="Evaluate vs random opponent every N learner steps")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="Episodes per evaluation round")
    p.add_argument("--checkpoint-interval", type=int, default=2_000,
                   help="Save a checkpoint every N learner steps")

    # ── network ───────────────────────────────────────────────────────────────
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 256],
                   help="MLP hidden layer widths")

    # ── optimiser ─────────────────────────────────────────────────────────────
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--clip-gradient", type=float, default=10_000)
    p.add_argument("--adam-b1",       type=float, default=0.0)
    p.add_argument("--adam-b2",       type=float, default=0.999)
    p.add_argument("--adam-eps",      type=float, default=1e-7)

    # ── trajectory ────────────────────────────────────────────────────────────
    p.add_argument("--batch-size",     type=int,   default=256,
                   help="Number of parallel environment instances")
    p.add_argument("--trajectory-max", type=int,   default=200,
                   help="Steps per trajectory (should match MAX_STEPS in shooter_env)")

    # ── R-NaD algorithm ───────────────────────────────────────────────────────
    p.add_argument("--target-network-avg",    type=float, default=0.001,
                   help="EMA rate for target network update")
    p.add_argument("--eta-reward-transform",  type=float, default=0.2)
    p.add_argument("--c-vtrace",              type=float, default=1.0)
    p.add_argument("--nerd-beta",             type=float, default=2.0)
    p.add_argument("--nerd-clip",             type=float, default=10_000)
    p.add_argument("--entropy-schedule-size",    type=int, nargs="+", default=[20_000])
    p.add_argument("--entropy-schedule-repeats", type=int, nargs="+", default=[1])

    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device: 'cpu', 'cuda', 'cuda:0', …")

    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── directory setup ───────────────────────────────────────────────────────
    run_name = _unique_run_name(args.run_name)
    run_dir  = Path(args.runs_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── config ────────────────────────────────────────────────────────────────
    cfg        = _build_rnad_config(args)
    cfg_dict   = _config_to_dict(run_name, args, cfg)
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(run_dir))

    # ── model ─────────────────────────────────────────────────────────────────
    env_fn = lambda: ShooterGymEnv(self_play=True)
    model  = RNaD(env_fn=env_fn, config=cfg, device=args.device)

    best_reward     = float("-inf")
    best_model_path = run_dir / "best_model.pt"
    final_path      = run_dir / "final_model.pt"
    t0              = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"  Run:    {run_name}")
    print(f"  Dir:    {run_dir}")
    print(f"  Device: {args.device}")
    print(f"  Steps:  {args.total_steps:,} actor steps")
    print(f"{'='*60}\n")
    print(f"Config saved : {config_path}")
    print(f"TensorBoard  : tensorboard --logdir {args.runs_dir}\n")

    # ── training loop ─────────────────────────────────────────────────────────
    try:
        while model.num_timesteps < args.total_steps:
            prev_actor = model.actor_steps
            logs       = model.step()
            model.num_timesteps += model.actor_steps - prev_actor

            ls            = model.learner_steps
            alpha, updated = model._entropy_schedule(ls - 1)

            # ── per-step scalars ───────────────────────────────────────────
            if ls % args.log_interval == 0:
                elapsed = time.perf_counter() - t0
                fps     = model.actor_steps / max(elapsed, 1e-8)

                writer.add_scalar("train/loss",         logs["loss"],        ls)
                writer.add_scalar("train/mean_reward",  logs["mean_reward"], ls)
                writer.add_scalar("train/alpha",        alpha,               ls)
                writer.add_scalar("train/actor_steps",  model.actor_steps,   ls)
                writer.add_scalar("train/fps",          fps,                 ls)
                if updated:
                    writer.add_scalar("train/target_net_updated", 1.0, ls)

                progress = model.num_timesteps / args.total_steps * 100
                print(
                    f"[{progress:5.1f}%] step={ls:>6d}  loss={logs['loss']:.4f}"
                    f"  rew={logs['mean_reward']:+.3f}"
                    f"  a={alpha:.3f}  actor={model.actor_steps:,}  fps={fps:.0f}"
                )

            # ── periodic evaluation ────────────────────────────────────────
            if ls % args.eval_interval == 0:
                stats = evaluate(model, args.eval_episodes)

                writer.add_scalar("eval/mean_reward", stats["mean_reward"], ls)
                writer.add_scalar("eval/std_reward",  stats["std_reward"],  ls)
                writer.add_scalar("eval/win_rate",    stats["win_rate"],    ls)
                writer.flush()

                print(
                    f"  [eval]  mean_reward={stats['mean_reward']:+.3f}"
                    f"  win_rate={stats['win_rate']:.0%}"
                )

                if stats["mean_reward"] > best_reward:
                    best_reward = stats["mean_reward"]
                    model.save(str(best_model_path))
                    print(f"  [best]  * new best={best_reward:+.3f} -> {best_model_path.name}")

            # ── checkpoints ───────────────────────────────────────────────
            if ls % args.checkpoint_interval == 0:
                ckpt = ckpt_dir / f"model_step_{ls:07d}.pt"
                model.save(str(ckpt))
                print(f"  [ckpt]  {ckpt.name}")

    except KeyboardInterrupt:
        print("\n[interrupted] Saving final model before exit...")

    # ── final save ────────────────────────────────────────────────────────────
    model.save(str(final_path))

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  Final model  : {final_path}")
    print(f"  Best model   : {best_model_path}  (reward={best_reward:+.3f})")
    print(f"  Wall time    : {elapsed/60:.1f} min")
    print(f"  Learner steps: {model.learner_steps:,}")
    print(f"  Actor steps:   {model.actor_steps:,}")
    print(f"{'='*60}")
    print(f"\ntensorboard --logdir {args.runs_dir}")

    writer.close()


if __name__ == "__main__":
    main()
