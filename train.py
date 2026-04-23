#!/usr/bin/env python3
"""
Unified training script for the Shooter environment.

Quickstart
----------
  python train.py rnad
  python train.py rnad --run-name my_run --total-steps 2_000_000 --device cuda
  python train.py ppo
  python train.py ppo --total-steps 2_000_000 --n-envs 8 --opponent scripted --device cuda

After training
--------------
  tensorboard --logdir runs/
  python animate.py rnad --run runs/<run_name>
  python animate.py ppo  --run runs/<run_name>

Run directory layout
--------------------
  runs/<algo>_<run_name>/   (suffixed _1, _2, … if the name is already taken)
    config.json            full reproducible config
    best_model.pt/.zip     checkpoint with highest eval reward
    final_model.pt/.zip    end-of-training snapshot
    checkpoints/           periodic snapshots
    events.out.tfevents.*  TensorBoard logs

TensorBoard tags (both algos)
------------------------------
  eval_random/mean_reward     mean episode reward, Red vs random Blue
  eval_random/std_reward      std of episode rewards
  eval_random/win_rate        fraction of eval episodes won by Red
  eval_scripted/mean_reward   mean episode reward, Red vs scripted Blue
  eval_scripted/std_reward    std of episode rewards
  eval_scripted/win_rate      fraction of eval episodes won by Red
  train/fps                   actor steps per second

R-NaD only
  train/loss          combined V + NeuRD loss (per learner step)
  train/mean_reward   mean batch trajectory reward (per learner step)
  train/alpha         entropy schedule alpha (per learner step)

PPO only (via SB3 internal logger)
  train/policy_gradient_loss, value_loss, entropy_loss, approx_kl, ...
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from pettingzoo_env.shooter_gym_env import ShooterGymEnv


# ── shared helpers ─────────────────────────────────────────────────────────────

def _make_run_name(algo: str, base: str | None, runs_dir: Path) -> str:
    stem = f"{algo}_{base}" if base else algo
    if not (runs_dir / stem).exists():
        return stem
    n = 1
    while (runs_dir / f"{stem}_{n}").exists():
        n += 1
    return f"{stem}_{n}"


def _setup_dirs(args) -> tuple[Path, Path]:
    runs_dir = Path(args.runs_dir)
    run_dir  = runs_dir / _make_run_name(args.algo, args.run_name, runs_dir)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    return run_dir, ckpt_dir


def _print_header(run_dir: Path, args):
    print(f"\n{'='*60}")
    print(f"  Algo:   {args.algo.upper()}")
    print(f"  Run:    {run_dir.name}")
    print(f"  Dir:    {run_dir}")
    print(f"  Device: {args.device}")
    print(f"  Steps:  {args.total_steps:,}")
    print(f"{'='*60}")
    print(f"TensorBoard: tensorboard --logdir {args.runs_dir}\n")


def evaluate(model, num_episodes: int, *, use_legal_mask: bool, opponent: str = "random") -> dict:
    """Run num_episodes games: Red = trained model vs Blue = opponent.

    Returns mean/std episode reward and win rate.
    A win requires episode reward > 15 (kill bonus must clear the
    step-penalty floor of -10 for a full 200-step episode).
    """
    env = ShooterGymEnv(self_play=False, opponent=opponent)
    rewards, wins = [], 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            if use_legal_mask:
                action, _ = model.predict(
                    obs, legal_actions=env.legal_actions_mask(), deterministic=True
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
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


def _log_eval(writer: SummaryWriter, stats: dict, step: int, prefix: str = "eval"):
    writer.add_scalar(f"{prefix}/mean_reward", stats["mean_reward"], step)
    writer.add_scalar(f"{prefix}/std_reward",  stats["std_reward"],  step)
    writer.add_scalar(f"{prefix}/win_rate",    stats["win_rate"],    step)
    writer.flush()


# ── R-NaD ─────────────────────────────────────────────────────────────────────

def train_rnad(args):
    from new_rnad import RNaD, RNaDConfig, AdamConfig, NerdConfig

    run_dir, ckpt_dir = _setup_dirs(args)

    cfg = RNaDConfig(
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

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({"algo": "rnad", "run_name": run_dir.name, **vars(args)}, f,
                  indent=2, default=str)

    writer = SummaryWriter(log_dir=str(run_dir))
    model  = RNaD(env_fn=lambda: ShooterGymEnv(self_play=True), config=cfg,
                  device=args.device)

    best_reward     = float("-inf")
    best_model_path = run_dir / "best_model.pt"
    final_path      = run_dir / "final_model.pt"
    t0              = time.perf_counter()

    _print_header(run_dir, args)
    print(f"Config saved: {config_path}\n")

    try:
        while model.num_timesteps < args.total_steps:
            prev_actor = model.actor_steps
            logs       = model.step()
            model.num_timesteps += model.actor_steps - prev_actor

            ls            = model.learner_steps
            alpha, updated = model._entropy_schedule(ls - 1)

            # ── per-learner-step logging ───────────────────────────────────
            if ls % args.log_interval == 0:
                elapsed = time.perf_counter() - t0
                fps     = model.actor_steps / max(elapsed, 1e-8)

                writer.add_scalar("train/loss",        logs["loss"],        ls)
                writer.add_scalar("train/mean_reward", logs["mean_reward"], ls)
                writer.add_scalar("train/alpha",       alpha,               ls)
                writer.add_scalar("train/actor_steps", model.actor_steps,   ls)
                writer.add_scalar("train/fps",         fps,                 ls)
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
                stats_random   = evaluate(model, args.eval_episodes, use_legal_mask=True, opponent="random")
                stats_scripted = evaluate(model, args.eval_episodes, use_legal_mask=True, opponent="scripted")
                _log_eval(writer, stats_random,   ls, prefix="eval_random")
                _log_eval(writer, stats_scripted, ls, prefix="eval_scripted")
                print(
                    f"  [eval vs random  ]  mean_reward={stats_random['mean_reward']:+.3f}"
                    f"  win_rate={stats_random['win_rate']:.0%}"
                )
                print(
                    f"  [eval vs scripted]  mean_reward={stats_scripted['mean_reward']:+.3f}"
                    f"  win_rate={stats_scripted['win_rate']:.0%}"
                )
                stats = stats_random  # use random stats for best-model tracking

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

    model.save(str(final_path))
    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  Final model  : {final_path}")
    print(f"  Best model   : {best_model_path}  (reward={best_reward:+.3f})")
    print(f"  Wall time    : {elapsed/60:.1f} min")
    print(f"  Learner steps: {model.learner_steps:,}")
    print(f"  Actor steps  : {model.actor_steps:,}")
    print(f"{'='*60}")
    print(f"\ntensorboard --logdir {args.runs_dir}")
    writer.close()


# ── PPO ───────────────────────────────────────────────────────────────────────

def train_ppo(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    run_dir, ckpt_dir = _setup_dirs(args)

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({"algo": "ppo", "run_name": run_dir.name, **vars(args)}, f,
                  indent=2, default=str)

    writer = SummaryWriter(log_dir=str(run_dir))

    def _env_fn():
        return ShooterGymEnv(self_play=False, opponent=args.opponent)

    train_env = make_vec_env(_env_fn, n_envs=args.n_envs, seed=args.seed)

    _print_header(run_dir, args)
    print(f"Config saved: {config_path}\n")

    if args.load:
        print(f"Resuming from {args.load}")
        model = PPO.load(args.load, env=train_env, device=args.device)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": list(args.hidden_layers)},
            tensorboard_log=str(run_dir),   # SB3 writes its own scalars here
            device=args.device,
            verbose=0,
            seed=args.seed,
        )

    t0 = time.perf_counter()

    class _Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best_reward  = float("-inf")
            self._last_eval   = 0
            self._last_ckpt   = 0

        def _on_step(self) -> bool:
            n = self.num_timesteps

            if n - self._last_eval >= args.eval_interval:
                self._last_eval = n
                stats_random   = evaluate(self.model, args.eval_episodes, use_legal_mask=False, opponent="random")
                stats_scripted = evaluate(self.model, args.eval_episodes, use_legal_mask=False, opponent="scripted")
                elapsed = time.perf_counter() - t0
                fps     = n / max(elapsed, 1e-8)

                _log_eval(writer, stats_random,   n, prefix="eval_random")
                _log_eval(writer, stats_scripted, n, prefix="eval_scripted")
                writer.add_scalar("train/fps", fps, n)

                progress = n / args.total_steps * 100
                print(
                    f"[{progress:5.1f}%] steps={n:>8,}  fps={fps:.0f}"
                )
                print(
                    f"  [eval vs random  ]  mean_reward={stats_random['mean_reward']:+.3f}"
                    f"  win_rate={stats_random['win_rate']:.0%}"
                )
                print(
                    f"  [eval vs scripted]  mean_reward={stats_scripted['mean_reward']:+.3f}"
                    f"  win_rate={stats_scripted['win_rate']:.0%}"
                )

                stats = stats_random  # use random stats for best-model tracking
                if stats["mean_reward"] > self.best_reward:
                    self.best_reward = stats["mean_reward"]
                    self.model.save(str(run_dir / "best_model"))
                    print(f"  [best]  * new best={self.best_reward:+.3f} -> best_model.zip")

            if n - self._last_ckpt >= args.checkpoint_interval:
                self._last_ckpt = n
                ckpt = ckpt_dir / f"ppo_step_{n:08d}"
                self.model.save(str(ckpt))
                print(f"  [ckpt]  {ckpt.name}.zip")

            return True

    callback = _Callback()
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=callback,
            reset_num_timesteps=not bool(args.load),
        )
    except KeyboardInterrupt:
        print("\n[interrupted] Saving final model before exit...")

    model.save(str(run_dir / "final_model"))
    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  Final model  : {run_dir / 'final_model'}.zip")
    print(f"  Best model   : {run_dir / 'best_model'}.zip  (reward={callback.best_reward:+.3f})")
    print(f"  Wall time    : {elapsed/60:.1f} min")
    print(f"  Timesteps    : {model.num_timesteps:,}")
    print(f"{'='*60}")
    print(f"\ntensorboard --logdir {args.runs_dir}")
    writer.close()
    train_env.close()


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train PPO or R-NaD on the Shooter environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="algo", required=True)

    def _add_common(sp):
        g = sp.add_argument_group("common")
        g.add_argument("--run-name",    type=str, default=None,
                       help="Base name; algo + timestamp always appended")
        g.add_argument("--runs-dir",    type=str, default="runs")
        g.add_argument("--total-steps", type=int, default=500_000,
                       help="Training budget in actor steps")
        g.add_argument("--eval-episodes", type=int, default=20,
                       help="Episodes per evaluation round (Red vs random Blue)")
        g.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 256],
                       help="MLP hidden layer widths")
        g.add_argument("--learning-rate", type=float, default=3e-4)
        g.add_argument("--seed",   type=int, default=42)
        g.add_argument("--device", type=str, default="auto",
                       help="Torch device: auto, cpu, cuda, cuda:0, …")

    # ── PPO ──────────────────────────────────────────────────────────────────
    ppo = sub.add_parser("ppo", help="Train with PPO (Stable Baselines 3)",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common(ppo)
    g = ppo.add_argument_group("ppo-specific")
    g.add_argument("--n-envs",    type=int, default=8,
                   help="Parallel training environments")
    g.add_argument("--opponent",  type=str, default="scripted",
                   choices=["random", "scripted"], help="Blue opponent policy")
    g.add_argument("--load",      type=str, default=None,
                   help="Path to a .zip checkpoint to resume from")
    g.add_argument("--eval-interval",       type=int, default=10_000,
                   help="Evaluate vs random every N actor steps")
    g.add_argument("--checkpoint-interval", type=int, default=50_000,
                   help="Save checkpoint every N actor steps")

    # ── R-NaD ────────────────────────────────────────────────────────────────
    rnad = sub.add_parser("rnad", help="Train with R-NaD (self-play Nash equilibrium)",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common(rnad)
    rnad.set_defaults(learning_rate=5e-5, device="cpu")
    g = rnad.add_argument_group("rnad-specific")
    g.add_argument("--log-interval",          type=int,   default=50,
                   help="Log training scalars every N learner steps")
    g.add_argument("--eval-interval",         type=int,   default=500,
                   help="Evaluate vs random every N learner steps")
    g.add_argument("--checkpoint-interval",   type=int,   default=2_000,
                   help="Save checkpoint every N learner steps")
    g.add_argument("--batch-size",            type=int,   default=256,
                   help="Parallel environment instances")
    g.add_argument("--trajectory-max",        type=int,   default=200,
                   help="Steps per trajectory")
    g.add_argument("--clip-gradient",         type=float, default=10_000)
    g.add_argument("--adam-b1",               type=float, default=0.0)
    g.add_argument("--adam-b2",               type=float, default=0.999)
    g.add_argument("--adam-eps",              type=float, default=1e-7)
    g.add_argument("--target-network-avg",    type=float, default=0.001,
                   help="EMA rate τ for target network")
    g.add_argument("--eta-reward-transform",  type=float, default=0.2)
    g.add_argument("--c-vtrace",              type=float, default=1.0)
    g.add_argument("--nerd-beta",             type=float, default=2.0)
    g.add_argument("--nerd-clip",             type=float, default=10_000)
    g.add_argument("--entropy-schedule-size",    type=int, nargs="+", default=[20_000])
    g.add_argument("--entropy-schedule-repeats", type=int, nargs="+", default=[1])

    return p.parse_args()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.algo == "ppo":
        train_ppo(args)
    else:
        train_rnad(args)
