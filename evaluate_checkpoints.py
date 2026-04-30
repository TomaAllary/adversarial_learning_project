"""
evaluate_checkpoints.py
=======================
Evaluate every model checkpoint in a run folder against three opponents:
  • random      — uniform random actions
  • scripted    — BFS ScriptedShooterAgent
  • exploiter   — a trained PPO exploiter (user-defined per checkpoint)

Discovers checkpoints automatically:
  • RNaD  : rnad_at_exploiter_gen*.pt   (in run_dir root)
  • PPO   : checkpoints/ppo_step_*.zip  (in run_dir/checkpoints/)

Usage
-----
    python evaluate_checkpoints.py \\
        --run-dir  runs/my_league_run \\
        --out      results.json \\
        --episodes 200 \\
        --device   cpu

Exploiter pairs
---------------
Edit the EXPLOITER_PAIRS constant below to specify which exploiter .zip
to use for each checkpoint.  Each entry is a tuple of:
    (checkpoint_filename_stem, exploiter_zip_path)

Example:
    EXPLOITER_PAIRS = [
        ("rnad_at_exploiter_gen0001", "runs/my_run/exploiter_gen0001/final_exploiter.zip"),
        ("ppo_step_00500000",         "runs/my_run/exploiter_gen0002/final_exploiter.zip"),
    ]

Checkpoints not listed in EXPLOITER_PAIRS are still evaluated against
random and scripted, but the exploiter column is skipped for them.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION  ← edit this section
# ─────────────────────────────────────────────────────────────────────────────

# Depends on the name of your runs and trained exploiters
EXPLOITER_PAIRS: list[tuple[str, str]] = [
    # ppo
    ("ppo_step_00050000",  "runs/minimax_exploiter_runs/exploiter_ppo_50k_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("ppo_step_00100000",  "runs/minimax_exploiter_runs/exploiter_ppo_100k_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("ppo_step_01000000",  "runs/minimax_exploiter_runs/exploiter_ppo_1M_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("ppo_step_02000000",  "runs/minimax_exploiter_runs/exploiter_ppo_2M_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("ppo_step_03500000",  "runs/minimax_exploiter_runs/exploiter_ppo_3p5M_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("ppo_step_04500000",  "runs/minimax_exploiter_runs/exploiter_ppo_4p5M_no_stop/exploiter_gen01/best_exploiter.zip"),
    # rnad
    ("model_step_0000500",  "runs/minimax_exploiter_runs/exploiter_rnad_500_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("model_step_0003000",  "runs/minimax_exploiter_runs/exploiter_rnad_3000_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("model_step_0010000",  "runs/minimax_exploiter_runs/exploiter_rnad_10_000_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("model_step_0060000",  "runs/minimax_exploiter_runs/exploiter_rnad_60_000_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("model_step_0080000",  "runs/minimax_exploiter_runs/exploiter_rnad_80_000_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("model_step_0100000",  "runs/minimax_exploiter_runs/exploiter_rnad_100_000_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("model_step_0140000",  "runs/minimax_exploiter_runs/exploiter_rnad_140_000_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("model_step_0180000",  "runs/minimax_exploiter_runs/exploiter_rnad_180_000_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("model_step_0220000",  "runs/minimax_exploiter_runs/exploiter_rnad_220_000_no_stop/exploiter_gen01/best_exploiter.zip"),
    ("model_step_0250000",  "runs/minimax_exploiter_runs/exploiter_rnad_250_000_no_stop/exploiter_gen01/best_exploiter.zip"),
    # league
    ("rnad_at_exploiter_gen0001",  "exploiter_gen0001/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0002",  "exploiter_gen0002/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0003",  "exploiter_gen0003/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0004",  "exploiter_gen0004/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0005",  "exploiter_gen0005/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0006",  "exploiter_gen0006/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0007",  "exploiter_gen0007/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0008",  "exploiter_gen0008/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0009",  "exploiter_gen0009/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0010",  "exploiter_gen0010/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0011",  "exploiter_gen0011/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0012",  "exploiter_gen0012/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0013",  "exploiter_gen0013/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0014",  "exploiter_gen0014/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0015",  "exploiter_gen0015/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0016",  "exploiter_gen0016/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0017",  "exploiter_gen0017/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0018",  "exploiter_gen0018/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0019",  "exploiter_gen0019/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0020",  "exploiter_gen0020/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0021",  "exploiter_gen0021/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0022",  "exploiter_gen0022/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0023",  "exploiter_gen0023/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0024",  "exploiter_gen0024/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0025",  "exploiter_gen0025/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0026",  "exploiter_gen0026/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0027",  "exploiter_gen0027/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0028",  "exploiter_gen0028/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0029",  "exploiter_gen0029/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0030",  "exploiter_gen0030/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0031",  "exploiter_gen0031/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0032",  "exploiter_gen0032/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0033",  "exploiter_gen0033/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0034",  "exploiter_gen0034/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0035",  "exploiter_gen0035/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0036",  "exploiter_gen0036/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0037",  "exploiter_gen0037/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0038",  "exploiter_gen0038/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0039",  "exploiter_gen0039/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0040",  "exploiter_gen0040/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0041",  "exploiter_gen0041/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0042",  "exploiter_gen0042/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0043",  "exploiter_gen0043/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0044",  "exploiter_gen0044/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0045",  "exploiter_gen0045/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0046",  "exploiter_gen0046/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0047",  "exploiter_gen0047/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0048",  "exploiter_gen0048/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0049",  "exploiter_gen0049/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0050",  "exploiter_gen0050/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0051",  "exploiter_gen0051/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0052",  "exploiter_gen0052/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0053",  "exploiter_gen0053/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0054",  "exploiter_gen0054/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0055",  "exploiter_gen0055/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0056",  "exploiter_gen0056/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0057",  "exploiter_gen0057/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0058",  "exploiter_gen0058/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0059",  "exploiter_gen0059/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0060",  "exploiter_gen0060/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0061",  "exploiter_gen0061/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0062",  "exploiter_gen0062/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0063",  "exploiter_gen0063/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0064",  "exploiter_gen0064/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0065",  "exploiter_gen0065/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0066",  "exploiter_gen0066/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0067",  "exploiter_gen0067/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0068",  "exploiter_gen0068/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0069",  "exploiter_gen0069/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0070",  "exploiter_gen0070/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0071",  "exploiter_gen0071/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0072",  "exploiter_gen0072/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0073",  "exploiter_gen0073/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0074",  "exploiter_gen0074/final_exploiter.zip"),
    ("rnad_at_exploiter_gen0075",  "exploiter_gen0075/final_exploiter.zip"),

]

# ─────────────────────────────────────────────────────────────────────────────
# Imports (after sys-path is known to include the project root)
# ─────────────────────────────────────────────────────────────────────────────

from environments.shooter_gym_env import ShooterGymEnv
from environments.shooter_env import OBS_DIM, N_AGENTS
from minimax_exploiter import (
    PPOMainAgent,
    PolicyNetwork,
    _RNaDMainAgentAdapter,
    N_ACTIONS,
    WIN_REWARD_THRESHOLD,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_rnad(path: Path, hidden_layers: tuple, device: str):
    """Load a RNaD .pt checkpoint → (PolicyNetwork, learner_steps, actor_steps)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Support both bare state-dicts and wrapped {"params": ...} dicts
    state_dict = ckpt.get("params", ckpt) if isinstance(ckpt, dict) else ckpt
    net = PolicyNetwork(
        obs_size=OBS_DIM,
        num_actions=N_ACTIONS,
        hidden_layers=hidden_layers,
    )
    net.load_state_dict(state_dict)
    net.eval()
    learner_steps = ckpt.get("learner_steps") if isinstance(ckpt, dict) else None
    actor_steps   = ckpt.get("actor_steps")   if isinstance(ckpt, dict) else None
    return net, learner_steps, actor_steps


def _load_agent(path: Path, hidden_layers: tuple, device: str):
    """Return (agent, kind, learner_steps, actor_steps)."""
    if path.suffix == ".pt":
        net, learner_steps, actor_steps = _load_rnad(path, hidden_layers, device)
        legal = torch.ones(1, N_ACTIONS, dtype=torch.float32,
                           device=torch.device(device))
        return _RNaDMainAgentAdapter(net, device, legal), "rnad", learner_steps, actor_steps
    elif path.suffix == ".zip":
        # Actor steps are encoded in the filename (e.g. ppo_step_00050000 → 50000)
        m = re.search(r"(\d+)$", path.stem)
        actor_steps = int(m.group(1)) if m else None
        return PPOMainAgent.from_checkpoint(str(path), device=device), "ppo", None, actor_steps
    else:
        raise ValueError(f"Unknown checkpoint extension: {path.suffix}")


def _evaluate(
    subject_agent,          # _RNaDMainAgentAdapter | PPOMainAgent
    opponent_fn: Callable,  # obs -> int
    num_episodes: int,
) -> dict:
    """Run num_episodes and return mean_reward, std_reward, win_rate."""
    env = ShooterGymEnv(self_play=False)
    env._opponent_fn = opponent_fn

    rewards, wins = [], 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            action = subject_agent.get_action(obs)
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += float(r)
            done = term or trunc
        rewards.append(ep_reward)
        if ep_reward > WIN_REWARD_THRESHOLD:
            wins += 1
    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "win_rate":    float(wins / num_episodes),
    }


def _random_opponent_fn() -> Callable:
    rng = np.random.default_rng()
    return lambda _obs: int(rng.integers(N_ACTIONS))


def _scripted_opponent_fn() -> Callable:
    from environments.scripted_shooter_agent import ScriptedShooterAgent
    agent = ScriptedShooterAgent(N_AGENTS * 2)
    return lambda obs: agent.get_action_and_value(obs)


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_checkpoints(run_dir: Path, max_gen: int | None = None) -> list[Path]:
    """
    Find all evaluable checkpoints under run_dir.

    RNaD : run_dir/rnad_at_exploiter_gen*.pt
    PPO  : run_dir/checkpoints/ppo_step_*.zip
    """
    ckpts: list[Path] = []

    # League checkpoints (optionally capped at max_gen)
    for p in sorted(run_dir.glob("rnad_at_exploiter_gen*.pt")):
        if max_gen is not None:
            m = re.search(r"gen(\d+)", p.stem)
            if m and int(m.group(1)) > max_gen:
                continue
        ckpts.append(p)

    # PPO checkpoints
    ckpts += sorted((run_dir / "checkpoints").glob("ppo_step_*.zip"))

    # RNAD checkpoints
    ckpts += sorted((run_dir / "checkpoints").glob("model_step_*.pt"))

    return ckpts


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(args) -> dict:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    hidden_layers = tuple(args.hidden_layers)
    device        = args.device
    num_episodes  = args.episodes

    # Build exploiter lookup: stem → zip path
    exploiter_lookup: dict[str, str] = {
        stem: zip_path for stem, zip_path in EXPLOITER_PAIRS
    }

    # Discover checkpoints
    checkpoints = discover_checkpoints(run_dir, max_gen=args.max_gen)
    if not checkpoints:
        raise RuntimeError(
            f"No checkpoints found in {run_dir}.\n"
            "Expected:\n"
            "  checkpoints/rnad_at_exploiter_gen*.pt  (in run_dir/)\n"
            "  checkpoints/ppo_step_*.zip (in run_dir/checkpoints/)"
        )

    print(f"Found {len(checkpoints)} checkpoint(s) in {run_dir}")
    print(f"Evaluating each with {num_episodes} episodes per opponent\n")

    # Pre-build stateless opponent functions (created once, reused for all ckpts)
    random_fn   = _random_opponent_fn()
    scripted_fn = _scripted_opponent_fn()

    results: dict = {}

    for i, ckpt_path in enumerate(checkpoints, 1):
        stem = ckpt_path.stem
        print(f"[{i}/{len(checkpoints)}] {ckpt_path.name}")
        t0 = time.perf_counter()

        # Load subject agent
        try:
            agent, kind, learner_steps, actor_steps = _load_agent(ckpt_path, hidden_layers, device)
        except Exception as e:
            print(f"  ERROR loading checkpoint: {e}")
            results[stem] = {"error": str(e)}
            continue

        entry: dict = {
            "kind":          kind,
            "path":          str(ckpt_path),
            "learner_steps": learner_steps,
            "actor_steps":   actor_steps,
        }

        # ── vs random ─────────────────────────────────────────────────────────
        entry["vs_random"] = _evaluate(agent, random_fn, num_episodes)
        print(
            f"  vs random   — win_rate={entry['vs_random']['win_rate']:.1%}"
            f"  mean_reward={entry['vs_random']['mean_reward']:+.2f}"
        )

        # ── vs scripted ───────────────────────────────────────────────────────
        entry["vs_scripted"] = _evaluate(agent, scripted_fn, num_episodes)
        print(
            f"  vs scripted — win_rate={entry['vs_scripted']['win_rate']:.1%}"
            f"  mean_reward={entry['vs_scripted']['mean_reward']:+.2f}"
        )

        # ── vs exploiter ──────────────────────────────────────────────────────
        # Resolve: explicit EXPLOITER_PAIRS first, then auto-discover from stem
        expl_path = None
        if stem in exploiter_lookup:
            _candidate = run_dir / exploiter_lookup[stem]
            expl_path = _candidate if _candidate.exists() else Path(exploiter_lookup[stem])
        else:
            m = re.search(r"rnad_at_exploiter_(gen\d+)", stem)
            if m:
                gen_dir = run_dir / f"exploiter_{m.group(1)}"
                found = sorted(gen_dir.glob("final_exploiter.*"))
                if found:
                    expl_path = found[0]

        if expl_path is not None:
            try:
                expl_agent, _, _, _ = _load_agent(expl_path, hidden_layers, device)
                expl_fn    = lambda obs, _a=expl_agent: _a.get_action(obs)
                entry["vs_exploiter"] = _evaluate(agent, expl_fn, num_episodes)
                entry["exploiter_path"] = str(expl_path)
                print(
                    f"  vs exploiter — win_rate={entry['vs_exploiter']['win_rate']:.1%}"
                    f"  mean_reward={entry['vs_exploiter']['mean_reward']:+.2f}"
                )
            except Exception as e:
                print(f"  vs exploiter — ERROR: {e}")
                entry["vs_exploiter"] = {"error": str(e)}
        else:
            entry["vs_exploiter"] = None

        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s\n")

        results[stem] = entry

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate all checkpoints in a league run folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir",       type=str, required=True,
                   help="Path to the league run folder")
    p.add_argument("--out",           type=str, default=None,
                   help="Output JSON file path (default: eval_<run_dir_name>.json)")
    p.add_argument("--episodes",      type=int, default=200,
                   help="Episodes per opponent per checkpoint")
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 256],
                   help="PolicyNetwork hidden layer widths (must match training)")
    p.add_argument("--device",        type=str, default="cpu")
    p.add_argument("--max-gen",       type=int, default=None,
                   help="Only evaluate rnad_at_exploiter_gen checkpoints up to this generation")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    results = run_evaluation(args)

    out_path = Path(args.out) if args.out else Path("output/evaluation") / f"eval_{Path(args.run_dir).name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")

    # Quick summary table
    print("\n── Summary ──────────────────────────────────────────────────────")
    header = f"{'Checkpoint':<40}  {'vs_random':>10}  {'vs_scripted':>11}  {'vs_exploiter':>12}"
    print(header)
    print("─" * len(header))
    for stem, entry in results.items():
        if "error" in entry:
            print(f"{stem:<40}  ERROR: {entry['error']}")
            continue
        r  = entry["vs_random"]["win_rate"]   if entry.get("vs_random")   else float("nan")
        s  = entry["vs_scripted"]["win_rate"] if entry.get("vs_scripted") else float("nan")
        e_entry = entry.get("vs_exploiter")
        e  = e_entry["win_rate"] if isinstance(e_entry, dict) and "win_rate" in e_entry else float("nan")
        print(f"{stem:<40}  {r:>9.1%}  {s:>10.1%}  {e:>11.1%}")
