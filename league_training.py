"""
Multiprocess League Training
=============================

Architecture
------------
                          ┌─────────────────────────────────┐
                          │         MAIN PROCESS            │
                          │                                 │
                          │  RNaD trains continuously       │
                          │  (self-play + population mix)   │
                          │                                 │
                          │  Policy population:             │
                          │   - last 3 RNaD snapshots       │
                          │   - latest trained exploiter    │
                          │                                 │
                          │  Every EXPLOITER_INTERVAL steps:│
                          │   → snapshot current RNaD       │
                          │   → send to exploiter worker    │
                          └───────────────┬─────────────────┘
                                          │  mp.Pipe
                          ┌───────────────▼─────────────────┐
                          │       EXPLOITER PROCESS         │
                          │                                 │
                          │  Trains until:                  │
                          │    • win_rate ≥ 1.0  OR         │
                          │    • actor_steps ≥ MAX_STEPS    │
                          │                                 │
                          │  Sends back: final win_rate +   │
                          │             trained net weights │
                          └─────────────────────────────────┘

Population sampling
-------------------
When RNaD collects trajectories, each of its `batch_size` environment
instances is paired with an opponent drawn from the population:

    population = [rnad_snap_1, rnad_snap_2, rnad_snap_3, exploiter]

Weights:
    rnad_weight     → controls how much previous RNaD snapshots are used
    exploiter_weight → controls how much the exploiter is used
    (normalised to sum to 1; missing slots are re-weighted automatically)

Population opponents are realised by hot-swapping the `_opponent_fn` of
each ShooterGymEnv(self_play=False) *before* trajectory collection.

Usage
-----
    python league_training.py --total-main-steps 2_000_000

    # Bump exploiter pressure
    python league_training.py --exploiter-weight 0.4 --rnad-weight 0.6

    # Quick smoke test
    python league_training.py --total-main-steps 50_000 \\
        --exploiter-interval 10000 --exploiter-max-steps 5000

    # Resume from checkpoint
    python league_training.py --resume runs/my_run/league_state.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from environments.shooter_gym_env import ShooterGymEnv
from environments.shooter_env import OBS_DIM
from minimax_exploiter import (
    MinimaxShooterGymEnv,
    evaluate_vs_main,
    WIN_REWARD_THRESHOLD,
)
from rnad import RNaD, RNaDConfig, AdamConfig, NerdConfig, PolicyNetwork

N_ACTIONS = 7
_EVAL_EPISODES = 50


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeagueConfig:
    # ── RNaD (main agent) ─────────────────────────────────────────────────────
    total_main_steps:       int   = 2_000_000
    hidden_layers:          tuple = (256, 256)
    batch_size:             int   = 256
    trajectory_max:         int   = 200
    learning_rate:          float = 5e-5
    log_interval:           int   = 100        # learner steps between console logs
    eval_interval_steps:    int   = 50_000     # actor steps between evaluations

    # ── Population / opponent sampling ────────────────────────────────────────
    population_size:        int   = 3          # number of past RNaD snapshots kept
    rnad_weight:            float = 0.7        # relative weight for RNaD snapshots
    exploiter_weight:       float = 0.3        # relative weight for exploiter slot
    snapshot_interval:      int   = 50_000     # actor steps between RNaD snapshots

    # ── Exploiter ────────────────────────────────────────────────────────────
    exploiter_interval:     int   = 100_000    # actor steps between exploiter launches
    exploiter_max_steps:    int   = 200_000    # hard budget per exploiter run
    exploiter_win_target:   float = 1.0        # stop early if win-rate reaches this
    exploiter_batch_size:   int   = 128
    exploiter_traj_max:     int   = 200

    # ── Minimax reward ────────────────────────────────────────────────────────
    alpha:   float = 0.05
    gamma:   float = 0.995
    v_shift: float = 25.0

    # ── Misc ─────────────────────────────────────────────────────────────────
    device:  str   = "cpu"
    seed:    int   = 42
    run_dir: str   = "runs/league"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: policy network → opponent callable
# ─────────────────────────────────────────────────────────────────────────────

def _net_to_opponent_fn(net: PolicyNetwork, device: str) -> Callable:
    """Return a fn(obs: np.ndarray) -> int that samples from `net`."""
    _dev = torch.device(device)
    _legal = torch.ones(1, N_ACTIONS, dtype=torch.float32, device=_dev)
    _net = copy.deepcopy(net).to(_dev)
    _net.eval()
    for p in _net.parameters():
        p.requires_grad_(False)

    def _fn(obs: np.ndarray) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=_dev).unsqueeze(0)
        with torch.no_grad():
            pi, _, _, _ = _net(obs_t, _legal)
        pi_np = pi.cpu().numpy().squeeze().astype(np.float64)
        pi_np /= pi_np.sum()
        return int(np.random.choice(N_ACTIONS, p=pi_np))

    return _fn


def _state_dict_cpu(net: PolicyNetwork) -> Dict:
    """Return a state dict with all tensors on CPU (safe to pickle/share)."""
    return {k: v.cpu() for k, v in net.state_dict().items()}


def _load_net(state_dict: Dict, hidden_layers: tuple, device: str) -> PolicyNetwork:
    net = PolicyNetwork(OBS_DIM, N_ACTIONS, hidden_layers)
    net.load_state_dict(state_dict)
    net.to(device)
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Population manager
# ─────────────────────────────────────────────────────────────────────────────

class PopulationManager:
    """
    Maintains a rolling window of past RNaD snapshots and the latest
    trained exploiter, and exposes a method to build per-environment
    opponent callables for trajectory collection.

    Sampling logic
    --------------
    Each call to `sample_opponents(n)` returns a list of n callables.
    The probability of drawing from each slot is:

        p_rnad_slot   = rnad_weight / num_rnad_snapshots  (per snapshot)
        p_exploiter   = exploiter_weight                   (if present)

    Weights are re-normalised when the exploiter slot is empty or when
    fewer than `population_size` RNaD snapshots have been collected.
    """

    def __init__(
        self,
        population_size:  int   = 3,
        rnad_weight:      float = 0.7,
        exploiter_weight: float = 0.3,
        hidden_layers:    tuple = (256, 256),
        device:           str   = "cpu",
    ):
        self._pop_size        = population_size
        self._rnad_w          = rnad_weight
        self._exploiter_w     = exploiter_weight
        self._hidden_layers   = hidden_layers
        self._device          = device

        self._rnad_snapshots:   Deque[Dict]        = deque(maxlen=population_size)
        self._exploiter_sd:     Optional[Dict]     = None   # latest exploiter state_dict

    # ── public API ────────────────────────────────────────────────────────────

    def add_rnad_snapshot(self, net: PolicyNetwork) -> None:
        """Push a new RNaD snapshot (CPU state dict) into the rolling window."""
        self._rnad_snapshots.append(_state_dict_cpu(net))

    def set_exploiter(self, state_dict: Dict) -> None:
        """Hot-swap the exploiter slot with a freshly trained network."""
        self._exploiter_sd = {k: v.cpu() for k, v in state_dict.items()}

    def sample_opponents(self, n: int) -> List[Callable]:
        """
        Return a list of n opponent callables sampled from the population.
        Falls back to a random opponent when the population is empty.
        """
        entries, weights = self._build_distribution()
        if not entries:
            rng = np.random.default_rng()
            return [lambda _obs, _r=rng: int(_r.integers(N_ACTIONS))] * n

        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr /= weights_arr.sum()

        chosen = np.random.choice(len(entries), size=n, p=weights_arr)
        # Build opponent fns lazily — one per unique index to avoid redundant copies
        cache: Dict[int, Callable] = {}
        result = []
        for idx in chosen:
            if idx not in cache:
                sd = entries[idx]
                net = _load_net(sd, self._hidden_layers, self._device)
                cache[idx] = _net_to_opponent_fn(net, self._device)
            result.append(cache[idx])
        return result

    def num_rnad_snapshots(self) -> int:
        return len(self._rnad_snapshots)

    def has_exploiter(self) -> bool:
        return self._exploiter_sd is not None

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_distribution(self) -> Tuple[List[Dict], List[float]]:
        """Return (list_of_state_dicts, list_of_weights) ready for np.random.choice."""
        entries: List[Dict]  = []
        weights: List[float] = []

        n_rnad = len(self._rnad_snapshots)
        if n_rnad > 0:
            w_per_rnad = self._rnad_w / n_rnad
            for sd in self._rnad_snapshots:
                entries.append(sd)
                weights.append(w_per_rnad)

        if self._exploiter_sd is not None:
            entries.append(self._exploiter_sd)
            weights.append(self._exploiter_w)

        return entries, weights


# ─────────────────────────────────────────────────────────────────────────────
# Mixed-opponent env factory
# ─────────────────────────────────────────────────────────────────────────────

class PopulationShooterEnv(ShooterGymEnv):
    """
    A ShooterGymEnv whose Blue opponent can be replaced at runtime.
    Used so the main RNaD can face a sampled population member per episode.
    """

    def __init__(self):
        super().__init__(self_play=False, opponent="random")

    def set_opponent(self, fn: Callable) -> None:
        self._opponent_fn = fn


def make_population_env_fn(population: PopulationManager, batch_size: int):
    """
    Returns an env_fn that, when called `batch_size` times, creates envs
    with opponents pre-sampled from `population`.

    We stash the sampled opponents in a thread-local list that is consumed
    in order.  This works because RNaD calls env_fn exactly `batch_size`
    times sequentially in __init__ / when rebuilding after reset.

    The trick: we sample all `batch_size` opponents once (per trajectory
    collection call) and rotate them.  Because the opponents are stateless
    callables this is safe.
    """
    _sampled: List[Callable] = []

    def resample():
        nonlocal _sampled
        _sampled = population.sample_opponents(batch_size)

    resample()  # initial sample

    call_count = [0]

    def env_fn():
        idx = call_count[0] % batch_size
        if idx == 0:
            resample()
        opp = _sampled[idx]
        call_count[0] += 1
        env = PopulationShooterEnv()
        env.set_opponent(opp)
        return env

    return env_fn


# ─────────────────────────────────────────────────────────────────────────────
# Exploiter worker (runs in a separate process)
# ─────────────────────────────────────────────────────────────────────────────

def _exploiter_worker(
    conn,            # multiprocessing.Connection  (child end)
    config: LeagueConfig,
    run_dir_str: str,
    generation: int,
):
    """
    Long-running worker process.

    Protocol
    --------
    Receives from parent:
        ("train", state_dict_cpu, gen_id)
            → trains exploiter against frozen RNaD snapshot
            → replies ("done", gen_id, final_win_rate, exploiter_state_dict_cpu)

        ("quit",)
            → exits cleanly
    """
    run_dir = Path(run_dir_str)
    torch.manual_seed(config.seed + 9999)
    np.random.seed(config.seed + 9999)

    print(f"[exploiter worker] ready (pid={mp.current_process().pid})")

    while True:
        msg = conn.recv()
        cmd = msg[0]

        if cmd == "quit":
            print("[exploiter worker] shutting down.")
            break

        if cmd != "train":
            print(f"[exploiter worker] unknown command: {cmd}")
            continue

        _, rnad_sd, gen_id = msg
        print(f"\n[exploiter worker] gen {gen_id}: starting exploiter training")

        # Rebuild the frozen main net from the received state dict
        main_net = PolicyNetwork(OBS_DIM, N_ACTIONS, config.hidden_layers)
        main_net.load_state_dict(rnad_sd)
        main_net.eval()
        for p in main_net.parameters():
            p.requires_grad_(False)

        ckpt_dir = run_dir / f"exploiter_gen{gen_id:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Build exploiter R-NaD (single-agent mode)
        max_learner_steps = (
            config.exploiter_max_steps
            // (config.exploiter_batch_size * config.exploiter_traj_max)
            + 1
        )
        expl_cfg = RNaDConfig(
            policy_network_layers=config.hidden_layers,
            batch_size=config.exploiter_batch_size,
            trajectory_max=config.exploiter_traj_max,
            learning_rate=config.learning_rate,
            adam=AdamConfig(b1=0.0, b2=0.999, eps=1e-7),
            clip_gradient=10_000,
            target_network_avg=0.001,
            entropy_schedule_repeats=(1,),
            entropy_schedule_size=(max_learner_steps,),
            eta_reward_transform=0.2,
            nerd=NerdConfig(beta=2.0, clip=10_000),
            c_vtrace=1.0,
            num_players=1,
            seed=config.seed + gen_id * 31,
        )

        def env_fn():
            return MinimaxShooterGymEnv(
                main_net=main_net,
                alpha=config.alpha,
                gamma=config.gamma,
                v_shift=config.v_shift,
                device=config.device,
            )

        expl_model = RNaD(env_fn=env_fn, config=expl_cfg, device=config.device)

        t0 = time.perf_counter()
        final_win_rate = 0.0
        eval_interval = config.exploiter_batch_size * config.exploiter_traj_max * 10

        while expl_model.actor_steps < config.exploiter_max_steps:
            prev = expl_model.actor_steps
            expl_model.step()
            expl_model.num_timesteps += expl_model.actor_steps - prev

            # Periodic eval
            if expl_model.actor_steps % eval_interval < (
                config.exploiter_batch_size * config.exploiter_traj_max
            ):
                stats = evaluate_vs_main(
                    expl_model.params, main_net,
                    num_episodes=_EVAL_EPISODES, device=config.device,
                )
                final_win_rate = stats["win_rate"]
                elapsed = (time.perf_counter() - t0) / 60
                print(
                    f"[exploiter gen {gen_id}] "
                    f"steps={expl_model.actor_steps:>7,}  "
                    f"win_rate={final_win_rate:.0%}  "
                    f"elapsed={elapsed:.1f}min"
                )
                if final_win_rate >= config.exploiter_win_target:
                    print(f"[exploiter gen {gen_id}] ✓ reached {final_win_rate:.0%} win-rate — stopping early")
                    break

        # Final eval regardless (ensures we always have an accurate number)
        stats = evaluate_vs_main(
            expl_model.params, main_net,
            num_episodes=_EVAL_EPISODES, device=config.device,
        )
        final_win_rate = stats["win_rate"]
        print(
            f"[exploiter gen {gen_id}] FINAL win_rate={final_win_rate:.0%}  "
            f"steps={expl_model.actor_steps:,}"
        )

        torch.save(
            expl_model.params.state_dict(),
            ckpt_dir / "final_exploiter.pt",
        )

        conn.send((
            "done",
            gen_id,
            final_win_rate,
            _state_dict_cpu(expl_model.params),
        ))

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_league(config: LeagueConfig):
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    writer = SummaryWriter(log_dir=str(run_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"\n{'='*65}")
    print(f"  Multiprocess League Training")
    print(f"  Run dir     : {run_dir}")
    print(f"  Total steps : {config.total_main_steps:,}")
    print(f"  Population  : {config.population_size} RNaD snapshots + 1 exploiter slot")
    print(f"  Weights     : rnad={config.rnad_weight}  exploiter={config.exploiter_weight}")
    print(f"  Exploiter   : every {config.exploiter_interval:,} steps, "
          f"budget {config.exploiter_max_steps:,} steps")
    print(f"{'='*65}\n")

    # ── Population ────────────────────────────────────────────────────────────
    population = PopulationManager(
        population_size=config.population_size,
        rnad_weight=config.rnad_weight,
        exploiter_weight=config.exploiter_weight,
        hidden_layers=config.hidden_layers,
        device=config.device,
    )

    # ── Exploiter subprocess ──────────────────────────────────────────────────
    parent_conn, child_conn = mp.Pipe(duplex=True)
    exploiter_proc = mp.Process(
        target=_exploiter_worker,
        args=(child_conn, config, str(run_dir), 0),
        daemon=True,
        name="exploiter_worker",
    )
    exploiter_proc.start()
    child_conn.close()   # parent only uses parent_conn

    exploiter_busy    = False   # True while worker is training
    exploiter_gen     = 0       # generation counter
    exploiter_history: List[Dict] = []   # [{gen, win_rate, actor_steps}, …]

    # ── RNaD main agent ───────────────────────────────────────────────────────
    # Build env factory that uses the population
    env_fn = make_population_env_fn(population, config.batch_size)

    main_cfg = RNaDConfig(
        policy_network_layers=config.hidden_layers,
        batch_size=config.batch_size,
        trajectory_max=config.trajectory_max,
        learning_rate=config.learning_rate,
        adam=AdamConfig(b1=0.0, b2=0.999, eps=1e-7),
        clip_gradient=10_000,
        target_network_avg=0.001,
        entropy_schedule_repeats=(1,),
        entropy_schedule_size=(
            config.total_main_steps // (config.batch_size * config.trajectory_max) + 1,
        ),
        eta_reward_transform=0.2,
        nerd=NerdConfig(beta=2.0, clip=10_000),
        c_vtrace=1.0,
        num_players=2,
        seed=config.seed,
    )

    main_model = RNaD(env_fn=env_fn, config=main_cfg, device=config.device)

    # ── Training state ────────────────────────────────────────────────────────
    t0                      = time.perf_counter()
    last_snapshot_steps     = 0
    last_exploiter_steps    = 0
    last_log_steps          = 0

    print("Training started. RNaD runs continuously; exploiter launches in background.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while main_model.actor_steps < config.total_main_steps:

        # ── 1. RNaD step ──────────────────────────────────────────────────────
        prev_actor = main_model.actor_steps
        logs = main_model.step()
        main_model.num_timesteps += main_model.actor_steps - prev_actor

        ls = main_model.learner_steps
        actor_steps = main_model.actor_steps

        writer.add_scalar("rnad/loss",        logs["loss"],        ls)
        writer.add_scalar("rnad/mean_reward", logs["mean_reward"], ls)
        writer.add_scalar("rnad/actor_steps", actor_steps,         ls)
        writer.add_scalar(
            "population/num_rnad_snapshots",
            population.num_rnad_snapshots(), ls,
        )
        writer.add_scalar(
            "population/has_exploiter",
            float(population.has_exploiter()), ls,
        )

        # ── 2. Console log ────────────────────────────────────────────────────
        if ls % config.log_interval == 0:
            elapsed = (time.perf_counter() - t0) / 60
            pop_desc = (
                f"{population.num_rnad_snapshots()} RNaD snap"
                + ("s" if population.num_rnad_snapshots() != 1 else "")
                + (f" + exploiter" if population.has_exploiter() else "")
            )
            expl_status = "busy" if exploiter_busy else "idle"
            print(
                f"[main] ls={ls:>6}  actor={actor_steps:>8,}  "
                f"loss={logs['loss']:.4f}  "
                f"pop=({pop_desc})  "
                f"exploiter={expl_status}  "
                f"elapsed={elapsed:.1f}min"
            )

        # ── 3. Snapshot RNaD into population ──────────────────────────────────
        if actor_steps - last_snapshot_steps >= config.snapshot_interval:
            population.add_rnad_snapshot(main_model.params)
            last_snapshot_steps = actor_steps
            print(
                f"  [snapshot] Stored RNaD snapshot "
                f"(population has {population.num_rnad_snapshots()} snapshots)"
            )
            writer.add_scalar(
                "population/snapshot_actor_steps", actor_steps, ls
            )

        # ── 4. Poll exploiter results (non-blocking) ──────────────────────────
        if exploiter_busy and parent_conn.poll():
            resp = parent_conn.recv()
            if resp[0] == "done":
                _, recv_gen, win_rate, expl_sd = resp
                exploiter_busy = False

                # Inject exploiter into population
                population.set_exploiter(expl_sd)

                # Log the *post-training* win-rate — key convergence metric
                writer.add_scalar("exploiter/final_win_rate",  win_rate, recv_gen)
                writer.add_scalar("exploiter/win_rate_vs_time", win_rate, actor_steps)
                writer.flush()

                record = {
                    "gen":         recv_gen,
                    "win_rate":    win_rate,
                    "actor_steps": actor_steps,
                }
                exploiter_history.append(record)

                print(
                    f"\n  [exploiter gen {recv_gen}] DONE  "
                    f"win_rate={win_rate:.0%}  "
                    f"(main actor_steps={actor_steps:,})\n"
                )

                # Save running history
                with open(run_dir / "exploiter_history.json", "w") as f:
                    json.dump(exploiter_history, f, indent=2)

                # Save current RNaD checkpoint alongside
                ckpt_path = run_dir / f"rnad_at_exploiter_gen{recv_gen:04d}.pt"
                main_model.save(str(ckpt_path))

        # ── 5. Launch exploiter if interval reached and worker idle ───────────
        ready_for_exploiter = (
            not exploiter_busy
            and (actor_steps - last_exploiter_steps >= config.exploiter_interval)
            and population.num_rnad_snapshots() > 0   # at least one snapshot exists
        )
        if ready_for_exploiter:
            exploiter_gen += 1
            last_exploiter_steps = actor_steps

            # Snapshot current RNaD weights to send to worker
            rnad_sd = _state_dict_cpu(main_model.params)

            print(
                f"\n  [launch exploiter gen {exploiter_gen}] "
                f"at actor_step={actor_steps:,}"
            )
            parent_conn.send(("train", rnad_sd, exploiter_gen))
            exploiter_busy = True

            writer.add_scalar(
                "exploiter/launch_actor_steps", actor_steps, exploiter_gen
            )

    # ── Training done — final cleanup ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Training complete — {main_model.actor_steps:,} actor steps")
    print(f"{'='*65}")

    # Wait for any in-flight exploiter
    if exploiter_busy:
        print("Waiting for in-flight exploiter to finish …")
        resp = parent_conn.recv()
        if resp[0] == "done":
            _, recv_gen, win_rate, expl_sd = resp
            population.set_exploiter(expl_sd)
            print(f"  Exploiter gen {recv_gen} finished: win_rate={win_rate:.0%}")
            writer.add_scalar("exploiter/final_win_rate", win_rate, recv_gen)
            exploiter_history.append({
                "gen": recv_gen, "win_rate": win_rate,
                "actor_steps": main_model.actor_steps,
            })

    # Shut down worker
    parent_conn.send(("quit",))
    exploiter_proc.join(timeout=30)
    if exploiter_proc.is_alive():
        exploiter_proc.terminate()

    # Save final artefacts
    main_model.save(str(run_dir / "final_rnad.pt"))
    with open(run_dir / "exploiter_history.json", "w") as f:
        json.dump(exploiter_history, f, indent=2)

    # Print convergence summary
    print("\nExploiter win-rate history (Nash convergence proxy):")
    print(f"  {'Gen':>5}  {'Win Rate':>10}  {'Actor Steps':>14}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*14}")
    for r in exploiter_history:
        print(f"  {r['gen']:>5}  {r['win_rate']:>10.1%}  {r['actor_steps']:>14,}")
    print("\n(Lower exploiter win-rate = closer to Nash equilibrium)")

    writer.flush()
    writer.close()

    return main_model, exploiter_history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multiprocess league: continuous RNaD + periodic exploiter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── RNaD ──────────────────────────────────────────────────────────────────
    p.add_argument("--total-main-steps",    type=int,   default=2_000_000)
    p.add_argument("--hidden-layers",       type=int,   nargs="+", default=[256, 256])
    p.add_argument("--batch-size",          type=int,   default=256)
    p.add_argument("--trajectory-max",      type=int,   default=200)
    p.add_argument("--learning-rate",       type=float, default=5e-5)
    p.add_argument("--log-interval",        type=int,   default=100)
    p.add_argument("--eval-interval-steps", type=int,   default=50_000)

    # ── Population ────────────────────────────────────────────────────────────
    p.add_argument("--population-size",     type=int,   default=3,
                   help="Number of past RNaD snapshots to keep in population")
    p.add_argument("--rnad-weight",         type=float, default=0.9,
                   help="Relative sampling weight for RNaD snapshot slots")
    p.add_argument("--exploiter-weight",    type=float, default=0.1,
                   help="Relative sampling weight for the exploiter slot")
    p.add_argument("--snapshot-interval",   type=int,   default=100_000,
                   help="Actor steps between RNaD population snapshots")

    # ── Exploiter ─────────────────────────────────────────────────────────────
    p.add_argument("--exploiter-interval",  type=int,   default=150_000,
                   help="Actor steps between exploiter launches")
    p.add_argument("--exploiter-max-steps", type=int,   default=2_000_000,
                   help="Max actor steps per exploiter run")
    p.add_argument("--exploiter-win-target",type=float, default=0.85,
                   help="Exploiter stops early when win-rate reaches this")
    p.add_argument("--exploiter-batch-size",type=int,   default=128)
    p.add_argument("--exploiter-traj-max",  type=int,   default=200)

    # ── Minimax reward ────────────────────────────────────────────────────────
    p.add_argument("--alpha",   type=float, default=0.05)
    p.add_argument("--gamma",   type=float, default=0.995)
    p.add_argument("--v-shift", type=float, default=25.0)

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--device",   type=str, default="cpu")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--run-dir",  type=str, default="runs/league")

    return p.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # required for CUDA + torch multiprocessing
    args = parse_args()

    cfg = LeagueConfig(
        total_main_steps        = args.total_main_steps,
        hidden_layers           = tuple(args.hidden_layers),
        batch_size              = args.batch_size,
        trajectory_max          = args.trajectory_max,
        learning_rate           = args.learning_rate,
        log_interval            = args.log_interval,
        eval_interval_steps     = args.eval_interval_steps,
        population_size         = args.population_size,
        rnad_weight             = args.rnad_weight,
        exploiter_weight        = args.exploiter_weight,
        snapshot_interval       = args.snapshot_interval,
        exploiter_interval      = args.exploiter_interval,
        exploiter_max_steps     = args.exploiter_max_steps,
        exploiter_win_target    = args.exploiter_win_target,
        exploiter_batch_size    = args.exploiter_batch_size,
        exploiter_traj_max      = args.exploiter_traj_max,
        alpha                   = args.alpha,
        gamma                   = args.gamma,
        v_shift                 = args.v_shift,
        device                  = args.device,
        seed                    = args.seed,
        run_dir                 = args.run_dir,
    )

    train_league(cfg)
