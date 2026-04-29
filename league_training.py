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
    PPOMainAgent,
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

    # ── Exploiter (PPO) ───────────────────────────────────────────────────────
    exploiter_interval:     int   = 100_000    # actor steps between exploiter launches
    exploiter_max_steps:    int   = 200_000    # hard budget per exploiter run
    exploiter_win_target:   float = 1.0        # stop early if win-rate reaches this
    # PPO hyperparameters
    exploiter_n_envs:       int   = 8
    exploiter_n_steps:      int   = 2048
    exploiter_batch_size:   int   = 64         # PPO mini-batch size
    exploiter_n_epochs:     int   = 10
    exploiter_lr:           float = 3e-4
    exploiter_clip_range:   float = 0.2
    exploiter_ent_coef:     float = 0.01

    # ── Minimax reward ────────────────────────────────────────────────────────
    alpha:   float = 0.05
    gamma:   float = 0.995
    v_shift: float = 25.0

    # ── Misc ─────────────────────────────────────────────────────────────────
    exploiter_feedback: bool = True   # if False, exploiter trains for eval only (not injected into population)
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
        # PPO exploiter: stored as a loaded PPOMainAgent (None until first gen done)
        self._exploiter_agent: Optional[PPOMainAgent] = None
        self._exploiter_fn:    Optional[Callable]     = None   # cached callable

    # ── public API ────────────────────────────────────────────────────────────

    def add_rnad_snapshot(self, net: PolicyNetwork) -> None:
        """Push a new RNaD snapshot (CPU state dict) into the rolling window."""
        self._rnad_snapshots.append(_state_dict_cpu(net))

    def set_exploiter(self, zip_path: str) -> None:
        """Hot-swap the exploiter slot with a freshly trained PPO model.

        ``zip_path`` is the path to the ``.zip`` file saved by SB3.
        The model is loaded immediately and a fresh opponent callable is built.
        """
        self._exploiter_agent = PPOMainAgent.from_checkpoint(zip_path,
                                                              device=self._device)
        # Build the callable once; reused until the next set_exploiter call
        agent = self._exploiter_agent
        self._exploiter_fn = lambda obs: agent.get_action(obs)

    def sample_opponents(self, n: int) -> List[Callable]:
        """
        Return a list of n opponent callables sampled from the population.
        Falls back to a random opponent when the population is empty.
        """
        # ── Build RNaD entries ────────────────────────────────────────────────
        rnad_entries: List[Dict]  = list(self._rnad_snapshots)
        rnad_weights: List[float] = []
        if rnad_entries:
            w_per = self._rnad_w / len(rnad_entries)
            rnad_weights = [w_per] * len(rnad_entries)

        has_expl = self._exploiter_fn is not None
        total_w  = sum(rnad_weights) + (self._exploiter_w if has_expl else 0.0)

        if total_w == 0.0:
            rng = np.random.default_rng()
            return [lambda _obs, _r=rng: int(_r.integers(N_ACTIONS))] * n

        # Normalise
        norm_rnad = [w / total_w for w in rnad_weights]
        norm_expl = (self._exploiter_w / total_w) if has_expl else 0.0

        # Sample indices: 0 … len(rnad)-1 are RNaD; index len(rnad) is exploiter
        all_weights = norm_rnad + ([norm_expl] if has_expl else [])
        chosen = np.random.choice(len(all_weights), size=n,
                                  p=np.array(all_weights, dtype=np.float64))

        # Build RNaD opponent callables lazily (one copy per unique snapshot index)
        rnad_cache: Dict[int, Callable] = {}
        result: List[Callable] = []
        expl_idx = len(rnad_entries)   # index that maps to the exploiter slot

        for idx in chosen:
            if idx == expl_idx:
                result.append(self._exploiter_fn)
            else:
                if idx not in rnad_cache:
                    net = _load_net(rnad_entries[idx], self._hidden_layers,
                                   self._device)
                    rnad_cache[idx] = _net_to_opponent_fn(net, self._device)
                result.append(rnad_cache[idx])

        # Accumulate counts; print ratio every 10 000 episodes
        n_expl = int(np.sum(chosen == expl_idx))
        self._sample_count      = getattr(self, "_sample_count",      0) + n
        self._sample_expl_total = getattr(self, "_sample_expl_total", 0) + n_expl
        if self._sample_count % 100 < n:
            total = self._sample_count
            ratio = self._sample_expl_total / total if total else 0.0
            print(
                f"[population] {total:,} episodes sampled — "
                f"exploiter: {ratio:.1%}  rnad: {1-ratio:.1%}"
            )

        return result

    def num_rnad_snapshots(self) -> int:
        return len(self._rnad_snapshots)

    def has_exploiter(self) -> bool:
        return self._exploiter_agent is not None


# ─────────────────────────────────────────────────────────────────────────────
# Mixed-opponent env factory
# ─────────────────────────────────────────────────────────────────────────────

class PopulationShooterEnv(ShooterGymEnv):
    """
    A ShooterGymEnv whose Blue opponent is resampled from the population
    at the start of every episode, so a newly-injected exploiter is picked
    up immediately without needing to reconstruct any envs.
    """

    def __init__(self, population: "PopulationManager"):
        super().__init__(self_play=False, opponent="random")
        self._population = population

    def reset(self, **kwargs):
        # Resample a fresh opponent from the current population before each episode
        self._opponent_fn = self._population.sample_opponents(1)[0]
        return super().reset(**kwargs)


def make_population_env_fn(population: PopulationManager, batch_size: int):
    """Returns an env_fn that creates PopulationShooterEnv instances.

    Each env resamples its own opponent from the population at every episode
    reset, so the full population (including any newly-arrived exploiter) is
    always reflected without requiring env reconstruction.
    """
    def env_fn():
        return PopulationShooterEnv(population)

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
            → trains a PPO exploiter against the frozen RNaD snapshot
            → replies ("done", gen_id, final_win_rate, zip_path_str)
              where zip_path_str is the on-disk path to the saved .zip

        ("quit",)
            → exits cleanly
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as e:
        raise ImportError(
            "stable_baselines3 is required for PPO exploiter training. "
            "Install it with:  pip install stable-baselines3"
        ) from e

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
        print(f"\n[exploiter worker] gen {gen_id}: starting PPO exploiter training")

        # ── Rebuild the frozen RNaD main net from the received state dict ─────
        main_net = PolicyNetwork(OBS_DIM, N_ACTIONS, config.hidden_layers)
        main_net.load_state_dict(rnad_sd)
        main_net.eval()
        for p in main_net.parameters():
            p.requires_grad_(False)

        ckpt_dir = run_dir / f"exploiter_gen{gen_id:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── Vectorised Minimax env ─────────────────────────────────────────────
        def _env_fn():
            return MinimaxShooterGymEnv(
                main_net=main_net,
                alpha=config.alpha,
                gamma=config.gamma,
                v_shift=config.v_shift,
                device="cpu",
            )

        train_env = make_vec_env(
            _env_fn,
            n_envs=config.exploiter_n_envs,
            seed=config.seed + gen_id * 31,
        )

        # ── PPO model ──────────────────────────────────────────────────────────
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            n_steps=config.exploiter_n_steps,
            batch_size=config.exploiter_batch_size,
            n_epochs=config.exploiter_n_epochs,
            learning_rate=config.exploiter_lr,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=config.exploiter_clip_range,
            ent_coef=config.exploiter_ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": list(config.hidden_layers)},
            device="cpu",
            verbose=0,
            seed=config.seed + gen_id * 31,
        )

        # ── Callback: periodic eval + early stopping ───────────────────────────
        t0 = time.perf_counter()
        final_win_rate = 0.0
        eval_interval  = config.exploiter_n_envs * config.exploiter_n_steps * 10

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self._last_eval = 0

            def _on_step(self) -> bool:
                nonlocal final_win_rate
                n = self.num_timesteps
                if n - self._last_eval < eval_interval:
                    return True
                self._last_eval = n

                stats = evaluate_vs_main(
                    PPOMainAgent(self.model), main_net,
                    num_episodes=_EVAL_EPISODES, device="cpu",
                )
                final_win_rate = stats["win_rate"]
                elapsed = (time.perf_counter() - t0) / 60
                print(
                    f"[exploiter gen {gen_id}] "
                    f"steps={n:>7,}  "
                    f"win_rate={final_win_rate:.0%}  "
                    f"elapsed={elapsed:.1f}min"
                )
                conn.send(("progress", gen_id, n, final_win_rate))
                if final_win_rate >= config.exploiter_win_target:
                    print(
                        f"[exploiter gen {gen_id}] ✓ reached "
                        f"{final_win_rate:.0%} win-rate — stopping early"
                    )
                    return False   # signal SB3 to stop
                return True

        model.learn(
            total_timesteps=config.exploiter_max_steps,
            callback=_Callback(),
            reset_num_timesteps=True,
        )
        train_env.close()

        # ── Final eval (always, even after early stop) ─────────────────────────
        stats = evaluate_vs_main(
            PPOMainAgent(model), main_net,
            num_episodes=_EVAL_EPISODES, device="cpu",
        )
        final_win_rate = stats["win_rate"]
        print(
            f"[exploiter gen {gen_id}] FINAL win_rate={final_win_rate:.0%}  "
            f"steps={model.num_timesteps:,}"
        )

        # ── Save .zip and send the path back to the parent ─────────────────────
        # SB3 model.save() appends .zip automatically
        zip_stem = str(ckpt_dir / "final_exploiter")
        model.save(zip_stem)
        zip_path = zip_stem + ".zip"

        conn.send(("done", gen_id, final_win_rate, model.num_timesteps, zip_path))

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
        while exploiter_busy and parent_conn.poll():
            resp = parent_conn.recv()
            if resp[0] == "progress":
                _, recv_gen, expl_steps, win_rate = resp
                writer.add_scalar("exploiter/win_rate_during_training", win_rate, actor_steps)
                continue
            if resp[0] != "done":
                continue
            _, recv_gen, win_rate, expl_steps_used, zip_path = resp
            exploiter_busy = False

            # Inject exploiter into population (skip if eval-only mode)
            if config.exploiter_feedback:
                population.set_exploiter(zip_path)

            # Log the *post-training* win-rate — key convergence metric
            writer.add_scalar("exploiter/final_win_rate",  win_rate,       recv_gen)
            writer.add_scalar("exploiter/win_rate_vs_time", win_rate,      actor_steps)
            writer.add_scalar("exploiter/training_steps",  expl_steps_used, recv_gen)
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
        while True:
            resp = parent_conn.recv()
            if resp[0] == "progress":
                continue
            if resp[0] == "done":
                _, recv_gen, win_rate, expl_steps_used, zip_path = resp
                population.set_exploiter(zip_path)
                print(f"  Exploiter gen {recv_gen} finished: win_rate={win_rate:.0%}")
                writer.add_scalar("exploiter/final_win_rate",  win_rate,        recv_gen)
                writer.add_scalar("exploiter/training_steps",  expl_steps_used, recv_gen)
                exploiter_history.append({
                    "gen": recv_gen, "win_rate": win_rate,
                    "actor_steps": main_model.actor_steps,
                })
                break

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
    p.add_argument("--total-main-steps",    type=int,   default=200_000_000)
    p.add_argument("--hidden-layers",       type=int,   nargs="+", default=[256, 256])
    p.add_argument("--batch-size",          type=int,   default=256)
    p.add_argument("--trajectory-max",      type=int,   default=200)
    p.add_argument("--learning-rate",       type=float, default=5e-5)
    p.add_argument("--log-interval",        type=int,   default=100)
    p.add_argument("--eval-interval-steps", type=int,   default=50_000)

    # ── Population ────────────────────────────────────────────────────────────
    p.add_argument("--population-size",     type=int,   default=3,
                   help="Number of past RNaD snapshots to keep in population")
    p.add_argument("--rnad-weight",         type=float, default=0.7,
                   help="Relative sampling weight for RNaD snapshot slots")
    p.add_argument("--exploiter-weight",    type=float, default=0.3,
                   help="Relative sampling weight for the exploiter slot")
    p.add_argument("--snapshot-interval",   type=int,   default=100_000,
                   help="Actor steps between RNaD population snapshots")

    # ── Exploiter (PPO) ───────────────────────────────────────────────────────
    p.add_argument("--exploiter-interval",   type=int,   default=100_000,
                   help="Actor steps between exploiter launches")
    p.add_argument("--exploiter-max-steps",  type=int,   default=20_000,
                   help="Max actor steps per exploiter run")
    p.add_argument("--exploiter-win-target", type=float, default=0.9,
                   help="Exploiter stops early when win-rate reaches this")
    p.add_argument("--exploiter-n-envs",     type=int,   default=8,
                   help="Parallel envs for PPO rollout collection")
    p.add_argument("--exploiter-n-steps",    type=int,   default=2048,
                   help="Steps per env per PPO rollout")
    p.add_argument("--exploiter-batch-size", type=int,   default=64,
                   help="PPO mini-batch size (must divide n_envs * n_steps)")
    p.add_argument("--exploiter-n-epochs",   type=int,   default=10,
                   help="Gradient epochs per PPO rollout")
    p.add_argument("--exploiter-lr",         type=float, default=3e-4,
                   help="PPO Adam learning rate")
    p.add_argument("--exploiter-clip-range", type=float, default=0.2,
                   help="PPO clip epsilon")
    p.add_argument("--exploiter-ent-coef",   type=float, default=0.01,
                   help="PPO entropy bonus coefficient")

    # ── Minimax reward ────────────────────────────────────────────────────────
    p.add_argument("--alpha",   type=float, default=0.05)
    p.add_argument("--gamma",   type=float, default=0.995)
    p.add_argument("--v-shift", type=float, default=25.0)

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--no-exploiter-feedback", action="store_true",
                   help="Train exploiter for evaluation only — do not inject it into the population")
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
        exploiter_n_envs        = args.exploiter_n_envs,
        exploiter_n_steps       = args.exploiter_n_steps,
        exploiter_batch_size    = args.exploiter_batch_size,
        exploiter_n_epochs      = args.exploiter_n_epochs,
        exploiter_lr            = args.exploiter_lr,
        exploiter_clip_range    = args.exploiter_clip_range,
        exploiter_ent_coef      = args.exploiter_ent_coef,
        alpha                   = args.alpha,
        gamma                   = args.gamma,
        v_shift                 = args.v_shift,
        exploiter_feedback      = not args.no_exploiter_feedback,
        device                  = args.device,
        seed                    = args.seed,
        run_dir                 = args.run_dir,
    )

    train_league(cfg)
