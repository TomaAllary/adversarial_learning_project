# Adversarial Learning — Tactical Shooter

A research project exploring adversarial multi-agent reinforcement learning in a custom 2D tactical shooter environment. Two algorithms are implemented and compared: **PPO** (Proximal Policy Optimization via [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)) for single-team training against a fixed opponent, and **R-NaD** (Regularized Nash Dynamics) for self-play convergence to a Nash equilibrium.

---

## Table of Contents

- [Environment](#environment)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training](#training)
- [Visualisation](#visualisation)

---

## Environment

### ShooterEnvironment (`pettingzoo_env/shooter_env.py`)

A symmetric N-vs-N tactical shooter built on [PettingZoo](https://pettingzoo.farama.org/) `ParallelEnv`. Two teams (Red and Blue) spawn on opposite corners of an 8×8 grid and fight until one team is eliminated or the step limit is reached.

**Default configuration:** 1 agent per team (1v1).

#### Actions — `Discrete(7)`

| ID | Action               |
| -- | -------------------- |
| 0  | Stay                 |
| 1  | Move North           |
| 2  | Move South           |
| 3  | Move West            |
| 4  | Move East            |
| 5  | Rotate left (−45°) |
| 6  | Rotate right (+45°) |

Moving into a wall cell is silently blocked (the action is consumed but position is unchanged).

#### Vision & Shooting

Each agent has a **90° vision cone** (±45° from heading) with a **4-cell range**. Walls block both vision and line-of-sight. When an enemy is inside the cone and has clear line-of-sight, the agent shoots with **100% hit probability** each step.

#### HP & Elimination

Each agent starts with **5 HP**. When HP reaches 0 the agent is eliminated. The episode ends when all agents on one side are dead or the step limit (200) is reached.

#### Rewards

| Event                       | Reward |
| --------------------------- | ------ |
| Per step                    | −0.05 |
| Hit on enemy                | +2.0   |
| Enemy team eliminated (win) | +20.0  |
| Own team eliminated (loss)  | −20.0 |

#### Observation — `Box(0, 1, shape=(11,))`

A flat float32 vector from the perspective of the observing agent (for the default 1v1 configuration):

| Features               | Dim          | Description                                             |
| ---------------------- | ------------ | ------------------------------------------------------- |
| `norm_x`, `norm_y` | 2×2 = 4     | Normalised grid positions (0–1) for self and enemy     |
| `hp_ratio`           | 2            | HP / 5 for self and enemy                               |
| `in_my_cone`         | 1            | 1 if enemy is currently inside this agent's vision cone |
| `heading (sin, cos)` | 2×2 = 4     | Unit-vector heading for self and enemy                  |
| **Total**        | **11** |                                                         |

#### Rendering

Pass `render_mode="human"` to open a Pygame window. The renderer shows agent positions, HP bars, vision cones, and a step/HP HUD.

---

### ShooterGymEnv (`pettingzoo_env/shooter_gym_env.py`)

A `gymnasium.Env` wrapper around `ShooterEnvironment` that makes it compatible with both **PPO** (single-agent) and **R-NaD** (self-play). Controlled via the `self_play` constructor argument.

#### Self-play mode (`self_play=True`) — for R-NaD

Converts the simultaneous PettingZoo game into a sequential two-player interface. Turns alternate: player 0 (Red) → player 1 (Blue) → …

- **Player 0's turn:** Red's action is buffered; the underlying env does not step yet. Returns Blue's current observation and `reward = [0, 0]`.
- **Player 1's turn:** Both buffered actions are applied; the env steps. Returns Red's new observation and `reward = [r_red, r_blue]`.

V-trace in R-NaD handles the one-turn delay by accumulating discounted returns between a player's own turns.

Extra methods for R-NaD:

- `current_player() → int` — 0 (Red) or 1 (Blue)
- `legal_actions_mask() → np.ndarray` — all-ones (all actions are always legal)

#### Single-agent mode (`self_play=False`) — for PPO / SB3

The agent controls Red; Blue is driven by an `opponent` policy. `step()` returns a scalar float reward for Red.

| `opponent`           | Behaviour                                  |
| ---------------------- | ------------------------------------------ |
| `"random"` (default) | Blue samples uniformly                     |
| `"scripted"`         | Blue uses the BFS `ScriptedShooterAgent` |
| `callable`           | Any `fn(obs: np.ndarray) -> int`         |

#### Constructor arguments

| Argument        | Type               | Default      | Description                            |
| --------------- | ------------------ | ------------ | -------------------------------------- |
| `self_play`   | `bool`           | `True`     | Mode selector                          |
| `opponent`    | `str \| callable` | `"random"` | Blue policy (single-agent mode only)   |
| `render_mode` | `str \| None`     | `None`     | `"human"` to enable Pygame rendering |
| `fps`         | `int`            | `10`       | Frames per second when rendering       |

---

## Algorithms

### PPO (`train.py ppo`)

**Proximal Policy Optimization** via [Stable Baselines 3](https://stable-baselines3.readthedocs.io/). Red is trained as a single agent against a fixed opponent (scripted BFS or random) using `ShooterGymEnv` with `self_play=False`.

| Hyperparameter             | Value         |
| -------------------------- | ------------- |
| Learning rate              | 3e-4          |
| Discount γ                | 0.99          |
| GAE λ                     | 0.95          |
| PPO clip ε                | 0.2           |
| Entropy coefficient        | 0.01          |
| Value function coefficient | 0.5           |
| Batch size                 | 64            |
| Update epochs              | 10            |
| Rollout steps per env      | 2048          |
| Network                    | MLP [256, 256] |

SB3 handles rollout collection, GAE computation, mini-batch updates, and TensorBoard logging automatically. An `EvalCallback` saves the best checkpoint and a `CheckpointCallback` saves periodic snapshots.

---

### R-NaD (`new_rnad.py`)

**Regularized Nash Dynamics** ([Perolat et al., 2022](https://arxiv.org/pdf/2206.15378.pdf)) — an algorithm that provably converges to a Nash equilibrium in two-player zero-sum games via self-play.

Key ideas:

- A **single shared policy network** is used for both players. The observation is always from the current player's perspective, so the same weights implement a strategy that is optimal regardless of which side you play.
- **V-trace** importance-weighted returns handle the off-policy correction introduced by the sequential wrapper.
- **NeuRD (NERD)** policy updates use a clipped advantage signal with a threshold to avoid large policy deviations.
- An **entropy schedule** (α) interpolates between two regularisation reference policies (`params_prev` and `params_prev_`), gradually tightening the constraint as training progresses.
- A slowly-updated **target network** (EMA, τ = 0.001) provides stable value targets.

#### Network architecture

```
obs (11) → Linear(256) → ReLU → Linear(256) → ReLU
                                               ├─ policy_head → Linear(7)  → masked softmax → π
                                               └─ value_head  → Linear(1)  → V
```

#### Key `RNaDConfig` parameters

| Parameter                    | Default        | Description                                |
| ---------------------------- | -------------- | ------------------------------------------ |
| `policy_network_layers`    | `(256, 256)` | Hidden layer widths                        |
| `batch_size`               | 256            | Parallel environment instances             |
| `trajectory_max`           | 200            | Steps per trajectory (match `MAX_STEPS`) |
| `learning_rate`            | 5e-5           | Adam learning rate                         |
| `target_network_avg`       | 0.001          | EMA rate τ for target network             |
| `entropy_schedule_size`    | `(20000,)`   | Steps per entropy phase                    |
| `entropy_schedule_repeats` | `(1,)`       | Repetitions of each phase                  |
| `eta_reward_transform`     | 0.2            | Entropy regularisation strength            |
| `nerd.beta`                | 2.0            | NeuRD gradient clipping threshold          |
| `c_vtrace`                 | 1.0            | V-trace importance weight clipping         |
| `num_players`              | 2              | Players (set to 1 for single-agent)        |
| `seed`                     | 42             | RNG seed                                   |

#### SB3-style API

```python
model = RNaD(env_fn=lambda: ShooterGymEnv(self_play=True), config=cfg)
model.learn(total_timesteps=500_000)
action, _ = model.predict(obs, legal_actions=mask)
model.save("model.pt")
model = RNaD.load("model.pt", env_fn=env_fn)
```

---

### Scripted Agent (`pettingzoo_env/scripted_shooter_agent.py`)

A deterministic rule-based opponent used as a training baseline for PPO. It uses **BFS pathfinding** to navigate toward the nearest living enemy and rotates to face the target when within 3 cells.

---

## Project Structure

```
adversarial_learning_project/
│
├── new_rnad.py                      # R-NaD algorithm (SB3-compatible)
├── train.py                         # Unified training script (PPO + R-NaD)
├── animate_rnad.py                  # Visualise a trained R-NaD model
│
├── pettingzoo_env/
│   ├── shooter_env.py               # Core PettingZoo environment
│   ├── shooter_gym_env.py           # Gymnasium wrapper (SB3 PPO + R-NaD)
│   ├── scripted_shooter_agent.py    # Rule-based BFS opponent
│   ├── utils.py                     # BFS pathfinder, map generator, helpers
│   └── prisoner_env.py              # Separate pursuit-evasion environment
│
├── legacy_rnad/                     # Archived JAX-based R-NaD (unmaintained)
├── test/                            # PettingZoo API validation tests
├── plotting/                        # Training metrics visualisation
├── assets/                          # Pygame sprites
│
├── runs/                            # R-NaD training outputs (gitignored)
├── requirement.txt
└── .gitignore
```

---

## Setup

### 1. Create and activate a Conda environment

```bash
conda create -n adversarial python=3.11
conda activate adversarial
```

### 2. Install PyTorch with GPU support

```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 3. Install remaining dependencies

```bash
pip install -r requirement.txt
```

`requirement.txt` includes: `pygame`, `numpy`, `pettingzoo`, `pymunk`, `SuperSuit`, `tensorboard`, `tqdm`, `matplotlib`, `gymnasium`, `stable-baselines3`.

---

## Training

Both algorithms are trained through a single unified script — `train.py` — which shares the evaluation loop, TensorBoard logging, checkpointing, and run-directory layout. Choose the algorithm with the first positional argument.

```bash
tensorboard --logdir runs/   # monitor any run
```

### R-NaD self-play — `python train.py rnad`

Trains a single shared policy to Nash equilibrium via self-play. Both Red and Blue are driven by the same network.

```bash
# Minimal — 500k actor steps
python train.py rnad

# Named run on GPU
python train.py rnad \
    --run-name experiment_1 \
    --total-steps 2_000_000 \
    --batch-size 256 \
    --learning-rate 5e-5 \
    --device cuda
```

#### All arguments — `rnad`

**Common**

| Argument           | Default   | Description                              |
| ------------------ | --------- | ---------------------------------------- |
| `--run-name`     | auto      | Base name; `rnad_` + timestamp appended  |
| `--runs-dir`     | `runs`  | Root directory for outputs               |
| `--total-steps`  | 500 000   | Stop after this many actor steps         |
| `--eval-episodes`| 20        | Episodes per evaluation round            |
| `--hidden-layers`| `256 256` | MLP hidden layer widths                 |
| `--learning-rate`| 5e-5      | Adam learning rate                       |
| `--seed`         | 42        | RNG seed                                 |
| `--device`       | `cpu`   | Torch device (`cpu`, `cuda`, `cuda:0`, …)|

**R-NaD specific**

| Argument                       | Default   | Description                                       |
| ------------------------------ | --------- | ------------------------------------------------- |
| `--log-interval`             | 50        | Log training scalars every N learner steps        |
| `--eval-interval`            | 500       | Evaluate vs random every N learner steps          |
| `--checkpoint-interval`      | 2 000     | Save checkpoint every N learner steps             |
| `--batch-size`               | 256       | Parallel environment instances                    |
| `--trajectory-max`           | 200       | Steps per trajectory                              |
| `--clip-gradient`            | 10 000    | Global gradient norm clip                         |
| `--adam-b1`                  | 0.0       | Adam β₁                                          |
| `--adam-b2`                  | 0.999     | Adam β₂                                          |
| `--adam-eps`                 | 1e-7      | Adam ε                                            |
| `--target-network-avg`       | 0.001     | EMA rate τ for target network                    |
| `--eta-reward-transform`     | 0.2       | Entropy regularisation strength η                |
| `--c-vtrace`                 | 1.0       | V-trace importance weight clip                    |
| `--nerd-beta`                | 2.0       | NeuRD gradient clip threshold                     |
| `--nerd-clip`                | 10 000    | NeuRD logit clip                                  |
| `--entropy-schedule-size`    | `20000`   | Steps per entropy phase                           |
| `--entropy-schedule-repeats` | `1`       | Repetitions of each phase                         |

---

### PPO vs scripted opponent — `python train.py ppo`

Trains Red with SB3 PPO while Blue uses the scripted BFS agent (default) or a random policy.

```bash
# Minimal — 500k steps, 8 envs, scripted opponent
python train.py ppo

# Longer run on GPU with a random opponent
python train.py ppo \
    --total-steps 5_000_000 \
    --n-envs 16 \
    --opponent random \
    --device cuda

# Resume from a saved checkpoint
python train.py ppo --load runs/ppo_experiment_1_20260101_120000/best_model.zip
```

#### All arguments — `ppo`

**Common** (same as R-NaD table above, with different defaults)

| Argument           | Default   | Description                              |
| ------------------ | --------- | ---------------------------------------- |
| `--learning-rate`| 3e-4      | PPO learning rate                        |
| `--device`       | `auto`  | Torch device                             |

**PPO specific**

| Argument                  | Default      | Description                                   |
| ------------------------- | ------------ | --------------------------------------------- |
| `--n-envs`              | 8            | Parallel training environments                |
| `--opponent`            | `scripted` | Blue policy: `scripted` or `random`         |
| `--load`                | —            | Path to a `.zip` checkpoint to resume from   |
| `--eval-interval`       | 10 000       | Evaluate vs random every N actor steps        |
| `--checkpoint-interval` | 50 000       | Save checkpoint every N actor steps           |

---

### Run directory layout (both algos)

```
runs/rnad_experiment_1_20260101_120000/
  config.json            # full reproducible config
  best_model.pt/.zip     # checkpoint with highest eval reward
  final_model.pt/.zip    # end-of-training snapshot
  checkpoints/
    model_step_0002000.pt   # R-NaD
    ppo_step_00050000.zip   # PPO
    ...
  events.out.tfevents.*  # TensorBoard logs
```

### TensorBoard scalars (both algos)

| Tag                   | Description                                |
| --------------------- | ------------------------------------------ |
| `eval/mean_reward`  | Mean episode reward (Red vs random Blue)   |
| `eval/std_reward`   | Std of episode rewards                     |
| `eval/win_rate`     | Fraction of evaluation episodes won by Red |
| `train/fps`         | Actor steps per second                     |

**R-NaD only**

| Tag                   | Description                                |
| --------------------- | ------------------------------------------ |
| `train/loss`        | Combined V + NeuRD loss                    |
| `train/mean_reward` | Mean batch trajectory reward               |
| `train/alpha`       | Current entropy schedule α                |
| `train/actor_steps` | Cumulative environment steps               |

**PPO only** (via SB3 internal logger)

`train/policy_gradient_loss`, `train/value_loss`, `train/entropy_loss`, `train/approx_kl`, `train/clip_fraction`, …

To load a trained model for inference:

```python
from stable_baselines3 import PPO
from pettingzoo_env.shooter_gym_env import ShooterGymEnv

env   = ShooterGymEnv(self_play=False, opponent="scripted")
model = PPO.load("runs/ppo_20260101_120000/best_model.zip")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
```

---

## Visualisation

### Animate R-NaD — `animate_rnad.py`

Opens a Pygame window and loops through shooter games using a trained model.

```bash
# Load best model from a run directory
python animate_rnad.py --run runs/experiment_1_20260101_120000

# Point to a specific checkpoint
python animate_rnad.py --model runs/experiment_1_20260101_120000/checkpoints/model_step_0010000.pt

# Red=trained vs Blue=random (shows learned advantage)
python animate_rnad.py --run runs/... --mode vs_random

# Slow down and use deterministic policy
python animate_rnad.py --run runs/... --fps 3 --deterministic

# Run exactly 10 episodes then print a summary
python animate_rnad.py --run runs/... --episodes 10
```

#### All arguments

| Argument            | Default       | Description                                                                           |
| ------------------- | ------------- | ------------------------------------------------------------------------------------- |
| `--run`           | —            | Path to run directory; loads `best_model.pt`, falls back to `final_model.pt`      |
| `--model`         | —            | Direct path to any `.pt` file                                                       |
| `--mode`          | `self_play` | `self_play`: both sides use trained policy. `vs_random`: Red=trained, Blue=random |
| `--fps`           | 5             | Pygame frames per second                                                              |
| `--episodes`      | 0 (∞)        | Episodes to play before exiting                                                       |
| `--deterministic` | off           | Use argmax policy instead of sampling                                                 |
| `--device`        | `cpu`       | Torch device                                                                          |

Close the window or press `Ctrl+C` to stop. A summary (mean reward, win rate) is printed at the end.
