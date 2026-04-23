"""
R-NaD (Regularized Nash Dynamics) with a Stable Baselines3-compatible interface.

Paper: https://arxiv.org/pdf/2206.15378.pdf

Differences from the OpenSpiel implementation:
- PyTorch instead of JAX/Haiku
- Accepts gym.Env instead of pyspiel games
- API follows SB3 conventions: learn(), predict(), save(), load()

Environment requirements:
- Standard gym.Env interface (reset, step, observation_space, action_space)
- Optional: current_player() -> int   (which player acts; defaults to 0)
- Optional: legal_actions_mask() -> np.ndarray  (valid actions; defaults to all-ones)
- For multi-player: step() should return per-player rewards as an array,
  or a scalar reward (zero-sum assumed for 2-player games).
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdamConfig:
    b1: float = 0.0
    b2: float = 0.999
    eps: float = 10e-8


@dataclass
class NerdConfig:
    beta: float = 2.0
    clip: float = 10_000


@dataclass
class FineTuningConfig:
    from_learner_steps: int = -1
    policy_threshold: float = 0.03
    policy_discretization: int = 32


@dataclass
class RNaDConfig:
    # Network architecture
    policy_network_layers: Sequence[int] = (256, 256)

    # Trajectory collection
    batch_size: int = 256        # number of parallel environment instances
    trajectory_max: int = 200    # steps per trajectory

    # Optimizer
    learning_rate: float = 0.00005
    adam: AdamConfig = field(default_factory=AdamConfig)
    clip_gradient: float = 10_000
    target_network_avg: float = 0.001  # EMA rate for target net

    # R-NaD algorithm
    entropy_schedule_repeats: Sequence[int] = (1,)
    entropy_schedule_size: Sequence[int] = (20_000,)
    eta_reward_transform: float = 0.2
    nerd: NerdConfig = field(default_factory=NerdConfig)
    c_vtrace: float = 1.0

    # Game structure
    num_players: int = 2  # set to 1 for single-agent environments

    # Fine-tuning
    finetune: FineTuningConfig = field(default_factory=FineTuningConfig)

    seed: int = 42


# ---------------------------------------------------------------------------
# Entropy schedule
# ---------------------------------------------------------------------------

class EntropySchedule:
    """Mirrors the original EntropySchedule exactly."""

    def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
        if len(repeats) != len(sizes):
            raise ValueError("`repeats` must be parallel to `sizes`.")
        if not sizes:
            raise ValueError("`sizes` and `repeats` must not be empty.")
        if any(r <= 0 for r in repeats):
            raise ValueError("All repeat values must be strictly positive.")
        if repeats[-1] != 1:
            raise ValueError("The last value in `repeats` must be 1.")

        schedule = [0]
        for size, repeat in zip(sizes, repeats):
            schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])
        self.schedule = np.array(schedule, dtype=np.int32)

    def __call__(self, learner_step: int) -> Tuple[float, bool]:
        last_size = int(self.schedule[-1] - self.schedule[-2])
        last_start = int(self.schedule[-1]) + (
            learner_step - int(self.schedule[-1])) // last_size * last_size

        within = self.schedule[self.schedule <= learner_step]
        start = int(within[-1]) if len(within) > 0 else int(self.schedule[0])

        future = self.schedule[self.schedule > learner_step]
        finish = int(future[0]) if len(future) > 0 else int(self.schedule[-1])
        size = finish - start

        beyond = int(self.schedule[-1]) <= learner_step
        iteration_start = last_start if beyond else start
        iteration_size = last_size if beyond else size

        update_target_net = (
            learner_step > 0
            and iteration_size > 0
            and learner_step == iteration_start + iteration_size - 1
        )
        alpha = (
            min(2.0 * (learner_step - iteration_start) / iteration_size, 1.0)
            if iteration_size > 0 else 1.0
        )
        return alpha, update_target_net


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden_layers: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        in_size = obs_size
        for h in hidden_layers:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        self.torso = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_size, num_actions)
        self.value_head = nn.Linear(in_size, 1)

    def forward(
        self, obs: torch.Tensor, legal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torso = self.torso(obs)
        logits = self.policy_head(torso)
        v = self.value_head(torso)
        pi = _legal_softmax(logits, legal)
        log_pi = _legal_log_softmax(logits, legal)
        return pi, v, log_pi, logits


def _legal_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    l_min = logits.min(dim=-1, keepdim=True).values
    logits = torch.where(legal.bool(), logits, l_min)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits = logits * legal
    exp_l = torch.where(legal.bool(), torch.exp(logits), torch.zeros_like(logits))
    return exp_l / exp_l.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _legal_log_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.full_like(logits, float("-inf"))
    logits_masked = torch.where(legal.bool(), logits, neg_inf)
    max_l = logits_masked.max(dim=-1, keepdim=True).values
    logits_masked = logits_masked - max_l
    exp_l = torch.exp(logits_masked)
    baseline = torch.log(exp_l.sum(dim=-1, keepdim=True).clamp(min=1e-8))
    # Avoid 0 * -inf = nan: only write log_policy where legal
    log_pi = legal * (logits - max_l - baseline)
    return log_pi


# ---------------------------------------------------------------------------
# V-Trace helpers
# ---------------------------------------------------------------------------

def _has_played(
    valid: torch.Tensor, player_id: torch.Tensor, player: int
) -> torch.Tensor:
    """Returns 1 at timestep t if `player` acts at t or any later valid step."""
    T, B = valid.shape
    carry = torch.zeros(B, device=valid.device)
    result = torch.zeros(T, B, device=valid.device)

    for t in reversed(range(T)):
        v = valid[t].bool()           # [B]
        is_ours = (player_id[t] == player)  # [B]

        res_t = torch.where(v, torch.where(is_ours, torch.ones(B, device=valid.device), carry), torch.zeros(B, device=valid.device))
        new_carry = torch.where(v, carry, torch.zeros(B, device=valid.device))

        result[t] = res_t
        carry = new_carry

    return result


def _policy_ratio(
    pi: torch.Tensor, mu: torch.Tensor,
    actions_oh: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    """pi/mu for the selected action; 1 on invalid steps."""
    def _sel(p):
        return (actions_oh * p).sum(dim=-1) * valid + (1.0 - valid)

    return _sel(pi) / _sel(mu).clamp(min=1e-8)


# ---------------------------------------------------------------------------
# V-Trace (backward scan translated to Python loop)
# ---------------------------------------------------------------------------

def v_trace(
    v: torch.Tensor,               # [T, B, 1]
    valid: torch.Tensor,           # [T, B]
    player_id: torch.Tensor,       # [T, B]
    acting_policy: torch.Tensor,   # [T, B, A]
    merged_policy: torch.Tensor,   # [T, B, A]
    merged_log_policy: torch.Tensor,  # [T, B, A]
    player_others: torch.Tensor,   # [T, B, 1]  (+1 ours, -1 theirs) * valid
    actions_oh: torch.Tensor,      # [T, B, A]
    reward: torch.Tensor,          # [T, B]
    player: int,
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T, B = valid.shape
    gamma = 1.0
    dev = valid.device

    has_played = _has_played(valid, player_id, player)

    policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid)   # [T, B]
    inv_mu = _policy_ratio(torch.ones_like(merged_policy), acting_policy, actions_oh, valid)  # [T, B]

    # Entropy regularisation terms
    eta_reg_entropy = (
        -eta * (merged_policy * merged_log_policy).sum(-1)
        * player_others.squeeze(-1)
    )  # [T, B]
    eta_log_policy = -eta * merged_log_policy * player_others  # [T, B, A]

    # Carry state initialised to zero / ones
    c_reward = torch.zeros(B, device=dev)
    c_reward_unc = torch.zeros(B, device=dev)
    c_next_v = torch.zeros(B, 1, device=dev)
    c_next_vt = torch.zeros(B, 1, device=dev)
    c_is = torch.ones(B, device=dev)

    v_targets_list: List[torch.Tensor] = []
    lo_list: List[torch.Tensor] = []

    for t in reversed(range(T)):
        cs = policy_ratio[t]       # [B]
        pid = player_id[t]         # [B]
        v_t = v[t]                 # [B, 1]
        r_t = reward[t]            # [B]
        ere_t = eta_reg_entropy[t] # [B]
        val_t = valid[t]           # [B]
        inv_mu_t = inv_mu[t]       # [B]
        aoh_t = actions_oh[t]      # [B, A]
        elp_t = eta_log_policy[t]  # [B, A]

        reward_unc = r_t + gamma * c_reward_unc + ere_t
        disc_reward = r_t + gamma * c_reward

        # Clipped importance weights
        is_prod = cs * c_is  # [B]
        if np.isinf(rho):
            rho_clipped = is_prod
        else:
            rho_clipped = torch.minimum(torch.full_like(is_prod, rho), is_prod)
        c_clipped = torch.minimum(torch.full_like(is_prod, c), is_prod)

        our_v_target = (
            v_t
            + rho_clipped.unsqueeze(-1) * (reward_unc.unsqueeze(-1) + gamma * c_next_v - v_t)
            + lambda_ * c_clipped.unsqueeze(-1) * gamma * (c_next_vt - c_next_v)
        )  # [B, 1]

        our_lo = (
            v_t
            + elp_t
            + aoh_t * inv_mu_t.unsqueeze(-1) * (
                disc_reward.unsqueeze(-1)
                + gamma * c_is.unsqueeze(-1) * c_next_vt
                - v_t
            )
        )  # [B, A]

        is_valid = val_t.bool()     # [B]
        is_ours = (pid == player).bool()  # [B]

        zeros_v = torch.zeros_like(our_v_target)
        zeros_lo = torch.zeros_like(our_lo)

        def sel2(iv, io, a, b, reset):
            return torch.where(iv, torch.where(io, a, b), reset)

        v_target_t = sel2(is_valid.unsqueeze(-1), is_ours.unsqueeze(-1), our_v_target, zeros_v, zeros_v)
        lo_t = sel2(is_valid.unsqueeze(-1), is_ours.unsqueeze(-1), our_lo, zeros_lo, zeros_lo)

        v_targets_list.insert(0, v_target_t)
        lo_list.insert(0, lo_t)

        # --- Update carry ---
        def sel1(iv, io, a, b, reset):
            return torch.where(iv, torch.where(io, a, b), reset)

        zeros_b = torch.zeros(B, device=dev)
        ones_b = torch.ones(B, device=dev)
        zeros_b1 = torch.zeros(B, 1, device=dev)

        new_c_reward = sel1(is_valid, is_ours, zeros_b, ere_t + cs * disc_reward, zeros_b)
        new_c_reward_unc = sel1(is_valid, is_ours, zeros_b, reward_unc, zeros_b)
        new_c_next_v = sel2(is_valid.unsqueeze(-1), is_ours.unsqueeze(-1), v_t, gamma * c_next_v, zeros_b1)
        new_c_next_vt = sel2(is_valid.unsqueeze(-1), is_ours.unsqueeze(-1), our_v_target, gamma * c_next_vt, zeros_b1)
        new_c_is = sel1(is_valid, is_ours, ones_b, cs * c_is, ones_b)

        c_reward = new_c_reward
        c_reward_unc = new_c_reward_unc
        c_next_v = new_c_next_v
        c_next_vt = new_c_next_vt
        c_is = new_c_is

    v_targets = torch.stack(v_targets_list, dim=0)  # [T, B, 1]
    learning_outputs = torch.stack(lo_list, dim=0)  # [T, B, A]
    return v_targets, has_played, learning_outputs


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def get_loss_v(
    v_list: List[torch.Tensor],
    v_target_list: List[torch.Tensor],
    mask_list: List[torch.Tensor],
) -> torch.Tensor:
    loss = torch.tensor(0.0)
    for v_n, v_target, mask in zip(v_list, v_target_list, mask_list):
        diff = (v_n - v_target.detach()) ** 2        # [T, B, 1]
        loss_v = mask.unsqueeze(-1) * diff
        norm = mask.sum()
        loss = loss + loss_v.sum() / (norm + (norm == 0).float())
    return loss


def _apply_force_with_threshold(
    decision_outputs: torch.Tensor,
    force: torch.Tensor,
    threshold: float,
    threshold_center: torch.Tensor,
) -> torch.Tensor:
    can_decrease = (decision_outputs - threshold_center) > -threshold
    can_increase = (decision_outputs - threshold_center) < threshold
    force_neg = torch.minimum(force, torch.zeros_like(force))
    force_pos = torch.maximum(force, torch.zeros_like(force))
    clipped = can_decrease * force_neg + can_increase * force_pos
    return decision_outputs * clipped.detach()


def _renormalize(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    total = (loss * mask).sum()
    norm = mask.sum()
    return total / (norm + (norm == 0).float())


def get_loss_nerd(
    logit_list: List[torch.Tensor],
    policy_list: List[torch.Tensor],
    q_vr_list: List[torch.Tensor],
    valid: torch.Tensor,
    player_ids: torch.Tensor,
    legal_actions: torch.Tensor,
    importance_sampling_correction: List[torch.Tensor],
    clip: float = 100,
    threshold: float = 2,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=valid.device)
    num_valid = legal_actions.sum(dim=-1, keepdim=True).clamp(min=1)

    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
        zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)
    ):
        adv_pi = q_vr - (pi * q_vr).sum(dim=-1, keepdim=True)
        adv_pi = is_c * adv_pi
        adv_pi = torch.clamp(adv_pi, -clip, clip).detach()

        mean_logit = (logit_pi * legal_actions).sum(dim=-1, keepdim=True) / num_valid
        logits = logit_pi - mean_logit

        threshold_center = torch.zeros_like(logits)
        nerd_loss = (
            legal_actions
            * _apply_force_with_threshold(logits, adv_pi, threshold, threshold_center)
        ).sum(dim=-1)  # [T, B]

        mask = valid * (player_ids == k).float()
        loss = loss - _renormalize(nerd_loss, mask)

    return loss


# ---------------------------------------------------------------------------
# Policy post-processing (fine-tuning)
# ---------------------------------------------------------------------------

def _post_process_policy(
    policy: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if threshold <= 0:
        return policy
    over = (policy >= threshold).float()
    all_below = (policy.max(dim=-1, keepdim=True).values < threshold).float()
    mask = mask * (over + all_below)
    denom = (mask * policy).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return mask * policy / denom


# ---------------------------------------------------------------------------
# Gym helpers
# ---------------------------------------------------------------------------

def _gym_reset(env: gym.Env) -> np.ndarray:
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _gym_step(env: gym.Env, action) -> Tuple[np.ndarray, Any, bool, Dict]:
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    obs, reward, done, info = result
    return obs, reward, done, info


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class RNaD:
    """
    R-NaD solver with a Stable Baselines3-compatible interface.

    Usage (2-player adversarial game):
        model = RNaD(env_fn=lambda: MyGameEnv(), config=RNaDConfig(num_players=2))
        model.learn(total_timesteps=1_000_000)
        action, _ = model.predict(obs)

    Usage (single-agent gym):
        model = RNaD(env_fn=lambda: gym.make("CartPole-v1"),
                     config=RNaDConfig(num_players=1, trajectory_max=500))
        model.learn(total_timesteps=500_000)
    """

    def __init__(
        self,
        env_fn: Union[Callable[[], gym.Env], gym.Env],
        config: Optional[RNaDConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or RNaDConfig()
        self.device = torch.device(device)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Build a throwaway env to read spaces
        if callable(env_fn) and not isinstance(env_fn, gym.Env):
            self._env_fn = env_fn
            _probe = env_fn()
        else:
            _env = env_fn
            self._env_fn = lambda: copy.deepcopy(_env)
            _probe = _env

        obs_space = _probe.observation_space
        self.obs_size: int = (
            int(np.prod(obs_space.shape))
            if hasattr(obs_space, "shape")
            else int(obs_space.n)
        )
        self.num_actions: int = int(_probe.action_space.n)

        self.learner_steps = 0
        self.actor_steps = 0
        self.num_timesteps = 0

        # Create batch_size env instances
        self._envs: List[gym.Env] = [
            self._env_fn() for _ in range(self.config.batch_size)
        ]

        self._setup_networks()
        self._entropy_schedule = EntropySchedule(
            sizes=self.config.entropy_schedule_size,
            repeats=self.config.entropy_schedule_repeats,
        )

    # ------------------------------------------------------------------
    # Network setup
    # ------------------------------------------------------------------

    def _setup_networks(self):
        def _make() -> PolicyNetwork:
            return PolicyNetwork(
                self.obs_size, self.num_actions, self.config.policy_network_layers
            ).to(self.device)

        self.params = _make()
        self.params_target = _make()
        self.params_prev = _make()
        self.params_prev_ = _make()

        for net in (self.params_target, self.params_prev, self.params_prev_):
            net.load_state_dict(self.params.state_dict())

        self.optimizer = optim.Adam(
            self.params.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
            eps=self.config.adam.eps,
        )

    # ------------------------------------------------------------------
    # Public SB3-like API
    # ------------------------------------------------------------------

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 100,
        verbose: int = 1,
    ) -> "RNaD":
        while self.num_timesteps < total_timesteps:
            prev_actor = self.actor_steps
            logs = self.step()
            self.num_timesteps += self.actor_steps - prev_actor

            if verbose > 0 and self.learner_steps % log_interval == 0:
                print(
                    f"learner_step={self.learner_steps}  "
                    f"loss={logs['loss']:.4f}  "
                    f"actor_steps={self.actor_steps}"
                )
        return self

    def predict(
        self,
        observation: np.ndarray,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        legal_actions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, None]:
        """SB3-compatible predict. Returns (actions, states)."""
        single = observation.ndim == 1
        if single:
            observation = observation[np.newaxis]

        B = len(observation)
        if legal_actions is None:
            legal_actions = np.ones((B, self.num_actions), dtype=np.float32)
        elif legal_actions.ndim == 1:
            legal_actions = legal_actions[np.newaxis]

        obs_t = torch.tensor(observation, dtype=torch.float32, device=self.device)
        leg_t = torch.tensor(legal_actions, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pi, _, _, _ = self.params_target(obs_t, leg_t)
            pi_np = pi.cpu().numpy().astype(np.float64)

        if deterministic:
            actions = pi_np.argmax(axis=-1)
        else:
            pi_np /= pi_np.sum(axis=-1, keepdims=True)
            actions = np.array([
                np.random.choice(self.num_actions, p=pi_np[i]) for i in range(B)
            ])

        return (actions[0] if single else actions), None

    def save(self, path: str):
        torch.save({
            "config": self.config,
            "obs_size": self.obs_size,
            "num_actions": self.num_actions,
            "params": self.params.state_dict(),
            "params_target": self.params_target.state_dict(),
            "params_prev": self.params_prev.state_dict(),
            "params_prev_": self.params_prev_.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learner_steps": self.learner_steps,
            "actor_steps": self.actor_steps,
            "num_timesteps": self.num_timesteps,
        }, path)

    @classmethod
    def load(
        cls,
        path: str,
        env_fn: Union[Callable[[], gym.Env], gym.Env],
        device: str = "cpu",
    ) -> "RNaD":
        data = torch.load(path, map_location=device, weights_only=False)
        model = cls(env_fn, config=data["config"], device=device)
        model.params.load_state_dict(data["params"])
        model.params_target.load_state_dict(data["params_target"])
        model.params_prev.load_state_dict(data["params_prev"])
        model.params_prev_.load_state_dict(data["params_prev_"])
        model.optimizer.load_state_dict(data["optimizer"])
        model.learner_steps = data["learner_steps"]
        model.actor_steps = data["actor_steps"]
        model.num_timesteps = data["num_timesteps"]
        return model

    # ------------------------------------------------------------------
    # Core algorithm step
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        timestep = self.collect_batch_trajectory()
        alpha, update_target_net = self._entropy_schedule(self.learner_steps)
        loss_val = self._update_parameters(timestep, alpha, update_target_net)
        self.learner_steps += 1
        return {"loss": loss_val, "actor_steps": self.actor_steps, "learner_steps": self.learner_steps}

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------

    def _update_parameters(
        self, timestep: Dict[str, np.ndarray], alpha: float, update_target_net: bool
    ) -> float:
        def _t(key, dtype=torch.float32):
            return torch.tensor(timestep[key], dtype=dtype, device=self.device)

        obs = _t("obs")            # [T, B, obs_size]
        legal = _t("legal")        # [T, B, A]
        valid = _t("valid")        # [T, B]
        player_id = _t("player_id")  # [T, B]
        action_oh = _t("action_oh")  # [T, B, A]
        actor_policy = _t("policy")  # [T, B, A]
        rewards = _t("rewards")    # [T, B, P]

        T, B = valid.shape

        # Flatten time+batch for a single network forward pass
        obs_flat = obs.view(T * B, -1)
        leg_flat = legal.view(T * B, -1)

        self.params.train()
        pi_flat, v_flat, log_pi_flat, logit_flat = self.params(obs_flat, leg_flat)

        with torch.no_grad():
            _, v_target_flat, _, _ = self.params_target(obs_flat, leg_flat)
            _, _, log_pi_prev_flat, _ = self.params_prev(obs_flat, leg_flat)
            _, _, log_pi_prev__flat, _ = self.params_prev_(obs_flat, leg_flat)

        def _unflat(x, last_dim):
            return x.view(T, B, last_dim)

        A = self.num_actions
        pi = _unflat(pi_flat, A)
        v = _unflat(v_flat, 1)
        log_pi = _unflat(log_pi_flat, A)
        logit = _unflat(logit_flat, A)
        v_target = _unflat(v_target_flat, 1)
        log_pi_prev = _unflat(log_pi_prev_flat, A)
        log_pi_prev_ = _unflat(log_pi_prev__flat, A)

        # Fine-tuned policy used as merged_policy in v-trace
        cfg = self.config.finetune
        if cfg.from_learner_steps >= 0 and self.learner_steps > cfg.from_learner_steps:
            policy_pprocessed = _post_process_policy(pi, legal, cfg.policy_threshold)
        else:
            policy_pprocessed = pi

        # Reward regularisation signal: log(pi / pi_reg)
        log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

        num_players = self.config.num_players
        v_target_list, has_played_list, lo_list = [], [], []

        for player in range(num_players):
            reward = rewards[:, :, player]  # [T, B]
            player_others = (
                (2 * (player_id == player).float() - 1) * valid
            ).unsqueeze(-1)  # [T, B, 1]

            with torch.no_grad():
                v_t_, hp, lo_ = v_trace(
                    v_target, valid, player_id,
                    actor_policy, policy_pprocessed, log_policy_reg,
                    player_others, action_oh, reward, player,
                    eta=self.config.eta_reward_transform,
                    lambda_=1.0,
                    c=self.config.c_vtrace,
                    rho=np.inf,
                )
            v_target_list.append(v_t_)
            has_played_list.append(hp)
            lo_list.append(lo_)

        loss_v = get_loss_v([v] * num_players, v_target_list, has_played_list)

        is_correction = [torch.ones_like(valid).unsqueeze(-1)] * num_players
        loss_nerd = get_loss_nerd(
            [logit] * num_players, [pi] * num_players, lo_list,
            valid, player_id, legal, is_correction,
            clip=self.config.nerd.clip, threshold=self.config.nerd.beta,
        )

        loss = loss_v + loss_nerd

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params.parameters(), self.config.clip_gradient)
        self.optimizer.step()

        # EMA update of target network
        with torch.no_grad():
            tau = self.config.target_network_avg
            for p, pt in zip(self.params.parameters(), self.params_target.parameters()):
                pt.data.add_(tau * (p.data - pt.data))

        # Roll prev networks forward when the entropy schedule says so
        if update_target_net:
            self.params_prev_.load_state_dict(self.params_prev.state_dict())
            self.params_prev.load_state_dict(self.params_target.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Trajectory collection
    # ------------------------------------------------------------------

    def collect_batch_trajectory(self) -> Dict[str, np.ndarray]:
        T = self.config.trajectory_max
        B = self.config.batch_size
        P = self.config.num_players
        A = self.num_actions

        all_obs = np.zeros((T, B, self.obs_size), dtype=np.float32)
        all_legal = np.ones((T, B, A), dtype=np.float32)
        all_valid = np.zeros((T, B), dtype=np.float32)
        all_player_id = np.zeros((T, B), dtype=np.float32)
        all_action_oh = np.zeros((T, B, A), dtype=np.float32)
        all_policy = np.zeros((T, B, A), dtype=np.float32)
        all_rewards = np.zeros((T, B, P), dtype=np.float32)

        # Reset all environments
        current_obs = np.array([_gym_reset(env) for env in self._envs], dtype=np.float32)
        done_flags = np.zeros(B, dtype=bool)

        for t in range(T):
            legal = self._get_legal(done_flags)          # [B, A]
            player_id = self._get_player_id(done_flags)  # [B]
            valid = (~done_flags).astype(np.float32)     # [B]

            with torch.no_grad():
                obs_t = torch.tensor(current_obs, dtype=torch.float32, device=self.device)
                leg_t = torch.tensor(legal, dtype=torch.float32, device=self.device)
                pi, _, _, _ = self.params(obs_t, leg_t)
                pi_np = pi.cpu().numpy().astype(np.float64)
            pi_np /= pi_np.sum(axis=-1, keepdims=True)

            # Sample actions (only act for non-done envs)
            actions = np.zeros(B, dtype=int)
            for i in range(B):
                if not done_flags[i]:
                    actions[i] = np.random.choice(A, p=pi_np[i])

            action_oh = np.zeros((B, A), dtype=np.float32)
            action_oh[np.arange(B), actions] = 1.0

            all_obs[t] = current_obs
            all_legal[t] = legal
            all_valid[t] = valid
            all_player_id[t] = player_id
            all_action_oh[t] = action_oh
            all_policy[t] = pi_np.astype(np.float32)

            new_obs = current_obs.copy()
            for i, env in enumerate(self._envs):
                if done_flags[i]:
                    continue
                obs_new, r, done, _ = _gym_step(env, actions[i])
                self.actor_steps += 1

                # Build per-player reward vector
                r_arr = np.zeros(P, dtype=np.float32)
                if np.isscalar(r):
                    pid = int(player_id[i]) % P
                    r_arr[pid] = float(r)
                    if P == 2 and done:
                        r_arr[1 - pid] = -float(r)
                else:
                    r_arr[:] = np.asarray(r, dtype=np.float32)[:P]

                all_rewards[t, i] = r_arr
                new_obs[i] = np.asarray(obs_new, dtype=np.float32)

                if done:
                    done_flags[i] = True

            current_obs = new_obs

        return {
            "obs": all_obs,
            "legal": all_legal,
            "valid": all_valid,
            "player_id": all_player_id,
            "action_oh": all_action_oh,
            "policy": all_policy,
            "rewards": all_rewards,
        }

    # ------------------------------------------------------------------
    # Environment query helpers
    # ------------------------------------------------------------------

    def _get_legal(self, done_flags: np.ndarray) -> np.ndarray:
        legal = np.ones((len(self._envs), self.num_actions), dtype=np.float32)
        for i, env in enumerate(self._envs):
            if not done_flags[i] and hasattr(env, "legal_actions_mask"):
                legal[i] = env.legal_actions_mask()
        return legal

    def _get_player_id(self, done_flags: np.ndarray) -> np.ndarray:
        player_id = np.zeros(len(self._envs), dtype=np.float32)
        for i, env in enumerate(self._envs):
            if not done_flags[i] and hasattr(env, "current_player"):
                player_id[i] = float(env.current_player())
        return player_id
