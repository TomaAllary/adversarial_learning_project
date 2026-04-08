import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# ───────────────────────────────PPO PARAMETERS───────────────────────────────
LR = 1e-3
GAMMA = 0.99
CLIP = 0.1
ENT_COEF = 0.1
VF_COEF = 0.1
BATCH_SIZE = 32
# ────────────────────────────────────────────────────────────────────────────


class PPO(nn.Module):
    def __init__(self, num_actions: int, obs_dim: int):
        super().__init__()
        self.lr, self.gamma, self.clip, self.ent_coef, self.vf_coef, self.batch_size = LR, GAMMA, CLIP, ENT_COEF, VF_COEF, BATCH_SIZE

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(obs_dim, 128)), nn.ReLU(),
            self._layer_init(nn.Linear(128, 128)), nn.ReLU(),
        )
        self.actor  = self._layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(128, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, eps=1e-5)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        probs  = Categorical(logits=self.actor(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def update(self, rb_obs, rb_actions, rb_logprobs, rb_rewards, rb_terms, rb_values, end_step):
        with torch.no_grad():
            advantages = torch.zeros_like(rb_rewards)
            for t in reversed(range(end_step)):
                delta = rb_rewards[t] + self.gamma * rb_values[t+1] * rb_terms[t+1] - rb_values[t]
                advantages[t] = delta + self.gamma ** 2 * advantages[t+1]
            returns = advantages + rb_values

        flat = lambda t: torch.flatten(t[:end_step], 0, 1)
        b_obs, b_act, b_lp, b_adv, b_ret, b_val = (
            flat(rb_obs), flat(rb_actions), flat(rb_logprobs),
            flat(advantages), flat(returns), flat(rb_values)
        )

        idx = np.arange(len(b_obs))
        for _ in range(3):
            np.random.shuffle(idx)
            for start in range(0, len(b_obs), self.batch_size):
                mb = idx[start:start+self.batch_size]
                _, newlp, entropy, value = self.get_action_and_value(b_obs[mb], b_act.long()[mb])
                ratio = (newlp - b_lp[mb]).exp()

                adv_std = b_adv[mb].std()
                if adv_std.isnan() or adv_std < 1e-6:
                    continue
                adv = (b_adv[mb] - b_adv[mb].mean()) / adv_std

                pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1-self.clip, 1+self.clip)).mean()

                value = value.flatten()
                v_clipped = b_val[mb] + (value - b_val[mb]).clamp(-self.clip, self.clip)
                v_loss = 0.5 * torch.max((value - b_ret[mb])**2, (v_clipped - b_ret[mb])**2).mean()

                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()

        return {"pg_loss": pg_loss.item(), "v_loss": v_loss.item(), "entropy": entropy.mean().item(), "loss": loss.item()}
