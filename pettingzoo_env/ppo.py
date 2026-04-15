from datetime import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# ───────────────────────────────PPO PARAMETERS───────────────────────────────
LR         = 3e-4
GAMMA      = 0.99
GAE_LAMBDA = 0.95
CLIP       = 0.2
ENT_COEF   = 0.01
VF_COEF    = 0.5
BATCH_SIZE = 64
UPDATE_EPOCHS = 4
MAX_GRAD_NORM = 0.5
# ────────────────────────────────────────────────────────────────────────────


class PPO(nn.Module):
    def __init__(self, num_actions: int, obs_dim: int, save_path, agent_name):
        super().__init__()
        self.lr = LR
        self.gamma = GAMMA
        self.gae_lambda = GAE_LAMBDA
        self.clip = CLIP
        self.ent_coef = ENT_COEF
        self.vf_coef = VF_COEF
        self.batch_size = BATCH_SIZE
        self.update_epochs = UPDATE_EPOCHS

        name = f"PPO_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self._folder = os.path.join(save_path, name, agent_name)
        os.makedirs(self._folder, exist_ok=True)

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            self._layer_init(nn.Linear(256, 256)),     nn.Tanh(),
        )
        self.actor  = self._layer_init(nn.Linear(256, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(256, 1), std=1.0)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, eps=1e-5)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        probs  = Categorical(logits=self.actor(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def update(self, rb_obs, rb_actions, rb_logprobs, rb_rewards, rb_terms, rb_values, end_step, log=False):
        """
        GAE advantage estimation + PPO update.
        All buffers are shape [MAX_CYCLES, 1, ...] — we strip the middle dim.
        end_step: the actual last filled index.
        """
        # Strip the extra dim added in train.py
        obs      = rb_obs     [:end_step].squeeze(1)   # [T, obs_dim]
        actions  = rb_actions [:end_step].squeeze(1)   # [T]
        logprobs = rb_logprobs[:end_step].squeeze(1)   # [T]
        rewards  = rb_rewards [:end_step].squeeze(1)   # [T]
        terms    = rb_terms   [:end_step].squeeze(1)   # [T]
        values   = rb_values  [:end_step].squeeze(1)   # [T]

        T = end_step

        with torch.no_grad():
            advantages = torch.zeros(T, device=obs.device)
            last_gae   = 0.0
            for t in reversed(range(T)):
                next_non_term = 1.0 - terms[t]
                next_val = values[t+1] if t + 1 < T else 0.0
                delta    = rewards[t] + self.gamma * next_val * next_non_term - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * next_non_term * last_gae
                advantages[t] = last_gae
            returns = advantages + values

        idx = np.arange(T)
        pg_loss_last = v_loss_last = entropy_last = loss_last = 0.0

        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, self.batch_size):
                mb = idx[start:start + self.batch_size]
                if len(mb) < 2:
                    continue

                _, newlp, entropy, value = self.get_action_and_value(obs[mb], actions.long()[mb])
                ratio = (newlp - logprobs[mb]).exp()

                adv = advantages[mb]
                adv_std = adv.std()
                if adv_std.isnan() or adv_std < 1e-6:
                    continue
                adv = (adv - adv.mean()) / (adv_std + 1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * ratio.clamp(1 - self.clip, 1 + self.clip)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                value = value.flatten()
                v_clipped = values[mb] + (value - values[mb]).clamp(-self.clip, self.clip)
                v_loss = 0.5 * torch.max(
                    (value - returns[mb]) ** 2,
                    (v_clipped - returns[mb]) ** 2,
                ).mean()

                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=MAX_GRAD_NORM)
                self.optimizer.step()

                pg_loss_last = pg_loss.item()
                v_loss_last  = v_loss.item()
                entropy_last = entropy.mean().item()
                loss_last    = loss.item()

        sum_of_rewards = rewards.sum().item()
        results = {
            "pg_loss": pg_loss_last,
            "v_loss":  v_loss_last,
            "entropy": entropy_last,
            "loss":    loss_last,
            "sum_of_rewards": sum_of_rewards,
        }
        if log:
            with open(os.path.join(self._folder, "metrics.txt"), "a", encoding="utf-8") as file:
                file.write(
                    f"pg_loss: {results['pg_loss']}," +
                    f"v_loss: {results['v_loss']}," +
                    f"entropy: {results['entropy']}," +
                    f"loss: {results['loss']}," +
                    f"sum_of_rewards: {results['sum_of_rewards']}\n")
                file.write("-"*50 + "\n")
        return results
    
    def save(self, ):
        full_path = os.path.join(self._folder, "model.pt")
        torch.save({
            "model_state":     self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, full_path)

        print(f"############ Saved model checkpoint to {full_path} ############")
 
    @classmethod
    def load(cls, path: str, num_actions: int, obs_dim: int, agent_name: str, device=torch.device("cpu")):
        agent = cls(
            num_actions=num_actions, 
            obs_dim=obs_dim,
            save_path="None",
            agent_name=agent_name,
        ).to(device)
        checkpoint = torch.load(path, map_location=device)
        agent.load_state_dict(checkpoint["model_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return agent
