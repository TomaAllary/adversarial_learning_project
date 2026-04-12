"""
trains 6 independent PPO agents on the 3v3 ShooterEnvironment.
"""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from pettingzoo_env.shooter_env import ShooterEnvironment, OBS_DIM
from pettingzoo_env.ppo import PPO


DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CYCLES     = 200        # must match shooter_env.MAX_STEPS
TOTAL_EPISODES = 1
VERBOSE_RATE   = 50


# ── helpers ───────────────────────────────────────────────────────────────────

def empty_buffers(agents, obs_dim, max_cycles):
    return {
        a: {
            "obs":      torch.zeros((max_cycles, obs_dim), device=DEVICE),
            "actions":  torch.zeros((max_cycles,),         device=DEVICE),
            "logprobs": torch.zeros((max_cycles,),         device=DEVICE),
            "rewards":  torch.zeros((max_cycles,),         device=DEVICE),
            "terms":    torch.zeros((max_cycles,),         device=DEVICE),
            "values":   torch.zeros((max_cycles,),         device=DEVICE),
        }
        for a in agents
    }


# ── training loop ─────────────────────────────────────────────────────────────

def train(env, agents):
    for episode in tqdm(range(TOTAL_EPISODES)):
        next_obs, _ = env.reset(seed=None)
        rb           = empty_buffers(env.possible_agents, OBS_DIM, MAX_CYCLES)
        total_ret    = {a: 0.0 for a in env.possible_agents}
        alive_agents = list(env.possible_agents)
        end_step     = 0

        with torch.no_grad():
            for step in range(MAX_CYCLES):
                actions, logprobs, values = {}, {}, {}

                for a in alive_agents:
                    obs_t = torch.tensor(
                        next_obs[a], dtype=torch.float32, device=DEVICE
                    ).unsqueeze(0)
                    act, lp, _, val = agents[a].get_action_and_value(obs_t)
                    actions[a]  = act.squeeze(0)
                    logprobs[a] = lp.squeeze(0)
                    values[a]   = val.squeeze(0)

                step_actions = {a: actions[a].item() for a in alive_agents}
                next_obs, rewards, terms, truncs, _ = env.step(step_actions)

                for a in alive_agents:
                    rb[a]["obs"][step]      = torch.tensor(next_obs[a], dtype=torch.float32, device=DEVICE)
                    rb[a]["actions"][step]  = actions[a]
                    rb[a]["logprobs"][step] = logprobs[a]
                    rb[a]["rewards"][step]  = torch.tensor(rewards[a], dtype=torch.float32, device=DEVICE)
                    rb[a]["terms"][step]    = float(terms[a])
                    rb[a]["values"][step]   = values[a].flatten()
                    total_ret[a]           += rewards[a]

                end_step = step + 1
                if any(terms.values()) or any(truncs.values()):
                    alive_agents = []   # episode done
                    break

        # PPO update for each agent
        metrics = {}
        for a in env.possible_agents:
            if end_step < 2:
                metrics[a] = {"pg_loss": 0, "v_loss": 0, "entropy": 0, "loss": 0}
                continue
            m = agents[a].update(
                rb[a]["obs"].unsqueeze(1),
                rb[a]["actions"].unsqueeze(1),
                rb[a]["logprobs"].unsqueeze(1),
                rb[a]["rewards"].unsqueeze(1),
                rb[a]["terms"].unsqueeze(1),
                rb[a]["values"].unsqueeze(1),
                end_step,
            )
            metrics[a] = m

        if episode % VERBOSE_RATE == 0:
            print(f"\n{'='*50}")
            print(f"  Episode {episode:4d}  |  steps: {end_step}")
            print(f"{'='*50}")
            for team in ("red", "blue"):
                team_ret = sum(total_ret[a] for a in env.possible_agents if team in a)
                print(f"  {team.upper()} total return: {team_ret:+.2f}")
            for a in env.possible_agents:
                print(f"  [{a}]  ret={total_ret[a]:+.2f}  "
                      f"loss={metrics[a]['loss']:.4f}  "
                      f"ent={metrics[a]['entropy']:.3f}")


# ── demo render ───────────────────────────────────────────────────────────────

def render_demo(agents, n_episodes=3):
    env = ShooterEnvironment(render_mode="human", fps=6)
    for a in agents.values():
        a.eval()

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            done   = False
            print(f"\n=== Demo episode {ep+1} ===")
            while not done:
                actions = {}
                for a in env.possible_agents:
                    obs_t = torch.tensor(obs[a], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    act, _, _, _ = agents[a].get_action_and_value(obs_t)
                    actions[a] = act.item()
                obs, _, terms, truncs, _ = env.step(actions)
                env.render()
                time.sleep(1.0 / 6)
                done = any(terms.values()) or any(truncs.values())
    env.close()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    env = ShooterEnvironment(render_mode=None)
    num_actions = env.action_space(env.possible_agents[0]).n

    agents = {
        a: PPO(num_actions=num_actions, obs_dim=OBS_DIM).to(DEVICE)
        for a in env.possible_agents
    }
    print(f"Training on {DEVICE}  |  agents: {list(agents.keys())}")

    train(env, agents)

    # Render an example
    render_demo(agents)
