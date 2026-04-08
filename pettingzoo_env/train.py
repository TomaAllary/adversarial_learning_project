import time
import numpy as np
import torch
from tqdm import tqdm

from pettingzoo_env.ppo import PPO
from pettingzoo_env.prisoner_env import PrisonerEnvironment

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CYCLES      = 125
TOTAL_EPISODES  = 200
VERBOSE_RATE    = 50

def batchify_obs(obs, device):
    return torch.tensor(np.stack([obs[a] for a in obs]).astype(np.float32), device=device)

def batchify(x, device):
    return torch.tensor(np.stack([x[a] for a in x]), dtype=torch.float32, device=device)

def unbatchify(x, env):
    x = x.detach().cpu().numpy()
    return {a: x[i] for i, a in enumerate(env.possible_agents)}


def train(env, agents):
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
 
    # One replay buffer per agent
    def empty_buffers():
        return {
            agent: {
                "obs":      torch.zeros((MAX_CYCLES, obs_dim), device=DEVICE),
                "actions":  torch.zeros((MAX_CYCLES,), device=DEVICE),
                "logprobs": torch.zeros((MAX_CYCLES,), device=DEVICE),
                "rewards":  torch.zeros((MAX_CYCLES,), device=DEVICE),
                "terms":    torch.zeros((MAX_CYCLES,), device=DEVICE),
                "values":   torch.zeros((MAX_CYCLES,), device=DEVICE),
            }
            for agent in env.possible_agents
        }
 
    for episode in tqdm(range(TOTAL_EPISODES)):
        next_obs, _ = env.reset(seed=None)
        rb = empty_buffers()
        total_returns = {a: 0 for a in env.possible_agents}
        end_step = 0
 
        with torch.no_grad():
            for step in range(MAX_CYCLES):
                actions, logprobs, values = {}, {}, {}
                for i, agent in enumerate(env.possible_agents):
                    obs_tensor = torch.tensor(np.array(next_obs[agent], dtype=np.float32), device=DEVICE).unsqueeze(0)
                    act, lp, _, val = agents[agent].get_action_and_value(obs_tensor)
                    actions[agent]  = act.squeeze(0)
                    logprobs[agent] = lp.squeeze(0)
                    values[agent]   = val.squeeze(0)
 
                next_obs, rewards, terms, truncs, _ = env.step({a: actions[a].item() for a in env.possible_agents})
 
                for a in env.possible_agents:
                    rb[a]["obs"][step]      = torch.tensor(np.array(next_obs[a], dtype=np.float32), device=DEVICE)
                    rb[a]["actions"][step]  = actions[a]
                    rb[a]["logprobs"][step] = logprobs[a]
                    rb[a]["rewards"][step]  = rewards[a]
                    rb[a]["terms"][step]    = float(terms[a])
                    rb[a]["values"][step]   = values[a].flatten()
                    total_returns[a]       += rewards[a]
 
                end_step = step
                if any(terms.values()) or any(truncs.values()):
                    break
 
        metrics = {}
        for a in env.possible_agents:
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
            print(f"#### Training episode {episode} ####")
            for a in env.possible_agents:
                print(f"------ Agent: {a} ------")
                print(f"Episodic Return: {total_returns[a]:.4f}")
                print(f"Episode Length: {end_step}")
                for k, v in metrics[a].items():
                    print(f"{k}: {v:.4f}")
                print("\n-----------------------\n")
 
 
if __name__ == "__main__":
    env = PrisonerEnvironment(render_mode=None, grid_size=7, cell_size=80, fps=5)
    num_actions = env.action_space(env.possible_agents[0]).n
    obs_dim     = env.observation_space(env.possible_agents[0]).shape[0]
 
    agents = {a: PPO(num_actions=num_actions, obs_dim=obs_dim).to(DEVICE) for a in env.possible_agents}
 
    train(env, agents)
 
    # Render trained policy
    env = PrisonerEnvironment(render_mode="human", grid_size=7, cell_size=80, fps=5)
    for a in agents.values():
        a.eval()
    with torch.no_grad():
        for episode in range(3):
            obs, _ = env.reset(seed=None)
            done = False
            while not done:
                actions = {}
                for a in env.possible_agents:
                    obs_tensor = torch.tensor(np.array(obs[a], dtype=np.float32), device=DEVICE).unsqueeze(0)
                    act, _, _, _ = agents[a].get_action_and_value(obs_tensor)
                    actions[a] = act.item()
                obs, _, terms, truncs, _ = env.step(actions)
                env.render()
                done = any(terms.values()) or any(truncs.values())
                if not done:
                    time.sleep(0.3)
