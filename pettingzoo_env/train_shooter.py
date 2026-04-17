"""
trains 6 independent PPO agents on the 3v3 ShooterEnvironment.
"""
import argparse
from datetime import datetime
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from pettingzoo_env.shooter_env import ShooterEnvironment, OBS_DIM
from pettingzoo_env.ppo import PPO
from pettingzoo_env.scripted_shooter_agent import ScriptedShooterAgent


DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CYCLES     = 1000        # must match shooter_env.MAX_STEPS
TOTAL_EPISODES = 3_000_000
MAX_TIME_MINUTES = 60 * 9
VERBOSE_RATE   = 100
SAVE_RATE      = 1000


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
def train(env, agents, fix_blue_team=False, fix_red_team=False):
    # Paths
    parent_folder = f"PPO_{datetime.now().strftime('%Y%m%d_%H%M')}"
    save_folder = {}
    for agent_name in env.possible_agents:
        save_folder[agent_name] = os.path.join("checkpoints/PPO", parent_folder, agent_name)
        os.makedirs(save_folder[agent_name], exist_ok=True)
     
    start_time = time.time()
    for episode in tqdm(range(TOTAL_EPISODES)):
        if (time.time() - start_time) > MAX_TIME_MINUTES * 60:
            print(f"\nReached max training time of {MAX_TIME_MINUTES} minutes. Ending training.")
            break
        
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
                    
                    if isinstance(agents[a], ScriptedShooterAgent):
                        act = agents[a].get_action_and_value(obs_t)
                        act = torch.Tensor([act])
                    else:
                        act, lp, _, val = agents[a].get_action_and_value(obs_t)
                        logprobs[a] = lp.squeeze(0)
                        values[a]   = val.squeeze(0)
                        
                    actions[a]  = act.squeeze(0)

                step_actions = {a: actions[a].item() for a in alive_agents}
                next_obs, rewards, terms, truncs, _ = env.step(step_actions)

                for a in alive_agents:
                    total_ret[a]           += rewards[a]
                    
                    if fix_blue_team and "blue" in a:
                        continue
                    if fix_red_team and "red" in a:
                        continue
                    
                    rb[a]["obs"][step]      = torch.tensor(next_obs[a], dtype=torch.float32, device=DEVICE)
                    rb[a]["actions"][step]  = actions[a]
                    rb[a]["logprobs"][step] = logprobs[a]
                    rb[a]["rewards"][step]  = torch.tensor(rewards[a], dtype=torch.float32, device=DEVICE)
                    rb[a]["terms"][step]    = float(terms[a])
                    rb[a]["values"][step]   = values[a].flatten()

                end_step = step + 1
                if any(terms.values()) or any(truncs.values()):
                    alive_agents = []   # episode done
                    break

        # PPO update for each agent
        metrics = {}
        for a in env.possible_agents:
            
            if fix_blue_team and "blue" in a:
                continue
            if fix_red_team and "red" in a:
                continue
                
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
                current_ep=episode,
                log = episode % VERBOSE_RATE == 0
            )
            metrics[a] = m

        if episode % VERBOSE_RATE == 0:
            for a in env.possible_agents:
                if not isinstance(agents[a], PPO):
                    continue

                with open(os.path.join(save_folder[a], "metrics.txt"), "a", encoding="utf-8") as file:
                    file.write(
                        f"## EPISODE {episode} ## " +
                        f"pg_loss: {metrics[a]['pg_loss']}," +
                        f"v_loss: {metrics[a]['v_loss']}," +
                        f"entropy: {metrics[a]['entropy']}," +
                        f"loss: {metrics[a]['loss']}," +
                        f"sum_of_rewards: {metrics[a]['sum_of_rewards']}\n")
                    file.write("-"*50 + "\n")

            print(f"\n{'='*50}")
            print(f"  Episode {episode:4d}  |  steps: {end_step}")
            print(f"{'='*50}")
            for team in ("red", "blue"):
                team_ret = sum(total_ret[a] for a in env.possible_agents if team in a)
                print(f"  {team.upper()} total return: {team_ret:+.2f}")
            for a in env.possible_agents:
                    print(f"  [{a}]  ret={total_ret[a]:+.2f}  ")
                    
        if episode % (SAVE_RATE) == 0:
            for a in env.possible_agents:
                if isinstance(agents[a], PPO):
                    agents[a].save(save_folder[a])


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
                    if isinstance(agents[a], ScriptedShooterAgent):
                        actions[a] = agents[a].get_action_and_value(obs_t)
                    else:
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
        env.possible_agents[i]: PPO.load("checkpoints/PPO/PPO_20260416_1047/red_0/model.pt", num_actions=num_actions, obs_dim=OBS_DIM, device=DEVICE)
        # env.possible_agents[i]: PPO(num_actions=num_actions, obs_dim=OBS_DIM).to(DEVICE)
        for i in range(int(len(env.possible_agents)  /2))
    }
    for i in range(int(len(env.possible_agents) / 2), len(env.possible_agents)):
       agents[env.possible_agents[i]] = ScriptedShooterAgent(num_agents=len(env.possible_agents))
        
    # print(f"Training on {DEVICE}  |  agents: {list(agents.keys())}")

    # train(env, agents, fix_blue_team=True)


    # Render an example
    render_demo(agents)
