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
MAX_CYCLES     = 200        # must match shooter_env.MAX_STEPS
TOTAL_EPISODES = 3_000_000
MAX_TIME_MINUTES = 60 * 15
VERBOSE_RATE   = 100
SAVE_RATE      = 1000
N_ENV          = 10
ROLLOUTS_PER_UPDATE = 20

# ── helpers ───────────────────────────────────────────────────────────────────
def empty_buffers(agents, obs_dim, max_cycles, n_rollouts):
    total = max_cycles * n_rollouts
    return {
        a: {
            "obs":      torch.zeros((total, obs_dim), device=DEVICE),
            "actions":  torch.zeros((total,), dtype=torch.long, device=DEVICE),
            "logprobs": torch.zeros((total,),         device=DEVICE),
            "rewards":  torch.zeros((total,),         device=DEVICE),
            "terms":    torch.zeros((total,),         device=DEVICE),
            "values":   torch.zeros((total,),         device=DEVICE),
        }
        for a in agents
    }


# ── training loop ─────────────────────────────────────────────────────────────
def train(env, agents, fix_blue_team=False, fix_red_team=False):
    should_verbose = False
    should_save = False

    # Paths
    parent_folder = f"PPO_{datetime.now().strftime('%Y%m%d_%H%M')}"
    save_folder = {}
    for agent_name in env.possible_agents:
        save_folder[agent_name] = os.path.join("checkpoints/PPO", parent_folder, agent_name)
        os.makedirs(save_folder[agent_name], exist_ok=True)
     
    start_time = time.time()
    episode = 0

    for episode in tqdm(range(TOTAL_EPISODES)):
        if (time.time() - start_time) > MAX_TIME_MINUTES * 60:
            print(f"\nReached max training time of {MAX_TIME_MINUTES} minutes. Ending training.")
            break

        # ── Reset accumulators every N rollouts ──────────────────────────────
        if episode % ROLLOUTS_PER_UPDATE == 0:
            rb = empty_buffers(env.possible_agents, OBS_DIM, MAX_CYCLES, ROLLOUTS_PER_UPDATE)
            global_step = 0  # write pointer into the big buffer

        next_obs, _ = env.reset(seed=None)
        total_ret    = {a: 0.0 for a in env.possible_agents}
        alive_agents = list(env.possible_agents)
        episode_steps = 0
        last_obs     = next_obs   # fallback for bootstrap
        last_truncs  = {}

        with torch.no_grad():
            for step in range(MAX_CYCLES):
                buf_idx = global_step + step   # absolute position in big buffer
                actions, logprobs, values = {}, {}, {}

                for a in alive_agents:
                    obs_t = torch.tensor(next_obs[a], dtype=torch.float32, device=DEVICE).unsqueeze(0)

                    if isinstance(agents[a], ScriptedShooterAgent):
                        act = torch.tensor([agents[a].get_action_and_value(obs_t)])
                    else:
                        act, lp, _, val = agents[a].get_action_and_value(obs_t)
                        logprobs[a] = lp.squeeze(0)
                        values[a]   = val.squeeze(0)

                    actions[a] = act.squeeze(0)

                step_actions = {a: actions[a].item() for a in alive_agents}
                next_obs, rewards, terms, truncs, _ = env.step(step_actions)
                last_obs   = next_obs
                last_truncs = truncs

                for a in alive_agents:
                    total_ret[a] += rewards[a]

                    if (fix_blue_team and "blue" in a) or (fix_red_team and "red" in a):
                        continue
                    if isinstance(agents[a], ScriptedShooterAgent):
                        continue

                    rb[a]["obs"][buf_idx]      = torch.tensor(next_obs[a], dtype=torch.float32, device=DEVICE)
                    rb[a]["actions"][buf_idx]  = actions[a]
                    rb[a]["logprobs"][buf_idx] = logprobs[a]
                    rb[a]["rewards"][buf_idx]  = rewards[a]
                    rb[a]["terms"][buf_idx]    = float(terms[a])
                    rb[a]["values"][buf_idx]   = values[a].flatten()

                episode_steps = step + 1
                if any(terms.values()) or any(truncs.values()):
                    alive_agents = []
                    break

        global_step += episode_steps

        if episode % VERBOSE_RATE == 0:
            should_verbose = True
        if episode % (SAVE_RATE) == 0:
            should_save = True

        # ── Only update every ROLLOUTS_PER_UPDATE episodes ──────────────────
        is_update_episode = (episode % ROLLOUTS_PER_UPDATE == ROLLOUTS_PER_UPDATE - 1)
        if not is_update_episode:
            continue

        total_steps = global_step  # how many timesteps are actually filled

        metrics = {}
        for a in env.possible_agents:
            if (fix_blue_team and "blue" in a) or (fix_red_team and "red" in a):
                continue
            if isinstance(agents[a], ScriptedShooterAgent):
                continue
            if total_steps < 2:
                metrics[a] = {"pg_loss": 0, "v_loss": 0, "entropy": 0, "loss": 0, "sum_of_rewards": 0}
                continue

            with torch.no_grad():
                last_obs_t = torch.tensor(last_obs[a], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                was_truncated = any(last_truncs.values())
                bootstrap_value = agents[a].get_value(last_obs_t).squeeze() if was_truncated else torch.tensor(0.0)

            # Pass flat buffers — PPO.update() expects [T, 1, ...] so unsqueeze
            m = agents[a].update(
                rb[a]["obs"][:total_steps].unsqueeze(1),
                rb[a]["actions"][:total_steps].unsqueeze(1),
                rb[a]["logprobs"][:total_steps].unsqueeze(1),
                rb[a]["rewards"][:total_steps].unsqueeze(1),
                rb[a]["terms"][:total_steps].unsqueeze(1),
                rb[a]["values"][:total_steps].unsqueeze(1),
                total_steps,
                bootstrap_value=bootstrap_value,
            )
            metrics[a] = m

        if should_verbose:
            should_verbose = False
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
            print(f"  Episode {episode:4d}")
            print(f"{'='*50}")
            for team in ("red", "blue"):
                team_ret = sum(total_ret[a] for a in env.possible_agents if team in a)
                print(f"  {team.upper()} total return: {team_ret:+.2f}")
            for a in env.possible_agents:
                    print(f"  [{a}]  ret={total_ret[a]:+.2f}  ")
                    
        if should_save:
            should_save = False
            for a in env.possible_agents:
                if isinstance(agents[a], PPO):
                    agents[a].save(os.path.join(save_folder[a], f"model_ep_{episode}.pt"))


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
        # env.possible_agents[i]: PPO.load("checkpoints/PPO/PPO_20260422_1424/red_0/model_ep_9019.pt", num_actions=num_actions, obs_dim=OBS_DIM, device=DEVICE)
        env.possible_agents[i]: PPO(num_actions=num_actions, obs_dim=OBS_DIM).to(DEVICE)
        for i in range(int(len(env.possible_agents)  /2))
    }
    for i in range(int(len(env.possible_agents) / 2), len(env.possible_agents)):
       agents[env.possible_agents[i]] = ScriptedShooterAgent(num_agents=len(env.possible_agents))
        
    print(f"Training on {DEVICE}  |  agents: {list(agents.keys())}")
    train(env, agents, fix_blue_team=True)

    # Render an example
    # render_demo(agents, n_episodes=10)
