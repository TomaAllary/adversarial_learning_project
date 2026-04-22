import matplotlib.pyplot as plt
import re

def parse_ppo_log(file_path):
    eps = []
    pg_losses = []
    v_losses = []
    entropies = []
    losses = []
    rewards = []

    with open(file_path, "r") as f:
        for line in f:
            if "pg_loss" in line:
                # Extract values using regex

                ## EPISODE 19 ## 
                ep = int(re.search(r"EPISODE\s*([-0-9.eE]+)", line).group(1))
                pg_loss = float(re.search(r"pg_loss:\s*([-0-9.eE]+)", line).group(1))
                v_loss = float(re.search(r"v_loss:\s*([-0-9.eE]+)", line).group(1))
                entropy = float(re.search(r"entropy:\s*([-0-9.eE]+)", line).group(1))
                loss = float(re.search(r"loss:\s*([-0-9.eE]+)", line).group(1))
                reward = float(re.search(r"sum_of_rewards:\s*([-0-9.eE]+)", line).group(1))

                eps.append(ep)
                pg_losses.append(pg_loss)
                v_losses.append(v_loss)
                entropies.append(entropy)
                losses.append(loss)
                rewards.append(reward)

    return eps, pg_losses, v_losses, entropies, losses, rewards

def plot_losses(eps, pg_losses, v_losses, entropies, losses):
    plt.figure()

    plt.xlabel("Training Iterations")
    plt.ylabel("Value")
    plt.title("PPO Training Losses")
    plt.legend()
    plt.grid()

    plt.plot(eps, pg_losses, label="Policy Loss (pg_loss)")
    plt.show()
    plt.plot(eps, v_losses, label="Value Loss (v_loss)")
    plt.show()
    plt.plot(eps, losses, label="Total Loss")
    plt.show()
    plt.plot(eps, entropies, label="Entropy")
    plt.show()


def plot_rewards(eps, rewards, smooth_window=10):
    plt.figure()

    # Raw rewards
    plt.plot(eps, rewards, label="Raw Reward", alpha=0.5)

    # Smoothed rewards
    if smooth_window > 1:
        smoothed = [
            sum(rewards[max(0, i - smooth_window):i + 1]) / (i - max(0, i - smooth_window) + 1)
            for i in range(len(rewards))
        ]
        plt.plot(eps, smoothed, label=f"Smoothed ({smooth_window})")

    plt.xlabel("Training Iterations")
    plt.ylabel("Sum of Rewards")
    plt.title("PPO Training Rewards")
    plt.legend()
    plt.grid()

    plt.show()


eps, pg, v, ent, loss, rew = parse_ppo_log("checkpoints/PPO/low_ent/PPO_20260422_1551/red_0/metrics.txt")

plot_losses(eps, pg, v, ent, loss)
plot_rewards(eps, rew)