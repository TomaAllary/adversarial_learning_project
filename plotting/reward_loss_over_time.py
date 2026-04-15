import matplotlib.pyplot as plt
import re

def parse_ppo_log(file_path):
    pg_losses = []
    v_losses = []
    entropies = []
    losses = []
    rewards = []

    with open(file_path, "r") as f:
        for line in f:
            if "pg_loss" in line:
                # Extract values using regex
                pg_loss = float(re.search(r"pg_loss:\s*([-0-9.eE]+)", line).group(1))
                v_loss = float(re.search(r"v_loss:\s*([-0-9.eE]+)", line).group(1))
                entropy = float(re.search(r"entropy:\s*([-0-9.eE]+)", line).group(1))
                loss = float(re.search(r"loss:\s*([-0-9.eE]+)", line).group(1))
                reward = float(re.search(r"sum_of_rewards:\s*([-0-9.eE]+)", line).group(1))

                pg_losses.append(pg_loss)
                v_losses.append(v_loss)
                entropies.append(entropy)
                losses.append(loss)
                rewards.append(reward)

    return pg_losses, v_losses, entropies, losses, rewards

def plot_losses(pg_losses, v_losses, entropies, losses):
    plt.figure()

    plt.plot(pg_losses, label="Policy Loss (pg_loss)")
    plt.plot(v_losses, label="Value Loss (v_loss)")
    plt.plot(losses, label="Total Loss")
    plt.plot(entropies, label="Entropy")

    plt.xlabel("Training Iterations")
    plt.ylabel("Value")
    plt.title("PPO Training Losses")
    plt.legend()
    plt.grid()

    plt.show()

def plot_rewards(rewards, smooth_window=10):
    plt.figure()

    # Raw rewards
    plt.plot(rewards, label="Raw Reward", alpha=0.5)

    # Smoothed rewards
    if smooth_window > 1:
        smoothed = [
            sum(rewards[max(0, i - smooth_window):i + 1]) / (i - max(0, i - smooth_window) + 1)
            for i in range(len(rewards))
        ]
        plt.plot(smoothed, label=f"Smoothed ({smooth_window})")

    plt.xlabel("Training Iterations")
    plt.ylabel("Sum of Rewards")
    plt.title("PPO Training Rewards")
    plt.legend()
    plt.grid()

    plt.show()


pg, v, ent, loss, rew = parse_ppo_log("checkpoints/PPO/PPO_20260415_1642/red_0/metrics.txt")

# plot_losses(pg, v, ent, loss)
plot_rewards(rew)