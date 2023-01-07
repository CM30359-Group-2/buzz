# Access the rewards stored in the specified file and plot them
# Usage: python plot.py <filename>

import sys
import matplotlib.pyplot as plt
import numpy as np

files = [
    "qlearning2023-01-06_17-17-27.txt",
    "dqn.txt",
    "ddqn.txt",
    "dqfd2023-01-07_01-00-42.txt",
]

# Crete a figure
fig = plt.figure()

# Set the figure size
fig.set_size_inches(18, 4)

# Spread the subplots out
fig.subplots_adjust(wspace=0.3)

# Remove left and right margins
fig.subplots_adjust(left=0.05, right=0.95)

# Create a subplot for each file
for i, file in enumerate(files):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.set_title(["Q-learning", "DQN", "DDQN", "DQfD"][i])

    # Read the rewards from the file
    with open(file, "r") as f:
        rewards = [float(line) for line in f.readlines()]

    # Get the 100 episode moving average
    avg_rewards = np.zeros(len(rewards))
    for j in range(len(rewards)):
        avg_rewards[j] = np.mean(rewards[max(0, j - 100) : j + 1])

    # Plot the rewards

    if (i != 0):
        ax.plot(rewards, label="Rewards")
    ax.plot(avg_rewards, label="100-episode moving average", color="orange")

    # Add axis labels
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    # Add a legend to bottom right
    ax.legend(loc="lower right")

# Save the plot to a file
plt.savefig("agg.png")
