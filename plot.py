# Access the rewards stored in the specified file and plot them
# Usage: python plot.py <filename>

import sys
import matplotlib.pyplot as plt
import numpy as np

# Read the rewards from the specified file
with open(sys.argv[1], 'r') as file:
    rewards = np.loadtxt(file)

# Get 100-episode running average
avg_rewards = np.zeros(len(rewards))
for i in range(len(rewards)):
    avg_rewards[i] = np.mean(rewards[max(0, i - 100):i + 1])

# Plot the rewards
plt.plot(rewards, label='Rewards')
plt.plot(avg_rewards, label='100-episode average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()

# Save the plot - output file name is specified as the first argument without the extension
plt.savefig(sys.argv[1].split('.')[0] + '.png')