import numpy as np

class QLearning:
    def __init__(self, env, alpha, gamma, epsilon):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = {}
        for state in self.env.observation_space:
            for action in self.env.action_space:
                self.Q[(state, action)] = 0

    def train(self):
        for episode in range(1000):
            S = self.env.reset()
            total_reward = 0

            # for _ in range():


# https://github.com/lazavgeridis/LunarLander-v2/blob/main/rl_landers.py

                    