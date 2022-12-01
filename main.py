import gym
from q_learning import QLearning

env = gym.make("LunarLander-v2", continuous=False)

agent = QLearning(env)
agent.train()
