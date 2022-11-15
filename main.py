import gym
import numpy as np
from gym.utils.play import play
from q_learning import QLearning


def car_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    print(obs_t)


env = gym.make("LunarLander-v2", render_mode="human", continuous=False)

#play(env, noop=np.array([0,0,0]), callback=car_callback)

qlearn = QLearning(env)
qlearn.train()