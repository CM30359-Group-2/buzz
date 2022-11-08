import gym
import numpy as np
from gym.utils.play import play

def car_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    print(obs_t)


env = gym.make("LunarLander-v2", render_mode="rgb_array")
play(env, noop=np.array([0,0,0]), callback=car_callback)
