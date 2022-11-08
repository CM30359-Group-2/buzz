# https://github.com/openai/mlsh/blob/master/gym/examples/agents/keyboard_agent.py

import gym
import numpy as np
from gym.utils.play import play


# Open file
f = open("data.txt", "w")

def car_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    # Write to file
    f.write(str(obs_t) + " " + str(obs_tp1) + " " + str(action) + " " + str(rew) + " " + str(terminated) + " " + str(truncated) + " " + str(info))

# Wrap env in PlayableGame
play(gym.make("LunarLander-v2", render_mode="rgb_array", continuous=True), keys_to_action={
"a": np.array([-1.0, -1.0]),
"w": np.array([1.0, 0.0]),
"d": np.array([-1.0, 1.0]),
"wa": np.array([1.0, -1.0]),
"wd": np.array([1.0, 1.0]),
}, noop=np.array([0,0]), callback=car_callback)
