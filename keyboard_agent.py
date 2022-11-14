# https://github.com/openai/mlsh/blob/master/gym/examples/agents/keyboard_agent.py

import gym
import numpy as np
from gym.utils.play import play

# Wrap env in PlayableGame
play(gym.make("LunarLander-v2", render_mode="rgb_array", continuous=True), keys_to_action={
"a": np.array([-1.0, -1.0]),
"w": np.array([1.0, 0.0]),
"d": np.array([-1.0, 1.0]),
"wa": np.array([1.0, -1.0]),
"wd": np.array([1.0, 1.0]),
}, noop=np.array([0,0]))
