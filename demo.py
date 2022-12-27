# https://github.com/openai/mlsh/blob/master/gym/examples/agents/keyboard_agent.py

import gym
import numpy as np
from gym.utils.play import play
import json
import time

transitions = []
filename = "demos/epNo" + str(time.time()) + ".json"


def writeFile(jsonFile,):
    dumped_transitions = json.dumps({
        "strategy": "example",  # Fill in whatever the chosen strategy is, if none, put 'N/A'
        "user": "Q",  # Fill in the user who is controlling this demonstration episode
        "data": transitions
    }, indent=4)
    jsonFile.write(dumped_transitions)

    jsonFile.close()


def game_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global filename
    global transitions

    # Write to file

    stateData = {
        "obs_t": obs_t.tolist(),
        "obs_tp1": obs_tp1.tolist(),
        "action": action.tolist(),
        "rew": rew,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }

    transitions.append(stateData)

    if terminated and len(transitions) > 0:
        jsonFile = open(filename, "w")
        writeFile(jsonFile)
        transitions = []
        filename = "demos/epNo" + str(time.time()) + ".json"
