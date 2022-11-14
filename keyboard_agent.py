# https://github.com/openai/mlsh/blob/master/gym/examples/agents/keyboard_agent.py

import gym
import numpy as np
from gym.utils.play import play
import json
import time

stateDataToDump = []

episodeData = {
            "strategy": "example", # Fill in whatever the chosen strategy is, if none, put 'N/A'
            "user": "Cam", # Fill in the user who is controlling this demonstration episode
            "data": stateDataToDump
    }

def writeFile(jsonFile):
    episodeDataToDump = json.dumps(episodeData, indent=4)
    jsonFile.write(episodeDataToDump)

    jsonFile.close()

    
jsonFile = open("epNo" + str(time.time()) + ".json", "w")


def car_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    # Write to file

    if terminated == False:

        print ("...")
        print ("Timestep")


        print (type(obs_tp1))
    
        print (obs_tp1)

        print (obs_tp1.tolist())

        stateData = {
            "obs_t": obs_t[0].tolist(),
            "obs_tp1": obs_tp1.tolist(),
            "action": action.tolist(),
            "rew": rew,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
        
        stateDataToDump.append(stateData)

    else:

        jsonFile = open("epNo" + str(time.time()) + ".json", "w")
        
        writeFile(jsonFile)

        stateDataToDump.clear()



        
        

# Wrap env in PlayableGame
play(gym.make("LunarLander-v2", render_mode="rgb_array", continuous=True), keys_to_action={
"a": np.array([-1.0, -1.0]),
"w": np.array([1.0, 0.0]),
"d": np.array([-1.0, 1.0]),
"wa": np.array([1.0, -1.0]),
"wd": np.array([1.0, 1.0]),
}, noop=np.array([0,0]), callback=car_callback)



