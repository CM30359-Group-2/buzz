import argparse
from datetime import datetime
from operator import itemgetter
import os
import gym
import numpy as np
from agents.dqn import DQN
from agents.ddqn import DDQN
from agents.dqfd import DQFD
from agents.q_learning import QLearning

AGENTS = [
    "dqn",
    "ddqn",
    "dqfd",
    "qlearning"
]

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        prog='ReinforcementLearning',
        description='Performs reinforcement learning on the Lunar Lander environment'
    )
    parser.add_argument('-a', '--agent', dest='algorithm', choices=AGENTS, default='dqn', help='the agent you want to use')
    parser.add_argument('-e', '--episodes', required=True, type=int, default=1000, help='the number of episodes to run')
    parser.add_argument('-d', '--demos', help='path to demonstrations directory (only for DQFD)')
    parser.add_argument('-s', '--save-progress', required=False, action='store_true', help='if you want the agent to checkpoint during training')
    parser.add_argument('-c', '--checkpoint', type=str, help='the checkpoint to load when the agent plays an episode')
    parser.add_argument('-r', '--render', action='store_true', help='if you want the environment to be rendered')
    algorithm, episodes, demos_path, save_progress, checkpoint_path, render = itemgetter('algorithm', 'episodes', 'demos', 'save_progress', 'checkpoint', 'render')(vars(parser.parse_args()))

    env = gym.make('LunarLander-v2')

    rewards = []

    if algorithm == 'dqn':
        print("Running DQN")
        agent = DQN(action_space=env.action_space.n, state_space=env.observation_space.shape[0])

        # If asked to load a checkpoint, run the agent in play mode
        if (checkpoint_path != None):
            rewards = agent.play(env, episodes, checkpoint_path, render)
        else:
        # Otherwise, run the agent in train mode
            rewards = agent.train(env, episodes, checkpoint=save_progress, render=render)
    elif algorithm == 'ddqn':
        print("Running Double DQN")
        agent = DDQN(action_space=env.action_space.n, state_space=env.observation_space.shape[0])
        
        if (checkpoint_path != None):
            rewards = agent.play(env, episodes, checkpoint_path, render)
        else:
            rewards = agent.train(env, episodes, checkpoint=save_progress, render=render)
    elif algorithm == 'dqfd':
        print("Running Deep Q-learning from Demonstrations")
        agent = DQFD(action_space=env.action_space.n, state_space=env.observation_space.shape[0])
        
        if (checkpoint_path != None):
            rewards = agent.play(env, episodes, checkpoint_path, render)
        else:
            rewards = agent.train(env, episodes, demos_path, checkpoint=save_progress, render=render)
    elif algorithm == 'qlearning':
        print("Running Q-learning")
        agent = QLearning(action_space=env.action_space.n, state_space=env.observation_space.shape[0])

        if (checkpoint_path != None):
            rewards = agent.play(env, episodes, checkpoint_path, render)
        else:
            rewards = agent.train(env, episodes, checkpoint=save_progress, render=render)
    else:
        print("That is not a permitted algorithm")
        exit(1)

    #  Get unix timestamp with datetime
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    with open(os.path.join(script_dir, f'{algorithm}{timestamp}.txt'), 'w') as file:
        rewards = np.array(rewards)
        np.savetxt(file, rewards)
    print('Done')
    
