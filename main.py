import argparse
import gym
from agents.dqfd import dqfd
from agents.dqn import dqn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ReinforcementLearning',
        description='Performs reinforcement learning on the Lunar Lander environment'
    )
    parser.add_argument('-a', '--agent', default='dqn')
    args = vars(parser.parse_args())
    agent = args['agent']
    env = gym.make('LunarLander-v2')

    if agent == 'dqn':
        dqn(env)
    elif agent == 'dqfd':
        dqfd(env)
